"""Rate-limit-aware router. Picks the first eligible provider for each call."""
import asyncio
import time
from collections import defaultdict, deque

from praxis.gateway.providers import BaseProvider


# Free-tier limits per provider. Values tagged "measured" were read live from the
# provider's x-ratelimit-* response headers on the configured keys (2026-06); the
# rest are the providers' published free-tier figures or conservative estimates.
# Every entry MUST keep: rpm, rpd, tpm, cooldown, max_ctx (tokens_per_day optional).
# NOTE: only per-minute and per-day windows are enforced (RateState has no hourly
# bucket), so any per-hour cap is approximated by the matching rpm/cooldown pacing.
LIMITS: dict[str, dict] = {
    "ollama":     {"rpm": 9999, "rpd": 9_999_999, "tpm": 99_999_999, "cooldown": 0,   "max_ctx": 32_000},
    # cerebras measured: 5 rpm / 150 rph / 2400 rpd, 30k tpm, 1M tpd. cooldown 12s ≈ 5/min pacing.
    "cerebras":   {"rpm": 5,    "rpd": 2400,      "tpm": 30_000,     "cooldown": 12,  "max_ctx": 8_000,    "tokens_per_day": 1_000_000},
    # groq: rpd + tpm measured (1000 rpd, 12k tpm); rpm 30 + tpd 100k from published free tier.
    "groq":       {"rpm": 30,   "rpd": 1000,      "tpm": 12_000,     "cooldown": 2,   "max_ctx": 100_000,  "tokens_per_day": 100_000},
    # nvidia: no rate-limit headers exposed — kept as conservative estimate.
    "nvidia":     {"rpm": 40,   "rpd": 9999,      "tpm": 100_000,    "cooldown": 2,   "max_ctx": 100_000},
    # gemini-2.5-flash-lite free tier (published); context window confirmed 1,048,576 in.
    "gemini":     {"rpm": 15,   "rpd": 1000,      "tpm": 250_000,    "cooldown": 4,   "max_ctx": 1_000_000},
    # openrouter :free models — key confirmed free tier; 20 rpm / 50 rpd policy.
    "openrouter": {"rpm": 20,   "rpd": 50,        "tpm": 99_999_999, "cooldown": 3,   "max_ctx": 100_000},
    # github models — headers report high caps but published free tier is far lower; kept conservative.
    "github":     {"rpm": 10,   "rpd": 50,        "tpm": 99_999_999, "cooldown": 6,   "max_ctx": 8_000},
}

SHORTCUTS: dict[str, str] = {
    "g": "gemini", "gem": "gemini", "gemini": "gemini",
    "n": "nvidia", "nv": "nvidia", "nvidia": "nvidia",
    "o": "ollama", "oll": "ollama", "ollama": "ollama",
    "gr": "groq", "groq": "groq",
    "c": "cerebras", "cer": "cerebras", "cerebras": "cerebras",
    "or": "openrouter", "opr": "openrouter", "openrouter": "openrouter",
    "gh": "github", "ghb": "github", "github": "github",
}


def resolve(name: str | None) -> str | None:
    if not name:
        return None
    return SHORTCUTS.get(name.lower())


class RateState:
    def __init__(self):
        self.calls_minute: deque[float] = deque()
        self.tokens_minute: deque[tuple[float, int]] = deque()
        self.calls_today = 0
        self.tokens_today = 0
        self.day_start = self._day_start()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    @staticmethod
    def _day_start() -> float:
        now = time.time()
        return now - (now % 86400)

    def gc(self) -> None:
        now = time.time()
        if now - self.day_start >= 86400:
            self.calls_today = 0
            self.tokens_today = 0
            self.day_start = self._day_start()
        cutoff = now - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()
        while self.tokens_minute and self.tokens_minute[0][0] < cutoff:
            self.tokens_minute.popleft()

    def can_use(self, limits: dict, est_tokens: int = 0) -> tuple[bool, str | None]:
        self.gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        wait = limits["cooldown"] - (now - self.last_call)
        if wait > 0:
            return False, f"cooldown ({wait:.1f}s)"
        if len(self.calls_minute) >= limits["rpm"]:
            return False, "RPM limit"
        if self.calls_today >= limits["rpd"]:
            return False, "RPD limit"
        tpm = sum(t for _, t in self.tokens_minute)
        if tpm + est_tokens > limits["tpm"]:
            return False, "TPM limit"
        if "tokens_per_day" in limits and self.tokens_today + est_tokens > limits["tokens_per_day"]:
            return False, "daily token cap"
        return True, None

    def record(self) -> None:
        now = time.time()
        self.calls_minute.append(now)
        self.calls_today += 1
        self.last_call = now

    def record_tokens(self, tokens: int) -> None:
        now = time.time()
        self.tokens_minute.append((now, tokens))
        self.tokens_today += tokens

    def mark_unavailable(self, seconds: float, reason: str) -> None:
        self.unavailable_until = time.time() + seconds
        self.unavailable_reason = reason

    def snapshot(self, limits: dict) -> dict:
        self.gc()
        now = time.time()
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": limits["rpm"],
            "rpd_used": self.calls_today,
            "rpd_limit": limits["rpd"],
            "tpm_used": sum(t for _, t in self.tokens_minute),
            "tpm_limit": limits["tpm"],
            "tokens_today": self.tokens_today,
            "tokens_per_day": limits.get("tokens_per_day"),
            "cooldown_remaining": max(0, limits["cooldown"] - (now - self.last_call)) if self.last_call else 0,
            "last_call": self.last_call,
            "backoff_remaining": max(0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
        }


class Router:
    def __init__(self, providers: dict[str, BaseProvider], order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self.state: dict[str, RateState] = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def candidates(self, override: str | None = None) -> list[str]:
        if override:
            r = resolve(override)
            return [r] if r and r in self.providers else []
        return list(self.order)

    def pick(self, est_tokens: int, candidates: list[str]) -> tuple[str | None, list[dict]]:
        attempts: list[dict] = []
        for name in candidates:
            limits = LIMITS[name]
            if est_tokens > limits["max_ctx"]:
                attempts.append({"provider": name, "reason": f"prompt {est_tokens} > max_ctx {limits['max_ctx']}"})
                continue
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why or "ineligible"})
        return None, attempts

    def status(self) -> dict[str, dict]:
        out = {}
        for name in self.providers:
            snap = self.state[name].snapshot(LIMITS[name])
            snap["model"] = self.providers[name].model
            out[name] = snap
        return out


def backoff_for(err: Exception) -> tuple[float, str]:
    """Returns (seconds, reason) — how long to lock a provider after this error."""
    msg = str(err).lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "queue_exceeded" in msg or "high traffic" in msg or "queue" in msg:
            return 15, "server queue full"
        if "quota" in msg or "rpm" in msg or "per minute" in msg:
            return 60, "RPM quota burned"
        if "rpd" in msg or "per day" in msg or "daily" in msg:
            return 3600, "RPD quota burned"
        return 30, "rate limited"
    if status and 500 <= status < 600:
        return 20, f"upstream {status}"
    if status == 408 or "timeout" in msg or "timed out" in msg:
        return 10, "timeout"
    if status in (401, 403):
        return 600, "auth error"
    return 0, ""
