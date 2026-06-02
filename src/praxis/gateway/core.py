"""Core dispatch logic — shared by /v1/chat and the OpenAI-compat shim."""
import time
from dataclasses import dataclass

from praxis.gateway import db
from praxis.gateway.providers import ProviderError
from praxis.gateway.router import Router, backoff_for


@dataclass
class DispatchResult:
    provider: str
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    attempted: list[dict]


class DispatchFailed(Exception):
    def __init__(self, attempts: list[dict], last_error: str | None):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"all providers unavailable. attempts: {attempts}. last_error: {last_error}")


def estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    chars = sum(len(m.get("content") or "") for m in messages)
    return chars // 4 + max_tokens


def attempts_str(attempts: list[dict]) -> str:
    return "; ".join(f"{a['provider']}:{a['reason']}" for a in attempts)


async def dispatch_chat(
    router: Router,
    messages: list[dict],
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> DispatchResult:
    """Run a chat call with failover. Raises DispatchFailed if no provider works."""
    prompt_text = "".join(m.get("content") or "" for m in messages)
    est = estimate_tokens(messages, max_tokens)
    explicit = bool(provider_override)

    candidates = router.candidates(provider_override) if provider_override else list(router.order)
    if provider_override and not candidates:
        raise DispatchFailed([], f"unknown provider '{provider_override}'")

    all_attempts: list[dict] = []
    last_err: str | None = None

    for _ in range(len(candidates) + 1):
        name, atts = router.pick(est, candidates)
        all_attempts.extend(atts)
        if name is None:
            break

        provider = router.providers[name]
        t0 = time.time()
        router.state[name].record()

        try:
            result = await provider.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model_override,
            )
            latency = int((time.time() - t0) * 1000)
            tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
            router.state[name].record_tokens(tokens)
            db.log_call(
                provider=name, model=result["model"],
                input_tokens=result["input_tokens"], output_tokens=result["output_tokens"],
                latency_ms=latency, status="ok",
                prompt_chars=len(prompt_text), response_chars=len(result["text"]),
                override=provider_override, attempted=attempts_str(all_attempts),
            )
            return DispatchResult(
                provider=name,
                model=result["model"],
                text=result["text"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                latency_ms=latency,
                attempted=all_attempts,
            )

        except Exception as e:
            last_err = str(e)
            secs, reason = backoff_for(e)
            if secs > 0:
                router.state[name].mark_unavailable(secs, reason)
            db.log_call(
                provider=name, model=model_override or provider.model,
                status="error", error=str(e)[:500],
                latency_ms=int((time.time() - t0) * 1000),
                prompt_chars=len(prompt_text),
                override=provider_override, attempted=attempts_str(all_attempts),
            )
            tag = f"failed: {str(e)[:100]}"
            if secs > 0:
                tag += f" → backoff {secs:.0f}s ({reason})"
            all_attempts.append({"provider": name, "reason": tag})

            retryable = getattr(e, "retryable", True) if isinstance(e, ProviderError) else True
            if explicit or not retryable:
                raise DispatchFailed(all_attempts, last_err)
            candidates = [c for c in candidates if c != name]

    raise DispatchFailed(all_attempts, last_err)
