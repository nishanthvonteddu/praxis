"""Provider adapters — each speaks a vendor's native HTTP API, returns a uniform shape."""
import json
from typing import AsyncIterator

import httpx

from praxis.config import settings


class ProviderError(Exception):
    def __init__(self, msg: str, status: int | None = None, retryable: bool = True):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


class BaseProvider:
    name: str = ""

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    async def chat(self, messages, max_tokens=2048, temperature=0.7, model=None) -> dict:
        raise NotImplementedError

    async def stream(self, messages, max_tokens=2048, temperature=0.7, model=None) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # pragma: no cover


class OpenAICompat(BaseProvider):
    """Anything that speaks OpenAI's /chat/completions schema."""

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, messages, max_tokens, temperature, model, stream) -> dict:
        return {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

    async def chat(self, messages, max_tokens=2048, temperature=0.7, model=None) -> dict:
        body = self._body(messages, max_tokens, temperature, model, False)
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
            if r.status_code != 200:
                raise ProviderError(
                    f"{self.name} HTTP {r.status_code}: {r.text[:300]}",
                    status=r.status_code,
                    retryable=r.status_code not in (400, 401),
                )
            d = r.json()
            usage = d.get("usage") or {}
            return {
                "text": d["choices"][0]["message"]["content"] or "",
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "model": body["model"],
            }

    async def stream(self, messages, max_tokens=2048, temperature=0.7, model=None):
        body = self._body(messages, max_tokens, temperature, model, True)
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", f"{self.base_url}/chat/completions",
                                headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "ignore")[:300]
                    raise ProviderError(f"{self.name} HTTP {r.status_code}: {text}", status=r.status_code)
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        return
                    try:
                        d = json.loads(payload)
                        delta = d["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue


class Groq(OpenAICompat):
    name = "groq"
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://api.groq.com/openai/v1")


class Cerebras(OpenAICompat):
    name = "cerebras"
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://api.cerebras.ai/v1")


class NVIDIA(OpenAICompat):
    name = "nvidia"
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://integrate.api.nvidia.com/v1")


class OpenRouter(OpenAICompat):
    name = "openrouter"
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://openrouter.ai/api/v1")

    def _headers(self) -> dict:
        h = super()._headers()
        h["HTTP-Referer"] = "http://localhost"
        h["X-Title"] = "Praxis"
        return h


class GitHubModels(OpenAICompat):
    name = "github"
    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://models.github.ai/inference")


class Gemini(BaseProvider):
    name = "gemini"

    def __init__(self, api_key, model):
        super().__init__(api_key, model, "https://generativelanguage.googleapis.com/v1beta")

    def _convert(self, messages) -> dict:
        contents, system = [], None
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                system = (system or "") + ("\n" if system else "") + content
            else:
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": content}],
                })
        body = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    async def chat(self, messages, max_tokens=2048, temperature=0.7, model=None) -> dict:
        m = model or self.model
        body = self._convert(messages)
        body["generationConfig"] = {"maxOutputTokens": max_tokens, "temperature": temperature}
        url = f"{self.base_url}/models/{m}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(url, json=body)
            if r.status_code != 200:
                raise ProviderError(
                    f"gemini HTTP {r.status_code}: {r.text[:300]}",
                    status=r.status_code,
                    retryable=r.status_code not in (400, 401),
                )
            d = r.json()
            cands = d.get("candidates") or []
            if not cands:
                raise ProviderError(f"gemini no candidates: {json.dumps(d)[:200]}", status=200, retryable=True)
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            usage = d.get("usageMetadata") or {}
            return {
                "text": text,
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "model": m,
            }

    async def stream(self, messages, max_tokens=2048, temperature=0.7, model=None):
        m = model or self.model
        body = self._convert(messages)
        body["generationConfig"] = {"maxOutputTokens": max_tokens, "temperature": temperature}
        url = f"{self.base_url}/models/{m}:streamGenerateContent?alt=sse&key={self.api_key}"
        async with httpx.AsyncClient(timeout=180) as c:
            async with c.stream("POST", url, json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "ignore")[:300]
                    raise ProviderError(f"gemini HTTP {r.status_code}: {text}", status=r.status_code)
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        d = json.loads(line[6:])
                        for cand in d.get("candidates") or []:
                            for p in cand.get("content", {}).get("parts", []):
                                t = p.get("text", "")
                                if t:
                                    yield t
                    except Exception:
                        continue


class Ollama(BaseProvider):
    name = "ollama"

    def __init__(self, model, base_url="http://localhost:11434"):
        super().__init__("", model, base_url)

    async def chat(self, messages, max_tokens=2048, temperature=0.7, model=None) -> dict:
        m = model or self.model
        body = {
            "model": m,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(f"{self.base_url}/api/chat", json=body)
            if r.status_code != 200:
                raise ProviderError(f"ollama HTTP {r.status_code}: {r.text[:300]}", status=r.status_code)
            d = r.json()
            return {
                "text": d.get("message", {}).get("content", ""),
                "input_tokens": d.get("prompt_eval_count", 0),
                "output_tokens": d.get("eval_count", 0),
                "model": m,
            }

    async def stream(self, messages, max_tokens=2048, temperature=0.7, model=None):
        m = model or self.model
        body = {
            "model": m,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=600) as c:
            async with c.stream("POST", f"{self.base_url}/api/chat", json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode("utf-8", "ignore")[:300]
                    raise ProviderError(f"ollama HTTP {r.status_code}: {text}", status=r.status_code)
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        t = d.get("message", {}).get("content", "")
                        if t:
                            yield t
                        if d.get("done"):
                            return
                    except Exception:
                        continue


def build_providers() -> dict[str, BaseProvider]:
    out: dict[str, BaseProvider] = {}
    s = settings
    if s.gemini_api_key:
        out["gemini"] = Gemini(s.gemini_api_key, s.gemini_model)
    if s.nvidia_api_key:
        out["nvidia"] = NVIDIA(s.nvidia_api_key, s.nvidia_model)
    if s.groq_api_key:
        out["groq"] = Groq(s.groq_api_key, s.groq_model)
    if s.cerebras_api_key:
        out["cerebras"] = Cerebras(s.cerebras_api_key, s.cerebras_model)
    if s.open_router_api_key:
        out["openrouter"] = OpenRouter(s.open_router_api_key, s.openrouter_model)
    if s.github_access_token:
        out["github"] = GitHubModels(s.github_access_token, s.github_model)
    if s.ollama_model:
        out["ollama"] = Ollama(s.ollama_model, s.ollama_url)
    return out
