"""HTTP endpoints: native /v1/chat + OpenAI-compatible /v1/openai/chat/completions."""
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from praxis.gateway import db
from praxis.gateway.core import DispatchFailed, dispatch_chat
from praxis.gateway.router import LIMITS, SHORTCUTS, resolve
from praxis.gateway.schemas import ChatRequest, OAIChatRequest


router = APIRouter()


def _normalize(req: ChatRequest) -> list[dict]:
    if req.messages:
        return [dict(m) for m in req.messages]
    msgs = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    msgs.append({"role": "user", "content": req.prompt or ""})
    return msgs


# ---------- Native endpoint ----------

@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    rt = request.app.state.router
    messages = _normalize(req)
    try:
        result = await dispatch_chat(
            rt, messages,
            provider_override=req.provider,
            model_override=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except DispatchFailed as e:
        raise HTTPException(503, f"all providers unavailable. attempts={e.attempts} last_error={e.last_error}")
    return {
        "provider": result.provider,
        "model": result.model,
        "text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_ms": result.latency_ms,
        "attempted": result.attempted,
    }


# ---------- OpenAI-compat shim ----------
# Translates an OpenAI /chat/completions request into our internal dispatch,
# and back. Lets Pydantic AI (and Cursor, Continue, LangChain, etc.) talk to
# us with zero custom code on their side.

def _coerce_content(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # vision-style; just stringify text parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c) if c is not None else ""


def _resolve_provider(model: str) -> tuple[str | None, str | None]:
    """Map an OpenAI 'model' string to (provider, model_override).

    Conventions:
      "auto"                -> failover order, default model
      "gemini" / "g"        -> provider=gemini, default model
      "gemini/<model_id>"   -> provider=gemini, model=<model_id>
    """
    if not model or model.lower() == "auto":
        return None, None
    if "/" in model:
        prov, mod = model.split("/", 1)
        return resolve(prov), mod
    return resolve(model), None


@router.post("/v1/openai/chat/completions")
async def openai_chat(req: OAIChatRequest, request: Request):
    rt = request.app.state.router
    messages = [{"role": m.role, "content": _coerce_content(m.content)} for m in req.messages]
    provider, model_override = _resolve_provider(req.model)

    if req.stream:
        return StreamingResponse(
            _openai_stream(rt, messages, provider, model_override, req),
            media_type="text/event-stream",
        )

    try:
        result = await dispatch_chat(
            rt, messages,
            provider_override=provider,
            model_override=model_override,
            max_tokens=req.max_tokens or 2048,
            temperature=req.temperature if req.temperature is not None else 0.7,
        )
    except DispatchFailed as e:
        raise HTTPException(503, f"all providers unavailable: {e.last_error}")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"{result.provider}/{result.model}",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
        "x_praxis": {
            "provider": result.provider,
            "latency_ms": result.latency_ms,
            "attempted": result.attempted,
        },
    }


async def _openai_stream(rt, messages, provider, model_override, req):
    """Streaming variant — emit OpenAI-shaped SSE chunks."""
    # Pick a provider once; no failover mid-stream (too messy).
    est_tokens = sum(len(m.get("content") or "") for m in messages) // 4 + (req.max_tokens or 2048)
    candidates = rt.candidates(provider) if provider else list(rt.order)
    name, atts = rt.pick(est_tokens, candidates)
    if name is None:
        yield f"data: {json.dumps({'error': f'no provider available: {atts}'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    p = rt.providers[name]
    rt.state[name].record()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    fingerprint = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": f"{name}/{p.model}"}
    agg = []
    t0 = time.time()

    try:
        async for chunk in p.stream(
            messages,
            max_tokens=req.max_tokens or 2048,
            temperature=req.temperature if req.temperature is not None else 0.7,
            model=model_override,
        ):
            agg.append(chunk)
            payload = {**fingerprint, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
            yield f"data: {json.dumps(payload)}\n\n"

        done = {**fingerprint, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

        latency = int((time.time() - t0) * 1000)
        text = "".join(agg)
        db.log_call(
            provider=name, model=p.model,
            input_tokens=est_tokens - (req.max_tokens or 2048),
            output_tokens=len(text) // 4,
            latency_ms=latency, status="ok",
            prompt_chars=sum(len(m.get('content') or '') for m in messages),
            response_chars=len(text),
            attempted="",
        )
    except Exception as e:
        db.log_call(
            provider=name, model=p.model,
            status="error", error=str(e)[:500],
            latency_ms=int((time.time() - t0) * 1000),
        )
        yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
        yield "data: [DONE]\n\n"


# ---------- Status & introspection ----------

@router.get("/v1/providers")
async def list_providers(request: Request):
    rt = request.app.state.router
    return {
        "order": rt.order,
        "providers": list(rt.providers),
        "shortcuts": SHORTCUTS,
        "limits": LIMITS,
        "models": {n: p.model for n, p in rt.providers.items()},
    }


@router.get("/v1/status")
async def status(request: Request):
    rt = request.app.state.router
    return {
        "order": rt.order,
        "live": rt.status(),
        "today": db.aggregate_today(),
        "limits": LIMITS,
    }


@router.get("/v1/calls")
async def calls(limit: int = 100, provider: str | None = None, status: str | None = None):
    return db.recent(limit=limit, provider=provider, status=status)
