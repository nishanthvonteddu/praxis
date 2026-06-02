# Praxis

> Agentic learning platform. Tell it what you want to learn — it builds a plan, quizzes you daily, tracks what you actually know, and adapts.

Built on top of a multi-provider LLM gateway (Gemini / Groq / Cerebras / NVIDIA / OpenRouter / GitHub / Ollama) with automatic failover, so free-tier rate limits never break a learning session.

## What's inside

- **`praxis.gateway`** — multi-provider LLM router with failover, rate limiting, per-call telemetry. Exposes both a native `/v1/chat` endpoint and an OpenAI-compatible `/v1/openai/chat/completions` shim.
- **`praxis.learning`** — curriculum agent (Pydantic AI) that generates day-by-day plans, runs daily check-ins, and updates a per-concept mastery model.
- **`praxis.web`** — minimal HTMX + Jinja UI.

## Quick start

```bash
cp .env.example .env       # add at least one provider API key
./run.sh                   # uv sync + start server on :8099
```

Open http://localhost:8099 → enter a learning goal → get a plan → do your first check-in.

## Architecture

```
Browser ──HTTP──▶ Praxis App ──HTTP──▶ Praxis Gateway ──HTTPS──▶ Gemini / Groq / etc.
              (Pydantic AI agents)    (failover + logging)
```

Two SQLite databases:
- `gateway.db` — one row per LLM call (provider, model, tokens, latency, status)
- `learning.db` — goals, plans, concepts, mastery, check-in history

## Stack

- **FastAPI** for HTTP
- **Pydantic v2** for validation everywhere
- **Pydantic AI** for agent loops with structured output
- **SQLite** for persistence (no external DB)
- **HTMX + Jinja** for UI (no SPA build step)
- **uv** for dependency management

## Provider choice per agent

Pin via `.env`:
- `PLANNER_PROVIDER=gemini` — long-context, careful reasoning
- `CHECKIN_PROVIDER=groq` — fast, multi-turn
- `GRADER_PROVIDER=groq` — fast, judging short answers

If the pinned provider is rate-limited, the gateway transparently falls through `LLM_ORDER`.
