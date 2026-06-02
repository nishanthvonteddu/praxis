"""FastAPI entry point.

Mounts:
  /                          → web UI
  /v1/chat                   → native gateway endpoint
  /v1/openai/chat/completions → OpenAI-compat shim (for Pydantic AI etc.)
  /api/...                   → learning app endpoints
  /static                    → CSS/assets
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from praxis.config import settings
from praxis.gateway import db as gateway_db
from praxis.gateway.providers import build_providers
from praxis.gateway.router import Router
from praxis.gateway.routes import router as gateway_router
from praxis.learning import db as learning_db
from praxis.learning.routes import router as learning_router
from praxis.web.routes import router as web_router


WEB_ROOT = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    gateway_db.init()
    learning_db.init()
    providers = build_providers()
    if not providers:
        print("WARNING: no providers configured — set at least one API key in .env")
    app.state.providers = providers
    app.state.router = Router(providers, settings.order_list)
    print(f"Praxis ready. Providers: {list(providers)} | Order: {app.state.router.order}")
    yield


app = FastAPI(title="Praxis", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")
app.include_router(gateway_router)
app.include_router(learning_router)
app.include_router(web_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("praxis.main:app", host="0.0.0.0", port=settings.praxis_port, reload=False)
