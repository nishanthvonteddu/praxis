"""Pydantic request/response models for gateway HTTP endpoints."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: Optional[list[dict]] = None
    prompt: Optional[str] = None
    system: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False


class Attempt(BaseModel):
    provider: str
    reason: str


class ChatResponse(BaseModel):
    provider: str
    model: str
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    attempted: list[Attempt] = Field(default_factory=list)


# --- OpenAI-compat shim shapes ---

class OAIMessage(BaseModel):
    role: str
    content: Any  # OpenAI sometimes uses lists for vision; we coerce to str


class OAIChatRequest(BaseModel):
    model: str  # we interpret as provider name (or shortcut, or "auto")
    messages: list[OAIMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    # Anything else (tools, response_format, etc.) is accepted and ignored for v1
    model_config = {"extra": "allow"}
