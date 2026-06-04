"""Centralized configuration loaded from .env via pydantic-settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    nvidia_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    open_router_api_key: str = ""
    github_access_token: str = ""

    gemini_model: str = "gemini-2.5-flash-lite"
    nvidia_model: str = "mistralai/mistral-nemotron"
    groq_model: str = "llama-3.3-70b-versatile"
    cerebras_model: str = "zai-glm-4.7"
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    github_model: str = "openai/gpt-4.1-mini"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""

    llm_order: str = "gemini,groq,cerebras,nvidia,openrouter,github"

    planner_provider: str = "gemini"
    checkin_provider: str = "groq"
    grader_provider: str = "groq"
    feynman_provider: str = "groq"
    verifier_provider: str = "groq"

    praxis_port: int = 8099

    @property
    def order_list(self) -> list[str]:
        return [x.strip() for x in self.llm_order.split(",") if x.strip()]

    @property
    def gateway_base_url(self) -> str:
        return f"http://localhost:{self.praxis_port}"


settings = Settings()
