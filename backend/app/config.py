from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # No longer required — Ollama runs locally; kept so nothing crashes if referenced
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    database_url: str
    similarity_threshold: float = 0.3
    top_k: int = 5
    max_chunk_tokens: int = 500
    chunk_overlap_ratio: float = 0.12
    log_level: str = "INFO"


settings = Settings()
