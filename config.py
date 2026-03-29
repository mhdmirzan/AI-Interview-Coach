import os

from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str

    # Model settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000

    # Interview settings
    max_questions: int = 5
    default_difficulty: Literal["easy", "medium", "hard"] = "medium"

    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 3

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure downstream SDKs that read from process env can always find the key.
if settings.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)