from functools import lru_cache
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _load_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        (
            "http://localhost:8000,http://127.0.0.1:8000,"
            "http://localhost:8001,http://127.0.0.1:8001"
        ),
    )
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    local_defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://sentiment-dmx59w11d-anilkumargolla444-7543s-projects.vercel.app"
    ]
    for origin in local_defaults:
        if origin not in origins:
            origins.append(origin)
    return origins


class Settings(BaseModel):
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://sentiment_user:sentiment_pass@localhost:5432/sentiment_db",
    )
    model_path: str = os.getenv("MODEL_PATH", "models/sentiment_model.joblib")
    api_title: str = os.getenv("API_TITLE", "Document Sentiment API")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
    cors_origins: list[str] = _load_cors_origins()
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")
    seeded_user_email: str = os.getenv("SEEDED_USER_EMAIL", "anilkumargolla444@gmail.com")
    seeded_user_password: str = os.getenv("SEEDED_USER_PASSWORD", "Anil2020@b")
    default_user_credits: int = int(os.getenv("DEFAULT_USER_CREDITS", "25"))
    unlimited_user_email: str = os.getenv(
        "UNLIMITED_USER_EMAIL",
        os.getenv("PREMIUM_UNLIMITED_EMAIL", "anilkumargolla444@gmail.com"),
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_relevance_temperature: float = float(os.getenv("OPENAI_RELEVANCE_TEMPERATURE", "0.1"))
    openai_learning_temperature: float = float(os.getenv("OPENAI_LEARNING_TEMPERATURE", "0.2"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
