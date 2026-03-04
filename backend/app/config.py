from functools import lru_cache
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


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
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:8001,http://127.0.0.1:8001",
        ).split(",")
        if origin.strip()
    ]
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "/opt/homebrew/bin/tesseract")
    seeded_user_email: str = os.getenv("SEEDED_USER_EMAIL", "anilkumargolla444@gmail.com")
    seeded_user_password: str = os.getenv("SEEDED_USER_PASSWORD", "Anil2020@b")


@lru_cache
def get_settings() -> Settings:
    return Settings()
