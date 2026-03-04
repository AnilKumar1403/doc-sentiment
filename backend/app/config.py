from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://sentiment_user:sentiment_pass@localhost:5432/sentiment_db",
    )
    model_path: str = os.getenv("MODEL_PATH", "models/sentiment_model.joblib")
    api_title: str = os.getenv("API_TITLE", "Document Sentiment API")


@lru_cache
def get_settings() -> Settings:
    return Settings()
