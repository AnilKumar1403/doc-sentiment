from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from .config import get_settings
from .database import Base, engine
from .models import User
from .security import hash_password


def initialize_database() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_backward_compatible_columns()


def seed_default_user(db: Session) -> None:
    settings = get_settings()
    existing = db.query(User).filter(User.email == settings.seeded_user_email).first()
    if existing:
        return

    user = User(
        email=settings.seeded_user_email,
        display_name="Anil Kumar",
        password_hash=hash_password(settings.seeded_user_password),
    )
    db.add(user)
    db.commit()


def _ensure_backward_compatible_columns() -> None:
    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("documents")}

    if "owner_id" not in columns:
        _safe_execute("ALTER TABLE documents ADD COLUMN owner_id INTEGER")
    if "source_type" not in columns:
        _safe_execute("ALTER TABLE documents ADD COLUMN source_type VARCHAR(32) DEFAULT 'text' NOT NULL")
    if "file_name" not in columns:
        _safe_execute("ALTER TABLE documents ADD COLUMN file_name VARCHAR(255)")
    if "mime_type" not in columns:
        _safe_execute("ALTER TABLE documents ADD COLUMN mime_type VARCHAR(128)")
    if "extracted_char_count" not in columns:
        _safe_execute("ALTER TABLE documents ADD COLUMN extracted_char_count INTEGER DEFAULT 0 NOT NULL")

    result_columns = {col["name"] for col in inspector.get_columns("sentiment_results")}
    if "emotion_scores_json" not in result_columns:
        _safe_execute("ALTER TABLE sentiment_results ADD COLUMN emotion_scores_json TEXT DEFAULT '{}' NOT NULL")
    if "selected_metrics_json" not in result_columns:
        _safe_execute("ALTER TABLE sentiment_results ADD COLUMN selected_metrics_json TEXT DEFAULT '[]' NOT NULL")
    if "summary_text" not in result_columns:
        _safe_execute("ALTER TABLE sentiment_results ADD COLUMN summary_text TEXT")
    if "suggestions_json" not in result_columns:
        _safe_execute("ALTER TABLE sentiment_results ADD COLUMN suggestions_json TEXT DEFAULT '[]' NOT NULL")


def _safe_execute(sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))
