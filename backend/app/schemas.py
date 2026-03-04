from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class HealthResponse(BaseModel):
    status: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=2, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    display_name: str
    created_at: datetime


class AuthResponse(BaseModel):
    user: UserResponse
    message: str
    access_token: str
    token_type: str = "bearer"


class AnalyzeTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    emotion_metrics: str | None = Field(
        default=None,
        description="Comma-separated metrics, e.g. drama,love,anger",
    )


class EmotionScore(BaseModel):
    emotion: str
    score: float


class AnalyzeResponse(BaseModel):
    document_id: int
    label: str
    confidence: float
    model_name: str
    model_version: str
    source_type: str
    extracted_char_count: int
    selected_metrics: list[str]
    emotion_scores: list[EmotionScore]
    summary: str
    suggestions: list[str]


class DocumentItem(BaseModel):
    id: int
    title: str
    content: str
    source_type: str
    file_name: str | None
    mime_type: str | None
    extracted_char_count: int
    created_at: datetime
    label: str | None = None
    confidence: float | None = None
    selected_metrics: list[str] = Field(default_factory=list)
    emotion_scores: list[EmotionScore] = Field(default_factory=list)
    summary: str | None = None
    suggestions: list[str] = Field(default_factory=list)


class DashboardSummary(BaseModel):
    total_documents: int
    high_alert_documents: int
    last_analysis_at: datetime | None
    top_emotions: list[EmotionScore]


class ModelDetailsResponse(BaseModel):
    model_name: str
    model_version: str
    labels: list[str]
    thresholds: dict[str, float]
    train_metrics: dict[str, float | int]
