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


class AnalyzeTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class AnalyzeResponse(BaseModel):
    document_id: int
    label: str
    confidence: float
    model_name: str
    model_version: str
    source_type: str
    extracted_char_count: int


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


class DashboardSummary(BaseModel):
    total_documents: int
    positive_documents: int
    negative_documents: int
    last_analysis_at: datetime | None
