from datetime import datetime
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class AnalyzeResponse(BaseModel):
    document_id: int
    label: str
    confidence: float
    model_name: str
    model_version: str


class DocumentItem(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    label: str | None = None
    confidence: float | None = None


class HealthResponse(BaseModel):
    status: str
