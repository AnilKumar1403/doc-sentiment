from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .database import Base, engine, get_db
from .models import Document, SentimentResult
from .schemas import AnalyzeRequest, AnalyzeResponse, DocumentItem, HealthResponse
from .ml import SentimentModel

settings = get_settings()
app = FastAPI(title=settings.api_title)
model = SentimentModel()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
def index():
    file_path = frontend_dir / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(file_path))


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/api/v1/documents/analyze", response_model=AnalyzeResponse)
def analyze_document(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        label, confidence = model.predict(payload.content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    document = Document(title=payload.title, content=payload.content)
    db.add(document)
    db.flush()

    result = SentimentResult(
        document_id=document.id,
        label=label,
        confidence=confidence,
        model_name=model.model_name,
        model_version=model.model_version,
    )
    db.add(result)
    db.commit()

    return AnalyzeResponse(
        document_id=document.id,
        label=label,
        confidence=confidence,
        model_name=model.model_name,
        model_version=model.model_version,
    )


@app.get("/api/v1/documents", response_model=list[DocumentItem])
def list_documents(db: Session = Depends(get_db)):
    stmt = (
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .order_by(Document.created_at.desc())
    )
    rows = db.execute(stmt).all()

    return [
        DocumentItem(
            id=doc.id,
            title=doc.title,
            content=doc.content,
            created_at=doc.created_at,
            label=result.label if result else None,
            confidence=result.confidence if result else None,
        )
        for doc, result in rows
    ]


@app.get("/api/v1/documents/{document_id}", response_model=DocumentItem)
def get_document(document_id: int, db: Session = Depends(get_db)):
    stmt = (
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .where(Document.id == document_id)
    )
    row = db.execute(stmt).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc, result = row
    return DocumentItem(
        id=doc.id,
        title=doc.title,
        content=doc.content,
        created_at=doc.created_at,
        label=result.label if result else None,
        confidence=result.confidence if result else None,
    )
