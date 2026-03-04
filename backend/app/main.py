from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .bootstrap import initialize_database, seed_default_user
from .config import get_settings
from .database import SessionLocal, get_db
from .document_extractor import ExtractionError, extract_text_from_upload
from .ml import SentimentModel
from .models import Document, SentimentResult, User
from .schemas import (
    AnalyzeResponse,
    AnalyzeTextRequest,
    AuthResponse,
    DashboardSummary,
    DocumentItem,
    HealthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)
from .security import create_access_token, decode_access_token, hash_password, verify_password

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

initialize_database()
with SessionLocal() as seed_session:
    seed_default_user(seed_session)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    file_path = frontend_dir / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(file_path))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _current_user(request: Request, db: Session) -> User:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    email = decode_access_token(token)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    return _current_user(request, db)


@app.post("/api/v1/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    existing = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")

    user = User(
        email=payload.email,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.email)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
    )

    return AuthResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            created_at=user.created_at,
        ),
        message="Account created",
    )


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.email)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
    )

    return AuthResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            created_at=user.created_at,
        ),
        message="Login successful",
    )


@app.post("/api/v1/auth/logout")
def logout(response: Response) -> dict[str, str]:
    response.delete_cookie("access_token")
    return {"message": "Logged out"}


@app.get("/api/v1/auth/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )


@app.get("/api/v1/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DashboardSummary:
    total_documents = db.execute(
        select(func.count(Document.id)).where(Document.owner_id == user.id)
    ).scalar_one()

    positive_documents = db.execute(
        select(func.count(SentimentResult.id))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id, SentimentResult.label == "positive")
    ).scalar_one()

    negative_documents = db.execute(
        select(func.count(SentimentResult.id))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id, SentimentResult.label == "negative")
    ).scalar_one()

    last_analysis_at = db.execute(
        select(func.max(SentimentResult.created_at))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).scalar_one()

    return DashboardSummary(
        total_documents=int(total_documents),
        positive_documents=int(positive_documents),
        negative_documents=int(negative_documents),
        last_analysis_at=last_analysis_at,
    )


def _save_analysis(
    *,
    db: Session,
    user: User,
    title: str,
    content: str,
    source_type: str,
    file_name: str | None,
    mime_type: str | None,
) -> AnalyzeResponse:
    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the document")

    try:
        label, confidence = model.predict(content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    document = Document(
        owner_id=user.id,
        title=title,
        content=content,
        source_type=source_type,
        file_name=file_name,
        mime_type=mime_type,
        extracted_char_count=len(content),
    )
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
        source_type=source_type,
        extracted_char_count=len(content),
    )


@app.post("/api/v1/documents/analyze-text", response_model=AnalyzeResponse)
def analyze_text(
    payload: AnalyzeTextRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalyzeResponse:
    return _save_analysis(
        db=db,
        user=user,
        title=payload.title,
        content=payload.content,
        source_type="text",
        file_name=None,
        mime_type="text/plain",
    )


@app.post("/api/v1/documents/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    title: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalyzeResponse:
    content_bytes = await file.read()
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 10MB.")

    try:
        extracted_text, detected_source = extract_text_from_upload(file.filename or "", content_bytes)
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _save_analysis(
        db=db,
        user=user,
        title=title,
        content=extracted_text,
        source_type=detected_source,
        file_name=file.filename,
        mime_type=file.content_type,
    )


@app.get("/api/v1/documents/history", response_model=list[DocumentItem])
def history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[DocumentItem]:
    stmt = (
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
    )
    rows = db.execute(stmt).all()

    return [
        DocumentItem(
            id=doc.id,
            title=doc.title,
            content=doc.content,
            source_type=doc.source_type,
            file_name=doc.file_name,
            mime_type=doc.mime_type,
            extracted_char_count=doc.extracted_char_count,
            created_at=doc.created_at,
            label=result.label if result else None,
            confidence=result.confidence if result else None,
        )
        for doc, result in rows
    ]


@app.get("/api/v1/documents/{document_id}", response_model=DocumentItem)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentItem:
    stmt = (
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .where(Document.id == document_id, Document.owner_id == user.id)
    )
    row = db.execute(stmt).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc, result = row
    return DocumentItem(
        id=doc.id,
        title=doc.title,
        content=doc.content,
        source_type=doc.source_type,
        file_name=doc.file_name,
        mime_type=doc.mime_type,
        extracted_char_count=doc.extracted_char_count,
        created_at=doc.created_at,
        label=result.label if result else None,
        confidence=result.confidence if result else None,
    )
