import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .bootstrap import initialize_database, seed_default_user
from .config import get_settings
from .database import SessionLocal, get_db
from .document_extractor import ExtractionError, extract_text_from_upload
from .emotion_taxonomy import ALERT_EMOTIONS
from .ml import SentimentModel
from .models import Document, SentimentResult, User
from .reporting import build_report
from .schemas import (
    AnalyzeResponse,
    AnalyzeTextRequest,
    AuthResponse,
    DashboardSummary,
    DocumentItem,
    EmotionScore,
    HealthResponse,
    LoginRequest,
    ModelDetailsResponse,
    RegisterRequest,
    UserResponse,
)
from .security import create_access_token, decode_access_token, hash_password, verify_password

settings = get_settings()
app = FastAPI(title=settings.api_title)
model = SentimentModel()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
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
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=307)


def _index_page() -> FileResponse:
    file_path = frontend_dir / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(file_path))


@app.get("/login", include_in_schema=False)
def login_page() -> FileResponse:
    return _index_page()


@app.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    return _index_page()


@app.get("/analyze", include_in_schema=False)
def analyze_page() -> FileResponse:
    return _index_page()


@app.get("/history", include_in_schema=False)
def history_page() -> FileResponse:
    return _index_page()


@app.get("/profile", include_in_schema=False)
def profile_page() -> FileResponse:
    return _index_page()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/v1/model/details", response_model=ModelDetailsResponse)
def model_details() -> ModelDetailsResponse:
    model._ensure_loaded()
    return ModelDetailsResponse(
        model_name=model.model_name,
        model_version=model.model_version,
        labels=list(model.labels),
        thresholds={label: float(model.get_threshold(label)) for label in model.labels},
        train_metrics=model.train_metrics,
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def _current_user(request: Request, db: Session, authorization: str | None = None) -> User:
    token = _extract_bearer_token(authorization) or request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    email = decode_access_token(token)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
    return _current_user(request, db, authorization)


def _token_response(user: User, message: str, token: str) -> AuthResponse:
    return AuthResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            created_at=user.created_at,
        ),
        message=message,
        access_token=token,
    )


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
    return _token_response(user, "Account created", token)


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
    return _token_response(user, "Login successful", token)


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


def _parse_metrics(metrics_text: str | None, fallback: list[str] | None = None) -> list[str]:
    if fallback:
        return model.sanitize_metrics(fallback)
    if not metrics_text:
        return model.sanitize_metrics(None)
    metrics = [part.strip().lower() for part in metrics_text.split(",") if part.strip()]
    return model.sanitize_metrics(metrics)


def _serialize_scores(scores: dict[str, float], limit: int = 8) -> list[EmotionScore]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [EmotionScore(emotion=name, score=float(value)) for name, value in ranked[:limit]]


def _save_analysis(
    *,
    db: Session,
    user: User,
    title: str,
    content: str,
    source_type: str,
    file_name: str | None,
    mime_type: str | None,
    metrics_text: str | None,
) -> AnalyzeResponse:
    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the document")

    try:
        scores = model.predict_scores(content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    selected_metrics = _parse_metrics(metrics_text)
    selected_scores = {metric: scores.get(metric, 0.0) for metric in selected_metrics}
    summary, suggestions, dominant_emotion, dominant_score = build_report(
        selected_scores,
        thresholds={metric: model.get_threshold(metric) for metric in selected_metrics},
    )

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
        label=dominant_emotion,
        confidence=dominant_score,
        emotion_scores_json=json.dumps(selected_scores),
        selected_metrics_json=json.dumps(selected_metrics),
        summary_text=summary,
        suggestions_json=json.dumps(suggestions),
        model_name=model.model_name,
        model_version=model.model_version,
    )
    db.add(result)
    db.commit()

    return AnalyzeResponse(
        document_id=document.id,
        label=dominant_emotion,
        confidence=dominant_score,
        model_name=model.model_name,
        model_version=model.model_version,
        source_type=source_type,
        extracted_char_count=len(content),
        selected_metrics=selected_metrics,
        emotion_scores=_serialize_scores(selected_scores, limit=len(selected_scores)),
        summary=summary,
        suggestions=suggestions,
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
        metrics_text=payload.emotion_metrics,
    )


@app.post("/api/v1/documents/analyze", response_model=AnalyzeResponse)
def analyze_text_legacy(
    payload: AnalyzeTextRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AnalyzeResponse:
    return analyze_text(payload=payload, db=db, user=user)


@app.post("/api/v1/documents/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    title: str = Form(...),
    emotion_metrics: str | None = Form(default=None),
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
        metrics_text=emotion_metrics,
    )


def _decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@app.get("/api/v1/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DashboardSummary:
    total_documents = db.execute(
        select(func.count(Document.id)).where(Document.owner_id == user.id)
    ).scalar_one()

    last_analysis_at = db.execute(
        select(func.max(SentimentResult.created_at))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).scalar_one()

    rows = db.execute(
        select(SentimentResult.emotion_scores_json)
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).all()

    totals: dict[str, float] = {}
    high_alert_documents = 0
    for (scores_json,) in rows:
        scores = _decode_json(scores_json, {})
        if not isinstance(scores, dict):
            continue

        has_alert = False
        for emotion, score in scores.items():
            value = float(score)
            totals[emotion] = totals.get(emotion, 0.0) + value
            if emotion in ALERT_EMOTIONS and value >= model.get_threshold(emotion):
                has_alert = True
        if has_alert:
            high_alert_documents += 1

    denominator = max(int(total_documents), 1)
    ranked = sorted(
        ((emotion, total / denominator) for emotion, total in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_emotions = [EmotionScore(emotion=name, score=float(score)) for name, score in ranked[:6]]

    return DashboardSummary(
        total_documents=int(total_documents),
        high_alert_documents=int(high_alert_documents),
        last_analysis_at=last_analysis_at,
        top_emotions=top_emotions,
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

    response: list[DocumentItem] = []
    for doc, result in rows:
        selected_metrics = []
        emotion_scores: list[EmotionScore] = []
        summary = None
        suggestions: list[str] = []

        if result:
            selected_metrics = _decode_json(result.selected_metrics_json, [])
            raw_scores = _decode_json(result.emotion_scores_json, {})
            if isinstance(raw_scores, dict):
                emotion_scores = _serialize_scores({k: float(v) for k, v in raw_scores.items()}, limit=10)
            summary = result.summary_text
            suggestions = _decode_json(result.suggestions_json, [])

        response.append(
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
                selected_metrics=selected_metrics,
                emotion_scores=emotion_scores,
                summary=summary,
                suggestions=suggestions,
            )
        )

    return response


@app.get("/api/v1/documents/{document_id}", response_model=DocumentItem)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentItem:
    row = db.execute(
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .where(Document.id == document_id, Document.owner_id == user.id)
    ).first()

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc, result = row
    selected_metrics = _decode_json(result.selected_metrics_json, []) if result else []
    raw_scores = _decode_json(result.emotion_scores_json, {}) if result else {}
    emotion_scores = (
        _serialize_scores({k: float(v) for k, v in raw_scores.items()}, limit=10)
        if isinstance(raw_scores, dict)
        else []
    )
    suggestions = _decode_json(result.suggestions_json, []) if result else []

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
        selected_metrics=selected_metrics,
        emotion_scores=emotion_scores,
        summary=result.summary_text if result else None,
        suggestions=suggestions,
    )
