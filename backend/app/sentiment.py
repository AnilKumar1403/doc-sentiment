import json

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import get_db
from .document_extractor import ExtractionError, extract_text_from_upload
from .ml import SentimentModel
from .models import Document, SentimentResult, User
from .reporting import build_report
from .schemas import AnalyzeResponse, AnalyzeTextRequest, DocumentItem, EmotionScore
from .security import decode_access_token

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])
model = SentimentModel()


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
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


def _parse_metrics(metrics_text: str | None) -> list[str]:
    if not metrics_text:
        return model.sanitize_metrics(None)
    metrics = [part.strip().lower() for part in metrics_text.split(",") if part.strip()]
    return model.sanitize_metrics(metrics)


def _serialize_scores(scores: dict[str, float], limit: int = 8) -> list[EmotionScore]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [EmotionScore(emotion=name, score=float(value)) for name, value in ranked[:limit]]


def _decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


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

    scores = model.predict_scores(content)
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


@router.post("/analyze-text", response_model=AnalyzeResponse)
def sentiment_analyze_text(
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


@router.post("/analyze-file", response_model=AnalyzeResponse)
async def sentiment_analyze_file(
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


@router.get("/history", response_model=list[DocumentItem])
def sentiment_history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[DocumentItem]:
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
