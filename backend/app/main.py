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
from .llm_assist import (
    llm_available,
    refine_learning_qa,
    refine_relevance_report,
    refine_resume_generation,
    refine_storytelling,
    solve_learning_question_with_llm,
)
from .learning_domain import analyze_learning_domain, solve_learning_question
from .ml import SentimentModel
from .models import AnalysisHistory, Document, SentimentResult, User
from .relevance import build_revised_resume_package, compute_relevance_report, generate_cover_letter
from .reporting import build_report
from .schemas import (
    AnalyzeResponse,
    AnalyzeTextRequest,
    AuthResponse,
    DashboardSummary,
    DocumentItem,
    EmotionScore,
    HealthResponse,
    HistoryItem,
    LearningQARequest,
    LearningQAResponse,
    LearningRequest,
    LearningResponse,
    LoginRequest,
    ModelDetailsResponse,
    RegisterRequest,
    ResumeGenerationRequest,
    ResumeGenerationResponse,
    RelevanceRequest,
    RelevanceResponse,
    UserResponse,
)
from .security import create_access_token, decode_access_token, hash_password, verify_password
from .sentiment import router as sentiment_router

settings = get_settings()
app = FastAPI(title=settings.api_title)
model = SentimentModel()
app.include_router(sentiment_router)

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


@app.get("/relevance", include_in_schema=False)
def relevance_page() -> FileResponse:
    return _index_page()


@app.get("/sentiment", include_in_schema=False)
def sentiment_page() -> FileResponse:
    return _index_page()


@app.get("/learning", include_in_schema=False)
def learning_page() -> FileResponse:
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
    _ensure_unlimited_account(db, user)
    return user


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
) -> User:
    return _current_user(request, db, authorization)


def _ensure_unlimited_account(db: Session, user: User) -> None:
    if user.email.lower() != settings.unlimited_user_email.lower():
        return
    changed = False
    if not user.is_unlimited:
        user.is_unlimited = True
        changed = True
    if (user.credits_remaining or 0) < settings.default_user_credits:
        user.credits_remaining = settings.default_user_credits
        changed = True
    if changed:
        db.add(user)
        db.commit()
        db.refresh(user)


def _token_response(user: User, message: str, token: str) -> AuthResponse:
    return AuthResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            credits_remaining=int(user.credits_remaining),
            is_unlimited=bool(user.is_unlimited),
            created_at=user.created_at,
        ),
        message=message,
        access_token=token,
        jwt_token=token,
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
        credits_remaining=settings.default_user_credits,
        is_unlimited=payload.email.lower() == settings.unlimited_user_email.lower(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _ensure_unlimited_account(db, user)

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
    _ensure_unlimited_account(db, user)

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
        credits_remaining=int(user.credits_remaining),
        is_unlimited=bool(user.is_unlimited),
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


def _save_emotion_analysis(
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
    return _save_emotion_analysis(
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

    return _save_emotion_analysis(
        db=db,
        user=user,
        title=title,
        content=extracted_text,
        source_type=detected_source,
        file_name=file.filename,
        mime_type=file.content_type,
        metrics_text=emotion_metrics,
    )


def _relevance_analyze_text_core(
    payload: RelevanceRequest,
    db: Session,
    user: User,
    source_type: str,
) -> RelevanceResponse:
    base = compute_relevance_report(
        document_text=payload.document_text,
        reference_text=payload.reference_text,
        analysis_type=payload.analysis_type,
        role=payload.role,
        context_notes=payload.context_notes,
    )

    cover_letter = None
    if payload.analysis_type == "resume_jd":
        cover_letter = generate_cover_letter(
            resume_text=payload.document_text,
            job_description=payload.reference_text,
            role=payload.role,
            company=payload.company,
            applicant_name=user.display_name,
        )

    summary = (
        f"The document shows an overall relevance score of {base['relevance_score']:.2f}%. "
        "The analysis indicates clear alignment opportunities with a practical path to improve strategic fit."
    )
    detailed_summary = str(base.get("detailed_summary", ""))
    suggestions = list(base["suggestions"])
    priority_actions = list(base.get("priority_actions", []))
    risk_flags = list(base.get("risk_flags", []))
    communication_tone = str(base.get("communication_tone", "Professional and clear."))
    strengths = list(base["strengths"])
    gaps = list(base["gaps"])
    llm_enhanced = False

    if llm_available():
        llm_patch = refine_relevance_report(
            base_report={
                "summary": summary,
                "detailed_summary": detailed_summary,
                "metrics": base["metrics"],
                "strengths": strengths,
                "gaps": gaps,
                "suggestions": suggestions,
                "priority_actions": priority_actions,
                "risk_flags": risk_flags,
                "communication_tone": communication_tone,
            },
            analysis_type=payload.analysis_type,
            role=payload.role,
        )
        if llm_patch:
            summary = str(llm_patch.get("summary", summary))
            detailed_summary = str(llm_patch.get("detailed_summary", detailed_summary))
            strengths = _safe_str_list(llm_patch.get("strengths"), strengths)
            gaps = _safe_str_list(llm_patch.get("gaps"), gaps)
            suggestions = _safe_str_list(llm_patch.get("suggestions"), suggestions)
            priority_actions = _safe_str_list(llm_patch.get("priority_actions"), priority_actions)
            risk_flags = _safe_str_list(llm_patch.get("risk_flags"), risk_flags)
            communication_tone = str(llm_patch.get("communication_tone", communication_tone))
            llm_enhanced = True

    response = RelevanceResponse(
        title=payload.title,
        analysis_type=payload.analysis_type,
        relevance_score=float(base["relevance_score"]),
        metrics={k: float(v) for k, v in base["metrics"].items()},
        strengths=strengths,
        gaps=gaps,
        summary=summary,
        suggestions=suggestions,
        priority_actions=priority_actions,
        risk_flags=risk_flags,
        communication_tone=communication_tone,
        detailed_summary=detailed_summary,
        generated_cover_letter=cover_letter,
        llm_enhanced=llm_enhanced,
    )

    _record_history_entry(
        db=db,
        user=user,
        module_name="relevance",
        title=payload.title,
        source_type=source_type,
        analysis_type=payload.analysis_type,
        label="relevance",
        score=response.relevance_score,
        summary=response.summary,
        suggestions=response.suggestions,
        details={
            "metrics": response.metrics,
            "strengths": response.strengths,
            "gaps": response.gaps,
            "priority_actions": response.priority_actions,
            "risk_flags": response.risk_flags,
            "communication_tone": response.communication_tone,
            "detailed_summary": response.detailed_summary,
        },
    )

    return response


@app.post("/api/v1/relevance/analyze-text", response_model=RelevanceResponse)
def relevance_analyze_text(
    payload: RelevanceRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RelevanceResponse:
    return _relevance_analyze_text_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/relevance/analyze-text", response_model=RelevanceResponse, include_in_schema=False)
def relevance_analyze_text_plain(
    payload: RelevanceRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RelevanceResponse:
    return _relevance_analyze_text_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/v1/relevance/analyze-file", response_model=RelevanceResponse)
async def relevance_analyze_file(
    title: str = Form(...),
    document_text: str | None = Form(default=None),
    reference_text: str | None = Form(default=None),
    analysis_type: str = Form(default="general"),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    document_file: UploadFile | None = File(default=None),
    reference_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RelevanceResponse:
    extracted_doc = _extract_upload_text(document_file)
    extracted_ref = _extract_upload_text(reference_file)

    final_doc = (document_text or "").strip() or extracted_doc.strip()
    final_ref = (reference_text or "").strip() or extracted_ref.strip()

    if not final_doc:
        raise HTTPException(status_code=400, detail="Provide document text or attach a document file.")
    if not final_ref:
        raise HTTPException(status_code=400, detail="Provide reference/JD text or attach a reference file.")

    return _relevance_analyze_text_core(
        payload=RelevanceRequest(
            title=title,
            document_text=final_doc,
            reference_text=final_ref,
            analysis_type=analysis_type,
            role=role,
            company=company,
            context_notes=context_notes,
        ),
        db=db,
        user=user,
        source_type="file" if (document_file or reference_file) else "text",
    )


@app.post("/relevance/analyze-file", response_model=RelevanceResponse, include_in_schema=False)
async def relevance_analyze_file_plain(
    title: str = Form(...),
    document_text: str | None = Form(default=None),
    reference_text: str | None = Form(default=None),
    analysis_type: str = Form(default="general"),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    document_file: UploadFile | None = File(default=None),
    reference_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RelevanceResponse:
    return await relevance_analyze_file(
        title=title,
        document_text=document_text,
        reference_text=reference_text,
        analysis_type=analysis_type,
        role=role,
        company=company,
        context_notes=context_notes,
        document_file=document_file,
        reference_file=reference_file,
        db=db,
        user=user,
    )


def _generate_resume_core(
    payload: ResumeGenerationRequest,
    db: Session,
    user: User,
    source_type: str,
) -> ResumeGenerationResponse:
    package = build_revised_resume_package(
        resume_text=payload.cv_text,
        job_description=payload.jd_text,
        role=payload.role,
        company=payload.company,
        candidate_name=payload.candidate_name or user.display_name,
        context_notes=payload.context_notes,
    )

    detailed_strategy = str(package.get("detailed_strategy", ""))
    revised_resume = str(package.get("revised_resume", ""))
    revision_rationale = _safe_str_list(package.get("revision_rationale"), [])
    ats_keywords_added = _safe_str_list(package.get("ats_keywords_added"), [])
    target_relevance_score = float(package.get("target_relevance_score", 75.0) or 75.0)
    gap_to_target = float(package.get("gap_to_target", 0.0) or 0.0)
    estimated_post_update_score = float(
        package.get("estimated_post_update_score", package.get("relevance_score", 0.0)) or 0.0
    )

    raw_actions = package.get("strategic_action_plan", [])
    strategic_action_plan: list[dict] = []
    if isinstance(raw_actions, list):
        for item in raw_actions[:20]:
            if not isinstance(item, dict):
                continue
            strategic_action_plan.append(
                {
                    "area": str(item.get("area", "")),
                    "where_to_add": str(item.get("where_to_add", "")),
                    "what_to_add": str(item.get("what_to_add", "")),
                    "why_it_matters": str(item.get("why_it_matters", "")),
                    "expected_impact": str(item.get("expected_impact", "")),
                    "estimated_score_lift": float(item.get("estimated_score_lift", 0.0) or 0.0),
                    "sample_line": (
                        str(item.get("sample_line", "")) if item.get("sample_line") is not None else None
                    ),
                    "priority": str(item.get("priority", "medium")),
                }
            )

    raw_keyword_coverage = package.get("jd_keyword_coverage", [])
    jd_keyword_coverage: list[dict] = []
    if isinstance(raw_keyword_coverage, list):
        for item in raw_keyword_coverage[:30]:
            if not isinstance(item, dict):
                continue
            jd_keyword_coverage.append(
                {
                    "keyword": str(item.get("keyword", "")),
                    "present_in_cv": bool(item.get("present_in_cv", False)),
                    "recommended_section": str(item.get("recommended_section", "")),
                    "action": str(item.get("action", "")),
                    "priority": str(item.get("priority", "medium")),
                }
            )

    raw_mods = package.get("line_level_modifications", [])
    line_level_modifications: list[dict] = []
    if isinstance(raw_mods, list):
        for item in raw_mods[:20]:
            if not isinstance(item, dict):
                continue
            line_level_modifications.append(
                {
                    "line_number": int(item.get("line_number", 0) or 0),
                    "current_line": str(item.get("current_line", "")),
                    "proposed_line": str(item.get("proposed_line", "")),
                    "why_change": str(item.get("why_change", "")),
                    "impact": str(item.get("impact", "")),
                    "priority": str(item.get("priority", "medium")),
                }
            )
    llm_enhanced = False

    if llm_available():
        refined = refine_resume_generation(
            base_resume_package={
                "target_relevance_score": target_relevance_score,
                "gap_to_target": gap_to_target,
                "estimated_post_update_score": estimated_post_update_score,
                "detailed_strategy": detailed_strategy,
                "revised_resume": revised_resume,
                "revision_rationale": revision_rationale,
                "ats_keywords_added": ats_keywords_added,
                "strategic_action_plan": strategic_action_plan,
                "jd_keyword_coverage": jd_keyword_coverage,
            },
            role=payload.role,
            company=payload.company,
        )
        if refined:
            detailed_strategy = str(refined.get("detailed_strategy", detailed_strategy))
            revised_resume = str(refined.get("revised_resume", revised_resume))
            revision_rationale = _safe_str_list(refined.get("revision_rationale"), revision_rationale)
            ats_keywords_added = _safe_str_list(refined.get("ats_keywords_added"), ats_keywords_added)
            target_relevance_score = float(refined.get("target_relevance_score", target_relevance_score) or target_relevance_score)
            gap_to_target = float(refined.get("gap_to_target", gap_to_target) or gap_to_target)
            estimated_post_update_score = float(
                refined.get("estimated_post_update_score", estimated_post_update_score) or estimated_post_update_score
            )
            refined_actions = refined.get("strategic_action_plan")
            if isinstance(refined_actions, list):
                normalized_actions: list[dict] = []
                for item in refined_actions[:20]:
                    if not isinstance(item, dict):
                        continue
                    normalized_actions.append(
                        {
                            "area": str(item.get("area", "")),
                            "where_to_add": str(item.get("where_to_add", "")),
                            "what_to_add": str(item.get("what_to_add", "")),
                            "why_it_matters": str(item.get("why_it_matters", "")),
                            "expected_impact": str(item.get("expected_impact", "")),
                            "estimated_score_lift": float(item.get("estimated_score_lift", 0.0) or 0.0),
                            "sample_line": (
                                str(item.get("sample_line", "")) if item.get("sample_line") is not None else None
                            ),
                            "priority": str(item.get("priority", "medium")),
                        }
                    )
                if normalized_actions:
                    strategic_action_plan = normalized_actions

            refined_keyword_coverage = refined.get("jd_keyword_coverage")
            if isinstance(refined_keyword_coverage, list):
                normalized_coverage: list[dict] = []
                for item in refined_keyword_coverage[:30]:
                    if not isinstance(item, dict):
                        continue
                    normalized_coverage.append(
                        {
                            "keyword": str(item.get("keyword", "")),
                            "present_in_cv": bool(item.get("present_in_cv", False)),
                            "recommended_section": str(item.get("recommended_section", "")),
                            "action": str(item.get("action", "")),
                            "priority": str(item.get("priority", "medium")),
                        }
                    )
                if normalized_coverage:
                    jd_keyword_coverage = normalized_coverage
            llm_enhanced = True

    generated_cover_letter = generate_cover_letter(
        resume_text=payload.cv_text,
        job_description=payload.jd_text,
        role=payload.role,
        company=payload.company,
        applicant_name=payload.candidate_name or user.display_name,
    )

    relevance_score = float(package.get("relevance_score", 0.0))
    baseline_summary = str(package.get("baseline_summary", ""))

    response = ResumeGenerationResponse(
        title=payload.title,
        role=payload.role,
        company=payload.company,
        relevance_score=relevance_score,
        target_relevance_score=target_relevance_score,
        gap_to_target=gap_to_target,
        estimated_post_update_score=estimated_post_update_score,
        baseline_summary=baseline_summary,
        detailed_strategy=detailed_strategy,
        revised_resume=revised_resume,
        revision_rationale=revision_rationale,
        ats_keywords_added=ats_keywords_added,
        strategic_action_plan=strategic_action_plan,
        jd_keyword_coverage=jd_keyword_coverage,
        line_level_modifications=line_level_modifications,
        generated_cover_letter=generated_cover_letter,
        credits_remaining=_credits_remaining_for_response(user),
        is_unlimited=bool(user.is_unlimited),
        llm_enhanced=llm_enhanced,
    )

    _record_history_entry(
        db=db,
        user=user,
        module_name="relevance",
        title=payload.title,
        source_type=source_type,
        analysis_type="resume_jd",
        label="resume_generation",
        score=response.relevance_score,
        summary=response.baseline_summary,
        suggestions=response.revision_rationale,
        details={
            "target_relevance_score": response.target_relevance_score,
            "gap_to_target": response.gap_to_target,
            "estimated_post_update_score": response.estimated_post_update_score,
            "detailed_strategy": response.detailed_strategy,
            "ats_keywords_added": response.ats_keywords_added,
            "strategic_action_plan": [
                item.model_dump() if hasattr(item, "model_dump") else dict(item)
                for item in response.strategic_action_plan
            ],
            "jd_keyword_coverage": [
                item.model_dump() if hasattr(item, "model_dump") else dict(item)
                for item in response.jd_keyword_coverage
            ],
            "line_level_modifications": [
                item.model_dump() if hasattr(item, "model_dump") else dict(item)
                for item in response.line_level_modifications
            ],
            "revised_resume": response.revised_resume,
            "credits_remaining": response.credits_remaining,
            "is_unlimited": response.is_unlimited,
        },
    )
    return response


@app.post("/api/v1/relevance/generate-resume", response_model=ResumeGenerationResponse)
def relevance_generate_resume(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/v1/relevance/generate_resume", response_model=ResumeGenerationResponse)
def relevance_generate_resume_alias(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/v1/relevance/resume-generator", response_model=ResumeGenerationResponse)
def relevance_generate_resume_alias_generator(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/relevance/generate-resume", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_plain(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/relevance/generate_resume", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_plain_alias(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/relevance/resume-generator", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_plain_generator(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/relevance/generate-resume", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_api_alias(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/relevance/generate_resume", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_api_alias_underscore(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/relevance/resume-generator", response_model=ResumeGenerationResponse, include_in_schema=False)
def relevance_generate_resume_api_alias_generator(
    payload: ResumeGenerationRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return _generate_resume_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/v1/relevance/generate-resume-file", response_model=ResumeGenerationResponse)
async def relevance_generate_resume_file(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    extracted_cv = _extract_upload_text(cv_file)
    extracted_jd = _extract_upload_text(jd_file)

    final_cv = (cv_text or "").strip() or extracted_cv.strip()
    final_jd = (jd_text or "").strip() or extracted_jd.strip()
    if not final_cv:
        raise HTTPException(status_code=400, detail="Provide CV text or attach a CV file.")
    if not final_jd:
        raise HTTPException(status_code=400, detail="Provide JD text or attach a JD file.")

    return _generate_resume_core(
        payload=ResumeGenerationRequest(
            title=title,
            cv_text=final_cv,
            jd_text=final_jd,
            role=role,
            company=company,
            candidate_name=candidate_name,
            context_notes=context_notes,
        ),
        db=db,
        user=user,
        source_type="file" if (cv_file or jd_file) else "text",
    )


@app.post("/api/v1/relevance/generate_resume_file", response_model=ResumeGenerationResponse)
async def relevance_generate_resume_file_alias(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return await relevance_generate_resume_file(
        title=title,
        cv_text=cv_text,
        jd_text=jd_text,
        role=role,
        company=company,
        candidate_name=candidate_name,
        context_notes=context_notes,
        cv_file=cv_file,
        jd_file=jd_file,
        db=db,
        user=user,
    )


@app.post("/relevance/generate-resume-file", response_model=ResumeGenerationResponse, include_in_schema=False)
async def relevance_generate_resume_file_plain(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return await relevance_generate_resume_file(
        title=title,
        cv_text=cv_text,
        jd_text=jd_text,
        role=role,
        company=company,
        candidate_name=candidate_name,
        context_notes=context_notes,
        cv_file=cv_file,
        jd_file=jd_file,
        db=db,
        user=user,
    )


@app.post("/relevance/generate_resume_file", response_model=ResumeGenerationResponse, include_in_schema=False)
async def relevance_generate_resume_file_plain_alias(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return await relevance_generate_resume_file_plain(
        title=title,
        cv_text=cv_text,
        jd_text=jd_text,
        role=role,
        company=company,
        candidate_name=candidate_name,
        context_notes=context_notes,
        cv_file=cv_file,
        jd_file=jd_file,
        db=db,
        user=user,
    )


@app.post("/api/relevance/generate-resume-file", response_model=ResumeGenerationResponse, include_in_schema=False)
async def relevance_generate_resume_file_api_alias(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return await relevance_generate_resume_file(
        title=title,
        cv_text=cv_text,
        jd_text=jd_text,
        role=role,
        company=company,
        candidate_name=candidate_name,
        context_notes=context_notes,
        cv_file=cv_file,
        jd_file=jd_file,
        db=db,
        user=user,
    )


@app.post("/api/relevance/generate_resume_file", response_model=ResumeGenerationResponse, include_in_schema=False)
async def relevance_generate_resume_file_api_alias_underscore(
    title: str = Form(...),
    cv_text: str | None = Form(default=None),
    jd_text: str | None = Form(default=None),
    role: str | None = Form(default=None),
    company: str | None = Form(default=None),
    candidate_name: str | None = Form(default=None),
    context_notes: str | None = Form(default=None),
    cv_file: UploadFile | None = File(default=None),
    jd_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResumeGenerationResponse:
    return await relevance_generate_resume_file_api_alias(
        title=title,
        cv_text=cv_text,
        jd_text=jd_text,
        role=role,
        company=company,
        candidate_name=candidate_name,
        context_notes=context_notes,
        cv_file=cv_file,
        jd_file=jd_file,
        db=db,
        user=user,
    )


def _learning_story_core(
    payload: LearningRequest,
    db: Session,
    user: User,
    source_type: str,
) -> LearningResponse:
    base = analyze_learning_domain(
        chapter_text=payload.chapter_text,
        subject=payload.subject,
        student_notes=payload.student_notes,
    )

    summary = str(base["storytelling_summary"])
    detailed_feedback = str(base.get("detailed_feedback", ""))
    weak_topics = list(base["weak_topics"])
    suggestions = list(base["suggestions"])
    study_plan = list(base.get("study_plan", []))
    mastery_score = float(base.get("mastery_score", 0.0))
    llm_enhanced = False

    if llm_available():
        refined = refine_storytelling(base_story=base, subject=payload.subject)
        if refined:
            summary = str(refined.get("storytelling_summary", summary))
            detailed_feedback = str(refined.get("detailed_feedback", detailed_feedback))
            weak_topics = _safe_str_list(refined.get("weak_topics"), weak_topics)
            suggestions = _safe_str_list(refined.get("suggestions"), suggestions)
            study_plan = _safe_str_list(refined.get("study_plan"), study_plan)
            llm_enhanced = True

    response = LearningResponse(
        subject=payload.subject,
        storytelling_summary=summary,
        detailed_feedback=detailed_feedback,
        retained_topics=list(base["retained_topics"]),
        weak_topics=weak_topics,
        suggestions=suggestions,
        study_plan=study_plan,
        mastery_score=mastery_score,
        llm_enhanced=llm_enhanced,
    )

    _record_history_entry(
        db=db,
        user=user,
        module_name="learning",
        title=f"{payload.subject.title()} Learning Analysis",
        source_type=source_type,
        analysis_type=payload.subject,
        label="learning",
        score=response.mastery_score,
        summary=response.storytelling_summary,
        suggestions=response.suggestions,
        details={
            "detailed_feedback": response.detailed_feedback,
            "retained_topics": response.retained_topics,
            "weak_topics": response.weak_topics,
            "study_plan": response.study_plan,
            "mastery_score": response.mastery_score,
        },
    )

    return response


def _learning_qa_core(
    payload: LearningQARequest,
    db: Session,
    user: User,
) -> LearningQAResponse:
    base = solve_learning_question(
        subject=payload.subject,
        question_text=payload.question_text,
        student_attempt=payload.student_attempt,
        assignment_context=payload.assignment_context,
        grade_level=payload.grade_level,
    )
    if bool(base.get("needs_llm_solver")) and llm_available():
        llm_solution = solve_learning_question_with_llm(
            subject=payload.subject,
            question_text=payload.question_text,
            student_attempt=payload.student_attempt,
            assignment_context=payload.assignment_context,
            grade_level=payload.grade_level,
        )
        if llm_solution:
            for key in (
                "concise_answer",
                "correct_answer",
                "answer_verdict",
                "answer_feedback",
                "references",
                "detailed_explanation",
                "logical_steps",
                "key_concepts",
                "common_mistakes",
                "practice_questions",
                "complexity_level",
            ):
                if key in llm_solution and llm_solution.get(key) not in (None, "", []):
                    base[key] = llm_solution.get(key)

    concise_answer = str(base.get("concise_answer", ""))
    current_answer = str(base.get("current_answer", payload.student_attempt or "Not provided"))
    correct_answer = str(base.get("correct_answer", concise_answer))
    answer_verdict = str(base.get("answer_verdict", "review_required"))
    answer_feedback = str(base.get("answer_feedback", ""))
    references = _safe_str_list(base.get("references"), [])
    is_numeric_math = bool(base.get("is_numeric_math", False))
    detailed_explanation = str(base.get("detailed_explanation", ""))
    logical_steps = _safe_str_list(base.get("logical_steps"), [])
    key_concepts = _safe_str_list(base.get("key_concepts"), [])
    common_mistakes = _safe_str_list(base.get("common_mistakes"), [])
    practice_questions = _safe_str_list(base.get("practice_questions"), [])
    complexity_level = str(base.get("complexity_level", "intermediate"))
    llm_enhanced = False

    if llm_available():
        refined = refine_learning_qa(base_answer=base, subject=payload.subject)
        if refined:
            if not is_numeric_math:
                concise_answer = str(refined.get("concise_answer", concise_answer))
                answer_feedback = str(refined.get("answer_feedback", answer_feedback))
            detailed_explanation = str(refined.get("detailed_explanation", detailed_explanation))
            logical_steps = _safe_str_list(refined.get("logical_steps"), logical_steps)
            key_concepts = _safe_str_list(refined.get("key_concepts"), key_concepts)
            common_mistakes = _safe_str_list(refined.get("common_mistakes"), common_mistakes)
            practice_questions = _safe_str_list(refined.get("practice_questions"), practice_questions)
            complexity_level = str(refined.get("complexity_level", complexity_level))
            references = _safe_str_list(refined.get("references"), references)
            if not is_numeric_math:
                verdict_candidate = str(refined.get("answer_verdict", answer_verdict)).strip().lower()
                if verdict_candidate in {"correct", "incorrect", "partial", "review_required", "not_provided"}:
                    answer_verdict = verdict_candidate
            llm_enhanced = True

    if is_numeric_math:
        concise_answer = f"The correct answer is {correct_answer}."
        answer_feedback = str(base.get("answer_feedback", answer_feedback))
        answer_verdict = str(base.get("answer_verdict", answer_verdict))
        if answer_verdict not in {"correct", "incorrect", "partial", "review_required", "not_provided"}:
            answer_verdict = "review_required"

    response = LearningQAResponse(
        subject=base.get("subject", payload.subject),
        question_text=payload.question_text,
        concise_answer=concise_answer,
        current_answer=current_answer,
        correct_answer=correct_answer,
        answer_verdict=answer_verdict,
        answer_feedback=answer_feedback,
        references=references,
        detailed_explanation=detailed_explanation,
        logical_steps=logical_steps,
        key_concepts=key_concepts,
        common_mistakes=common_mistakes,
        practice_questions=practice_questions,
        complexity_level=complexity_level,
        llm_enhanced=llm_enhanced,
    )

    title_prefix = payload.question_text.strip().replace("\n", " ")
    title = f"Q&A: {title_prefix[:72]}{'...' if len(title_prefix) > 72 else ''}"
    _record_history_entry(
        db=db,
        user=user,
        module_name="learning",
        title=title,
        source_type="text",
        analysis_type=f"{response.subject}:qa",
        label="learning_qa",
        score=None,
        summary=response.concise_answer,
        suggestions=response.practice_questions,
        details={
            "question_text": response.question_text,
            "current_answer": response.current_answer,
            "correct_answer": response.correct_answer,
            "answer_verdict": response.answer_verdict,
            "answer_feedback": response.answer_feedback,
            "references": response.references,
            "detailed_explanation": response.detailed_explanation,
            "logical_steps": response.logical_steps,
            "key_concepts": response.key_concepts,
            "common_mistakes": response.common_mistakes,
            "practice_questions": response.practice_questions,
            "complexity_level": response.complexity_level,
            "student_attempt": payload.student_attempt,
            "assignment_context": payload.assignment_context,
            "grade_level": payload.grade_level,
        },
    )
    return response


@app.post("/api/v1/learning/story", response_model=LearningResponse)
def learning_story(
    payload: LearningRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningResponse:
    return _learning_story_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/learning/story", response_model=LearningResponse, include_in_schema=False)
def learning_story_plain(
    payload: LearningRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningResponse:
    return _learning_story_core(payload=payload, db=db, user=user, source_type="text")


@app.post("/api/v1/learning/question-answer", response_model=LearningQAResponse)
def learning_question_answer(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/v1/learning/qa", response_model=LearningQAResponse)
def learning_question_answer_alias(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/v1/learning/question_answer", response_model=LearningQAResponse)
def learning_question_answer_alias_underscore(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/learning/question-answer", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_plain(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/learning/qa", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_plain_alias(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/learning/question_answer", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_plain_alias_underscore(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/learning/question-answer", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_api_alias(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/learning/qa", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_api_alias_short(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/learning/question_answer", response_model=LearningQAResponse, include_in_schema=False)
def learning_question_answer_api_alias_underscore(
    payload: LearningQARequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningQAResponse:
    return _learning_qa_core(payload=payload, db=db, user=user)


@app.post("/api/v1/learning/story-file", response_model=LearningResponse)
async def learning_story_file(
    subject: str = Form(...),
    chapter_text: str | None = Form(default=None),
    student_notes: str | None = Form(default=None),
    chapter_file: UploadFile | None = File(default=None),
    student_notes_file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> LearningResponse:
    extracted_chapter = _extract_upload_text(chapter_file)
    extracted_notes = _extract_upload_text(student_notes_file)

    final_chapter = (chapter_text or "").strip() or extracted_chapter.strip()
    final_notes = (student_notes or "").strip() or extracted_notes.strip()

    if not final_chapter:
        raise HTTPException(status_code=400, detail="Provide chapter text or attach a chapter file.")

    return _learning_story_core(
        payload=LearningRequest(subject=subject, chapter_text=final_chapter, student_notes=final_notes),
        db=db,
        user=user,
        source_type="file" if (chapter_file or student_notes_file) else "text",
    )


def _decode_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _safe_str_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned if cleaned else fallback
    return fallback


def _credits_remaining_for_response(user: User) -> int | None:
    if user.is_unlimited:
        return None
    return int(max(user.credits_remaining, 0))


def _record_history_entry(
    *,
    db: Session,
    user: User,
    module_name: str,
    title: str,
    source_type: str,
    analysis_type: str | None,
    label: str | None,
    score: float | None,
    summary: str | None,
    suggestions: list[str],
    details: dict,
) -> None:
    db.add(
        AnalysisHistory(
            owner_id=user.id,
            module_name=module_name,
            title=title,
            source_type=source_type,
            analysis_type=analysis_type,
            label=label,
            score=score,
            summary_text=summary,
            suggestions_json=json.dumps(suggestions),
            details_json=json.dumps(details),
        )
    )
    db.commit()


def _extract_upload_text(upload: UploadFile | None) -> str:
    if upload is None:
        return ""
    content_bytes = upload.file.read()
    if len(content_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 12MB.")
    try:
        text, _source = extract_text_from_upload(upload.filename or "", content_bytes)
        return text
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DashboardSummary:
    total_documents = db.execute(select(func.count(Document.id)).where(Document.owner_id == user.id)).scalar_one()

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

    sentiment_count = db.execute(
        select(func.count(SentimentResult.id))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).scalar_one()
    sentiment_avg_conf = db.execute(
        select(func.avg(SentimentResult.confidence))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).scalar_one()
    sentiment_last_run = db.execute(
        select(func.max(SentimentResult.created_at))
        .join(Document, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
    ).scalar_one()

    module_analytics_map: dict[str, dict] = {
        "sentiment": {
            "module": "sentiment",
            "total_analyses": int(sentiment_count or 0),
            "last_run_at": sentiment_last_run,
            "average_score": round(float(sentiment_avg_conf or 0.0) * 100, 2) if sentiment_avg_conf is not None else None,
        }
    }

    extra_module_rows = db.execute(
        select(
            AnalysisHistory.module_name,
            func.count(AnalysisHistory.id),
            func.max(AnalysisHistory.created_at),
            func.avg(AnalysisHistory.score),
        )
        .where(AnalysisHistory.owner_id == user.id)
        .group_by(AnalysisHistory.module_name)
    ).all()

    total_analyses = int(sentiment_count or 0)
    for module_name, count_value, last_run_value, avg_value in extra_module_rows:
        module_analytics_map[str(module_name)] = {
            "module": str(module_name),
            "total_analyses": int(count_value or 0),
            "last_run_at": last_run_value,
            "average_score": round(float(avg_value), 2) if avg_value is not None else None,
        }
        total_analyses += int(count_value or 0)

    for module_name in ("relevance", "learning"):
        if module_name not in module_analytics_map:
            module_analytics_map[module_name] = {
                "module": module_name,
                "total_analyses": 0,
                "last_run_at": None,
                "average_score": None,
            }

    module_analytics = [module_analytics_map[name] for name in ("sentiment", "relevance", "learning")]

    return DashboardSummary(
        total_documents=int(total_documents),
        high_alert_documents=int(high_alert_documents),
        last_analysis_at=last_analysis_at,
        top_emotions=top_emotions,
        module_analytics=module_analytics,
        total_analyses=total_analyses,
    )


@app.get("/api/v1/documents/history", response_model=list[HistoryItem])
def history(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[HistoryItem]:
    stmt = (
        select(Document, SentimentResult)
        .outerjoin(SentimentResult, Document.id == SentimentResult.document_id)
        .where(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
    )
    rows = db.execute(stmt).all()

    response: list[HistoryItem] = []
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
            HistoryItem(
                id=doc.id,
                module="sentiment",
                title=doc.title,
                source_type=doc.source_type,
                analysis_type="emotion",
                label=result.label if result else None,
                score=result.confidence if result else None,
                summary=summary,
                suggestions=suggestions,
                details={
                    "selected_metrics": selected_metrics,
                    "emotion_scores": [{"emotion": item.emotion, "score": float(item.score)} for item in emotion_scores],
                    "file_name": doc.file_name,
                    "mime_type": doc.mime_type,
                },
                created_at=doc.created_at,
            )
        )

    extra_rows = db.execute(
        select(AnalysisHistory)
        .where(AnalysisHistory.owner_id == user.id, AnalysisHistory.module_name.in_(["relevance", "learning"]))
        .order_by(AnalysisHistory.created_at.desc())
    ).scalars().all()

    for row in extra_rows:
        response.append(
            HistoryItem(
                id=10_000_000 + row.id,
                module=row.module_name,
                title=row.title,
                source_type=row.source_type,
                analysis_type=row.analysis_type,
                label=row.label,
                score=row.score,
                summary=row.summary_text,
                suggestions=_decode_json(row.suggestions_json, []),
                details=_decode_json(row.details_json, {}),
                created_at=row.created_at,
            )
        )

    response.sort(key=lambda item: item.created_at, reverse=True)
    return response


@app.get("/api/v1/documents/{document_id}", response_model=DocumentItem)
def get_document(document_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> DocumentItem:
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
