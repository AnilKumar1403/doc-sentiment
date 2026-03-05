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
    credits_remaining: int
    is_unlimited: bool
    created_at: datetime


class AuthResponse(BaseModel):
    user: UserResponse
    message: str
    access_token: str
    token_type: str = "bearer"
    jwt_token: str | None = None


class AnalyzeTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    emotion_metrics: str | None = Field(default=None, description="Comma-separated metrics")


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


class RelevanceRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    document_text: str = Field(min_length=1)
    reference_text: str = Field(min_length=1)
    analysis_type: str = Field(default="general")
    role: str | None = None
    company: str | None = None
    context_notes: str | None = None


class RelevanceResponse(BaseModel):
    title: str
    analysis_type: str
    relevance_score: float
    metrics: dict[str, float]
    strengths: list[str]
    gaps: list[str]
    summary: str
    suggestions: list[str]
    priority_actions: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    communication_tone: str | None = None
    detailed_summary: str | None = None
    generated_cover_letter: str | None = None
    llm_enhanced: bool = False


class ResumeGenerationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    cv_text: str = Field(min_length=1)
    jd_text: str = Field(min_length=1)
    role: str | None = None
    company: str | None = None
    candidate_name: str | None = None
    context_notes: str | None = None


class ResumeLineModification(BaseModel):
    line_number: int
    current_line: str
    proposed_line: str
    why_change: str
    impact: str
    priority: str


class ResumeStrategicAction(BaseModel):
    area: str
    where_to_add: str
    what_to_add: str
    why_it_matters: str
    expected_impact: str
    estimated_score_lift: float = 0.0
    sample_line: str | None = None
    priority: str = "medium"


class ResumeKeywordCoverage(BaseModel):
    keyword: str
    present_in_cv: bool
    recommended_section: str
    action: str
    priority: str = "medium"


class ResumeGenerationResponse(BaseModel):
    title: str
    role: str | None = None
    company: str | None = None
    relevance_score: float
    target_relevance_score: float = 75.0
    gap_to_target: float = 0.0
    estimated_post_update_score: float = 0.0
    baseline_summary: str
    detailed_strategy: str
    revised_resume: str
    revision_rationale: list[str] = Field(default_factory=list)
    ats_keywords_added: list[str] = Field(default_factory=list)
    strategic_action_plan: list[ResumeStrategicAction] = Field(default_factory=list)
    jd_keyword_coverage: list[ResumeKeywordCoverage] = Field(default_factory=list)
    line_level_modifications: list[ResumeLineModification] = Field(default_factory=list)
    generated_cover_letter: str | None = None
    credits_remaining: int | None = None
    is_unlimited: bool = False
    llm_enhanced: bool = False


class LearningRequest(BaseModel):
    subject: str = Field(description="mathematics or indian social")
    chapter_text: str = Field(min_length=1)
    student_notes: str | None = None


class LearningResponse(BaseModel):
    subject: str
    storytelling_summary: str
    detailed_feedback: str | None = None
    retained_topics: list[str]
    weak_topics: list[str]
    suggestions: list[str]
    study_plan: list[str] = Field(default_factory=list)
    mastery_score: float | None = None
    llm_enhanced: bool = False


class LearningQARequest(BaseModel):
    subject: str = Field(description="mathematics or indian social")
    question_text: str = Field(min_length=3)
    student_attempt: str | None = None
    assignment_context: str | None = None
    grade_level: str | None = None


class LearningQAResponse(BaseModel):
    subject: str
    question_text: str
    concise_answer: str
    current_answer: str | None = None
    correct_answer: str | None = None
    answer_verdict: str = "review_required"
    answer_feedback: str | None = None
    references: list[str] = Field(default_factory=list)
    detailed_explanation: str
    logical_steps: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    practice_questions: list[str] = Field(default_factory=list)
    complexity_level: str = "intermediate"
    llm_enhanced: bool = False


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


class ModuleAnalytics(BaseModel):
    module: str
    total_analyses: int
    last_run_at: datetime | None = None
    average_score: float | None = None


class DashboardSummary(BaseModel):
    total_documents: int
    high_alert_documents: int
    last_analysis_at: datetime | None
    top_emotions: list[EmotionScore]
    module_analytics: list[ModuleAnalytics] = Field(default_factory=list)
    total_analyses: int = 0


class ModelDetailsResponse(BaseModel):
    model_name: str
    model_version: str
    labels: list[str]
    thresholds: dict[str, float]
    train_metrics: dict[str, float | int]


class HistoryItem(BaseModel):
    id: int
    module: str
    title: str
    source_type: str
    analysis_type: str | None = None
    label: str | None = None
    score: float | None = None
    summary: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)
    created_at: datetime
