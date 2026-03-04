import json
from typing import Any

from .config import get_settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _client() -> Any | None:
    settings = get_settings()
    if not settings.openai_api_key or OpenAI is None:
        return None
    return OpenAI(api_key=settings.openai_api_key)


def llm_available() -> bool:
    return _client() is not None


def _parse_json_output(text: str) -> dict | None:
    payload = (text or "").strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = payload[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def refine_relevance_report(*, base_report: dict, analysis_type: str, role: str | None) -> dict | None:
    client = _client()
    if client is None:
        return None

    settings = get_settings()
    prompt = (
        "Refine this relevance-analysis report with strategic, professional, and practical guidance. "
        "Return strict JSON with keys: summary, detailed_summary, suggestions(list), priority_actions(list), "
        "risk_flags(list), strengths(list), gaps(list), communication_tone. "
        "Use concise, polite, executive language and preserve factual realism. "
        f"Analysis type: {analysis_type}. Role: {role or 'n/a'}. "
        f"Base report JSON: {json.dumps(base_report)}"
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior document strategist. Write professional, polite, crisp, and highly useful reports. "
                        "Focus on decision quality, clarity, and practical next steps."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.openai_relevance_temperature,
        )
        text = (response.output_text or "").strip()
        parsed = _parse_json_output(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def refine_storytelling(*, base_story: dict, subject: str) -> dict | None:
    client = _client()
    if client is None:
        return None

    settings = get_settings()
    prompt = (
        "Improve this student learning report for clarity, motivation, and actionable outcomes. "
        "Return strict JSON with keys: storytelling_summary, detailed_feedback, weak_topics(list), "
        "suggestions(list), study_plan(list). "
        "Use polite and encouraging tone with clear strategy. "
        f"Subject: {subject}. Base report: {json.dumps(base_story)}"
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an elite educational coach for students. "
                        "Provide structured, clear, and emotionally intelligent guidance."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.openai_learning_temperature,
        )
        text = (response.output_text or "").strip()
        parsed = _parse_json_output(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def refine_learning_qa(*, base_answer: dict, subject: str) -> dict | None:
    client = _client()
    if client is None:
        return None

    settings = get_settings()
    prompt = (
        "Improve this student question-answer response for clarity, correctness, politeness, and educational impact. "
        "Return strict JSON with keys: concise_answer, current_answer, correct_answer, answer_verdict, answer_feedback, "
        "references(list), detailed_explanation, logical_steps(list), key_concepts(list), common_mistakes(list), "
        "practice_questions(list), complexity_level. "
        "If the base answer already contains a deterministic numeric correct_answer, do not change it. "
        "Make the explanation easy to follow and useful for homework and assignments. "
        f"Subject: {subject}. Base answer: {json.dumps(base_answer)}"
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a patient teacher and academic mentor. "
                        "Provide accurate, structured, and supportive guidance."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.openai_learning_temperature,
        )
        text = (response.output_text or "").strip()
        parsed = _parse_json_output(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def refine_resume_generation(*, base_resume_package: dict, role: str | None, company: str | None) -> dict | None:
    client = _client()
    if client is None:
        return None

    settings = get_settings()
    prompt = (
        "Refine this resume rewrite package. Return strict JSON with keys: "
        "detailed_strategy, revised_resume, revision_rationale(list), ats_keywords_added(list). "
        "Output should be professional, crisp, and highly practical for real job applications. "
        f"Target role: {role or 'n/a'}. Company: {company or 'n/a'}. "
        f"Base package: {json.dumps(base_resume_package)}"
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert resume strategist and hiring advisor. "
                        "Produce recruiter-ready, ATS-aware content with clear structure and practical tone."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.openai_relevance_temperature,
        )
        text = (response.output_text or "").strip()
        parsed = _parse_json_output(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def solve_learning_question_with_llm(
    *,
    subject: str,
    question_text: str,
    student_attempt: str | None,
    assignment_context: str | None,
    grade_level: str | None,
) -> dict | None:
    client = _client()
    if client is None:
        return None

    settings = get_settings()
    prompt = (
        "Solve the student question and return strict JSON with keys: "
        "concise_answer, correct_answer, answer_verdict, answer_feedback, references(list), "
        "detailed_explanation, logical_steps(list), key_concepts(list), common_mistakes(list), "
        "practice_questions(list), complexity_level. "
        "Judge whether the student's current answer is correct/incorrect/partial/review_required/not_provided. "
        "For mathematics, compute exactly and include a definitive final answer. "
        f"Subject: {subject}. Question: {question_text}. "
        f"Student attempt: {student_attempt or 'Not provided'}. "
        f"Assignment context: {assignment_context or 'n/a'}. Grade level: {grade_level or 'n/a'}."
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert tutor. Provide accurate, logically structured solutions. "
                        "Be explicit, polite, and educational. Return only JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=settings.openai_learning_temperature,
        )
        text = (response.output_text or "").strip()
        parsed = _parse_json_output(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None
