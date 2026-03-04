from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOMAIN_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "learning_domain_model.joblib"
DOMAIN_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "domain"
INDIAN_SOCIAL_DATA_PATH = DOMAIN_DATA_DIR / "indian_social_studies_k12.json"
INDIAN_GK_PATH = DOMAIN_DATA_DIR / "indian_general_knowledge.json"

WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text or "") if len(t) > 2}


def _safe_eval_math(expr: str) -> float | None:
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.USub, ast.UAdd, ast.FloorDiv, ast.Load, ast.Call, ast.Name,
    }
    allowed_funcs = {"sqrt": np.sqrt, "abs": abs}

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return None

    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            return None
        if isinstance(node, ast.Name) and node.id not in allowed_funcs:
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                return None

    try:
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, allowed_funcs)
        if isinstance(value, (int, float, np.number)):
            return float(value)
    except Exception:
        return None
    return None


def _format_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _parse_attempt_value(text: str) -> float | None:
    candidate = (text or "").strip()
    if not candidate:
        return None

    normalized = candidate.lower().replace(",", "")
    normalized = normalized.replace("answer", "").replace("=", " ").strip()

    direct = _safe_eval_math(normalized)
    if direct is not None:
        return direct

    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _normalize_math_prompt(text: str) -> str:
    prompt = (text or "").strip().lower()
    prompt = prompt.replace("÷", "/").replace("×", "*").replace("^", "**")
    prompt = prompt.replace(":", " ")
    prompt = re.sub(r"\bwhat\s+is\b", "", prompt)
    prompt = re.sub(r"\bsolve\s+for\s+x\b", "", prompt)
    prompt = re.sub(r"\bsolve\b", "", prompt)
    prompt = re.sub(r"\bfind\b", "", prompt)
    prompt = re.sub(r"\bcalculate\b", "", prompt)
    prompt = re.sub(r"[?]", "", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    return prompt


def _parse_linear_equation_x(prompt: str) -> tuple[float, float, float] | None:
    normalized = re.sub(r"\s+", "", prompt)
    match = re.fullmatch(r"([+\-]?\d*\.?\d*)x([+\-]\d*\.?\d+)?=([+\-]?\d*\.?\d+)", normalized)
    if not match:
        return None

    coeff_raw, offset_raw, rhs_raw = match.groups()
    if coeff_raw in {"", "+", None}:
        a = 1.0
    elif coeff_raw == "-":
        a = -1.0
    else:
        a = float(coeff_raw)
    b = float(offset_raw) if offset_raw else 0.0
    c = float(rhs_raw)
    if abs(a) <= 1e-12:
        return None
    return a, b, c


def _solve_percentage_of(prompt: str) -> float | None:
    match = re.search(r"([+\-]?\d*\.?\d+)\s*%\s*of\s*([+\-]?\d*\.?\d+)", prompt)
    if not match:
        return None
    pct = float(match.group(1))
    base = float(match.group(2))
    return (pct / 100.0) * base


def _deterministic_math_solver(question_text: str) -> dict | None:
    prompt = _normalize_math_prompt(question_text)
    if not prompt:
        return None

    linear = _parse_linear_equation_x(prompt)
    if linear is not None:
        a, b, c = linear
        x_value = (c - b) / a
        x_text = _format_number(x_value)
        return {
            "solver_mode": "deterministic_linear",
            "correct_answer": x_text,
            "concise_answer": f"The correct answer is x = {x_text}.",
            "detailed_explanation": (
                "I solved the linear equation by isolating x: move constant terms to the right side and divide by the x-coefficient."
            ),
            "logical_steps": [
                f"Start with the equation in standard form: {prompt}.",
                "Move the constant term from the left side to the right side.",
                "Divide both sides by the coefficient of x.",
                "Substitute x back into the equation to verify.",
            ],
            "key_concepts": ["Linear equations", "Balancing equations", "Isolation of variable", "Verification"],
            "references": [
                "Algebra reference: linear equation ax + b = c, therefore x = (c - b) / a.",
                "NCERT algebra method: perform identical operations on both sides to preserve equality.",
            ],
            "complexity_level": "intermediate",
        }

    pct_value = _solve_percentage_of(prompt)
    if pct_value is not None:
        pct_text = _format_number(pct_value)
        return {
            "solver_mode": "deterministic_percentage",
            "correct_answer": pct_text,
            "concise_answer": f"The correct answer is {pct_text}.",
            "detailed_explanation": (
                "I converted the percentage to a fraction over 100 and multiplied by the base quantity."
            ),
            "logical_steps": [
                "Identify percentage p and base value n from the question.",
                "Use formula: (p/100) * n.",
                "Simplify and verify units/context.",
            ],
            "key_concepts": ["Percentages", "Fraction conversion", "Multiplicative reasoning"],
            "references": [
                "Percentage rule: p% of n = (p/100) * n.",
                "NCERT arithmetic method: convert percent to decimal/fraction before multiplication.",
            ],
            "complexity_level": "intermediate",
        }

    expr = prompt
    expr = re.sub(r"=\s*$", "", expr).strip()
    if "=" in expr:
        return None

    value = _safe_eval_math(expr)
    if value is None:
        return None

    value_text = _format_number(value)
    return {
        "solver_mode": "deterministic_arithmetic",
        "correct_answer": value_text,
        "concise_answer": f"The correct answer is {value_text}.",
        "detailed_explanation": (
            "I simplified the expression by applying arithmetic precedence (BODMAS/PEMDAS), "
            "evaluating brackets/powers first and then multiplication/division before addition/subtraction."
        ),
        "logical_steps": [
            "Rewrite the expression clearly and identify operations.",
            "Apply order of operations: bracket -> exponent -> multiply/divide -> add/subtract.",
            "Compute intermediate values carefully and verify the final value.",
        ],
        "key_concepts": ["Order of operations", "Arithmetic simplification", "Verification"],
        "references": [
            "Arithmetic rule reference: BODMAS/PEMDAS (order of operations).",
            "NCERT Mathematics practice guidance: show intermediate steps before final value.",
        ],
        "complexity_level": "intermediate",
    }

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _rank_text_candidates(
    query: str,
    candidates: list[dict],
    *,
    text_key: str,
    keywords_key: str | None = None,
    limit: int = 5,
    min_score: float = 1.0,
) -> list[dict]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    ranked: list[tuple[float, dict]] = []
    for candidate in candidates:
        text = str(candidate.get(text_key, "")).strip()
        if not text:
            continue
        text_tokens = _tokens(text)
        overlap = len(query_tokens & text_tokens)

        keyword_overlap = 0
        if keywords_key:
            raw_keywords = candidate.get(keywords_key) or []
            if isinstance(raw_keywords, list):
                keyword_tokens: set[str] = set()
                for raw in raw_keywords:
                    keyword_tokens |= _tokens(str(raw))
                keyword_overlap = len(query_tokens & keyword_tokens)

        score = float(overlap + (1.6 * keyword_overlap))
        if score < min_score:
            continue
        ranked.append((score, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:limit]]


def _social_answer_context(question_text: str) -> dict:
    social_data = _load_json(INDIAN_SOCIAL_DATA_PATH)
    gk_data = _load_json(INDIAN_GK_PATH)

    social_topics: list[dict] = []
    for cls, topics in (social_data.get("classes") or {}).items():
        for topic in topics:
            social_topics.append(
                {
                    "topic": topic,
                    "reference": f"Indian Social Studies (Class {cls}): {topic}",
                    "keywords": [topic],
                }
            )
    for topic in (social_data.get("advanced_topics") or []):
        social_topics.append(
            {
                "topic": topic,
                "reference": f"Indian Social Studies (Advanced): {topic}",
                "keywords": [topic],
            }
        )

    matched_topics = _rank_text_candidates(
        question_text,
        social_topics,
        text_key="topic",
        keywords_key="keywords",
        limit=4,
        min_score=2.0,
    )

    gk_items = gk_data.get("items") if isinstance(gk_data, dict) else []
    if not isinstance(gk_items, list):
        gk_items = []

    matched_gk = _rank_text_candidates(
        question_text,
        [item for item in gk_items if isinstance(item, dict)],
        text_key="fact",
        keywords_key="keywords",
        limit=2,
        min_score=1.3,
    )

    references: list[str] = []
    for item in matched_gk:
        topic_name = str(item.get("topic", "Indian GK")).strip()
        fact = str(item.get("fact", "")).strip()
        if fact:
            references.append(f"Indian GK - {topic_name}: {fact}")
    for topic in matched_topics:
        references.append(str(topic.get("reference", "")).strip())

    seen: set[str] = set()
    deduped_refs: list[str] = []
    for ref in references:
        key = ref.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_refs.append(ref)

    factual_lines: list[str] = []
    for item in matched_gk[:2]:
        fact = str(item.get("fact", "")).strip()
        if fact:
            factual_lines.append(fact)

    factual_answer = " ".join(factual_lines).strip()
    return {
        "references": deduped_refs[:6],
        "factual_answer": factual_answer,
    }


def _build_fallback_artifact() -> dict:
    math_data = _load_json(DOMAIN_DATA_DIR / "mathematics_k12_competitive.json")
    social_data = _load_json(DOMAIN_DATA_DIR / "indian_social_studies_k12.json")

    rows: list[dict] = []

    for cls, topics in (math_data.get("classes") or {}).items():
        for topic in topics:
            rows.append({"subject": "mathematics", "class": cls, "topic": topic, "text": f"class {cls} mathematics {topic}"})
    for topic in (math_data.get("competitive_exam_topics") or []):
        rows.append({"subject": "mathematics", "class": "competitive", "topic": topic, "text": f"competitive mathematics {topic}"})

    for cls, topics in (social_data.get("classes") or {}).items():
        for topic in topics:
            rows.append({"subject": "indian social", "class": cls, "topic": topic, "text": f"class {cls} indian social studies {topic}"})
    for topic in (social_data.get("advanced_topics") or []):
        rows.append({"subject": "indian social", "class": "advanced", "topic": topic, "text": f"advanced indian social {topic}"})

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    matrix = vectorizer.fit_transform([row["text"] for row in rows])

    artifact = {"rows": rows, "vectorizer": vectorizer, "matrix": matrix}
    return artifact


def _artifact() -> dict:
    if DOMAIN_MODEL_PATH.exists():
        try:
            loaded = joblib.load(DOMAIN_MODEL_PATH)
            if isinstance(loaded, dict) and {"rows", "vectorizer", "matrix"}.issubset(loaded.keys()):
                return loaded
        except Exception:
            pass
    return _build_fallback_artifact()


def analyze_learning_domain(*, subject: str, chapter_text: str, student_notes: str | None) -> dict:
    subject_key = subject.strip().lower()
    if subject_key not in {"mathematics", "indian social"}:
        subject_key = "mathematics"

    art = _artifact()
    rows = [r for r in art["rows"] if r["subject"] == subject_key]

    if not rows:
        return {
            "storytelling_summary": "No domain data available for this subject yet.",
            "retained_topics": [],
            "weak_topics": [],
            "suggestions": ["Add domain curriculum data and retrain the learning domain model."],
        }

    vectorizer = art["vectorizer"]
    chapter_vec = vectorizer.transform([chapter_text])

    indices = [idx for idx, r in enumerate(art["rows"]) if r["subject"] == subject_key]
    sub_matrix = art["matrix"][indices]
    sims = cosine_similarity(chapter_vec, sub_matrix)[0]

    ranked_positions = np.argsort(-sims)
    top = []
    for pos in ranked_positions[:18]:
        score = float(sims[pos])
        if score < 0.03 and len(top) >= 8:
            continue
        row = rows[int(pos)]
        top.append({"topic": row["topic"], "class": row["class"], "score": score})

    notes_tokens = _tokens(student_notes or "")
    retained = []
    weak = []

    for item in top:
        topic_tokens = _tokens(item["topic"])
        overlap = len(topic_tokens & notes_tokens)
        if overlap > 0:
            retained.append(item["topic"])
        else:
            weak.append(item["topic"])

    retained = retained[:12]
    weak = weak[:12]
    mastery_score = 0.0
    if top:
        mastery_score = round((len(retained) / max(len(top), 1)) * 100, 2)

    if subject_key == "mathematics":
        story = (
            "Imagine this chapter as a problem-solving journey. You begin with concept clarity, then move into pattern recognition, "
            "and finally convert that understanding into timed execution. The winning habit in mathematics is not just solving once, "
            "but solving repeatedly with rising speed and accuracy under pressure."
        )
        detailed_feedback = (
            "Your current performance indicates where conceptual confidence exists and where retrieval gaps still appear. "
            "To improve rapidly, treat weak topics as a training loop: learn the core rule, solve standard questions, then solve mixed "
            "exam-style questions under a timer. Use mistakes as data points, not setbacks."
        )
        suggestions = [
            "Daily pattern: 20 minutes concept refresh, 40 minutes focused weak-topic drills, 30 minutes timed mixed problems.",
            "Maintain an error log with mistake type, corrected method, and prevention rule; revise this log every day.",
            "Twice weekly, run competitive-style sectional tests across arithmetic, algebra, geometry, and DI.",
            "End each study block with a 5-line self-explanation to strengthen long-term retention.",
        ]
        study_plan = [
            "Day 1-2: Concept reconstruction and formula mapping for weak topics.",
            "Day 3-4: Standard and moderate problems with strict accuracy target.",
            "Day 5: Timed mixed set and post-test error analysis.",
            "Day 6: Advanced/competitive question set on recurring weak areas.",
            "Day 7: Revision sprint + short self-assessment quiz.",
        ]
    else:
        story = (
            "Read this chapter as a connected human story: causes, events, policies, and consequences. "
            "When you connect people, timelines, and institutional impact, social studies becomes easier to remember and explain in exams."
        )
        detailed_feedback = (
            "Your understanding improves when facts are linked to cause-effect logic instead of memorized in isolation. "
            "Build memory anchors around dates, reforms, institutions, and outcomes. This approach improves both short answers "
            "and long analytical responses."
        )
        suggestions = [
            "Create a one-page timeline and a cause-effect map for each lesson.",
            "Use flashcards for key terms, constitutional principles, leaders, and reforms.",
            "Practice structured answers: context -> key point -> evidence -> implication.",
            "Do a weekly recall session without notes to test retention depth.",
        ]
        study_plan = [
            "Day 1-2: Timeline + event linkage mapping.",
            "Day 3-4: Topic-wise short answer drills with factual evidence.",
            "Day 5: Long answer practice with structure and argument quality.",
            "Day 6: Mixed revision using flashcards and oral recall.",
            "Day 7: Mini test and reflection on weak themes.",
        ]

    return {
        "storytelling_summary": story,
        "detailed_feedback": detailed_feedback,
        "retained_topics": retained,
        "weak_topics": weak,
        "suggestions": suggestions,
        "study_plan": study_plan,
        "mastery_score": mastery_score,
    }


def solve_learning_question(
    *,
    subject: str,
    question_text: str,
    student_attempt: str | None = None,
    assignment_context: str | None = None,
    grade_level: str | None = None,
) -> dict:
    subject_key = subject.strip().lower()
    if subject_key not in {"mathematics", "indian social"}:
        subject_key = "mathematics"

    attempt_text = (student_attempt or "").strip()
    context_text = (assignment_context or "").strip()
    grade_text = (grade_level or "school").strip()
    q = question_text.strip()
    current_answer = attempt_text if attempt_text else "Not provided"
    correct_answer = ""
    answer_verdict = "not_provided" if not attempt_text else "review_required"
    references: list[str] = []
    is_numeric_math = False
    needs_llm_solver = False
    solver_mode = "heuristic"
    answer_feedback = (
        "No current answer was provided. Review the model answer and logical steps below."
        if not attempt_text
        else "Your current answer is being compared with the expected solution."
    )

    if subject_key == "mathematics":
        deterministic = _deterministic_math_solver(q)

        if deterministic is not None:
            solver_mode = str(deterministic.get("solver_mode", "deterministic"))
            is_numeric_math = True
            correct_value = str(deterministic.get("correct_answer", "")).strip()
            concise_answer = str(deterministic.get("concise_answer", f"The correct answer is {correct_value}."))
            correct_answer = correct_value
            detailed_explanation = str(deterministic.get("detailed_explanation", ""))
            references = [str(item).strip() for item in (deterministic.get("references") or []) if str(item).strip()]
            logical_steps = [str(item).strip() for item in (deterministic.get("logical_steps") or []) if str(item).strip()]
            key_concepts = [str(item).strip() for item in (deterministic.get("key_concepts") or []) if str(item).strip()]
            complexity_level = str(deterministic.get("complexity_level", "intermediate"))
            practice_questions = [
                "Solve: (18 / 3) + 4 * 2",
                "Solve: 5^2 - 3 * 4 + 6",
                "Solve: sqrt(144) + 18 / 3",
            ]
            common_mistakes = [
                "Ignoring operation order.",
                "Rounding too early.",
                "Skipping a final verification step.",
            ]

            if attempt_text:
                attempt_value = _parse_attempt_value(attempt_text)
                correct_numeric = _parse_attempt_value(correct_value)
                if attempt_value is None or correct_numeric is None:
                    answer_verdict = "review_required"
                    answer_feedback = (
                        f"Your current answer was '{attempt_text}', but it could not be validated as a numeric final value. "
                        f"The correct answer is {correct_value}. Please enter only the final numeric result."
                    )
                else:
                    parsed_attempt = _format_number(attempt_value)
                    if abs(attempt_value - correct_numeric) <= 1e-6:
                        answer_verdict = "correct"
                        answer_feedback = (
                            f"Your current answer ({parsed_attempt}) is correct. "
                            "Keep showing each logical step clearly."
                        )
                    else:
                        answer_verdict = "incorrect"
                        answer_feedback = (
                            f"Your current answer is {parsed_attempt}, but the correct answer is {correct_value}. "
                            "Recheck the operation sequence and recompute each step."
                        )
        else:
            solver_mode = "unresolved_math"
            needs_llm_solver = True
            concise_answer = "The problem requires a step-wise method. Use the structured approach below."
            correct_answer = "A complete step-wise computation is required to derive the final value."
            detailed_explanation = (
                "For complex mathematics questions, first identify the concept family (algebra, geometry, arithmetic, trigonometry), "
                "then solve in stages: given data, required output, formula/method, substitution, and validation."
            )
            references = [
                "Problem-solving framework: identify knowns, unknowns, method, substitution, validation.",
                "NCERT/competitive math method: justify each transition in algebraic steps.",
            ]
            logical_steps = [
                "Write the given data and define the exact unknown.",
                "Choose the relevant formula/theorem and explain why it applies.",
                "Substitute values step-by-step and simplify without skipping transitions.",
                "Check units/sign and confirm reasonableness of final answer.",
            ]
            key_concepts = ["Concept identification", "Formula selection", "Step-wise substitution", "Answer validation"]
            practice_questions = [
                "Create one similar problem and solve it fully.",
                "Create one harder variant with changed values and solve it.",
                "Explain your final method in 5 lines without symbols.",
            ]
            common_mistakes = [
                "Applying the wrong formula to the problem type.",
                "Skipping algebraic simplification steps.",
                "Not validating the final result against question conditions.",
            ]
            complexity_level = "advanced" if len(_tokens(q)) > 15 else "intermediate"
            if attempt_text:
                answer_verdict = "review_required"
                answer_feedback = (
                    "A final value cannot be validated automatically for this question format. "
                    "Use the logical steps below to compare each intermediate step with your current answer."
                )
    else:
        solver_mode = "social_retrieval"
        focus_terms = list(_tokens(q))[:8]
        social_context = _social_answer_context(q)
        factual_answer = str(social_context.get("factual_answer", "")).strip()
        if factual_answer:
            concise_answer = factual_answer
            correct_answer = factual_answer
        else:
            concise_answer = "This question can be answered using a cause -> event -> impact explanation structure."
            correct_answer = concise_answer
        detailed_explanation = (
            "In social studies, strong answers explain context first, then the key event/idea, and finally the impact on society, "
            "governance, rights, economy, or culture. This approach improves assignment quality and exam scoring."
        )
        references = social_context.get("references", [])
        if not isinstance(references, list):
            references = []
        references = [str(item).strip() for item in references if str(item).strip()]
        if not references:
            references = [
                "Indian Social Studies answer structure: context -> evidence -> impact.",
                "Indian GK revision approach: validate key dates, institutions, and constitutional references.",
            ]
        logical_steps = [
            "Start with brief historical/civic context in 1-2 lines.",
            "State the core concept/event clearly.",
            "Add evidence: dates, institutions, reforms, personalities, or constitutional references.",
            "Conclude with impact and present-day relevance.",
        ]
        if factual_answer:
            detailed_explanation += " A factual answer has been grounded using matched Indian GK references."
        key_concepts = focus_terms or ["Cause-effect", "Governance", "Social impact", "Evidence-based writing"]
        practice_questions = [
            "Write a short answer using context -> event -> impact format.",
            "Write one long answer with two supporting examples.",
            "Compare two related concepts in a table (similarities vs differences).",
        ]
        common_mistakes = [
            "Listing facts without linking cause and effect.",
            "Missing evidence (date, reform, policy, or institution).",
            "No concluding insight on impact or significance.",
        ]
        complexity_level = "intermediate" if len(_tokens(q)) < 18 else "advanced"
        if attempt_text:
            attempt_terms = _tokens(attempt_text)
            if factual_answer:
                expected_terms = _tokens(factual_answer)
            else:
                expected_terms = set(focus_terms)
            coverage = (len(expected_terms & attempt_terms) / max(len(expected_terms), 1)) if expected_terms else 0.0
            if coverage >= 0.75:
                answer_verdict = "correct"
                answer_feedback = (
                    "Your current answer captures most expected concepts. Strengthen evidence with dates/policies for full marks."
                )
            elif coverage >= 0.35:
                answer_verdict = "partial"
                answer_feedback = (
                    "Your current answer is partially correct. Add clearer context, factual evidence, and impact statements."
                )
            else:
                answer_verdict = "incorrect"
                answer_feedback = (
                    "Your current answer misses key concepts expected in this question. Follow the model structure below."
                )

            if factual_answer and answer_verdict == "incorrect":
                expected_years = set(re.findall(r"\b\d{4}\b", factual_answer))
                attempt_years = set(re.findall(r"\b\d{4}\b", attempt_text))
                if expected_years & attempt_years:
                    answer_verdict = "partial"
                    answer_feedback = (
                        "Your current answer captures part of the timeline, but misses complete factual detail. "
                        "Add both adoption and effective dates with constitutional context."
                    )

    if attempt_text:
        detailed_explanation += (
            " Your attempt was considered, and the recommendation is to keep your reasoning visible at each intermediate step "
            "so errors can be diagnosed early."
        )
    if context_text:
        detailed_explanation += " The response is aligned with your assignment context to improve practical usefulness."

    detailed_explanation += f" Suggested level focus: {grade_text}."

    return {
        "subject": subject_key,
        "question_text": q,
        "concise_answer": concise_answer,
        "current_answer": current_answer,
        "correct_answer": correct_answer,
        "answer_verdict": answer_verdict,
        "answer_feedback": answer_feedback,
        "references": references,
        "is_numeric_math": is_numeric_math,
        "needs_llm_solver": needs_llm_solver,
        "solver_mode": solver_mode,
        "detailed_explanation": detailed_explanation,
        "logical_steps": logical_steps,
        "key_concepts": key_concepts,
        "common_mistakes": common_mistakes,
        "practice_questions": practice_questions,
        "complexity_level": complexity_level,
    }
