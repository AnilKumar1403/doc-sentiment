import math
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WORD_RE = re.compile(r"[a-zA-Z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "to", "of", "in", "for", "on",
    "at", "with", "as", "it", "this", "that", "be", "by", "from", "has", "have", "had", "will",
    "would", "can", "could", "should", "we", "you", "they", "he", "she", "i", "our", "their", "your",
}


def _tokens(text: str) -> list[str]:
    words = [w.lower() for w in WORD_RE.findall(text or "")]
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _sentences(text: str) -> list[str]:
    items = [s.strip() for s in SENTENCE_RE.split(text or "") if s.strip()]
    return items if items else ([text.strip()] if text.strip() else [])


def _top_keywords(text: str, top_n: int = 30) -> list[str]:
    counts = Counter(_tokens(text))
    return [k for k, _ in counts.most_common(top_n)]


def _cosine(doc: str, reference: str) -> float:
    if not doc.strip() or not reference.strip():
        return 0.0
    vect = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    try:
        matrix = vect.fit_transform([doc, reference])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def _keyword_coverage(doc: str, reference: str) -> float:
    ref_keywords = set(_top_keywords(reference, 45))
    if not ref_keywords:
        return 0.0
    doc_tokens = set(_tokens(doc))
    overlap = len(ref_keywords.intersection(doc_tokens))
    return float(overlap / len(ref_keywords))


def _length_balance(doc: str, reference: str) -> float:
    doc_len = max(1, len(_tokens(doc)))
    ref_len = max(1, len(_tokens(reference)))
    ratio = min(doc_len, ref_len) / max(doc_len, ref_len)
    return float(math.sqrt(ratio))


def _sentence_clarity_score(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    sentence_lengths = [len(_tokens(s)) for s in sents]
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    # best readability band for business writing is roughly 12-24 informative words per sentence
    if avg_len < 8:
        return 0.4
    if avg_len > 30:
        return 0.55
    distance = abs(avg_len - 18)
    return max(0.0, min(1.0, 1.0 - (distance / 18)))


def _evidence_score(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    evidence_hits = 0
    for sentence in sents:
        has_number = bool(re.search(r"\d", sentence))
        has_outcome = bool(
            re.search(
                r"\b(improv|reduc|increas|deliver|built|launched|optimized|saved|grew|scaled|led)\w*\b",
                sentence.lower(),
            )
        )
        if has_number or has_outcome:
            evidence_hits += 1
    return float(evidence_hits / len(sents))


def _tone_profile(text: str) -> str:
    lower = text.lower()
    direct_terms = ["please", "thank", "appreciate", "kindly", "respectfully"]
    action_terms = ["built", "delivered", "led", "improved", "created", "launched", "optimized", "solved"]
    direct_score = sum(term in lower for term in direct_terms)
    action_score = sum(term in lower for term in action_terms)

    if direct_score >= 2 and action_score >= 3:
        return "Professional, collaborative, and outcome-driven."
    if action_score >= 3:
        return "Confident and execution-focused."
    if direct_score >= 2:
        return "Polite and considerate, but could include stronger achievement language."
    return "Neutral; strengthen clarity, confidence, and audience intent alignment."


def compute_relevance_report(
    *,
    document_text: str,
    reference_text: str,
    analysis_type: str,
    role: str | None = None,
    context_notes: str | None = None,
) -> dict:
    cosine = _cosine(document_text, reference_text)
    keyword = _keyword_coverage(document_text, reference_text)
    balance = _length_balance(document_text, reference_text)
    clarity = _sentence_clarity_score(document_text)
    evidence = _evidence_score(document_text)

    role_bonus = 0.0
    if role:
        role_tokens = set(_tokens(role))
        doc_tokens = set(_tokens(document_text))
        if role_tokens:
            role_bonus = 0.14 * (len(role_tokens & doc_tokens) / len(role_tokens))

    score = (0.40 * cosine) + (0.25 * keyword) + (0.10 * balance) + (0.10 * clarity) + (0.15 * evidence) + role_bonus
    score = float(max(0.0, min(1.0, score)))

    doc_keywords = set(_top_keywords(document_text, 50))
    ref_keywords = set(_top_keywords(reference_text, 50))
    strengths = sorted(doc_keywords.intersection(ref_keywords))[:14]
    gaps = sorted(ref_keywords - doc_keywords)[:14]

    priority_actions: list[str] = []
    risk_flags: list[str] = []
    suggestions: list[str] = []

    if analysis_type == "resume_jd":
        if gaps:
            priority_actions.append(
                "Add or strengthen quantified bullets that include these JD-critical terms: "
                + ", ".join(gaps[:8])
                + "."
            )
        priority_actions.append("Reorder experience bullets so the first two bullets per role match top JD priorities.")
        priority_actions.append("Demonstrate business impact with explicit metrics (percentage, revenue, cost, speed, scale).")
        suggestions.append("Use one-line role summary at the top tailored to the target JD.")
        suggestions.append("Ensure each skill in the JD appears in Skills, Projects, or Experience with evidence.")
    elif analysis_type == "love_letter":
        priority_actions.append("Anchor every emotional paragraph to one specific memory, place, or moment.")
        priority_actions.append("Use a clear emotional arc: gratitude -> reflection -> commitment.")
        suggestions.append("Keep tone sincere and personal; avoid generic praise and repetitive adjectives.")
    elif analysis_type in {"email", "proposal", "contract"}:
        priority_actions.append("Open with recipient objective and desired outcome in the first 2 lines.")
        priority_actions.append("Use concise section blocks with explicit decisions, owners, and timelines.")
        suggestions.append("Add a closing action request with next-step date and accountability.")
    else:
        priority_actions.append("Align section headings and sequence to the target reference intent.")
        priority_actions.append("Increase keyword depth on missing concepts without reducing readability.")
        suggestions.append("Use examples or proof points to strengthen credibility.")

    if context_notes and len(_tokens(context_notes)) > 4:
        suggestions.append("Mirror context notes in a dedicated section to signal intent awareness.")

    if keyword < 0.35:
        risk_flags.append("Low keyword alignment with the reference text.")
    if evidence < 0.28:
        risk_flags.append("Insufficient evidence-based statements (numbers, outcomes, measurable achievements).")
    if clarity < 0.50:
        risk_flags.append("Sentence structure can be simplified for better readability and decision impact.")
    if not risk_flags:
        risk_flags.append("No major structural risks detected; focus on precision and impact amplification.")

    metrics = {
        "overall_relevance": round(score * 100, 2),
        "cosine_similarity": round(cosine * 100, 2),
        "keyword_coverage": round(keyword * 100, 2),
        "content_balance": round(balance * 100, 2),
        "clarity_score": round(clarity * 100, 2),
        "evidence_score": round(evidence * 100, 2),
        "role_alignment_bonus": round(role_bonus * 100, 2),
    }

    tone = _tone_profile(document_text)
    detailed_summary = (
        "Executive Assessment\n"
        f"- Relevance score: {metrics['overall_relevance']:.2f}%.\n"
        f"- Strategic fit level: {'High' if metrics['overall_relevance'] >= 75 else 'Moderate' if metrics['overall_relevance'] >= 50 else 'Developing'}.\n"
        f"- Communication tone: {tone}\n\n"
        "Alignment Snapshot\n"
        f"- Strength anchors: {', '.join(strengths[:8]) if strengths else 'No strong overlap terms detected yet.'}\n"
        f"- Priority gaps: {', '.join(gaps[:8]) if gaps else 'No major gap terms detected.'}\n\n"
        "Decision Guidance\n"
        "- Prioritize high-impact edits first (title, opening section, top 3 evidence bullets).\n"
        "- Keep content concise, role-aware, and outcome-driven.\n"
        "- Align language with stakeholder intent and expected decision criteria."
    )

    return {
        "relevance_score": metrics["overall_relevance"],
        "metrics": metrics,
        "strengths": strengths,
        "gaps": gaps,
        "suggestions": suggestions,
        "priority_actions": priority_actions,
        "risk_flags": risk_flags,
        "communication_tone": tone,
        "detailed_summary": detailed_summary,
    }


def _extract_strong_evidence_lines(text: str, limit: int = 3) -> list[str]:
    candidates = []
    for sentence in _sentences(text):
        sent = sentence.strip()
        if len(sent) < 30:
            continue
        has_number = bool(re.search(r"\d", sent))
        has_action = bool(
            re.search(
                r"\b(built|delivered|led|optimized|improved|launched|created|reduced|increased|scaled|designed)\w*\b",
                sent.lower(),
            )
        )
        if has_number or has_action:
            candidates.append(sent)
    return candidates[:limit]


def generate_cover_letter(
    *,
    resume_text: str,
    job_description: str,
    role: str | None,
    company: str | None,
    applicant_name: str,
) -> str:
    resume_terms = _top_keywords(resume_text, 28)
    jd_terms = _top_keywords(job_description, 28)
    overlap = [term for term in jd_terms if term in set(resume_terms)][:10]
    role_title = role or "the role"
    company_name = company or "your organization"
    highlights = _extract_strong_evidence_lines(resume_text, limit=3)

    overlap_line = ", ".join(overlap[:8]) if overlap else "problem-solving, execution quality, cross-functional collaboration"
    achievement_block = "\n".join([f"- {item}" for item in highlights]) if highlights else (
        "- Delivered measurable project outcomes through structured execution.\n"
        "- Collaborated effectively across teams to ship reliable results.\n"
        "- Maintained quality while improving speed and stakeholder communication."
    )

    return (
        f"Dear Hiring Team at {company_name},\n\n"
        f"I am pleased to apply for {role_title}. After reviewing the role requirements, I am confident my background aligns strongly with your priorities, especially in {overlap_line}. "
        "I bring a disciplined approach that combines technical depth, business context, and clear communication.\n\n"
        "Relevant evidence from my recent work includes:\n"
        f"{achievement_block}\n\n"
        "If selected, my near-term focus would be to understand your delivery goals quickly, align on measurable outcomes, and contribute high-quality execution from the first sprint cycle. "
        "I am particularly motivated by environments that value ownership, collaboration, and customer impact.\n\n"
        "Thank you for your time and consideration. I would value the opportunity to discuss how I can contribute to your team.\n\n"
        f"Sincerely,\n{applicant_name}"
    )


def _candidate_name_from_resume(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        top = lines[0]
        if len(top.split()) <= 5 and len(top) <= 60:
            return top
    return "Candidate"


def _resume_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for idx, raw in enumerate((text or "").splitlines(), start=1):
        cleaned = raw.strip()
        if cleaned:
            lines.append((idx, cleaned))
    return lines


def _is_bullet(line: str) -> bool:
    stripped = (line or "").strip()
    return stripped.startswith(("-", "*", "•"))


def _line_priority(line_no: int, line: str) -> str:
    lower = line.lower()
    if line_no <= 8 or "summary" in lower or "objective" in lower:
        return "high"
    if _is_bullet(line):
        return "high" if line_no <= 30 else "medium"
    return "medium"


def _has_any_term(text: str, terms: list[str]) -> bool:
    lower = (text or "").lower()
    return any(term.lower() in lower for term in terms if term)


def _proposed_line_update(
    *,
    current_line: str,
    role: str | None,
    company: str | None,
    jd_terms: list[str],
) -> tuple[str, str, str]:
    role_text = role or "target role"
    company_text = company or "target company"
    top_terms = [term for term in jd_terms[:6] if term]
    term_a = top_terms[0] if top_terms else "role-relevant capability"
    term_b = top_terms[1] if len(top_terms) > 1 else "business outcome"

    stripped = current_line.strip()
    if _is_bullet(stripped):
        payload = re.sub(r"^[-*•]\s*", "", stripped)
        if not re.search(r"\d", payload):
            proposed = (
                f"- {payload}. Achieved measurable impact for {company_text} by improving {term_a} and {term_b}; "
                "replace placeholders with exact numbers (e.g., +18% conversion, -22% cycle time)."
            )
            why = "Current bullet is descriptive but not measurable. Recruiters and ATS score quantified outcomes higher."
            impact = "Increases credibility, improves interview conversion, and strengthens ATS relevance."
            return proposed, why, impact
        if not _has_any_term(payload, top_terms):
            proposed = (
                f"- {payload}. Explicitly connect this result to {term_a} and {term_b} to match {role_text} requirements."
            )
            why = "Achievement exists but missing JD language alignment."
            impact = "Improves semantic match with JD and reduces rejection risk in initial screening."
            return proposed, why, impact
        proposed = (
            f"- {payload}. Keep the metric and add one decision-impact phrase linked to {role_text} ownership."
        )
        why = "Bullet is strong; refine for sharper business context."
        impact = "Improves readability and executive clarity for hiring managers."
        return proposed, why, impact

    lower = stripped.lower()
    if "summary" in lower or "objective" in lower:
        proposed = (
            f"Professional Summary: Results-focused candidate targeting {role_text} at {company_text}; "
            f"demonstrated strengths in {term_a}, {term_b}, and measurable execution."
        )
        why = "Top section should communicate fit in one recruiter scan."
        impact = "Improves first-impression relevance and profile positioning."
        return proposed, why, impact

    if not _has_any_term(stripped, top_terms):
        proposed = (
            f"{stripped} | Add role-matched phrasing for {term_a} and {term_b} where applicable."
        )
        why = "Line lacks target-role terminology used in the JD."
        impact = "Raises keyword coverage and alignment with hiring criteria."
        return proposed, why, impact

    proposed = stripped
    why = "Line is already relevant; minor wording polish only."
    impact = "Maintains content quality while improving consistency."
    return proposed, why, impact


def _line_level_modification_plan(
    *,
    resume_text: str,
    jd_terms: list[str],
    role: str | None,
    company: str | None,
) -> list[dict]:
    lines = _resume_lines(resume_text)
    if not lines:
        return []

    candidates: list[tuple[int, str]] = []
    bullets = [(ln, tx) for ln, tx in lines if _is_bullet(tx)]
    headers = [(ln, tx) for ln, tx in lines if re.search(r"\b(summary|objective|experience|project|skills)\b", tx, re.I)]

    for ln, tx in headers[:3]:
        candidates.append((ln, tx))
    for ln, tx in bullets[:5]:
        candidates.append((ln, tx))

    if len(candidates) < 8:
        for ln, tx in lines[:20]:
            if (ln, tx) not in candidates:
                candidates.append((ln, tx))
            if len(candidates) >= 8:
                break

    plan: list[dict] = []
    seen_lines: set[int] = set()
    for line_no, current in candidates:
        if line_no in seen_lines:
            continue
        seen_lines.add(line_no)
        proposed, why, impact = _proposed_line_update(
            current_line=current,
            role=role,
            company=company,
            jd_terms=jd_terms,
        )
        plan.append(
            {
                "line_number": int(line_no),
                "current_line": current,
                "proposed_line": proposed,
                "why_change": why,
                "impact": impact,
                "priority": _line_priority(line_no, current),
            }
        )
        if len(plan) >= 10:
            break
    return plan


def _section_presence(text: str) -> dict[str, bool]:
    lower = (text or "").lower()
    return {
        "summary": bool(re.search(r"\b(summary|objective|profile)\b", lower)),
        "skills": bool(re.search(r"\b(skills|technical skills|core skills)\b", lower)),
        "experience": bool(re.search(r"\b(experience|employment|work history)\b", lower)),
        "projects": bool(re.search(r"\b(projects|portfolio)\b", lower)),
        "education": bool(re.search(r"\b(education|academic)\b", lower)),
        "certifications": bool(re.search(r"\b(certification|certifications)\b", lower)),
    }


def _recommended_section_for_keyword(keyword: str) -> str:
    term = (keyword or "").lower()
    if re.search(r"\b(aws|azure|gcp|python|java|sql|spark|tableau|power bi|excel|react|node)\b", term):
        return "CORE SKILLS + EXPERIENCE"
    if re.search(r"\b(api|architecture|microservice|cloud|pipeline|automation|model|analytics)\b", term):
        return "EXPERIENCE + PROJECTS"
    if re.search(r"\b(lead|stakeholder|ownership|mentor|cross[- ]functional|communication)\b", term):
        return "PROFESSIONAL SUMMARY + EXPERIENCE"
    if re.search(r"\b(cert|scrum|pmp|itil)\b", term):
        return "CERTIFICATIONS + SUMMARY"
    return "EXPERIENCE BULLETS"


def _keyword_priority(index: int, present: bool) -> str:
    if index < 8:
        return "high"
    if index < 16:
        return "medium" if present else "high"
    return "medium"


def _build_jd_keyword_coverage(
    *,
    jd_terms: list[str],
    resume_terms: list[str],
) -> list[dict]:
    resume_set = set(resume_terms)
    coverage: list[dict] = []
    for idx, keyword in enumerate(jd_terms[:24]):
        present = keyword in resume_set
        coverage.append(
            {
                "keyword": keyword,
                "present_in_cv": present,
                "recommended_section": _recommended_section_for_keyword(keyword),
                "action": (
                    f"Keep '{keyword}' and support it with one measurable proof point."
                    if present
                    else f"Add '{keyword}' in the recommended section with context + measurable impact."
                ),
                "priority": _keyword_priority(idx, present),
            }
        )
    return coverage


def _build_strategic_action_plan(
    *,
    role: str | None,
    company: str | None,
    jd_terms: list[str],
    gaps: list[str],
    overlap_keywords: list[str],
    analysis_metrics: dict,
    section_presence: dict[str, bool],
) -> list[dict]:
    role_text = role or "target role"
    company_text = company or "target company"
    top_gaps = gaps[:12]
    top_overlap = overlap_keywords[:8]
    evidence_score = float(analysis_metrics.get("evidence_score", 0.0))
    keyword_coverage = float(analysis_metrics.get("keyword_coverage", 0.0))

    actions: list[dict] = []

    actions.append(
        {
            "area": "Role Positioning",
            "where_to_add": "Professional Summary (first 3-4 lines)",
            "what_to_add": (
                f"Rewrite the summary for {role_text} at {company_text}, and include 3 JD capabilities: "
                + ", ".join((top_overlap + top_gaps)[:3] or jd_terms[:3] or ["role fit", "delivery", "impact"])
                + "."
            ),
            "why_it_matters": "Recruiters decide relevance in the first scan. The summary sets the fit narrative.",
            "expected_impact": "Improves first-pass shortlist rate and boosts semantic role alignment.",
            "estimated_score_lift": 4.5,
            "sample_line": (
                f"Results-focused professional targeting {role_text}, delivering measurable outcomes in "
                + ", ".join((top_overlap + top_gaps)[:3] or ["execution", "analysis", "stakeholder delivery"])
                + "."
            ),
            "priority": "high",
        }
    )

    if not section_presence.get("skills", False):
        actions.append(
            {
                "area": "Missing Skills Section",
                "where_to_add": "Create a CORE SKILLS section below summary",
                "what_to_add": "Add a JD-matched skills matrix with grouped categories (tools, methods, business capabilities).",
                "why_it_matters": "ATS parsers and recruiters rely on a clear skills block for fast filtering.",
                "expected_impact": "Raises keyword extraction coverage and structured readability.",
                "estimated_score_lift": 5.0,
                "sample_line": "CORE SKILLS: Python | SQL | Data Analysis | Stakeholder Management | Process Improvement",
                "priority": "high",
            }
        )

    if top_gaps:
        actions.append(
            {
                "area": "ATS Keyword Gap Closure",
                "where_to_add": "Skills + Top bullets in Experience",
                "what_to_add": "Integrate missing JD terms naturally: " + ", ".join(top_gaps[:8]) + ".",
                "why_it_matters": "Missing exact and semantic JD terms reduce ATS recall and recruiter confidence.",
                "expected_impact": "Substantial increase in JD keyword coverage and overall match confidence.",
                "estimated_score_lift": 7.5 if keyword_coverage < 55 else 5.5,
                "sample_line": (
                    f"Led {top_gaps[0]} initiatives with cross-functional teams, delivering measurable business outcomes."
                    if top_gaps
                    else None
                ),
                "priority": "high",
            }
        )

    actions.append(
        {
            "area": "Evidence Strengthening",
            "where_to_add": "Top 2 bullets under each recent role",
            "what_to_add": "Convert task-based bullets into outcome bullets: action + metric + business impact.",
            "why_it_matters": "Recruiters trust evidence-backed statements more than responsibility lists.",
            "expected_impact": "Improves credibility, interview conversion, and hiring-manager confidence.",
            "estimated_score_lift": 8.0 if evidence_score < 45 else 4.0,
            "sample_line": "Improved process cycle time by 28% through workflow redesign and automation.",
            "priority": "high",
        }
    )

    actions.append(
        {
            "area": "Experience Mapping to JD",
            "where_to_add": "Experience bullets aligned to job requirements",
            "what_to_add": (
                "Map each major JD requirement to one explicit achievement bullet. Cover at least 6 requirements."
            ),
            "why_it_matters": "Direct requirement-to-proof mapping reduces ambiguity in fit evaluation.",
            "expected_impact": "Raises relevance score consistency across ATS and human reviews.",
            "estimated_score_lift": 6.0,
            "sample_line": (
                f"Delivered {top_overlap[0]} program aligned to business goals, improving KPI performance by 18%."
                if top_overlap
                else "Delivered role-critical initiative with measurable KPI uplift."
            ),
            "priority": "high",
        }
    )

    if not section_presence.get("projects", False):
        actions.append(
            {
                "area": "Project Proof Expansion",
                "where_to_add": "Add PROJECTS section",
                "what_to_add": "Add 2 role-relevant projects with objective, stack/method, execution steps, and results.",
                "why_it_matters": "Projects help bridge missing direct experience and demonstrate practical capability.",
                "expected_impact": "Improves role-specific relevance depth and strengthens candidate story.",
                "estimated_score_lift": 4.5,
                "sample_line": "Built an end-to-end analytics project that reduced reporting turnaround by 40%.",
                "priority": "medium",
            }
        )

    actions.append(
        {
            "area": "Leadership and Collaboration Signals",
            "where_to_add": "Experience bullets + summary",
            "what_to_add": "Show ownership, stakeholder management, and decision influence in at least 3 bullets.",
            "why_it_matters": "Most JDs evaluate collaboration and ownership alongside technical skills.",
            "expected_impact": "Improves seniority perception and cross-functional fit.",
            "estimated_score_lift": 3.0,
            "sample_line": "Partnered with product and operations leaders to prioritize roadmap and deliver critical milestones.",
            "priority": "medium",
        }
    )

    actions.append(
        {
            "area": "Application-Specific Tailoring",
            "where_to_add": "Before each submission (final pass checklist)",
            "what_to_add": "Adjust headline, summary, top 6 skills, and first 5 bullets to mirror exact JD priorities.",
            "why_it_matters": "Generic resumes underperform against role-specific evaluation criteria.",
            "expected_impact": "Consistent improvement in application response rate.",
            "estimated_score_lift": 3.5,
            "sample_line": f"Tailored for {role_text}: focus on delivery ownership, measurable impact, and domain alignment.",
            "priority": "medium",
        }
    )

    return actions[:12]


def _estimate_post_update_score(
    *,
    current_score: float,
    action_plan: list[dict],
    target_score: float,
) -> float:
    cumulative_gain = 0.0
    for action in action_plan:
        try:
            cumulative_gain += float(action.get("estimated_score_lift", 0.0))
        except Exception:
            continue
    estimated = min(96.0, current_score + cumulative_gain)
    if current_score < target_score and estimated < target_score:
        estimated = min(96.0, current_score + (target_score - current_score) + 3.0)
    return round(estimated, 2)


def build_revised_resume_package(
    *,
    resume_text: str,
    job_description: str,
    role: str | None,
    company: str | None,
    candidate_name: str | None,
    context_notes: str | None = None,
) -> dict:
    analysis = compute_relevance_report(
        document_text=resume_text,
        reference_text=job_description,
        analysis_type="resume_jd",
        role=role,
        context_notes=context_notes,
    )

    jd_terms = _top_keywords(job_description, 40)
    resume_terms = _top_keywords(resume_text, 40)
    resume_set = set(resume_terms)
    ats_keywords_added = [term for term in jd_terms if term not in resume_set][:14]
    overlap_keywords = [term for term in jd_terms if term in resume_set][:12]
    evidence_lines = _extract_strong_evidence_lines(resume_text, limit=6)
    section_presence = _section_presence(resume_text)
    target_relevance_score = 75.0

    keyword_coverage_plan = _build_jd_keyword_coverage(
        jd_terms=jd_terms,
        resume_terms=resume_terms,
    )
    strategic_action_plan = _build_strategic_action_plan(
        role=role,
        company=company,
        jd_terms=jd_terms,
        gaps=analysis.get("gaps", []),
        overlap_keywords=overlap_keywords,
        analysis_metrics=analysis.get("metrics", {}),
        section_presence=section_presence,
    )
    estimated_post_update_score = _estimate_post_update_score(
        current_score=float(analysis.get("relevance_score", 0.0)),
        action_plan=strategic_action_plan,
        target_score=target_relevance_score,
    )
    gap_to_target = round(max(0.0, target_relevance_score - float(analysis.get("relevance_score", 0.0))), 2)

    final_name = (candidate_name or "").strip() or _candidate_name_from_resume(resume_text)
    target_role = role or "Target Role"
    target_company = company or "Target Company"

    professional_summary = (
        f"Results-focused professional targeting {target_role}. Strong alignment with {target_company} priorities in "
        + ", ".join(overlap_keywords[:6] if overlap_keywords else jd_terms[:6])
        + ". Known for execution quality, stakeholder collaboration, and measurable delivery outcomes."
    )

    core_skills = sorted(set((overlap_keywords + ats_keywords_added)[:16]))
    if not core_skills:
        core_skills = jd_terms[:12]

    impact_bullets = evidence_lines[:4] if evidence_lines else [
        "Delivered measurable improvements through structured problem-solving and execution discipline.",
        "Collaborated across teams to improve delivery quality and reliability.",
        "Converted business objectives into actionable technical outcomes."
    ]
    mapped_experience = [
        f"Aligned project execution with JD priority area: {term}."
        for term in (overlap_keywords[:6] if overlap_keywords else jd_terms[:6])
    ]
    if context_notes:
        mapped_experience.append(f"Integrated application context: {context_notes.strip()[:180]}")

    revised_resume = (
        f"{final_name}\n"
        f"Target Role: {target_role}\n"
        "Location: [Your City] | Email: [Your Email] | Phone: [Your Number] | LinkedIn: [Your Link]\n\n"
        "PROFESSIONAL SUMMARY\n"
        f"{professional_summary}\n\n"
        "CORE SKILLS\n"
        + " | ".join(core_skills[:16])
        + "\n\n"
        "IMPACT HIGHLIGHTS\n"
        + "\n".join([f"- {line}" for line in impact_bullets])
        + "\n\n"
        "TAILORED EXPERIENCE BULLETS\n"
        + "\n".join([f"- {line}" for line in mapped_experience[:8]])
        + "\n\n"
        "PROJECTS\n"
        "- Add 2-3 JD-aligned projects with technology stack, objective, action, and measurable result.\n"
        "- Ensure each project explicitly demonstrates one or more critical JD capabilities.\n\n"
        "EDUCATION\n"
        "- Degree, Institution, Year\n\n"
        "CERTIFICATIONS (Optional)\n"
        "- Add role-relevant certifications if they support JD keywords and responsibilities."
    )

    revision_rationale = [
        "Strengthened top-of-resume positioning to match role intent and recruiter scan behavior.",
        "Expanded ATS-relevant skill vocabulary using JD-critical terms.",
        "Converted generic statements into impact-oriented bullets with evidence framing.",
        "Reordered emphasis toward decision-driving criteria: relevance, outcomes, and role-fit clarity.",
        f"Built an execution roadmap to move relevance from {float(analysis.get('relevance_score', 0.0)):.2f}% toward {target_relevance_score:.2f}%+.",
    ]
    if ats_keywords_added:
        revision_rationale.append(
            "Added missing ATS terms to reduce keyword rejection risk: " + ", ".join(ats_keywords_added[:8]) + "."
        )

    detailed_strategy = (
        "Strategic Resume Upgrade Plan (JD vs CV)\n"
        f"Current relevance: {float(analysis.get('relevance_score', 0.0)):.2f}% | "
        f"Target relevance: {target_relevance_score:.2f}%+ | "
        f"Projected post-update relevance: {estimated_post_update_score:.2f}%\n\n"
        "Phase 1: Critical alignment (highest lift)\n"
        "- Rewrite summary for role-fit and business outcomes.\n"
        "- Close top ATS keyword gaps in Skills + top Experience bullets.\n"
        "- Convert top bullets to quantified outcomes with decision impact.\n\n"
        "Phase 2: Proof depth and differentiation\n"
        "- Map each JD requirement to one strong proof bullet.\n"
        "- Add JD-aligned projects/case studies where direct experience is thin.\n"
        "- Show ownership, stakeholder influence, and delivery accountability.\n\n"
        "Phase 3: Submission quality control\n"
        "- Run final JD checklist for exact terminology, readability, and consistency.\n"
        "- Keep resume concise while preserving metrics and problem-solution-outcome clarity.\n"
        "- Tailor every application version to the specific role and company context."
    )

    line_level_modifications = _line_level_modification_plan(
        resume_text=resume_text,
        jd_terms=jd_terms,
        role=role,
        company=company,
    )

    return {
        "relevance_score": float(analysis["relevance_score"]),
        "target_relevance_score": target_relevance_score,
        "gap_to_target": gap_to_target,
        "estimated_post_update_score": estimated_post_update_score,
        "baseline_summary": str(analysis.get("detailed_summary") or analysis.get("summary") or ""),
        "detailed_strategy": detailed_strategy,
        "revised_resume": revised_resume,
        "revision_rationale": revision_rationale,
        "ats_keywords_added": ats_keywords_added,
        "strategic_action_plan": strategic_action_plan,
        "jd_keyword_coverage": keyword_coverage_plan,
        "line_level_modifications": line_level_modifications,
        "analysis": analysis,
    }
