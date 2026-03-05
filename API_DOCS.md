# API Integration Guide - Anil's aquaanalysis

Base URL (local):
- `http://127.0.0.1:8000`

API version:
- All integration endpoints are under `/api/v1`

Authentication:
- Obtain token from `POST /api/v1/auth/login` or `POST /api/v1/auth/register`
- Send token in header: `Authorization: Bearer <access_token>`
- Cookie auth is also supported for browser flows

Credits:
- `anilkumargolla444@gmail.com` is configured as unlimited plan
- other users default to 25 credits
- resume generation is available as a common feature (no per-request credit deduction)

Content types:
- JSON endpoints: `application/json`
- File endpoints: `multipart/form-data`

## 1) Auth APIs

### `POST /api/v1/auth/register`
Create a new account and return JWT.

Request:
```json
{
  "display_name": "Anil Kumar",
  "email": "anil@example.com",
  "password": "StrongPass123!"
}
```

Response:
```json
{
  "user": {
    "id": 2,
    "email": "anil@example.com",
    "display_name": "Anil Kumar",
    "credits_remaining": 25,
    "is_unlimited": false,
    "created_at": "2026-03-04T22:10:58.001Z"
  },
  "message": "Account created",
  "access_token": "<jwt>",
  "jwt_token": "<jwt>",
  "token_type": "bearer"
}
```

### `POST /api/v1/auth/login`
Login and return JWT.

Request:
```json
{
  "email": "anilkumargolla444@gmail.com",
  "password": "Anil2020@b"
}
```

### `GET /api/v1/auth/me`
Get current user profile.

### `POST /api/v1/auth/logout`
Clear auth cookie (token-based integrations should discard their local token).

## 2) Dashboard APIs

### `GET /api/v1/dashboard/summary`
Returns total docs, high-alert count, last analysis time, and top emotions.

Additional analytics fields:
- `module_analytics`: per-module analytics blocks for `sentiment`, `relevance`, `learning`
- `total_analyses`: total analyses across modules

### `GET /api/v1/model/details`
Returns model metadata, labels, thresholds, and train metrics.

## 3) Sentiment Module APIs

### `POST /api/v1/sentiment/analyze-text`
Request:
```json
{
  "title": "Customer Email",
  "content": "I am worried about the delay in delivery...",
  "emotion_metrics": "worried,stressed,polite"
}
```

### `POST /api/v1/sentiment/analyze-file`
`multipart/form-data`:
- `title` (string, required)
- `emotion_metrics` (string, optional; comma-separated)
- `file` (binary, required)

Supported file families:
- txt, pdf, docx, image (OCR via Tesseract)

### `GET /api/v1/sentiment/history`
Returns historical sentiment analysis entries for logged-in user.

## 4) Relevance Module APIs

### `POST /api/v1/relevance/analyze-text`
Request:
```json
{
  "title": "Resume Match",
  "analysis_type": "resume_jd",
  "role": "Data Analyst",
  "company": "ExampleCorp",
  "document_text": "Resume text...",
  "reference_text": "Job description text...",
  "context_notes": "Targeting product analytics role"
}
```

Key response sections:
- `summary` (executive one-paragraph assessment)
- `detailed_summary` (structured strategic narrative)
- `priority_actions` (ordered high-impact actions)
- `risk_flags` (quality risks to address)
- `communication_tone` (tone diagnosis)
- `generated_cover_letter` (for `resume_jd`)

### `POST /api/v1/relevance/analyze-file`
`multipart/form-data`:
- `title` (required)
- `analysis_type` (optional; default `general`)
- `role`, `company`, `context_notes` (optional)
- `document_text` and/or `document_file` (at least one required)
- `reference_text` and/or `reference_file` (at least one required)

### `POST /api/v1/relevance/generate-resume`
Generate a revised ATS-aware resume based on CV + JD.

Request:
```json
{
  "title": "Resume rewrite for Data Analyst role",
  "cv_text": "Current resume content...",
  "jd_text": "Job description content...",
  "role": "Data Analyst",
  "company": "ExampleCorp",
  "candidate_name": "Anil Kumar",
  "context_notes": "Targeting analytics + stakeholder-facing role"
}
```

Response highlights:
- `relevance_score`
- `target_relevance_score`, `gap_to_target`, `estimated_post_update_score`
- `baseline_summary`
- `detailed_strategy`
- `revised_resume`
- `revision_rationale`
- `ats_keywords_added`
- `strategic_action_plan` (where to add, what to add, why it matters, expected impact, estimated score lift, sample line)
- `jd_keyword_coverage` (keyword present in CV, recommended section, action, priority)
- `line_level_modifications` (line number, current line, proposed line, reason, impact, priority)
- `generated_cover_letter`
- `credits_remaining`, `is_unlimited`

### `POST /api/v1/relevance/generate-resume-file`
`multipart/form-data`:
- `title` (required)
- `cv_text` and/or `cv_file` (at least one required)
- `jd_text` and/or `jd_file` (at least one required)
- `role`, `company`, `candidate_name`, `context_notes` (optional)

Compatibility aliases:
- `POST /api/v1/relevance/generate_resume`
- `POST /api/v1/relevance/generate_resume_file`
- `POST /api/v1/relevance/resume-generator`
- `POST /api/relevance/generate-resume`
- `POST /api/relevance/generate_resume`
- `POST /api/relevance/resume-generator`
- `POST /api/relevance/generate-resume-file`
- `POST /api/relevance/generate_resume_file`
- `POST /relevance/generate-resume`
- `POST /relevance/generate_resume`
- `POST /relevance/resume-generator`
- `POST /relevance/generate-resume-file`
- `POST /relevance/generate_resume_file`

## 5) Learning Module APIs

### `POST /api/v1/learning/story`
Request:
```json
{
  "subject": "mathematics",
  "chapter_text": "Quadratic equations chapter...",
  "student_notes": "I understand factorization but not discriminant."
}
```

Key response sections:
- `storytelling_summary` (narrative explanation)
- `detailed_feedback` (diagnostic learning guidance)
- `study_plan` (step-by-step training plan)
- `mastery_score` (0-100 readiness indicator)

### `POST /api/v1/learning/story-file`
`multipart/form-data`:
- `subject` (required; e.g. `mathematics`, `indian social`)
- `chapter_text` and/or `chapter_file` (at least one required)
- `student_notes` and/or `student_notes_file` (optional)

### `POST /api/v1/learning/question-answer`
Student homework/assignment Q&A solver for Mathematics and Indian Social.

Request:
```json
{
  "subject": "mathematics",
  "question_text": "Solve: (24/3) + 5*2",
  "student_attempt": "I got 26",
  "assignment_context": "Class 8 homework, chapter arithmetic",
  "grade_level": "Class 8"
}
```

Response highlights:
- `current_answer`
- `correct_answer`
- `answer_verdict` (`correct`/`incorrect`/`partial`/`review_required`/`not_provided`)
- `answer_feedback`
- `references` (topic/GK references used for the answer)
- `concise_answer`
- `detailed_explanation`
- `logical_steps`
- `key_concepts`
- `common_mistakes`
- `practice_questions`
- `complexity_level`

Math correctness guard:
- for deterministic arithmetic expressions, backend computes the numeric answer first and locks `correct_answer` / `answer_verdict` before any LLM enhancement.
- deterministic solver currently covers arithmetic expressions, linear equations (single variable `x`), and percentage-of questions.

Compatibility aliases:
- `POST /api/v1/learning/qa`
- `POST /api/v1/learning/question_answer`
- `POST /api/learning/question-answer`
- `POST /api/learning/qa`
- `POST /api/learning/question_answer`
- `POST /learning/question-answer`
- `POST /learning/qa`
- `POST /learning/question_answer`
- `POST /learning/story` (fallback mode)

## 6) Unified/History APIs

### `GET /api/v1/documents/history`
Returns a unified timeline of sentiment + relevance + learning analyses.

Includes:
- sentiment text/file analysis entries
- relevance text/file analysis entries
- learning story analysis entries
- learning question-answer entries

Response shape:
```json
[
  {
    "id": 10000021,
    "module": "relevance",
    "title": "Resume Match",
    "source_type": "file",
    "analysis_type": "resume_jd",
    "label": "relevance",
    "score": 74.12,
    "summary": "Executive summary ...",
    "suggestions": ["..."],
    "details": {"priority_actions": ["..."], "risk_flags": ["..."]},
    "created_at": "2026-03-04T21:57:22.401Z"
  }
]
```

### `GET /api/v1/documents/{document_id}`
Returns single document entry for logged-in user.

## 7) Error Contract

Standard API errors:
- `400` invalid payload, unsupported content, or missing required text/file combo
- `401` unauthenticated or invalid/expired token
- `404` missing route/resource
- `405` wrong HTTP method on valid route
- `413` file too large (>10MB)
- `500` server/model failures

Typical error payload:
```json
{
  "detail": "Error message"
}
```

## 8) Integration Examples

### Login + Reuse Token
```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"anilkumargolla444@gmail.com","password":"Anil2020@b"}' | jq -r '.access_token')
```

### Sentiment Text Analysis
```bash
curl -X POST http://127.0.0.1:8000/api/v1/sentiment/analyze-text \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Mail","content":"I am stressed about deadlines.","emotion_metrics":"stressed,worried"}'
```

### Relevance with Files
```bash
curl -X POST http://127.0.0.1:8000/api/v1/relevance/analyze-file \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Resume vs JD" \
  -F "analysis_type=resume_jd" \
  -F "role=Backend Engineer" \
  -F "document_file=@/absolute/path/resume.pdf" \
  -F "reference_file=@/absolute/path/jd.pdf"
```

### Resume Generation
```bash
curl -X POST http://127.0.0.1:8000/api/v1/relevance/generate-resume \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title":"Resume rewrite for Backend Engineer",
    "cv_text":"Current resume text...",
    "jd_text":"JD text...",
    "role":"Backend Engineer",
    "company":"ExampleCorp",
    "candidate_name":"Anil Kumar"
  }'
```

### Learning with Attachments
```bash
curl -X POST http://127.0.0.1:8000/api/v1/learning/story-file \
  -H "Authorization: Bearer $TOKEN" \
  -F "subject=mathematics" \
  -F "chapter_file=@/absolute/path/chapter.pdf" \
  -F "student_notes_file=@/absolute/path/answers.txt"
```

### Learning Q&A (Homework/Assignments)
```bash
curl -X POST http://127.0.0.1:8000/api/v1/learning/question-answer \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "subject":"mathematics",
    "question_text":"Solve (24/3)+5*2",
    "student_attempt":"26",
    "assignment_context":"Class 8 homework",
    "grade_level":"Class 8"
  }'
```

## 9) OpenAPI/Swagger

Interactive docs:
- `/docs`

OpenAPI JSON:
- `/openapi.json`
