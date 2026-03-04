# Anil's aquaanalysis (Single Source of Reference)

Anil's aquaanalysis is a login-first AI application for:
- Relevance analysis (resume vs JD, love letter, email, proposal, contract, general)
- CV-to-JD revised resume generation for job applications
- Learning analysis (Mathematics + Indian Social)
- Student Q&A solver for homework/assignments (detailed logical answers)
- Separate Sentiment module (same flow as earlier)
- OCR extraction from uploaded documents

This README is the authoritative reference for architecture, APIs, schema, model logic, data sources, and setup.

Integration API details with request/response examples are documented in:
- `API_DOCS.md`

## 1) Product Navigation and Separation

Login-first behavior:
- `/` redirects to `/login`
- analysis endpoints require authentication (`401` if not logged in)

Pages are strictly separated:
- `/dashboard` -> analytics only
- `/sentiment` -> sentiment-only workflows
- `/relevance` -> relevance-only workflows
- `/learning` -> learning-only workflows
- `/history` -> history only
- `/profile` -> profile/account info only
- `/login` -> general intro content + login component + footer

Frontend enforcement:
- each page is rendered in a dedicated `.view`
- hidden views use `.view.hidden { display: none !important; }` to prevent cross-page leakage

## 2) UI Theme

App name: **Anil's aquaanalysis**

Theme style:
- light green + light purple + light blue palette
- clean, low-noise visual layout
- watercolor-like gradients
- section-specific screens without mixed content

## 3) Tech Stack and Technologies

Frontend:
- HTML5, CSS3, Vanilla JavaScript
- Route-state UI with login guard

Backend:
- FastAPI
- SQLAlchemy
- Pydantic
- JWT auth (`python-jose`)
- Password hashing (`passlib` + `bcrypt`)

Document and OCR:
- `pypdf`, `python-docx`
- `PyMuPDF`
- `pytesseract` + `Pillow`

AI/ML:
- scikit-learn (`TfidfVectorizer`, `OneVsRestClassifier`, `LogisticRegression`)
- keyword lexicon hybrid boost
- per-label threshold calibration

Public datasets and data tools:
- Hugging Face `datasets`
- GoEmotions mapped dataset for stronger emotion robustness
- Curriculum domain data for Math and Indian Social (classes 1–12 + competitive topics)

LLM enhancement (optional):
- OpenAI SDK (`openai`)
- default model: `gpt-4o-mini`
- configurable temperatures:
  - relevance: `OPENAI_RELEVANCE_TEMPERATURE`
  - learning: `OPENAI_LEARNING_TEMPERATURE`

Data:
- SQLite local default
- PostgreSQL supported via `DATABASE_URL`

## 4) Models and Training

### 4.1 Emotion/Sentiment model
Current:
- `tfidf-ovr-logreg-keyword-hybrid`
- `v3-multi-emotion`

Training sources:
1. synthetic multi-label corpus (`scripts/generate_seed_dataset.py`)
2. public GoEmotions mapping (`scripts/fetch_public_dataset.py`)

Training script:
- `scripts/train_small_model.py`

### 4.2 Learning domain model (Math + Indian Social)
Curriculum datasets included:
- `backend/data/domain/mathematics_k12_competitive.json`
- `backend/data/domain/indian_social_studies_k12.json`
- `backend/data/domain/indian_general_knowledge.json`

Domain training script:
- `scripts/train_learning_domain_model.py`

Domain inference module:
- `backend/app/learning_domain.py`

Coverage:
- Mathematics class 1–12 topics + competitive exam topics
- Indian Social class 1–12 topics + advanced civics/history themes

## 5) API Structure

### Auth
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`
  - returns plan and credits metadata (`is_unlimited`, `credits_remaining`)
  - auth response includes both `access_token` and `jwt_token`

### Dashboard/Model
- `GET /api/v1/dashboard/summary`
- `GET /api/v1/model/details`
  - dashboard summary now includes `module_analytics` and `total_analyses`
  - module analytics are separated for `sentiment`, `relevance`, and `learning`

### Sentiment module (separate)
- `POST /api/v1/sentiment/analyze-text`
- `POST /api/v1/sentiment/analyze-file`
- `GET /api/v1/sentiment/history`

### Relevance module
- `POST /api/v1/relevance/analyze-text`
- `POST /api/v1/relevance/analyze-file`
- `POST /api/v1/relevance/generate-resume`
- `POST /api/v1/relevance/generate-resume-file` (file support)
  - compatibility aliases:
    - `POST /api/v1/relevance/generate_resume`
    - `POST /api/v1/relevance/generate_resume_file`
    - `POST /api/v1/relevance/resume-generator`
    - `POST /api/relevance/generate-resume`
    - `POST /api/relevance/generate_resume`
    - `POST /api/relevance/resume-generator`
    - `POST /api/relevance/generate-resume-file`
    - `POST /api/relevance/generate_resume_file`
  - plain-path compatibility aliases for proxy/static setups:
    - `POST /relevance/generate-resume`
    - `POST /relevance/generate_resume`
    - `POST /relevance/resume-generator`
    - `POST /relevance/generate-resume-file`
    - `POST /relevance/generate_resume_file`
  - accepts text and/or attachments for both sides:
    - `document_text` and/or `document_file`
    - `reference_text` and/or `reference_file`
  - returns richer strategic output:
    - `summary`, `detailed_summary`
    - `priority_actions`, `risk_flags`
    - `communication_tone`
    - stronger `generated_cover_letter` for `resume_jd`
  - resume generation returns:
    - `revised_resume`
    - `detailed_strategy`
    - `line_level_modifications` (line number + current text + proposed text + why + impact + priority)
    - `revision_rationale`
    - `ats_keywords_added`
    - `generated_cover_letter`
    - account metadata (`is_unlimited`, `credits_remaining`)

### Learning module
- `POST /api/v1/learning/story`
- `POST /api/v1/learning/story-file`
- `POST /api/v1/learning/question-answer`
  - aliases available for compatibility:
    - `POST /api/v1/learning/qa`
    - `POST /api/v1/learning/question_answer`
    - `POST /api/learning/question-answer`
    - `POST /api/learning/qa`
    - `POST /api/learning/question_answer`
    - `POST /learning/question-answer`
    - `POST /learning/qa`
    - `POST /learning/question_answer`
  - student question/assignment solver for Mathematics and Indian Social
  - returns current answer, correct answer, verdict, feedback, references, detailed explanation, logical steps, mistakes, and practice questions
  - deterministic math correctness layer for arithmetic, linear equations, and percentage questions (LLM cannot override computed numeric answer)
  - accepts text and/or attachments:
    - `chapter_text` and/or `chapter_file`
    - `student_notes` and/or `student_notes_file`
  - returns richer coaching output:
    - `storytelling_summary`, `detailed_feedback`
    - `study_plan`, `mastery_score`

### Unified history
- `GET /api/v1/documents/history`
  - includes sentiment + relevance + learning runs in one chronological feed
  - learning story and learning Q&A are both tracked in history

### System
- `GET /health`
- OpenAPI: `/docs`, `/openapi.json`
- Integration guide with curl examples: `API_DOCS.md`

## 6) Environment Variables

Use `backend/.env`:
- `DATABASE_URL`
- `MODEL_PATH`
- `API_TITLE`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `CORS_ORIGINS`
- `TESSERACT_CMD`
- `SEEDED_USER_EMAIL`
- `SEEDED_USER_PASSWORD`
- `DEFAULT_USER_CREDITS` (default `25`)
- `UNLIMITED_USER_EMAIL` (default `anilkumargolla444@gmail.com`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `OPENAI_RELEVANCE_TEMPERATURE` (default `0.1`)
- `OPENAI_LEARNING_TEMPERATURE` (default `0.2`)

## 7) Setup and Run

```bash
cd /Users/anilkumar/Documents/Playground/doc-sentiment/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# emotion model data
python scripts/generate_seed_dataset.py
python scripts/fetch_public_dataset.py
python scripts/train_small_model.py

# learning domain model
python scripts/train_learning_domain_model.py

# run app
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/login`

## 8) OCR Notes

Install Tesseract:
```bash
brew install tesseract
```

If needed:
```bash
export TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

Supported file formats:
- `.txt`, `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.tiff`

## 9) Database and Schema

Tables:
- `users` (includes `credits_remaining`, `is_unlimited`)
- `documents`
- `sentiment_results`
- `analysis_history` (relevance + learning timeline)

Credit policy:
- `anilkumargolla444@gmail.com` -> unlimited credits
- all other users -> `DEFAULT_USER_CREDITS` (default 25)
- resume generation is available as a common feature (no per-request credit deduction)

Inspect local DB:
```bash
sqlite3 /Users/anilkumar/Documents/Playground/doc-sentiment/backend/sentiment.db
```

Useful SQL:
```sql
.tables
select id,email,display_name,created_at from users;
select id,email,is_unlimited,credits_remaining,created_at from users;
select id,title,source_type,created_at from documents order by id desc limit 20;
select document_id,label,confidence,model_name,model_version from sentiment_results order by id desc limit 20;
select id,module_name,title,analysis_type,score,created_at from analysis_history order by id desc limit 20;
```

## 10) Security Note

Do not store real API keys in `.env.example` or committed files.
Keep secrets only in local `.env`.
