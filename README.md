# Document Emotion Intelligence Platform (Single Source of Truth)

This repository is an end-to-end full-stack system for authenticated document analysis using OCR + multi-emotion AI.

It includes:
- Login-first web app (`/login` is master entry page)
- Account creation + authentication (JWT + HttpOnly cookie + bearer support)
- Document ingestion (text + file upload with OCR)
- Custom emotion metrics selection at analysis time
- Multi-emotion classification + summary + suggestions report
- Dashboard + history + profile
- Persisted DB schema for users, documents, emotion results

---

## 1) Product Behavior

### 1.1 Page routing and auth UX
- Root `/` redirects to `/login`.
- Routes:
  - `/login`
  - `/dashboard`
  - `/analyze`
  - `/history`
  - `/profile`
- UI hides all application views unless user is authenticated.
- After successful login/register, UI routes to dashboard and enables navigation.

### 1.2 Create account and login
- Create account endpoint inserts a new user with hashed password.
- Duplicate emails are rejected with `409`.
- Login validates credentials and returns both:
  - `access_token` (bearer)
  - HttpOnly cookie (`access_token`)
- Frontend stores bearer token and uses it for all API calls.

---

## 2) AI Model (Current v3)

### 2.1 Model architecture
`tfidf-ovr-logreg-keyword-hybrid` (`v3-multi-emotion`)

Pipeline:
1. `TfidfVectorizer` with 1-3 gram features
2. `OneVsRestClassifier(LogisticRegression)` for multi-label emotion scoring
3. Label threshold calibration per emotion on validation split
4. Keyword-signal adjustment layer to improve short text robustness

### 2.2 Emotion labels
- anger
- sweet
- fear
- sad
- nice
- emotional
- love
- drama
- depressed
- worried
- tensed
- stressed
- sick
- fight
- calm
- polite
- joy
- frustrated
- hopeful
- confused
- grateful

### 2.3 Training dataset
Generated synthetic multi-label dataset (`scripts/generate_seed_dataset.py`) with:
- single-emotion examples
- mixed-emotion examples
- keyword-driven examples
- contextual prefixes/suffixes

Current training stats are exposed by API and stored in artifact metadata.

### 2.4 Model details API
`GET /api/v1/model/details`
Returns:
- `model_name`
- `model_version`
- `labels`
- `thresholds`
- `train_metrics` (`micro_f1`, `macro_f1`, `samples`)

---

## 3) Analysis Logic

### 3.1 Inputs
You can analyze:
- Direct text (`/api/v1/documents/analyze-text`)
- Uploaded file (`/api/v1/documents/analyze-file`)

Optional metric filter:
- `emotion_metrics`: comma-separated list
- Example: `drama,love,anger`

### 3.2 Output report
Each analysis returns:
- Dominant emotion and confidence
- Selected metrics
- Per-metric emotion scores
- Summary paragraph
- Actionable suggestions list

### 3.3 File and OCR support
Supported files:
- `.txt`, `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.tiff`

OCR requirement:
- Tesseract binary installed locally
- Optional env: `TESSERACT_CMD=/opt/homebrew/bin/tesseract`

---

## 4) API Structure

### 4.1 Auth
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`

### 4.2 Model
- `GET /api/v1/model/details`

### 4.3 Documents
- `POST /api/v1/documents/analyze-text`
- `POST /api/v1/documents/analyze-file`
- `POST /api/v1/documents/analyze` (legacy alias)
- `GET /api/v1/documents/history`
- `GET /api/v1/documents/{document_id}`

### 4.4 Dashboard
- `GET /api/v1/dashboard/summary`

### 4.5 Health
- `GET /health`

OpenAPI docs:
- `/docs`
- `/openapi.json`

---

## 5) Database Schema

Tables:
- `users`
  - email unique
  - display_name
  - password_hash
  - timestamps
- `documents`
  - owner_id -> users
  - title, content
  - source metadata (`source_type`, `file_name`, `mime_type`)
  - extracted_char_count
- `sentiment_results`
  - document_id unique FK
  - dominant label + confidence
  - `emotion_scores_json`
  - `selected_metrics_json`
  - `summary_text`
  - `suggestions_json`
  - model metadata

Backward-compatible column migration is applied at startup in `backend/app/bootstrap.py`.

---

## 6) Setup & Run

```bash
cd /Users/anilkumar/Documents/Playground/doc-sentiment/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/generate_seed_dataset.py
python scripts/train_small_model.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/login`

---

## 7) Seeded account

- Email: `anilkumargolla444@gmail.com`
- Password: `Anil2020@b`

Used for quick initial login. New accounts can be created from UI.

---

## 8) Database Access

Current local DB:
- `/Users/anilkumar/Documents/Playground/doc-sentiment/backend/sentiment.db`

Inspect:
```bash
sqlite3 /Users/anilkumar/Documents/Playground/doc-sentiment/backend/sentiment.db
```

Useful SQL:
```sql
.tables
select id,email,display_name,created_at from users;
select id,title,source_type,created_at from documents order by id desc limit 20;
select document_id,label,confidence,model_name,model_version from sentiment_results order by id desc limit 20;
```

---

## 9) Quality Notes and Next Accuracy Upgrade

Current model is strong as a practical baseline for your listed emotions and supports targeted metric filtering.

For best production accuracy on your domain:
1. Collect real labeled datasets from your own documents.
2. Add human-reviewed label guidelines per emotion.
3. Fine-tune transformer multi-label model (DeBERTa/BERT) on real data.
4. Calibrate per-label thresholds on held-out validation data.
5. Add continuous feedback loop from user corrections.
