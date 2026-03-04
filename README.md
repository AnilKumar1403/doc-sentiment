# Document Sentiment Intelligence Platform

Production-style starter for document sentiment analysis with:

- Account system (register/login/logout/profile)
- Seeded admin login
- Document upload + OCR/text extraction
- Sentiment model inference and persistence
- Dashboard, Analyze, History UI
- FastAPI backend + SQLAlchemy + PostgreSQL/SQLite

## Features

- Authenticated multi-user data ownership
- File ingestion support: `.txt`, `.pdf`, `.docx`, `.png`, `.jpg`, `.jpeg`, `.tiff`
- OCR for image files and scanned PDFs
- Model inference: `TF-IDF + Logistic Regression`
- User dashboard with key stats
- Full analysis history per user

## Seeded Account

Seeded automatically at backend startup:

- Username: `anilkumargolla444@gmail.com`
- Password: `Anil2020@b`

For production, change seeded credentials and rotate secrets.

## Architecture

- Backend: `backend/app/main.py`
- Security/JWT/password hashing: `backend/app/security.py`
- OCR + file extraction: `backend/app/document_extractor.py`
- Model inference: `backend/app/ml.py`
- DB bootstrap + seed: `backend/app/bootstrap.py`
- Frontend: `frontend/index.html`, `frontend/styles.css`, `frontend/app.js`
- SQL schema: `database/schema.sql`

## Setup (Local SQLite mode)

1. Create and activate virtual env

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env

```bash
cp .env.example .env
# For local no-docker run:
# DATABASE_URL=sqlite:///./sentiment.db
```

3. Train sentiment model

```bash
python scripts/generate_seed_dataset.py
python scripts/train_small_model.py
```

4. Start API/UI

```bash
uvicorn app.main:app --reload --port 8000
```

5. Open

- `http://localhost:8000`

## Setup (PostgreSQL via Docker)

From project root:

```bash
docker compose up -d
```

Then in `backend/`:

```bash
cp .env.example .env
source .venv/bin/activate
psql "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db" -f ../database/schema.sql
python scripts/generate_seed_dataset.py
python scripts/train_small_model.py
uvicorn app.main:app --reload --port 8000
```

## OCR Prerequisite

`pytesseract` requires the Tesseract binary on host machine.

macOS:

```bash
brew install tesseract
```

If Tesseract is installed but not on PATH, set:

```bash
export TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

Without Tesseract, image/scanned-PDF OCR endpoints return a clear error message.

## API Overview

### Auth

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/logout`
- `GET /api/v1/auth/me`

### Documents

- `POST /api/v1/documents/analyze-text`
- `POST /api/v1/documents/analyze-file`
- `GET /api/v1/documents/history`
- `GET /api/v1/documents/{document_id}`

### Dashboard

- `GET /api/v1/dashboard/summary`

### System

- `GET /health`

## Security Notes

- Passwords are hashed with bcrypt (`passlib`)
- Sessions use signed JWT in HttpOnly cookie
- Document access is scoped to authenticated owner

## Quality and Extensibility

- Strongly typed request/response schemas
- Separation of concerns across auth, extraction, model, persistence
- Backward-compatible table bootstrap for existing local DBs

## Next Upgrades

1. Add Alembic migrations for strict schema evolution.
2. Add async task queue for heavy OCR jobs.
3. Add model registry + A/B testing across multiple models.
4. Add automated tests for auth, OCR extraction, and API contracts.
