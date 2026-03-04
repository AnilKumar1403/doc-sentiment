# End-to-End Implementation Details

## 1) Backend Implementation

### 1.1 Auth and Session

- File: `backend/app/security.py`
- Password hashing: `passlib` (`bcrypt`)
- Token: JWT (`python-jose`)
- Session transport: HttpOnly cookie (`access_token`)

Flow:
1. User registers or logs in.
2. API signs JWT with email subject and expiry.
3. Cookie is set by response.
4. Protected endpoints decode cookie token and load user from DB.

### 1.2 Data Model

- `users`: account identity and password hash
- `documents`: uploaded or text documents owned by user
- `sentiment_results`: one sentiment record per document

Ownership enforcement is applied server-side using `owner_id` filters in queries.

### 1.3 Document Ingestion and OCR

- File: `backend/app/document_extractor.py`

Supported extraction modes:
- `.txt` -> decode text
- `.docx` -> parse paragraphs
- `.pdf` -> direct text extraction (`pypdf`)
- scanned `.pdf` fallback -> render pages (`PyMuPDF`) + OCR (`pytesseract`)
- image files (`.png`, `.jpg`, `.jpeg`, `.tiff`) -> OCR (`pytesseract`)

If Tesseract binary is unavailable, extraction returns controlled error.

### 1.4 Sentiment Model

- Training script: `backend/scripts/train_small_model.py`
- Pipeline: `TfidfVectorizer(ngram 1-2) + LogisticRegression`
- Inference wrapper: `backend/app/ml.py`

### 1.5 Startup Bootstrapping

- File: `backend/app/bootstrap.py`

Startup does:
1. `create_all` for known tables.
2. Adds missing columns for backward compatibility with older `documents` table.
3. Seeds default user if absent.

## 2) Frontend Implementation

Single-page authenticated UI with section switching:

- Dashboard: summary metrics from `/api/v1/dashboard/summary`
- Analyze: text analysis + file upload analysis
- History: list of document analyses
- Profile: account details from `/api/v1/auth/me`
- Logout: cookie invalidation

Files:
- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

## 3) Database Schema

Authoritative SQL file: `database/schema.sql`

Includes indexes on:
- `users.email`
- `documents(owner_id, created_at)`
- `sentiment_results.label`

## 4) Production Hardening Recommendations

1. Move seeded credentials to one-time setup script only.
2. Enforce HTTPS and secure cookies (`Secure=True`).
3. Add CSRF token strategy for cookie-authenticated mutations.
4. Add rate limiting for login and upload endpoints.
5. Add malware scanning for uploaded files.
6. Add observability (structured logs, tracing, error metrics).
7. Move OCR and inference to background worker for large files.
8. Add Alembic migrations and CI checks.

## 5) Verification Checklist

- `GET /health` returns `ok`
- Can login with seeded account
- Can analyze text and get sentiment
- Can upload image/pdf/docx and get extracted text sentiment
- Dashboard stats update
- History only shows user-owned records
- Logout blocks protected endpoint access
