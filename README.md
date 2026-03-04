# Document Sentiment Analysis (End-to-End Starter)

This project gives you a full learning path and runnable starter for:

- Backend API (`FastAPI`)
- Database (`PostgreSQL`)
- Database schema + persistence
- Small AI model you can train yourself (`TF-IDF + Logistic Regression`)
- Simple UI (HTML/CSS/JS)

## 1) Project Structure

```
doc-sentiment/
  backend/
    app/
      config.py
      database.py
      models.py
      schemas.py
      ml.py
      main.py
    scripts/
      generate_seed_dataset.py
      train_small_model.py
    data/
      sentiment_seed.csv
    models/
      sentiment_model.joblib   # generated after training
    requirements.txt
    .env.example
  database/
    schema.sql
  frontend/
    index.html
    styles.css
    app.js
  docker-compose.yml
```

## 2) Prerequisites

- Python `3.10+`
- Docker + Docker Compose

## 3) Start PostgreSQL

From `doc-sentiment/`:

```bash
docker compose up -d
```

This starts:
- `postgres` on `localhost:5432`

## 4) Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Create schema:

```bash
psql "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db" -f ../database/schema.sql
```

## 5) Train Your Small Model (From Scratch)

Generate seed dataset + train model:

```bash
python scripts/generate_seed_dataset.py
python scripts/train_small_model.py
```

This saves model artifact to:
- `backend/models/sentiment_model.joblib`

## 6) Run API + UI

From `backend/`:

```bash
uvicorn app.main:app --reload --port 8000
```

Open:
- `http://localhost:8000`

## 7) Key API Endpoints

- `GET /health`
- `POST /api/v1/documents/analyze`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}`

## 8) Database Schema

See `database/schema.sql`.

Core tables:
- `documents`: source text and metadata
- `sentiment_results`: prediction label + confidence + model metadata

## 9) Next Learning Steps

1. Replace seed dataset with a real one (IMDB, Yelp, Amazon reviews).
2. Add neutral class and rebalance classes.
3. Evaluate with precision/recall/F1 + confusion matrix.
4. Try transformer baseline (`distilbert-base-uncased-finetuned-sst-2-english`).
5. Add auth and role-based access for production.

## 10) AI Models You Can Try

### Small + Fast (good for learning)
- `TF-IDF + Logistic Regression` (already implemented)
- `TF-IDF + Linear SVM`
- `FastText` supervised classifier

### Better quality (heavier)
- `distilbert-base-uncased-finetuned-sst-2-english`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

Start with the small model in this repo, then compare to a transformer.
