# Implementation Notes (v2 Multi-Emotion)

## 1. Login-First Routing

- Backend root `/` redirects to `/login`.
- Frontend route guard blocks all views when unauthenticated.
- URL paths are synced with views: `/dashboard`, `/analyze`, `/history`, `/profile`.

## 2. Multi-Emotion Model

- Emotion taxonomy in `backend/app/emotion_taxonomy.py`.
- Seed generator (`backend/scripts/generate_seed_dataset.py`) now creates multi-label training data.
- Training script (`backend/scripts/train_small_model.py`) trains a Classifier Chain model.
- Inference (`backend/app/ml.py`) outputs emotion score map instead of binary sentiment.

## 3. Report Generation

- `backend/app/reporting.py` creates:
  - summary
  - actionable suggestions
  - dominant emotion + score
- Analysis supports user-selected metric subset via comma-separated textbox input.

## 4. Persistence Model

`sentiment_results` now stores:
- dominant label + confidence
- `emotion_scores_json`
- `selected_metrics_json`
- `summary_text`
- `suggestions_json`

Backward-compatible bootstrap migration is handled in `backend/app/bootstrap.py`.

## 5. API Contract Changes

Analyze response now returns:
- `selected_metrics`
- `emotion_scores`
- `summary`
- `suggestions`

Dashboard now returns:
- `total_documents`
- `high_alert_documents`
- `top_emotions`
- `last_analysis_at`

## 6. OCR and Files

- Extraction supports text/docx/pdf/image formats.
- OCR path requires Tesseract binary.
- File endpoint accepts multipart form with `title`, optional `emotion_metrics`, and `file`.

## 7. Remaining Best-Model Upgrade Path

To move from strong baseline to best quality:
1. Collect real labeled multi-emotion corpus for your domain.
2. Fine-tune a transformer (e.g., DeBERTa/BERT multi-label head).
3. Calibrate per-label thresholds on validation set.
4. Add evaluation dashboard and active-learning feedback loop.
