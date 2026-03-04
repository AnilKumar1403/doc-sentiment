from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.emotion_taxonomy import EMOTION_LABELS, KEYWORD_LEXICON

base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "sentiment_seed.csv"
public_data_path = base_dir / "data" / "public_emotions.csv"
model_path = base_dir / "models" / "sentiment_model.joblib"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. Run scripts/generate_seed_dataset.py first."
    )

df = pd.read_csv(data_path)
required_cols = {"text", "labels"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset must have columns: {required_cols}")

if public_data_path.exists():
    public_df = pd.read_csv(public_data_path)
    if required_cols.issubset(public_df.columns):
        df = pd.concat([df, public_df], ignore_index=True)
        print(f"Merged public dataset rows: {len(public_df)}")

df["labels_list"] = df["labels"].fillna("").apply(
    lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
)

mlb = MultiLabelBinarizer(classes=EMOTION_LABELS)
Y = mlb.fit_transform(df["labels_list"])

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], Y, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                min_df=1,
                max_features=90000,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=3.0,
                    solver="liblinear",
                )
            ),
        ),
    ]
)

pipeline.fit(X_train, y_train)
probs = pipeline.predict_proba(X_test)

thresholds: dict[str, float] = {}
preds = np.zeros_like(probs, dtype=int)
for idx, label in enumerate(EMOTION_LABELS):
    best_threshold = 0.42
    best_score = -1.0
    for candidate in np.arange(0.35, 0.81, 0.02):
        current_pred = (probs[:, idx] >= candidate).astype(int)
        score = f1_score(y_test[:, idx], current_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)

    thresholds[label] = best_threshold
    preds[:, idx] = (probs[:, idx] >= best_threshold).astype(int)

micro_f1 = f1_score(y_test, preds, average="micro", zero_division=0)
macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

print(f"Micro-F1: {micro_f1:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

artifact = {
    "pipeline": pipeline,
    "labels": list(mlb.classes_),
    "thresholds": thresholds,
    "model_name": "tfidf-ovr-logreg-keyword-hybrid",
    "model_version": "v3-multi-emotion",
    "train_metrics": {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "samples": int(len(df)),
    },
    "keyword_lexicon": KEYWORD_LEXICON,
}

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, model_path)
print(f"Model saved to {model_path}")
