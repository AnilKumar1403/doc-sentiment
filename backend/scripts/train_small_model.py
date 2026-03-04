from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "sentiment_seed.csv"
model_path = base_dir / "models" / "sentiment_model.joblib"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. Run scripts/generate_seed_dataset.py first."
    )

df = pd.read_csv(data_path)
required_cols = {"text", "label"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset must have columns: {required_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000,
            ),
        ),
        ("clf", LogisticRegression(max_iter=400, class_weight="balanced", random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

print("Classification report:")
print(classification_report(y_test, preds))

model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
