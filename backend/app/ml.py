from pathlib import Path
import joblib
from .config import get_settings


class SentimentModel:
    def __init__(self):
        settings = get_settings()
        self.model_path = Path(settings.model_path)
        self.model_name = "tfidf-logreg"
        self.model_version = "v1"
        self.pipeline = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run scripts/generate_seed_dataset.py then scripts/train_small_model.py"
            )
        self.pipeline = joblib.load(self.model_path)

    def predict(self, text: str) -> tuple[str, float]:
        if self.pipeline is None:
            self.load()

        proba = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        max_idx = proba.argmax()
        label = str(labels[max_idx])
        confidence = float(proba[max_idx])
        return label, confidence
