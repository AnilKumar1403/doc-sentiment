from pathlib import Path
import re

import joblib

from .config import get_settings
from .emotion_taxonomy import EMOTION_LABELS, KEYWORD_LEXICON


WORD_RE = re.compile(r"[a-zA-Z']+")


class SentimentModel:
    def __init__(self):
        settings = get_settings()
        self.model_path = Path(settings.model_path)
        self.model_name = "tfidf-ovr-logreg-keyword-hybrid"
        self.model_version = "v3-multi-emotion"
        self.pipeline = None
        self.labels: list[str] = EMOTION_LABELS
        self.thresholds: dict[str, float] = {label: 0.42 for label in EMOTION_LABELS}
        self.keyword_lexicon: dict[str, list[str]] = KEYWORD_LEXICON
        self.train_metrics: dict[str, float | int] = {}

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run scripts/generate_seed_dataset.py then scripts/train_small_model.py"
            )

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            self.pipeline = artifact["pipeline"]
            self.labels = list(artifact.get("labels", EMOTION_LABELS))
            self.thresholds = {
                label: float(artifact.get("thresholds", {}).get(label, 0.42))
                for label in self.labels
            }
            self.model_name = str(artifact.get("model_name", self.model_name))
            self.model_version = str(artifact.get("model_version", self.model_version))
            self.keyword_lexicon = artifact.get("keyword_lexicon", KEYWORD_LEXICON)
            self.train_metrics = artifact.get("train_metrics", {})
            return

        self.pipeline = artifact
        self.labels = ["positive", "negative"]
        self.thresholds = {"positive": 0.5, "negative": 0.5}
        self.model_name = "legacy-tfidf-logreg"
        self.model_version = "v1"
        self.keyword_lexicon = {}
        self.train_metrics = {}

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self.load()

    def predict_scores(self, text: str) -> dict[str, float]:
        self._ensure_loaded()

        probs_row = self.pipeline.predict_proba([text])[0]
        base_scores = {label: float(probs_row[idx]) for idx, label in enumerate(self.labels)}
        return self._apply_keyword_adjustments(text, base_scores)

    def _apply_keyword_adjustments(self, text: str, base_scores: dict[str, float]) -> dict[str, float]:
        normalized_text = text.lower()
        tokens = set(WORD_RE.findall(normalized_text))

        adjusted: dict[str, float] = {}
        for label, score in base_scores.items():
            keywords = self.keyword_lexicon.get(label, [])
            hits = 0
            for keyword in keywords:
                keyword_norm = keyword.lower()
                if " " in keyword_norm:
                    if keyword_norm in normalized_text:
                        hits += 1
                elif keyword_norm in tokens:
                    hits += 1

            boost = min(0.32, hits * 0.07)
            adjusted[label] = min(1.0, max(0.0, score + boost))

        return adjusted

    def sanitize_metrics(self, metrics: list[str] | None) -> list[str]:
        if not metrics:
            return list(self.labels)

        normalized: list[str] = []
        allowed = set(self.labels)
        for metric in metrics:
            key = metric.strip().lower()
            if key and key in allowed and key not in normalized:
                normalized.append(key)
        return normalized or list(self.labels)

    def get_threshold(self, emotion: str) -> float:
        self._ensure_loaded()
        return float(self.thresholds.get(emotion, 0.42))
