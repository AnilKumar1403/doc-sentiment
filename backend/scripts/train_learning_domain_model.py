from pathlib import Path
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

base = Path(__file__).resolve().parents[1]
domain_dir = base / "data" / "domain"
out_path = base / "models" / "learning_domain_model.joblib"

math_data = json.loads((domain_dir / "mathematics_k12_competitive.json").read_text())
social_data = json.loads((domain_dir / "indian_social_studies_k12.json").read_text())

rows = []

for cls, topics in math_data["classes"].items():
    for topic in topics:
        rows.append({"subject": "mathematics", "class": cls, "topic": topic, "text": f"class {cls} mathematics {topic}"})
for topic in math_data["competitive_exam_topics"]:
    rows.append({"subject": "mathematics", "class": "competitive", "topic": topic, "text": f"competitive mathematics {topic}"})

for cls, topics in social_data["classes"].items():
    for topic in topics:
        rows.append({"subject": "indian social", "class": cls, "topic": topic, "text": f"class {cls} indian social studies {topic}"})
for topic in social_data["advanced_topics"]:
    rows.append({"subject": "indian social", "class": "advanced", "topic": topic, "text": f"advanced indian social {topic}"})

vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
matrix = vectorizer.fit_transform([r["text"] for r in rows])

artifact = {"rows": rows, "vectorizer": vectorizer, "matrix": matrix}
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, out_path)
print(f"Learning domain model saved -> {out_path}")
print(f"Rows: {len(rows)}")
