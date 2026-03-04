from pathlib import Path
import random
import pandas as pd

random.seed(42)

positive_templates = [
    "I love this product, it is excellent and works perfectly.",
    "The service was amazing and I am very satisfied.",
    "Great experience overall, highly recommended.",
    "Fantastic quality and very helpful support team.",
    "This made my day better, truly wonderful.",
    "I am impressed by the performance and value.",
]

negative_templates = [
    "I hate this, it is terrible and disappointing.",
    "Very bad experience, I regret buying this.",
    "The quality is poor and support is unhelpful.",
    "This is frustrating and not worth the money.",
    "Awful service, I am not satisfied at all.",
    "The product failed quickly and caused issues.",
]

intensifiers = ["really", "extremely", "quite", "honestly", "surprisingly", "consistently"]

rows = []
for _ in range(300):
    sentence = random.choice(positive_templates)
    sentence = sentence.replace("is", f"is {random.choice(intensifiers)}", 1)
    rows.append({"text": sentence, "label": "positive"})

for _ in range(300):
    sentence = random.choice(negative_templates)
    sentence = sentence.replace("is", f"is {random.choice(intensifiers)}", 1)
    rows.append({"text": sentence, "label": "negative"})

random.shuffle(rows)

df = pd.DataFrame(rows)
out_path = Path(__file__).resolve().parents[1] / "data" / "sentiment_seed.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Seed dataset saved to {out_path} with {len(df)} rows")
