from pathlib import Path
import sys

import pandas as pd
from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.emotion_taxonomy import EMOTION_LABELS

GOEMOTION_MAP = {
    "anger": "anger",
    "annoyance": "frustrated",
    "disappointment": "sad",
    "disapproval": "worried",
    "disgust": "sick",
    "embarrassment": "emotional",
    "fear": "fear",
    "grief": "depressed",
    "joy": "joy",
    "love": "love",
    "nervousness": "tensed",
    "optimism": "hopeful",
    "relief": "calm",
    "remorse": "sad",
    "sadness": "sad",
    "surprise": "emotional",
    "gratitude": "grateful",
    "caring": "nice",
    "amusement": "sweet",
    "curiosity": "confused",
    "admiration": "polite",
}


def main() -> None:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    train = ds["train"]

    label_names = train.features["labels"].feature.names

    rows = []
    for row in train:
        text = str(row["text"] or "").strip()
        if not text:
            continue

        mapped = []
        for idx in row["labels"]:
            name = label_names[idx]
            mapped_name = GOEMOTION_MAP.get(name)
            if mapped_name and mapped_name in EMOTION_LABELS and mapped_name not in mapped:
                mapped.append(mapped_name)

        if mapped:
            rows.append({"text": text, "labels": ",".join(sorted(mapped))})

    out_path = Path(__file__).resolve().parents[1] / "data" / "public_emotions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Public dataset mapped rows: {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
