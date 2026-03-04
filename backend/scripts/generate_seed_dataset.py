from pathlib import Path
import random
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.emotion_taxonomy import EMOTION_LABELS, KEYWORD_LEXICON

random.seed(42)

EMOTION_SNIPPETS = {
    "anger": ["I am furious about the repeated failures and delays", "This makes me angry and I want immediate action"],
    "sweet": ["Your words were sweet and comforting", "It felt gentle, warm, and sweet"],
    "fear": ["I am scared this situation may get worse", "There is fear about what could happen next"],
    "sad": ["I feel deeply sad about this loss", "This is a painful and sad moment"],
    "nice": ["The interaction was nice and respectful", "You handled this in a very nice way"],
    "emotional": ["I became emotional while reading this", "This event was emotional and overwhelming"],
    "love": ["I love how you supported me today", "There is love and care in this message"],
    "drama": ["The discussion turned into unnecessary drama", "There is constant drama around this issue"],
    "depressed": ["I feel depressed and disconnected", "This week feels depressing and heavy"],
    "worried": ["I am worried about the final result", "This makes me worried and uncertain"],
    "tensed": ["The room was tensed before the decision", "I am tensed and cannot relax"],
    "stressed": ["I am stressed by constant pressure", "This workload is stressful and draining"],
    "sick": ["I feel sick and weak today", "The patient looks sick and tired"],
    "fight": ["They started a fight over a small issue", "The argument escalated into a fight"],
    "calm": ["She stayed calm during the crisis", "I feel calm after the clarification"],
    "polite": ["Thank you for the polite response", "He remained polite under pressure"],
    "joy": ["I felt pure joy after hearing the news", "This result brings joy and excitement"],
    "frustrated": ["I am frustrated by repeated blockers", "This process is frustrating and slow"],
    "hopeful": ["I am hopeful that this will improve", "We remain hopeful about recovery"],
    "confused": ["I am confused by conflicting instructions", "This update is confusing and incomplete"],
    "grateful": ["I am grateful for your support", "We are thankful and grateful for your help"],
}

PREFIXES = [
    "In this project",
    "During this week",
    "In today's discussion",
    "From my perspective",
    "In this situation",
]
SUFFIXES = ["right now", "for our team", "for this release", "for this case", "today"]

rows: list[dict[str, str]] = []

for emotion in EMOTION_LABELS:
    snippets = EMOTION_SNIPPETS[emotion]
    keywords = KEYWORD_LEXICON[emotion]

    for _ in range(280):
        sentence = f"{random.choice(PREFIXES)} {random.choice(snippets)} {random.choice(SUFFIXES)}."
        rows.append({"text": sentence, "labels": emotion})

    for _ in range(180):
        sentence = f"{random.choice(PREFIXES)} the tone is {random.choice(keywords)} {random.choice(SUFFIXES)}."
        rows.append({"text": sentence, "labels": emotion})

# Mixed-label sentences
for _ in range(3000):
    emotions = random.sample(EMOTION_LABELS, random.choice([2, 2, 3]))
    parts = []
    for emotion in emotions:
        source = random.choice(["snippet", "keyword"])
        if source == "snippet":
            parts.append(random.choice(EMOTION_SNIPPETS[emotion]))
        else:
            parts.append(f"the tone feels {random.choice(KEYWORD_LEXICON[emotion])}")

    text = f"{random.choice(PREFIXES)}. " + ". ".join(parts) + f" {random.choice(SUFFIXES)}."
    labels = ",".join(sorted(emotions))
    rows.append({"text": text, "labels": labels})

random.shuffle(rows)
df = pd.DataFrame(rows)

out_path = Path(__file__).resolve().parents[1] / "data" / "sentiment_seed.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Seed dataset saved to {out_path} with {len(df)} rows")
print(f"Emotion labels: {len(EMOTION_LABELS)} -> {', '.join(EMOTION_LABELS)}")
