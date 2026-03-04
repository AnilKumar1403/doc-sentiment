from app.database import SessionLocal
from app.models import User, Document, SentimentResult
from datetime import datetime
import json


def seed():
    db = SessionLocal()

    # Create a user
    user = User(
        email="test@example.com",
        display_name="Test User",
        password_hash="fake_hashed_password",  # normally bcrypt
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create a document
    document = Document(
        owner_id=user.id,
        title="Sample Document",
        content="This is a test document for sentiment analysis.",
        source_type="text",
        extracted_char_count=48,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Create sentiment result
    sentiment = SentimentResult(
        document_id=document.id,
        label="positive",
        confidence=0.92,
        emotion_scores_json=json.dumps({"joy": 0.8, "sadness": 0.1}),
        selected_metrics_json=json.dumps(["polarity", "subjectivity"]),
        summary_text="Overall positive tone.",
        suggestions_json=json.dumps(["Keep the tone consistent"]),
        model_name="distilbert-base-uncased",
        model_version="v1",
    )
    db.add(sentiment)
    db.commit()

    db.close()
    print("✅ Database seeded successfully!")


if __name__ == "__main__":
    seed()