from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal
from app.models import AnalysisHistory, Document, SentimentResult, User


def _copy_sentiment_documents(
    *,
    source_user_id: int | None,
    target_user_id: int,
) -> int:
    inserted = 0
    with SessionLocal() as db:
        docs = db.execute(
            select(Document).where(Document.owner_id == source_user_id).order_by(Document.created_at.asc())
        ).scalars().all()

        for doc in docs:
            existing = db.execute(
                select(Document.id).where(
                    Document.owner_id == target_user_id,
                    Document.title == doc.title,
                    Document.created_at == doc.created_at,
                    Document.content == doc.content,
                )
            ).first()
            if existing:
                continue

            cloned_doc = Document(
                owner_id=target_user_id,
                title=doc.title,
                content=doc.content,
                source_type=doc.source_type,
                file_name=doc.file_name,
                mime_type=doc.mime_type,
                extracted_char_count=doc.extracted_char_count,
                created_at=doc.created_at,
            )
            db.add(cloned_doc)
            db.flush()

            result = db.execute(select(SentimentResult).where(SentimentResult.document_id == doc.id)).scalar_one_or_none()
            if result:
                db.add(
                    SentimentResult(
                        document_id=cloned_doc.id,
                        label=result.label,
                        confidence=result.confidence,
                        emotion_scores_json=result.emotion_scores_json,
                        selected_metrics_json=result.selected_metrics_json,
                        summary_text=result.summary_text,
                        suggestions_json=result.suggestions_json,
                        model_name=result.model_name,
                        model_version=result.model_version,
                        created_at=result.created_at,
                    )
                )
            inserted += 1

        db.commit()
    return inserted


def _copy_analysis_history(
    *,
    source_user_id: int | None,
    target_user_id: int,
) -> int:
    inserted = 0
    with SessionLocal() as db:
        rows = db.execute(
            select(AnalysisHistory)
            .where(AnalysisHistory.owner_id == source_user_id)
            .order_by(AnalysisHistory.created_at.asc())
        ).scalars().all()

        for row in rows:
            duplicate = db.execute(
                select(AnalysisHistory.id).where(
                    AnalysisHistory.owner_id == target_user_id,
                    AnalysisHistory.module_name == row.module_name,
                    AnalysisHistory.title == row.title,
                    AnalysisHistory.analysis_type == row.analysis_type,
                    AnalysisHistory.label == row.label,
                    AnalysisHistory.created_at == row.created_at,
                )
            ).first()
            if duplicate:
                continue

            db.add(
                AnalysisHistory(
                    owner_id=target_user_id,
                    module_name=row.module_name,
                    title=row.title,
                    source_type=row.source_type,
                    analysis_type=row.analysis_type,
                    label=row.label,
                    score=row.score,
                    summary_text=row.summary_text,
                    suggestions_json=row.suggestions_json or json.dumps([]),
                    details_json=row.details_json or json.dumps({}),
                    created_at=row.created_at,
                )
            )
            inserted += 1

        db.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed local history records into a target user account.")
    parser.add_argument("--email", required=True, help="Target account email")
    parser.add_argument(
        "--include-all-users",
        action="store_true",
        help="Also copy records owned by other users into the target account",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        target = db.execute(select(User).where(User.email == args.email)).scalar_one_or_none()
        if target is None:
            raise SystemExit(f"Target user not found: {args.email}")
        target_id = int(target.id)

        owner_ids: list[int | None] = [None]
        if args.include_all_users:
            ids = db.execute(select(User.id).where(User.id != target_id)).scalars().all()
            owner_ids.extend(int(v) for v in ids)

    total_docs = 0
    total_history = 0
    for owner_id in owner_ids:
        total_docs += _copy_sentiment_documents(source_user_id=owner_id, target_user_id=target_id)
        total_history += _copy_analysis_history(source_user_id=owner_id, target_user_id=target_id)

    print(
        json.dumps(
            {
                "target_email": args.email,
                "target_user_id": target_id,
                "owner_ids_scanned": owner_ids,
                "documents_seeded": total_docs,
                "history_seeded": total_history,
            }
        )
    )


if __name__ == "__main__":
    main()
