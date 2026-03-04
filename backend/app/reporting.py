from .emotion_taxonomy import ALERT_EMOTIONS, EMOTION_HINTS, POSITIVE_EMOTIONS


def build_report(
    selected_scores: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> tuple[str, list[str], str, float]:
    if not selected_scores:
        return (
            "No emotion metrics selected. Provide metrics to generate an emotion report.",
            ["Select at least one emotion metric such as love, drama, anger, or calm."],
            "unknown",
            0.0,
        )

    ranked = sorted(selected_scores.items(), key=lambda item: item[1], reverse=True)
    dominant_emotion, dominant_score = ranked[0]
    threshold_map = thresholds or {}
    high = [
        name
        for name, score in ranked
        if score >= float(threshold_map.get(name, 0.55))
    ]
    positive_pressure = sum(score for emotion, score in ranked if emotion in POSITIVE_EMOTIONS)
    alert_pressure = sum(score for emotion, score in ranked if emotion in ALERT_EMOTIONS)

    if high:
        summary = (
            f"Primary emotion is '{dominant_emotion}' ({dominant_score:.2f}). "
            f"High-intensity emotions detected: {', '.join(high)}."
        )
    else:
        top3 = ", ".join(f"{name} ({score:.2f})" for name, score in ranked[:3])
        summary = (
            f"Primary emotion is '{dominant_emotion}' ({dominant_score:.2f}). "
            f"No high-intensity emotion crossed 0.55. Top scores: {top3}."
        )
    if alert_pressure > positive_pressure:
        summary += " Emotional risk profile is elevated."
    elif positive_pressure > alert_pressure:
        summary += " Tone profile is mostly constructive."

    suggestions: list[str] = []
    for emotion, score in ranked[:5]:
        if score >= 0.35:
            hint = EMOTION_HINTS.get(emotion)
            if hint:
                suggestions.append(f"{emotion}: {hint}")

    if any(
        emotion in ALERT_EMOTIONS and score >= float(threshold_map.get(emotion, 0.55))
        for emotion, score in ranked[:5]
    ):
        suggestions.append("Risk alert: include human-review workflow for sensitive/emotional content.")

    if not suggestions:
        suggestions.append("Tone is relatively balanced. Continue with clear and context-aware response style.")

    return summary, suggestions, dominant_emotion, float(dominant_score)
