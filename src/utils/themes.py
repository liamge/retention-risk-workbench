from __future__ import annotations


_THEME_KEYWORDS = {
    "engagement": [
        "login",
        "activity",
        "session",
        "crm",
        "usage",
        "adoption",
    ],
    "production": [
        "policy",
        "premium",
        "written",
        "submitted",
        "sold",
        "quote",
    ],
    "tenure": [
        "tenure",
        "age",
        "months_since",
        "weeks_since",
        "days_since",
    ],
    "retention_risk": [
        "churn",
        "lapse",
        "drop",
        "inactive",
        "decline",
    ],
    "profile": [
        "state",
        "channel",
        "agent",
        "segment",
        "type",
    ],
}


def map_feature_to_theme(feature_name: str) -> str:
    name = feature_name.lower()
    for theme, keywords in _THEME_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return theme
    return "other"