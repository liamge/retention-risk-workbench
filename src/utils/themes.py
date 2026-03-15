from __future__ import annotations


_THEME_KEYWORDS = [
    (
        "Engagement & Listening",
        [
            "songs_played",
            "total_secs",
            "completion_rate",
            "skip_rate",
            "near_completion",
            "quality_score",
            "repeat_ratio",
            "log_day",
            "secs_per_unique",
            "avg_song",
        ],
    ),
    (
        "Subscription & Billing",
        [
            "payment_plan",
            "plan_list_price",
            "actual_amount_paid",
            "amount_paid",
            "payment_method",
            "auto_renew",
            "paid_to_list",
            "amount_paid_per_txn",
            "cancel",
        ],
    ),
    (
        "Renewal & Expiry",
        [
            "membership_expire",
            "post_expiry",
            "early_renewal",
            "latest_cancel",
            "latest_auto_renew",
        ],
    ),
    (
        "Recency & Activity",
        [
            "days_since_last",
            "latest_log",
            "latest_transaction",
            "last_log_date",
            "last_transaction_date",
        ],
    ),
    (
        "Content Breadth",
        [
            "num_unq",
            "secs_per_unique",
        ],
    ),
    (
        "Billing & Contract",
        [
            "contract",
            "paperless",
            "payment",
            "month_to_month",
            "electronic_check",
            "auto_pay",
        ],
    ),
    (
        "Customer Lifecycle",
        [
            "tenure",
            "customer_stage",
            "tenure_group",
            "months_since",
            "weeks_since",
        ],
    ),
    (
        "Pricing & Value",
        [
            "monthlycharges",
            "totalcharges",
            "avg_monthly_spend",
            "price_ratio",
            "high_bill",
            "revenue",
        ],
    ),
    (
        "Support & Protection",
        [
            "techsupport",
            "onlinesecurity",
            "onlinebackup",
            "deviceprotection",
            "support",
        ],
    ),
    (
        "Product Usage",
        [
            "internetservice",
            "fiber",
            "streaming",
            "phoneservice",
            "multipleservices",
            "num_services",
            "service",
        ],
    ),
    (
        "Engagement & Usage",
        [
            "login",
            "activity",
            "session",
            "crm",
            "usage",
            "adoption",
        ],
    ),
    (
        "Production",
        [
            "policy",
            "premium",
            "written",
            "submitted",
            "sold",
            "quote",
        ],
    ),
    (
        "Retention Risk",
        [
            "churn",
            "lapse",
            "drop",
            "inactive",
            "decline",
        ],
    ),
    (
        "Customer Profile",
        [
            "state",
            "channel",
            "agent",
            "segment",
            "type",
            "city",
            "age",
            "gender",
            "registered_via",
            "registration_init",
            "account_age",
            "senior",
            "partner",
            "dependents",
        ],
    ),
]


def map_feature_to_theme(feature_name: str) -> str:
    name = feature_name.lower()
    for theme, keywords in _THEME_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            return theme
    return "Other"
