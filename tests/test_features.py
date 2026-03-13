import pandas as pd

from src.features import engineer_features


def test_engineer_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "Churn": ["Yes", "No"],
            "tenure": [1, 24],
            "MonthlyCharges": [70.0, 40.0],
            "TotalCharges": [70.0, 960.0],
            "Contract": ["Month-to-month", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "PaperlessBilling": ["Yes", "No"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
        }
    )

    out = engineer_features(df)

    expected_cols = {
        "avg_monthly_spend",
        "price_ratio",
        "tenure_group",
        "customer_stage",
        "is_month_to_month",
        "auto_pay",
        "paperless",
        "num_services",
        "service_adoption_rate",
        "fiber_customer",
        "high_bill",
    }
    assert expected_cols.issubset(set(out.columns))
    assert "ChurnFlag" in out.columns
