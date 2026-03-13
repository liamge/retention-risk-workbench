from src.predict import assign_risk_tier, recommended_action


def test_assign_risk_tier_respects_threshold():
    assert assign_risk_tier(0.8, threshold=0.5) == "High"
    assert assign_risk_tier(0.5, threshold=0.5) == "High"
    assert assign_risk_tier(0.4, threshold=0.5) == "Medium"
    assert assign_risk_tier(0.1, threshold=0.5) == "Low"


def test_recommended_action_matches_risk_levels():
    assert "Priority" in recommended_action(0.9, threshold=0.4)
    assert "Monitor" in recommended_action(0.5, threshold=0.6)
    assert "No intervention" in recommended_action(0.1, threshold=0.4)
