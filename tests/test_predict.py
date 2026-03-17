from src.cli.predict import assign_risk_tier, recommended_action


def test_assign_risk_tier_respects_threshold():
    assert assign_risk_tier(0.8, threshold=0.5) == "High"
    assert assign_risk_tier(0.7, threshold=0.8) == "Medium"  # below max(threshold, 0.66)
    assert assign_risk_tier(0.66, threshold=0.5) == "High"   # hits 0.66 floor for high tier
    assert assign_risk_tier(0.4, threshold=0.5) == "Medium"
    assert assign_risk_tier(0.1, threshold=0.5) == "Low"


def test_recommended_action_matches_risk_levels():
    assert "Save offer" in recommended_action(0.9, threshold=0.4)
    assert "High-touch" in recommended_action(0.6, threshold=0.5)
    assert "Monitor" in recommended_action(0.4, threshold=0.6)
    assert "No intervention" in recommended_action(0.1, threshold=0.4)
