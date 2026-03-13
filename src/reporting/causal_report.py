from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.causal import (
    ExperimentSchema,
    build_uplift_table,
    estimate_ate,
    estimate_cate_by_segment,
    recommend_policy,
    standardize_experiment_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate causal uplift artifacts for the dashboard.")
    parser.add_argument("--input", type=str, default="data/processed/retention_experiment.csv")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/causal")
    parser.add_argument("--treatment-col", type=str, default="treatment")
    parser.add_argument("--outcome-col", type=str, default="outcome")
    parser.add_argument("--segment-col", type=str, default="segment", help="Optional segment column for CATE.")
    parser.add_argument("--score-col", type=str, default="", help="Predicted uplift or priority score column")
    parser.add_argument("--budget", type=float, default=0.3, help="Budget fraction to target top uplift bins")
    parser.add_argument("--treatment-cost", type=float, default=0.0)
    parser.add_argument("--value-per-save", type=float, default=1.0, help="Value of preventing a churn event")
    return parser.parse_args()


def choose_score_column(df: pd.DataFrame, explicit: Optional[str]) -> pd.Series:
    candidates = [c for c in [explicit, "uplift_score", "priority_score", "churn_probability"] if c]
    for col in candidates:
        if col in df.columns:
            return df[col]
    # default: uniform scores retain ordering but keep table generation simple
    return pd.Series(1.0, index=df.index, name="uniform_score")


def generate_causal_artifacts(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    schema = ExperimentSchema(
        treatment_col=args.treatment_col,
        outcome_col=args.outcome_col,
        segment_col=args.segment_col if args.segment_col in df.columns else None,
    )

    standardized = standardize_experiment_frame(df, schema)
    score_series = choose_score_column(standardized, args.score_col or None)

    ate = estimate_ate(standardized, treatment_col=schema.treatment_col, outcome_col=schema.outcome_col)

    cate = pd.DataFrame()
    if schema.segment_col:
        cate = estimate_cate_by_segment(
            standardized,
            segment_cols=schema.segment_col,
            treatment_col=schema.treatment_col,
            outcome_col=schema.outcome_col,
        )

    uplift_table = build_uplift_table(
        scores=score_series,
        treatment=standardized[schema.treatment_col],
        outcome=standardized[schema.outcome_col],
    )

    policy = recommend_policy(
        uplift_table,
        treatment_cost=args.treatment_cost,
        value_per_success=args.value_per_save,
        budget_fraction=args.budget,
    )

    summary = {
        "ate": ate.ate,
        "treated_mean": ate.treated_mean,
        "control_mean": ate.control_mean,
        "uplift_direction": ate.uplift_direction,
        "qini": float(uplift_table["cumulative_uplift"].iloc[-1]) if not uplift_table.empty else 0.0,
        "uplift_top_bin": float(uplift_table.iloc[0]["uplift"]) if not uplift_table.empty else 0.0,
        "budget_fraction": args.budget,
    }

    return {
        "summary": summary,
        "cate": cate,
        "uplift_table": uplift_table,
        "policy": policy,
    }


def save_artifacts(bundle: dict, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "summary.json").write_text(json.dumps(bundle["summary"], indent=2), encoding="utf-8")
    if not bundle["cate"].empty:
        bundle["cate"].to_csv(artifact_dir / "cate.csv", index=False)
    bundle["uplift_table"].to_csv(artifact_dir / "uplift_table.csv", index=False)
    bundle["policy"].to_csv(artifact_dir / "policy_recommendations.csv", index=False)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input experiment file not found at {input_path}. Provide a CSV with treatment, outcome, and optional score columns."
        )

    df = pd.read_csv(input_path)
    bundle = generate_causal_artifacts(df, args)
    save_artifacts(bundle, Path(args.artifact_dir))
    print(f"Saved causal artifacts to {Path(args.artifact_dir).resolve()}")


if __name__ == "__main__":
    main()
