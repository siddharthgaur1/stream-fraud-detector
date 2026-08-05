"""Promote a registered model version to a stage, with gates.

Promotion is deliberate and checked, not `cp model.joblib prod/`. The gates
below are the documented criteria from RUNBOOK.md; `--force` skips them and
says so loudly in the log, because "we overrode the gate on the 14th" is a
thing you want to be able to find afterwards.

    python -m scripts.promote_model --version 7 --stage Staging
    python -m scripts.promote_model --version 7 --stage Production
"""
from __future__ import annotations

import argparse
import sys

import structlog

from common.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME

log = structlog.get_logger()

# Promotion criteria. These are floors, not targets — a model below any of
# them is worse than what it would replace on the axis that matters.
GATES = {
    # Ranking quality. Below this the ensemble is barely better than the
    # XGBoost head alone and the extra complexity is not paying for itself.
    "ensemble_auc": 0.90,
    # Precision-recall area matters more than AUC at a ~2% fraud rate: AUC
    # stays flattering when the positive class is rare, average precision
    # does not.
    "ensemble_avg_precision": 0.50,
    # A model trained on too little data can clear both metrics by luck.
    "train_rows": 5000,
}

STAGES = ("Staging", "Production", "Archived")


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def check_gates(metrics: dict[str, float]) -> list[str]:
    failures = []
    for name, floor in GATES.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not logged on this run")
        elif value < floor:
            failures.append(f"{name}: {value:.4f} < required {floor}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Registered model version number.")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--force", action="store_true", help="Promote despite failing gates.")
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="Do not archive the version currently at this stage. Leaves two "
             "versions at one stage, which the scorer resolves arbitrarily — "
             "only useful mid-migration.",
    )
    args = parser.parse_args()

    if not MLFLOW_TRACKING_URI:
        print("MLFLOW_TRACKING_URI is not set — nothing to promote against.", file=sys.stderr)
        return 2

    client = _client()
    mv = client.get_model_version(MODEL_REGISTRY_NAME, args.version)
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics

    print(f"{MODEL_REGISTRY_NAME} v{args.version} (run {mv.run_id[:8]}), currently: {mv.current_stage}")
    for name in GATES:
        print(f"  {name}: {metrics.get(name, 'not logged')}")

    # Archived is a demotion — gating it would mean you cannot retire a model
    # precisely when it is performing badly, which is when you need to most.
    if args.stage != "Archived":
        failures = check_gates(metrics)
        if failures:
            print("\nPromotion gates failed:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            if not args.force:
                print("\nRefusing to promote. Re-run with --force to override.", file=sys.stderr)
                return 1
            log.warning("promotion_gates_overridden", version=args.version,
                        stage=args.stage, failures=failures)

    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=args.version,
        stage=args.stage,
        archive_existing_versions=not args.keep_existing,
    )
    log.info("model_promoted", name=MODEL_REGISTRY_NAME, version=args.version, stage=args.stage)
    print(f"\nv{args.version} -> {args.stage}")
    if args.stage == "Production":
        print("Restart the scorer to pick it up: docker compose restart scorer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
