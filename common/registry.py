"""MLflow model registry access.

The registry is optional: with MLFLOW_TRACKING_URI unset everything falls back
to the local joblib bundle, so the compose stack and CI still work without an
MLflow server. When it *is* set, the deployed model is whatever sits at
MODEL_REGISTRY_STAGE — promotion is a stage transition, not a file copy, so
"which model is in production" has an auditable answer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from common.config import (
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODEL_REGISTRY_NAME,
    MODEL_REGISTRY_STAGE,
    MODEL_VERSION,
)

log = structlog.get_logger()

ARTIFACT_SUBDIR = "model"


def registry_enabled() -> bool:
    return bool(MLFLOW_TRACKING_URI)


def _client():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    from mlflow.tracking import MlflowClient

    return mlflow, MlflowClient()


def load_bundle() -> tuple[dict[str, Any], str, str]:
    """Returns (bundle, version, source).

    `source` is "registry:<stage>:<version>" or "file:<path>", and it is
    surfaced on /health and as a Prometheus label — during an incident the
    first question is which model is actually loaded, and that should not
    require reading the deployment config to answer.
    """
    import joblib

    if not registry_enabled():
        bundle = joblib.load(MODEL_PATH)  # trusted artifact, produced by training/train.py
        return bundle, bundle.get("version", MODEL_VERSION), f"file:{MODEL_PATH}"

    mlflow, client = _client()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[MODEL_REGISTRY_STAGE])
    if not versions:
        raise RuntimeError(
            f"no model registered at stage {MODEL_REGISTRY_STAGE!r} for "
            f"{MODEL_REGISTRY_NAME!r}. Promote one, or unset MLFLOW_TRACKING_URI "
            f"to fall back to {MODEL_PATH}."
        )
    mv = versions[0]
    # Download the directory and find the bundle by extension rather than by a
    # hardcoded filename: train.py's --out is configurable, so pinning the name
    # here would break the moment someone trains to a different path.
    local_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source))
    candidates = sorted(local_dir.rglob("*.joblib"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one .joblib artifact under {mv.source}, found {len(candidates)}"
        )
    local = candidates[0]
    # joblib.load unpickles, i.e. it executes whatever is in the artifact. The
    # artifact URI comes from the registry, so this is exactly as trusted as
    # write access to your MLflow server: anyone who can register a model can
    # run code in the scorer. Treat registry write access accordingly.
    bundle = joblib.load(local)
    log.info("model_loaded_from_registry", name=MODEL_REGISTRY_NAME, stage=MODEL_REGISTRY_STAGE,
             version=mv.version, run_id=mv.run_id)
    return bundle, str(mv.version), f"registry:{MODEL_REGISTRY_STAGE}:{mv.version}"


def promote(version: str, stage: str, archive_existing: bool = True) -> None:
    """Move a registered version to a stage. Used by scripts/promote_model.py."""
    _, client = _client()
    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    log.info("model_promoted", name=MODEL_REGISTRY_NAME, version=version, stage=stage)
