"""Central env-driven config. Every service reads the same env vars set by docker-compose."""
import os

KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:19092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "transactions")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# User history keys expire after this long without a transaction. Without a TTL
# the key space grows with the number of users ever seen, which is unbounded in
# production even though it looks fine in a demo with 200 synthetic users.
REDIS_HISTORY_TTL_SEC = int(os.environ.get("REDIS_HISTORY_TTL_SEC", str(90 * 24 * 3600)))

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"
)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

# MLflow model registry. When MLFLOW_TRACKING_URI is set the scorer loads the
# model at MODEL_REGISTRY_STAGE instead of the local joblib file, which makes
# promotion a registry transition rather than a file copy.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "stream-fraud-detector")
MODEL_REGISTRY_STAGE = os.environ.get("MODEL_REGISTRY_STAGE", "Production")

REPORTS_DIR = os.environ.get("REPORTS_DIR", "/reports")

FRAUD_SCORE_THRESHOLD = float(os.environ.get("FRAUD_SCORE_THRESHOLD", "0.7"))
XGB_WEIGHT = float(os.environ.get("XGB_WEIGHT", "0.7"))  # ensemble weight, iso gets 1-XGB_WEIGHT

# 0 disables rate limiting.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "600"))

# Drift alerting / retrain trigger (monitor service).
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.5"))
DRIFT_CONSECUTIVE_WINDOWS = int(os.environ.get("DRIFT_CONSECUTIVE_WINDOWS", "2"))
