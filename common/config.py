"""Central env-driven config. Every service reads the same env vars set by docker-compose."""
import os

KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:19092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "transactions")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5432/fraud"
)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

REPORTS_DIR = os.environ.get("REPORTS_DIR", "/reports")

FRAUD_SCORE_THRESHOLD = float(os.environ.get("FRAUD_SCORE_THRESHOLD", "0.7"))
XGB_WEIGHT = float(os.environ.get("XGB_WEIGHT", "0.7"))  # ensemble weight, iso gets 1-XGB_WEIGHT
