# stream-fraud-detector

[![CI](https://github.com/siddharthgaur1/stream-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/stream-fraud-detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

Real-time fraud/anomaly detection for synthetic UPI/fintech transactions:
Kafka-compatible streaming ingestion, rolling per-user feature engineering in
Redis, an XGBoost + IsolationForest ensemble scored behind a FastAPI service,
scored transactions persisted to Postgres, and Evidently drift/performance
reports.

See [SECURITY.md](SECURITY.md) for the security model and known limits of a
local-only demo deployment.

## Architecture

```
                         ┌────────────────────────────────────────────┐
                         │                 producer                    │
                         │  simulates a UPI transaction stream with    │
                         │  ~2% injected fraud (odd hour + high amount │
                         │  + never-seen device)                       │
                         └───────────────────┬──────────────────────-─┘
                                             │ produces
                                             ▼
                         ┌────────────────────────────────────────────┐
                         │        redpanda (Kafka-compatible)          │
                         │              topic: transactions            │
                         └───────────────────┬──────────────────────-─┘
                                             │ consumes
                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ scorer  (docker-compose service "scorer" = the api/ FastAPI app)          │
│                                                                            │
│  ┌───────────────────────────┐                                           │
│  │ stream_consumer thread     │  every message: scorer/model.py scores   │
│  │ (scorer/stream_consumer.py)│─▶ it (features rebuilt from redis history│
│  │ group "stream-scorer"      │  via common/features.py, same code path  │
│  └──────────────┬─────────────┘  as offline training) and persists it    │
│                 │                 to postgres — this is what makes the   │
│                 │                 pipeline actually real-time, not just  │
│                 │                 "scores whatever a client POSTs"       │
│                 ▼                                                        │
│         redis: per-user rolling amounts (last 30) + known device ids,    │
│         updated by scorer/model.py AFTER each score so later             │
│         transactions see it but a transaction never sees itself          │
│                 ▲                                                        │
│                 │ read + update                                          │
│  POST /score ───┘  (same scorer/model.py path, for ad-hoc client calls) │
│       │                                                                   │
│       ▼                                                                  │
│  postgres: scored_transactions                                           │
│       ▲                                                                  │
│  GET /stats ────┘   GET /health   GET /monitoring ──▶ reads report       │
└──────────────────────────────────────────────────────────────────────────┘
                                             ▲
                                             │ reads scored_transactions
                         ┌───────────────────┴──────────────────────-─┐
                         │                 monitor                     │
                         │  every MONITOR_INTERVAL_SEC: Evidently       │
                         │  DataDriftPreset(reference vs. production)   │
                         │  writes reports/latest.{html,json}           │
                         └────────────────────────────────────────────-─┘
```

Packages: `producer/` (stream simulation), `scorer/` (Kafka stream consumer +
model + SHAP, no HTTP), `api/` (FastAPI wiring + Postgres), `monitor/`
(Evidently reports), `common/` (pydantic contracts, feature engineering,
config — shared so training and serving can't drift apart).

## Setup

Requires Docker + Docker Compose.

```bash
# 1. Train the model once (generates synthetic data, fits XGBoost +
#    IsolationForest, writes models/model.joblib + models/reference.csv)
docker compose run --rm trainer

# 2. Bring the rest of the system up
docker compose up -d

# 3. Watch the producer generate traffic and the API score it
docker compose logs -f producer scorer
```

The API is at `http://localhost:8000`. Redpanda, Redis and Postgres are also
published on their default ports for local debugging.

To retrain (e.g. after changing `common/features.py`), rerun step 1 — the
scorer must be restarted afterwards since it loads the model once at startup.

## API

### `POST /score`

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-demo-1",
    "user_id": "user_00042",
    "amount": 4200.50,
    "merchant_category": "electronics",
    "hour": 2,
    "device_id": "brand-new-device",
    "location_hash": "560001",
    "user_age_days": 900
  }'
```

Real captured response (`docker compose up`, full stack, against the default-trained model):

```json
{
  "transaction_id": "tx-demo-1",
  "fraud_score": 0.9209,
  "is_fraud": true,
  "shap_explanation": {
    "amount_ratio_avg10": 5.7798,
    "is_new_device": 3.6656,
    "tx_count_seen": -1.2081,
    "hour": 0.8315,
    "amount": 0.7614
  },
  "latency_ms": 32.37
}
```

### `GET /stats`

```bash
curl http://localhost:8000/stats
```

```json
{"window": "1h", "transaction_count": 192, "fraud_count": 9, "fraud_rate": 0.0469, "avg_score": 0.1316}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_version": "v1", "kafka_connected": true, "redis_connected": true, "db_connected": true}
```

### `GET /monitoring`

```bash
curl http://localhost:8000/monitoring            # JSON summary
curl http://localhost:8000/monitoring?format=html # full Evidently report
```

```json
{"generated_at": 1785251258.42, "rows_compared": 163, "dataset_drift": false, "drift_share": 0.5, "current_fraud_rate": 0.0429, "current_avg_score": 0.13}
```

### `GET /monitoring/ml-monitor`

Second, optional monitoring backend: the sibling
[ml-monitor](https://github.com/siddharthgaur1/ml-monitor) package, run
alongside Evidently on the same timer (`monitor/ml_monitor_backend.py`),
giving KS/PSI drift on `amount`/`is_fraud` plus PSI-based prediction drift
on `fraud_score`. Not a hard dependency — `pip install -e ../ml-monitor` to
enable it; without it, `monitor/main.py` silently skips this second report
and `/monitoring/ml-monitor` 404s.

```bash
curl http://localhost:8000/monitoring/ml-monitor
```

## Local development (no Docker)

```bash
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# feature engineering self-check
PYTHONPATH=. python tests/test_features.py

# train a model from synthetic data
PYTHONPATH=. python -m training.generate_data
PYTHONPATH=. python -m training.train

# lint
ruff check .
```

Running `api`/`producer`/`monitor` outside Docker additionally needs a local
Kafka-compatible broker, Redis and Postgres reachable at the URLs in
`.env.example`.

## Load testing

```bash
pip install -r loadtest/requirements.txt
locust -f loadtest/locustfile.py --host http://localhost:8000

# or headless, for a fixed number to drop in a report:
locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 50 -r 10 -t 60s --csv loadtest/results
```

Generates `/score` requests with the same field ranges as `producer/stream.py`'s
synthetic traffic (see `common/features.MERCHANT_CATEGORIES` etc.), against
the running `docker compose up -d` stack.

**Measured run** (50 users, ramped 10/s, 60s, 12-core host, full `docker compose`
stack including the producer's background traffic):

| Metric | Value |
|---|---|
| Requests | 1733, 0 failures |
| Throughput | 29.7 req/s |
| Median latency | 1400 ms |
| p95 / p99 | 2300 ms / 2700 ms |
| Min / Max | 22 ms / 3146 ms |

The min (22 ms) matches the single-request example above — the model itself
is fast. The gap to the median under load is `uvicorn api.main:app` running
with no `--workers` flag (see `docker-compose.yml`): one Python process
serializes every request, so 50 concurrent clients queue behind each other
rather than run in parallel. Numbers are from this machine, not a claimed
production SLA — rerun `--csv loadtest/results` on your own hardware before
citing a number anywhere that matters.

## Project structure

```
common/     pydantic schemas, feature engineering, env config (shared, no train/serve skew)
producer/   simulates the transaction stream -> Kafka/Redpanda
scorer/     Kafka->Redis feature consumer + XGBoost/IsolationForest ensemble + SHAP
api/        FastAPI app (/score /stats /health /monitoring) + Postgres
monitor/    Evidently drift/performance reports on a timer
training/   offline data generation + model training (source of models/model.joblib)
notebooks/  training.ipynb — EDA/ROC/feature-importance/SHAP, reuses training/*
tests/      feature engineering self-check
loadtest/   locust load test for POST /score
```

## License

MIT — see [LICENSE](LICENSE).

## Notes / known limits

- Single shared `Dockerfile` for producer/scorer/monitor — same deps, only
  the compose `command:` differs. Simplest thing that works for 3 small
  Python services; split if they ever need divergent dependencies.
- `docker-compose` uses `fraud`/`fraud` Postgres creds — fine for local
  demo, replace before any real deployment.
- Feature engineering online (Redis, `scorer/redis_store.py`) and offline
  (in-memory replay, `training/train.py`) both call the exact same
  `common/features.compute_features`, so there's no train/serve skew.
- Drift monitoring compares `amount`/`fraud_score`/`is_fraud` only (the
  columns actually persisted to Postgres), not the full feature vector.
