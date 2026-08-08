> **Superseded — merged into
> [github.com/siddharthgaur1/ml-monitor](https://github.com/siddharthgaur1/ml-monitor)
> as the `ml_platform.fraud` subsystem.**
>
> This repository is archived and read-only. Its full commit history was carried
> over with `git subtree`, so nothing here is lost — the commits live in the
> flagship's history with their original dates.
>
> The merge removed a `try/except ImportError` around `import ml_monitor` that
> let a whole monitoring backend silently no-op when the sibling package was
> absent. Inside one package it is always present, so a failure there is now a
> real failure.
>
> The caveats below still hold and are repeated in the flagship's README: the
> transaction data is synthetic, and the service is not deployed anywhere.

# stream-fraud-detector

[![CI](https://github.com/siddharthgaur1/stream-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/stream-fraud-detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

Real-time fraud/anomaly detection for synthetic UPI/fintech transactions:
Kafka-compatible streaming ingestion, rolling per-user feature engineering in
Redis, an XGBoost + IsolationForest ensemble scored behind a FastAPI service,
scored transactions persisted to Postgres, and Evidently drift/performance
reports.

**Not deployed.** This runs as a local docker-compose stack. There is no public
endpoint — see [Deployment status](#deployment-status) for what exists, what
does not, and why. [RUNBOOK.md](RUNBOOK.md) is the operational guide;
[SECURITY.md](SECURITY.md) covers the security model.

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
export API_KEYS="$(openssl rand -hex 24)"   # required — see below
make train                                  # once: synthetic data + model bundle
make up                                     # builds, starts, waits for /health
make smoke                                  # end-to-end: /health + a real POST /score
```

`make help` lists every target. `make up` blocks until `/health` returns 200
rather than returning while containers are still starting.

**The scorer refuses to start without `API_KEYS`.** That is deliberate: a
scoring endpoint that silently works with no auth is how a demo ends up
exposed. For local work without a key, set `ALLOW_UNAUTHENTICATED=true`
instead — it has to be explicit and visible in the process environment.

The API is at `http://localhost:8000`. Redpanda, Redis and Postgres are also
published on their default ports for local debugging.

To retrain (e.g. after changing `common/features.py`), rerun `make train` and
`docker compose restart scorer` — the scorer loads the model once at startup.

### Optional: MLflow model registry

With `MLFLOW_TRACKING_URI` unset (the default) the scorer loads
`models/model.joblib`. Point it at an MLflow server and the deployed model
becomes whatever is promoted to `MODEL_REGISTRY_STAGE`:

```bash
python -m training.train --data data/transactions.csv --register   # registers, unstaged
python -m scripts.promote_model --version 3 --stage Staging
python -m scripts.promote_model --version 3 --stage Production
docker compose restart scorer
```

Registering is not promoting. Promotion is gated on `ensemble_auc >= 0.90`,
`ensemble_avg_precision >= 0.50` and `train_rows >= 5000`; `--force` overrides
and logs that it did. Criteria and reasoning: [RUNBOOK](RUNBOOK.md#3-rollback).

## API

`/score`, `/stats` and `/monitoring` require `X-API-Key`. `/health` and
`/metrics` do not — load balancers and Prometheus scrapers rarely send custom
headers, and neither endpoint exposes transaction data. Keep `/metrics` off a
public listener in a real deployment.

Rate limit: `RATE_LIMIT_PER_MIN` per key, default 600, `429` with `Retry-After`
past it. `0` disables.

### `POST /score`

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEYS" \
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
curl http://localhost:8000/stats -H "X-API-Key: $API_KEYS"
```

```json
{"window": "1h", "transaction_count": 192, "fraud_count": 9, "fraud_rate": 0.0469, "avg_score": 0.1316}
```

### `GET /health`

```bash
curl -i http://localhost:8000/health
```

```json
{"status": "ok", "model_version": "v1", "kafka_connected": true, "redis_connected": true, "db_connected": true}
```

**Returns `503` with `"status": "degraded"` when Redis or Postgres is
unreachable** — those are hard dependencies of correct scoring, and an
endpoint that returns 200 unconditionally cannot tell an orchestrator to pull
the replica. Kafka down is reported as `kafka_connected: false` but stays
`200`, because `POST /score` still works without the broker.

### `GET /metrics`

Prometheus exposition format. Scoring counts by source (`http`/`stream`) and
outcome, a latency histogram, DB insert failures, stream message age, drift
share, and the loaded model's identity as a label — so "which model is
serving?" is answerable without reading deployment config.

```bash
curl -s http://localhost:8000/metrics | grep fraud_
```

### `GET /monitoring`

```bash
curl http://localhost:8000/monitoring -H "X-API-Key: $API_KEYS"
curl "http://localhost:8000/monitoring?format=html" -H "X-API-Key: $API_KEYS"
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
pip install -r requirements-dev.txt

make test        # or: pytest
make lint        # or: ruff check .

# train a model from synthetic data
PYTHONPATH=. python -m training.generate_data
PYTHONPATH=. python -m training.train
```

The 60-test suite needs no Kafka, Redis or Postgres — those are faked. It
covers feature computation, ensemble scoring and thresholding, the HTTP
surface including auth and the 503 path, rate limiting, and the drift
trigger's alerting decisions.

Running `api`/`producer`/`monitor` outside Docker additionally needs a local
Kafka-compatible broker, Redis and Postgres reachable at the URLs in
`.env.example`.

## Results (load test)

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

## Monitoring and operations

Drift is checked every `MONITOR_INTERVAL_SEC` (default 60). The **retrain
trigger** fires a Slack alert when `drift_share >= DRIFT_SHARE_THRESHOLD`
(default 0.5) for `DRIFT_CONSECUTIVE_WINDOWS` (default 2) consecutive windows.

Two windows rather than one is deliberate: single windows cross the threshold
routinely — a burst from one merchant, a deploy, a quiet window with a small
sample — and paging on those trains people to ignore the alert. Two windows
costs ~2 minutes of detection latency and removes most of that noise. The
alert fires once on the transition and re-arms on recovery, so a condition
that persists for an hour pages once rather than sixty times.

With `SLACK_WEBHOOK_URL` unset, drift is still detected and logged but nobody
is notified — **absence of alerts is not evidence of health.**

**[RUNBOOK.md](RUNBOOK.md)** covers deploy, rollback (including registry
rollback), what to do when drift fires at 2am, what each alert means, and the
known failure modes. The one worth knowing here: **Redis being down is the
quietest failure in the system** — the feature store returns empty history, so
every transaction looks like a user's first and the model returns
confident-looking wrong scores. That is why `/health` 503s on it.

## Deployment status

**Not deployed. There is no public endpoint.** This is a local docker-compose
stack, and the README will say so until that changes.

Deploy-ready in the ways that matter: non-root multi-stage image, API-key auth,
per-key rate limiting, `/health` with real 503 semantics, `/metrics`,
structured JSON logs, images published to GHCR per commit SHA. Not provisioned
anywhere.

The honest blocker is cost, and it shapes the architecture: **managed Kafka is
out of budget.** The substitute would be dropping the broker in the deployed
environment and running HTTP-only scoring — `POST /score` works standalone;
the stream consumer is the part that needs a broker. That is a smaller system
than what runs locally, and saying so beats implying a Kafka deployment that
does not exist. Also needed: managed Redis and Postgres, secrets in a secret
manager rather than compose env vars, and per-consumer API keys so one can be
revoked without breaking everyone.

## Project structure

```
common/     pydantic schemas, feature engineering, registry access, env config
producer/   simulates the transaction stream -> Kafka/Redpanda
scorer/     Kafka->Redis feature consumer + XGBoost/IsolationForest ensemble + SHAP
api/        FastAPI app (/score /stats /health /metrics /monitoring) + auth + Postgres
monitor/    Evidently drift reports on a timer + the retrain trigger and Slack alerting
training/   offline data generation + model training (+ optional MLflow registration)
scripts/    promote_model.py — gated registry stage transitions
notebooks/  training.ipynb — EDA/ROC/feature-importance/SHAP, reuses training/*
tests/      60 tests, no infrastructure required
loadtest/   locust load test for POST /score
```

## License

MIT — see [LICENSE](LICENSE).

## Limitations

- **The data is synthetic.** Fraud is injected as odd-hour + high-amount +
  unseen-device, which is exactly what the features are built to detect, so
  the model's accuracy on it says very little about real fraud. Every metric
  here measures the pipeline, not the modelling.
- **Not deployed** — see [Deployment status](#deployment-status).
- **Load-test numbers are from one machine** and predate the auth and metrics
  work; the median was dominated by single-process uvicorn, not the model.
  Rerun before citing them.
- Single shared `Dockerfile` for producer/scorer/monitor — same deps, only
  the compose `command:` differs. Simplest thing that works for 3 small
  Python services; split if they ever need divergent dependencies.
- Base image is pinned to `-slim-bookworm`, not a digest. Digest pinning is
  stronger and is a `TODO` in the Dockerfile.
- `docker-compose` uses `fraud`/`fraud` Postgres creds — fine for local
  demo, replace before any real deployment.
- Feature engineering online (Redis, `scorer/redis_store.py`) and offline
  (in-memory replay, `training/train.py`) both call the exact same
  `common/features.compute_features`, so there's no train/serve skew.
- Drift monitoring compares `amount`/`fraud_score`/`is_fraud` only (the
  columns actually persisted to Postgres), not the full feature vector. A
  feature can drift without any of those three moving.
- The Kafka consumer uses `auto_offset_reset=latest` with automatic commits,
  so a scorer restart drops messages published while it was down. Acceptable
  for a demo stream; not acceptable if the audit log has to be complete.
- Rate limiting is a fixed window, so a client can send 2x the limit across a
  window boundary. Fine for abuse protection, not for billing.
