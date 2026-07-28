# Security notes

This is a demo/portfolio project (synthetic data, local docker-compose only).
Notes below are for anyone extending it toward a real deployment.

## Data

- All transaction data is synthetically generated (`Faker` + a scripted fraud
  pattern) — no real user or payment data is ever involved.
- `docker-compose.yml` ships hardcoded `fraud`/`fraud` Postgres credentials.
  **These are dev-only.** Replace with secrets-managed credentials (env file,
  Docker secrets, or a real secrets manager) before exposing this anywhere
  beyond localhost.

## Model artifact loading

`scorer/model.py` loads `models/model.joblib` via `joblib.load`, which can
execute arbitrary code if given an untrusted pickle. This is safe here
because the artifact is produced by `training/train.py` in this same repo
(the `trainer` compose service) — never load a `model.joblib` from an
external or untrusted source without re-checking this.

## Network exposure

- Only `scorer` (FastAPI, port 8000) is intended to be reachable by clients.
  Redpanda, Redis and Postgres ports are published in `docker-compose.yml`
  for local debugging convenience only — do not publish them on a
  network-reachable host without adding auth (Redis/Kafka have none
  configured here).
- The FastAPI app has no authentication/rate-limiting — add both before
  putting `/score` behind anything but a trusted internal network.

## Reporting

This is a personal/portfolio project with no SLA. Open a GitHub issue for
anything you find.
