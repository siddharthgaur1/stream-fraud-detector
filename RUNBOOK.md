# Runbook

Operational notes for the local docker-compose stack. Scope is honest: this
is a local demo deployment (see [Notes / known limits](README.md#notes--known-limits)
in the README — `fraud`/`fraud` Postgres creds, no auth on the API), not a
public production endpoint. Steps below assume `docker compose` on the host
described in [Setup](README.md#setup).

## Deploy

```bash
docker compose run --rm trainer   # writes models/model.joblib + models/reference.csv
docker compose up -d
curl http://localhost:8000/health # expect {"status": "ok", ...}
```

If `/health` reports `kafka_connected`, `redis_connected`, or `db_connected`
as `false`, the dependent container isn't ready yet — `docker compose logs`
that service before touching the scorer or API.

## Roll back a bad model

There's no model registry — `models/model.joblib` is a single file the
scorer loads once at startup. Rollback is therefore git-level, not a
registry promotion:

```bash
git checkout <last-good-commit> -- training/ common/features.py
docker compose run --rm trainer   # regenerates model.joblib from that code
docker compose restart scorer     # scorer only loads the model at startup
```

`# ponytail: file-based model, not a registry — move to MLflow (or similar)
if multiple model versions ever need to be compared or promoted independently
of the training code.`

## When drift fires

`GET /monitoring` returns `dataset_drift` (bool) and `drift_share` (fraction
of monitored columns — `amount`, `fraud_score`, `is_fraud` — that drifted).

- **`dataset_drift: true`** — confirm it isn't a one-off window by checking
  `GET /monitoring` again after the next timer tick (`monitor/main.py`).
  Two consecutive drifted windows is the retrain trigger.
- **Retrain**: rerun `docker compose run --rm trainer` against current
  traffic patterns, then `docker compose restart scorer`.
- **Before retraining**, diff `GET /stats` (`fraud_rate`, `avg_score`)
  against the values in the README's captured example — a fraud-rate jump
  usually means the producer's synthetic fraud pattern changed, not that the
  model degraded, and retraining won't fix that.

## Known gaps

- No auth on `/score` or `/monitoring` — do not expose this stack past
  localhost without adding one.
- No cloud deployment target documented yet — this runbook covers the local
  stack only.
