# Runbook — stream-fraud-detector

Written to be usable by someone who did not build this, at 2am, without
waking anyone up.

**Scope, honestly:** this is a docker-compose stack. It is not deployed to a
cloud provider — see [Deployment status](#deployment-status). Everything below
is real and executable against the local stack; the cloud sections say plainly
what would change.

---

## 1. Orientation

| Service | What it does | Fails how |
| --- | --- | --- |
| `scorer` | FastAPI. `POST /score`, plus a background Kafka consumer scoring the stream. | Scoring stops; `/health` goes 503 |
| `producer` | Publishes synthetic transactions to Redpanda. | Stream goes quiet; HTTP scoring unaffected |
| `monitor` | Evidently drift report every 60s; fires the retrain alert. | Drift goes undetected; scoring unaffected |
| `redis` | Per-user rolling features (last 30 amounts, known devices). | **Scoring degrades silently** — see [Redis down](#redis-is-down) |
| `postgres` | Scored-transaction audit log; source for `/stats` and drift. | Scores still returned, nothing recorded |
| `redpanda` | Kafka-compatible broker. | Stream scoring stops; HTTP scoring unaffected |

Dependency direction: **Redis and Postgres are hard dependencies of a healthy
scorer. Kafka is not.** `/health` reflects exactly that — Kafka down is
reported but stays 200, because `POST /score` still works.

---

## 2. Deploy

```bash
export API_KEYS="$(openssl rand -hex 24)"     # never run without this
make train                                    # first time only: builds models/model.joblib
make up                                       # waits for /health, fails loudly if it never comes
make smoke                                    # end-to-end: /health + a real POST /score
```

`make up` blocks until `/health` returns 200 or 120s elapse. If it times out:

```bash
docker compose ps           # which container is not up
docker compose logs scorer  # almost always the answer
```

### Verifying what is actually running

```bash
curl -s localhost:8000/health | python -m json.tool
```

`model_version` and the `fraud_model_info` metric label carry the loaded
model's identity — `file:/models/model.joblib` or
`registry:Production:7`. **Check this first in any incident.** "Which model is
serving" should never require reading deployment config to answer.

---

## 3. Rollback

### If MLflow is configured

Promotion is a registry transition, so rollback is one too:

```bash
python -m scripts.promote_model --version <last-good> --stage Production
docker compose restart scorer      # the model is loaded once, at startup
```

Find the last good version in the MLflow UI, or:

```bash
mlflow models list -n stream-fraud-detector
```

### If it is not (default)

The model is a single file, so rollback is git-level:

```bash
git checkout <last-good-commit> -- training/ common/features.py
make train
docker compose restart scorer
```

### Rolling back code, not the model

```bash
docker compose down
git checkout <last-good-commit>
make up
```

Images are published to GHCR per commit SHA, so you can also pin one:

```bash
docker pull ghcr.io/<owner>/stream-fraud-detector:sha-<full-sha>
```

---

## 4. When drift fires at 2am

You were paged because **`drift_share` stayed at or above `0.5` for 2
consecutive 60-second windows.**

### What that threshold means, and why it is what it is

`drift_share` is the fraction of monitored columns (`amount`, `fraud_score`,
`is_fraud`) that Evidently flagged as drifted. `0.5` means at least two of the
three moved.

The **2 consecutive windows** requirement exists because single windows cross
the line routinely and for boring reasons — a burst from one merchant, a
deploy, a quiet window with a small sample. Alerting on the first one trains
people to ignore the alert, and an ignored alert is worse than none. Two
windows costs ~2 minutes of detection latency and removes almost all of that
noise.

Tune with `DRIFT_SHARE_THRESHOLD` and `DRIFT_CONSECUTIVE_WINDOWS`. Raising the
window count costs detection latency; lowering it costs the alert's
credibility.

The alert fires **once** on the transition into a drifted state, not every
window, and re-arms when drift clears. State lives in
`reports/drift_state.json` and survives a monitor restart on purpose — a
crash-looping monitor must still be able to accumulate windows.

### Triage, in order

**1. Is this real, or did the input change?**

```bash
curl -s localhost:8000/stats -H "X-API-Key: $API_KEYS" | python -m json.tool
```

Compare `fraud_rate` against the README's captured baseline. A large jump in
`fraud_rate` usually means the *traffic* changed, not that the model
degraded. In the demo stack that means the producer's synthetic pattern
changed. Retraining will not fix a changed input distribution — it will bake
it in.

**2. Which column moved?**

```bash
curl -s "localhost:8000/monitoring?format=html" -H "X-API-Key: $API_KEYS" > /tmp/drift.html
open /tmp/drift.html
```

- **`amount` drifted, `fraud_score` did not** → input distribution shifted and
  the model absorbed it. Usually benign. Watch, do not retrain.
- **`fraud_score` drifted, `amount` did not** → the model is behaving
  differently on similar inputs. This is the one that matters.
- **`is_fraud` rate moved sharply** → check the threshold did not change.
  It is baked into the model bundle, so a change means a new model got
  deployed. Confirm `model_version` against what you expect.

**3. Did something get deployed?**

```bash
docker compose logs scorer | grep model_loaded
```

If `version` or `source` changed recently, drift is a deployment, not decay.
Roll back (§3) rather than retrain.

**4. Is the sample real?**

`rows_compared` under a few hundred makes drift statistics unreliable. If
traffic was low, wait for the next window before acting.

### If it is genuine model decay

```bash
make train                                                    # retrain
python -m training.train --data data/transactions.csv --register   # if MLflow is on
python -m scripts.promote_model --version <new> --stage Staging
# observe, then:
python -m scripts.promote_model --version <new> --stage Production
docker compose restart scorer
```

Promotion gates (`scripts/promote_model.py`) enforce `ensemble_auc >= 0.90`,
`ensemble_avg_precision >= 0.50`, `train_rows >= 5000`. `--force` overrides
them and logs that it did — use it only with a reason you would repeat out
loud.

### If you just want the paging to stop

Drift alerts are not a scoring outage. Nothing is down. It is safe to
acknowledge and pick it up in the morning **unless** `fraud_score` drift is
accompanied by a `fraud_rate` moving toward 0 or 1 — that means the model has
collapsed to one class and is no longer discriminating.

---

## 5. Alerts and what each one means

| Signal | Where | Means | Urgency |
| --- | --- | --- | --- |
| Slack: `drift detected` | `monitor` → `SLACK_WEBHOOK_URL` | 2+ consecutive drifted windows | Next morning, unless §4's collapse case |
| Slack: `drift cleared` | same | Drift returned below threshold | None, informational |
| `/health` 503 | Load balancer / compose healthcheck | Redis or Postgres unreachable | **Now** |
| `fraud_db_insert_failures_total` rising | `/metrics` | Scoring works, nothing is being recorded | Within the hour — `/stats` and drift are blind |
| `fraud_stream_last_message_age_seconds` climbing | `/metrics` | Consumer thread died while the process stayed up | Within the hour |
| `fraud_score_requests_total{outcome="error"}` rising | `/metrics` | Model raising on real input | **Now** |
| 429s from `/score` | Client | Rate limit (`RATE_LIMIT_PER_MIN`, default 600) | Only if unexpected |

Slack alerting no-ops with a logged warning when `SLACK_WEBHOOK_URL` is unset,
so **absence of alerts is not evidence of health**. Verify with:

```bash
docker compose logs monitor | grep -E "slack_webhook_unset|drift_alert"
```

---

## 6. Known failure modes

### Redis is down

**Worst failure mode in the system, because it is quiet.** The feature store
returns empty history, so every transaction looks like a user's first: `avg10`
is 0, `amount_ratio_avg10` falls back to a neutral 1.0, `is_new_device` is
always 1. The model still returns confident-looking scores. They are wrong.

`/health` returns 503 on Redis being unreachable precisely so a load balancer
pulls the replica rather than serving quiet garbage.

```bash
docker compose logs redis
docker compose restart redis
```

History rebuilds from the stream — no restore needed, but scores are degraded
until users accumulate transactions again.

### Postgres is down

`POST /score` still returns 200 with a valid score; the row is not persisted
and `fraud_db_insert_failures_total` climbs. `/stats` and drift detection go
blind. `/health` returns 503.

### Kafka / Redpanda is down

Stream scoring stops. `POST /score` is unaffected and `/health` stays 200.
`kafka_connected: false` and `fraud_stream_last_message_age_seconds` climbing.
The consumer retries connection 30 times at 2s intervals on startup; once
running, a broker outage leaves the thread alive but idle, so watch the age
gauge rather than the boolean.

### The scorer will not start

```
RuntimeError: API_KEYS is unset.
```

Intentional. Set `API_KEYS`, or `ALLOW_UNAUTHENTICATED=true` for local work
only. There is no default key on purpose.

```
no model registered at stage 'Production'
```

MLflow is configured but nothing is promoted. Promote one, or unset
`MLFLOW_TRACKING_URI` to fall back to the local bundle.

### Redis memory climbing

Per-user keys carry a 90-day sliding TTL (`REDIS_HISTORY_TTL_SEC`) and the
container is capped at 256mb with `allkeys-lru`. If memory still climbs, check
user-ID cardinality — an ID that is unique per transaction defeats both.

### Latency spike

SHAP runs per request and is the dominant cost. Check
`fraud_score_latency_seconds` by source. If p99 rose without traffic rising,
suspect Redis latency first (every score does a read and a write) before the
model.

---

## 7. Deployment status

**Not currently deployed to any cloud provider.** No public endpoint exists.

The application is deploy-ready in the ways that matter — non-root image,
API-key auth, rate limiting, `/health` with real semantics, `/metrics`,
structured JSON logs, images published to GHCR — but nothing has been
provisioned.

What a real deployment would need, none of which is done:

- **Managed Kafka is out of budget.** The honest substitute is dropping the
  broker entirely in the deployed environment and running HTTP-only scoring:
  `POST /score` works standalone, and the stream consumer is what needs the
  broker. Say so in the README rather than implying a Kafka deployment.
- Managed Redis and Postgres (Cloud Memorystore / Cloud SQL, or the AWS
  equivalents), replacing the compose containers.
- Secrets in Secret Manager, not env vars in a compose file. The
  `fraud`/`fraud` Postgres credentials are local-only and must not survive
  contact with anything internet-facing.
- A real `API_KEYS` value, rotated, one key per consumer so a key can be
  revoked without breaking everyone.
- `/metrics` on a separate port or behind network policy — it is
  unauthenticated by design for scrapers.

---

## 8. Quick reference

```bash
make help          # every target
make up            # start, wait for healthy
make down          # stop, remove volumes
make smoke         # end-to-end check
make load-test     # 60s locust run at 50 users
make test          # unit tests, no infrastructure needed
docker compose logs -f scorer

curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/metrics | grep fraud_
curl -s localhost:8000/stats -H "X-API-Key: $API_KEYS"
```
