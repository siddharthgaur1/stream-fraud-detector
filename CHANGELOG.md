# Changelog

## [Unreleased]

Baseline snapshot as of the portfolio hygiene pass (2026-08-04):

- Kafka-compatible streaming ingestion (Redpanda) -> Redis rolling features -> XGBoost + IsolationForest ensemble -> FastAPI scoring -> Postgres persistence -> Evidently drift monitoring.
- Measured load test: 1733 requests / 0 failures, 29.7 req/s throughput, p95/p99 2300ms/2700ms on a single-worker uvicorn process (see README Results — the gap to the 22ms single-request floor is the known `--workers` bottleneck).
- Fixed pre-existing CI failures: 2 ruff line-length violations and one unsorted import block (unrelated to this session's other changes — these were already failing before the README hygiene pass touched this repo).
