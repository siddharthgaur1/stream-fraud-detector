.DEFAULT_GOAL := help
.PHONY: help up down logs train test lint load-test clean rebuild ps smoke

# API_KEYS is generated once and reused so `make up` twice does not invalidate
# the key you just curled with. Override by exporting API_KEYS yourself.
export API_KEYS ?= local-dev-key

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

train: ## Train the model (required once before `make up`)
	docker compose run --rm trainer

up: ## Start the full stack
	docker compose up -d --build
	@echo "waiting for the scorer to report healthy..."
	@for i in $$(seq 1 60); do \
	  if curl -sf http://localhost:8000/health > /dev/null; then echo "ready"; exit 0; fi; \
	  sleep 2; \
	done; \
	echo "scorer did not become healthy in 120s — check: docker compose logs scorer"; exit 1

down: ## Stop the stack and remove volumes
	docker compose down -v

ps: ## Show container status
	docker compose ps

logs: ## Tail logs from every service
	docker compose logs -f

lint: ## Run ruff
	ruff check .

test: ## Run the test suite
	pytest -q

smoke: ## End-to-end check against a running stack
	@curl -sf http://localhost:8000/health | python -m json.tool
	@curl -sf -X POST http://localhost:8000/score \
	  -H "Content-Type: application/json" \
	  -H "X-API-Key: $$API_KEYS" \
	  -d '{"transaction_id":"smoke-1","user_id":"user_00001","amount":48000, \
	       "merchant_category":"electronics","hour":3,"device_id":"brand-new-device", \
	       "location_hash":"560001","user_age_days":12}' | python -m json.tool

load-test: ## Headless locust run against a running stack (60s, 50 users)
	locust -f loadtest/locustfile.py --host http://localhost:8000 \
	  --headless -u 50 -r 10 -t 60s --csv loadtest/results

rebuild: ## Rebuild images without cache
	docker compose build --no-cache

clean: ## Remove generated artifacts
	rm -rf reports/ loadtest/results_*.csv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
