# ─────────────────────────────────────────────────────────────────────────────
# Makefile — GraphRAG vs VectorRAG developer shortcuts
#
# Backend:
#   make dev              start backend with hot reload
#   make test             run full test suite with coverage
#   make test-unit        run unit tests only
#   make test-integration run integration tests only
#   make lint             ruff + black check
#   make format           auto-format with black + ruff --fix
#   make typecheck        run mypy
#   make install          install all Python dependencies
#
# Frontend:
#   make frontend-install install npm packages
#   make frontend-dev     start Vite dev server
#   make frontend-build   build React SPA for production
#   make frontend-lint    run eslint
#
# Pipeline:
#   make index            run full indexing pipeline
#   make evaluate         run evaluation suite
#   make seed             copy sample corpus to data/raw/
#
# Docker:
#   make docker-up        start dev stack (backend + frontend + redis)
#   make docker-down      stop dev stack
#   make docker-prod      start production stack
#   make docker-build     build all images
#   make docker-logs      tail backend logs
#
# Misc:
#   make clean            remove caches and build artifacts
#   make help             show this message
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: dev test test-unit test-integration lint format typecheck install \
        frontend-install frontend-dev frontend-build frontend-lint \
        index evaluate seed \
        docker-up docker-down docker-prod docker-build docker-logs \
        clean help

BACKEND_DIR  := backend
FRONTEND_DIR := frontend
PYTHON       := python3
PYTEST       := pytest
UVICORN      := uvicorn
NPM          := npm

# ── Backend: development ──────────────────────────────────────────────────────
dev:
	@echo ">>> Starting backend with hot reload..."
	cd $(BACKEND_DIR) && $(UVICORN) app.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir app \
		--log-level debug

# ── Backend: testing ──────────────────────────────────────────────────────────
test:
	@echo ">>> Running full test suite with coverage..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -v \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

test-unit:
	@echo ">>> Running unit tests..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/unit/ -v

test-integration:
	@echo ">>> Running integration tests..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/integration/ -v

test-fast:
	@echo ">>> Running tests without coverage (fast)..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -v --no-cov -x

# ── Backend: code quality ─────────────────────────────────────────────────────
lint:
	@echo ">>> Checking with ruff..."
	cd $(BACKEND_DIR) && ruff check app/ tests/
	@echo ">>> Checking formatting with black..."
	cd $(BACKEND_DIR) && black --check app/ tests/
	@echo ">>> Lint passed."

format:
	@echo ">>> Formatting with black..."
	cd $(BACKEND_DIR) && black app/ tests/
	@echo ">>> Fixing with ruff..."
	cd $(BACKEND_DIR) && ruff check --fix app/ tests/
	@echo ">>> Done."

typecheck:
	@echo ">>> Running mypy..."
	cd $(BACKEND_DIR) && mypy app/

# ── Backend: dependencies ─────────────────────────────────────────────────────
install:
	@echo ">>> Installing Python production dependencies..."
	pip install -r requirements.txt
	@echo ">>> Installing Python dev dependencies..."
	pip install -r requirements-dev.txt
	@echo ">>> Installing pre-commit hooks..."
	pre-commit install
	@echo ">>> Done."

# ── Frontend ──────────────────────────────────────────────────────────────────
frontend-install:
	@echo ">>> Installing frontend npm packages..."
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo ">>> Done."

frontend-dev:
	@echo ">>> Starting Vite dev server on http://localhost:5173 ..."
	cd $(FRONTEND_DIR) && $(NPM) run dev

frontend-build:
	@echo ">>> Building React SPA for production..."
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo ">>> Build output: frontend/dist/"

frontend-lint:
	@echo ">>> Running ESLint..."
	cd $(FRONTEND_DIR) && $(NPM) run lint

frontend-preview:
	@echo ">>> Previewing production build on http://localhost:4173 ..."
	cd $(FRONTEND_DIR) && $(NPM) run preview

# ── Pipeline scripts ──────────────────────────────────────────────────────────
index:
	@echo ">>> Running full indexing pipeline..."
	$(PYTHON) scripts/run_indexing.py --data-dir data/raw/articles

index-dev:
	@echo ">>> Running dev indexing (100 chunks, no gleaning)..."
	$(PYTHON) scripts/run_indexing.py --data-dir data/raw/articles --max-chunks 100 --gleaning-rounds 0

evaluate:
	@echo ">>> Running evaluation suite..."
	$(PYTHON) scripts/run_evaluation.py \
		--questions-file data/evaluation/questions.json \
		--output-dir data/evaluation/

seed:
	@echo ">>> Copying sample corpus to data/raw/ ..."
	cp -n data/raw/sample_corpus/*.json data/raw/ 2>/dev/null || true
	@echo ">>> Sample corpus ready in data/raw/"

prepare:
	@echo ">>> Preparing MultiHopRAG dataset..."
	$(PYTHON) scripts/prepare_data.py --corpus data/raw/corpus.json --multihop data/raw/MultiHopRAG.json
	@echo ">>> Done. Run 'make index' to start indexing."

prepare-dev:
	@echo ">>> Preparing MultiHopRAG dataset (dev mode: 50 articles)..."
	$(PYTHON) scripts/prepare_data.py --corpus data/raw/corpus.json --multihop data/raw/MultiHopRAG.json --max-articles 50
	@echo ">>> Done."

prepare-dry:
	@echo ">>> Dry run — previewing what prepare would write..."
	$(PYTHON) scripts/prepare_data.py --corpus data/raw/corpus.json --multihop data/raw/MultiHopRAG.json --dry-run

# ── Docker: development ───────────────────────────────────────────────────────
docker-build:
	@echo ">>> Building Docker images..."
	docker-compose build

docker-up:
	@echo ">>> Starting dev stack (backend + frontend + redis)..."
	docker-compose up --build -d
	@echo ">>> Backend:  http://localhost:8000/api/v1/health"
	@echo ">>> Frontend: http://localhost:5173"
	@echo ">>> API docs: http://localhost:8000/docs"

docker-down:
	@echo ">>> Stopping dev stack..."
	docker-compose down

docker-logs:
	docker-compose logs -f backend

docker-logs-all:
	docker-compose logs -f

docker-shell:
	docker-compose exec backend bash

# ── Docker: production ────────────────────────────────────────────────────────
docker-prod:
	@echo ">>> Starting production stack..."
	docker-compose -f docker-compose.prod.yml up --build -d
	@echo ">>> App: http://localhost:80"

docker-prod-down:
	docker-compose -f docker-compose.prod.yml down

docker-prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo ">>> Cleaning Python caches..."
	find . -type d -name "__pycache__"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"        -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"    -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"                  -delete 2>/dev/null || true
	find . -name ".coverage"              -delete 2>/dev/null || true
	@echo ">>> Cleaning frontend build..."
	rm -rf $(FRONTEND_DIR)/dist 2>/dev/null || true
	@echo ">>> Done."

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' || \
	echo "Run 'make <target>' — see Makefile header for full list."