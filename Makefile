# ─────────────────────────────────────────────────────────────────────────────
# Makefile — GraphRAG vs VectorRAG developer shortcuts
#
# Usage:
#   make dev          start backend with hot reload
#   make test         run full test suite with coverage
#   make test-unit    run unit tests only
#   make lint         run ruff + black check
#   make format       auto-format with black + ruff --fix
#   make typecheck    run mypy
#   make install      install all dependencies
#   make index        run full indexing pipeline on the dataset
#   make clean        remove caches and build artifacts
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: dev test test-unit test-integration lint format typecheck install \
        index evaluate clean docker-build docker-up docker-down

BACKEND_DIR := backend
PYTHON      := python3
PYTEST      := pytest
UVICORN     := uvicorn

# ── Development ───────────────────────────────────────────────────────────────
dev:
	@echo ">>> Starting backend with hot reload..."
	cd $(BACKEND_DIR) && $(UVICORN) app.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir app \
		--log-level debug

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo ">>> Running full test suite..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/ -v

test-unit:
	@echo ">>> Running unit tests..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/unit/ -v -m "not integration"

test-integration:
	@echo ">>> Running integration tests..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/integration/ -v -m integration

test-stage1:
	@echo ">>> Running Stage 1 foundation tests..."
	cd $(BACKEND_DIR) && $(PYTEST) tests/unit/test_stage1_foundation.py -v

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	@echo ">>> Running ruff..."
	cd $(BACKEND_DIR) && ruff check app/ tests/
	@echo ">>> Checking black formatting..."
	cd $(BACKEND_DIR) && black --check app/ tests/

format:
	@echo ">>> Auto-formatting with black..."
	cd $(BACKEND_DIR) && black app/ tests/
	@echo ">>> Auto-fixing with ruff..."
	cd $(BACKEND_DIR) && ruff check --fix app/ tests/

typecheck:
	@echo ">>> Running mypy..."
	cd $(BACKEND_DIR) && mypy app/

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	@echo ">>> Installing production dependencies..."
	cd $(BACKEND_DIR) && pip install -r requirements.txt
	@echo ">>> Installing dev dependencies..."
	cd $(BACKEND_DIR) && pip install -r requirements-dev.txt

# ── Pipeline ──────────────────────────────────────────────────────────────────
index:
	@echo ">>> Running full indexing pipeline..."
	cd $(BACKEND_DIR) && $(PYTHON) ../scripts/run_indexing.py

evaluate:
	@echo ">>> Running evaluation suite (125 questions)..."
	cd $(BACKEND_DIR) && $(PYTHON) ../scripts/run_evaluation.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f backend

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo ">>> Cleaning caches and build artifacts..."
	find . -type d -name "__pycache__"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"        -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"    -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"                  -delete 2>/dev/null || true
	find . -name ".coverage"              -delete 2>/dev/null || true
	@echo ">>> Done."