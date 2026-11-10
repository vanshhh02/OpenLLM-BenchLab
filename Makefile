.PHONY: setup test test-unit run-backend run-frontend run-all docker-up docker-down clean help

VENV_BIN = .venv/bin
PYTHON = $(VENV_BIN)/python
PIP = $(VENV_BIN)/pip

help:
	@echo "OpenLLM BenchLab Make Targets:"
	@echo "  setup         - Initialize virtualenv and install dependencies"
	@echo "  run-backend   - Start FastAPI backend server"
	@echo "  run-frontend  - Start Vite React frontend"
	@echo "  test          - Run all unit and integration tests"
	@echo "  docker-up     - Start all services with Docker Compose"
	@echo "  docker-down   - Stop Docker Compose services"
	@echo "  clean         - Clean temporary build and cache files"

setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cd frontend && npm install

run-backend:
	$(PYTHON) -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

run-frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest backend/tests tests -v

test-unit:
	$(PYTHON) -m pytest tests/test_hardware.py tests/test_config.py tests/test_cost.py -v

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

