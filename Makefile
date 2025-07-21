.PHONY: help install install-dev test lint format type-check security clean pre-commit setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup development environment
	uv sync --all-extras
	uv run pre-commit install

install: ## Install production dependencies
	uv sync

install-dev: ## Install development dependencies
	uv sync --extra dev

test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest -m unit

test-integration: ## Run integration tests only
	uv run pytest -m integration

test-cov: ## Run tests with coverage
	uv run pytest --cov=. --cov-report=html --cov-report=term

lint: ## Run all linting tools
	uv run flake8 .
	uv run bandit -r . --skip B101
	uv run safety check

format: ## Format code with black and isort
	uv run isort .
	uv run black .

type-check: ## Run type checking with mypy
	uv run mypy .

security: ## Run security checks
	uv run bandit -r . --skip B101
	uv run safety check

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -f .coverage

# Application commands
run: ## Run the main application
	uv run conversation_intelligence_main.py

run-capture: ## Run in capture mode
	uv run conversation_intelligence_main.py --mode capture

run-bot: ## Run the voice bot
	uv run main_bot.py

# Docker commands (if you want to add Docker later)
docker-build: ## Build Docker image
	docker build -t conversation-intelligence .

docker-run: ## Run Docker container
	docker run -p 8000:8000 conversation-intelligence