.PHONY: help install dev test lint format typecheck clean run docker

help:
	@echo "install    Install runtime dependencies + package"
	@echo "dev        Install dev dependencies (tests, linters)"
	@echo "test       Run the unit test suite"
	@echo "lint       Run ruff linter"
	@echo "format     Auto-format with ruff"
	@echo "typecheck  Run mypy static type checks"
	@echo "run        Run on a sample: make run INPUT=path/to/video.mp4"
	@echo "docker     Build the Docker image"
	@echo "clean      Remove caches and build artifacts"

install:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	mypy src

run:
	depthtrack $(INPUT) --config config/default.yaml

docker:
	docker build -t depthtrack .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
