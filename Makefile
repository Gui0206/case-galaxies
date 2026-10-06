.PHONY: install demo serve test lint experiments
install:
	uv sync --frozen --group dev
demo:
	uv run galaxies demo
serve:
	uv run galaxies serve
test:
	uv run pytest
lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
experiments:
	uv run galaxies experiments .runtime/datasets/demo.jsonl
