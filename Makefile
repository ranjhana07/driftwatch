PYTHON ?= python

.PHONY: install test compile serve ui scheduler demo bench up down clean

install:
	$(PYTHON) -m pip install -e ".[server,dev]"

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q src tests
	for f in ui/js/*.js; do node --check $$f; done

serve:
	$(PYTHON) -m driftwatch serve --ui-dir ui

scheduler:
	$(PYTHON) -m driftwatch scheduler

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf .pytest_cache .driftwatch build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
