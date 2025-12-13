PYTHON ?= python3
MODULE ?= trmnl_server
ARGS ?=
export SERVER_PORT ?= 4567

.PHONY: serve test clean
serve:
	$(PYTHON) -m $(MODULE) $(ARGS)

test:
	$(PYTHON) -m pytest -q tests

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache .coverage
	rm -rf var/generated
	rm -rf var/db
