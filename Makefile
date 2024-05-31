.PHONY: clean
clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@find . -type d -name '*.ropeproject' | xargs rm -rf
	@rm -rf .pytest_cache/
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg*
	@rm -f MANIFEST
	@rm -f .coverage.*

.PHONY: run_api
run_api:
	@bash run_api.sh

.PHONY: test
test:
	@pytest

.PHONY: lint
lint:
	@flake8 .

.PHONY: build
build:
	@pip install build
	@python -m build
