.PHONY: install install-classifier test run run-llm clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Create the virtualenv and install core dependencies.
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Optional, heavy: the transformers prompt-injection classifier defense.
install-classifier:
	$(PIP) install -r requirements-classifier.txt

# Deterministic test suite: MockEmbedding + mock LLM, no torch, no downloads.
test:
	$(PY) -m pytest -q

# Reproduce the deterministic mock-baseline breach matrix offline from the committed
# cache: no torch, no downloads, no model calls. Covers the three torch-free defenses;
# the injection_classifier column and real numbers come from `make run-llm`.
run:
	$(PY) -m ragpoison.runner --from-cache --defense none --defense context_fencing --defense provenance_filter

# Full real run: HuggingFace embeddings + Ollama generation. Heavy; run separately.
# Renders the resulting breach-rate matrix and worked bypasses into the README,
# replacing the TBD placeholders with real numbers.
run-llm:
	$(PY) -m ragpoison.runner --embedding hf --llm ollama --render-readme

clean:
	rm -rf .pytest_cache **/__pycache__ __pycache__ *.egg-info
