# RAG eval harness

Runs a fixed set of questions with known-good expectations through the real
RAG pipeline and reports how many pass. Catches answer-quality regressions
(bad chunking, a broken reranker, a model swap that degrades citations) that
a passing unit-test suite wouldn't — everything in `tests/unit/` mocks the
models out entirely.

Requires a live stack (Qdrant + Ollama + the real embedding/reranker
models), so it doesn't run in CI or GitHub Actions. Run it on the server.

## Setup (one-time per environment)

```bash
uv run python scripts/generate_test_data.py

uv run python -m src.cli ingest \
  --input data/raw/test_docs/pandas_docs --collection pandas_docs \
  --library pandas --version 2.0.0

uv run python -m src.cli ingest \
  --input data/raw/test_docs/numpy_docs --collection numpy_docs \
  --library numpy --version 1.24.0
```

## Run

```bash
uv run python scripts/run_eval.py
```

Prints a per-case PASS/FAIL line, a summary, and writes a JSON report to
`eval/results/latest.json`. Exits non-zero if the pass rate drops below
`--fail-under` (default 0.8) — useful as a manual gate before/after a model
or prompt change.

## Adding cases

Edit `eval/golden_set.yaml`. Each case is either:
- an **answer case**: `expected_keywords` (substrings the answer must
  contain, case-insensitive) and `min_citations`
- a **refusal case**: `expect_refusal: true` — the query should be rejected
  (e.g. off-topic for the collection) with no citations

Keep cases grounded in what's actually ingested — `generate_test_data.py`'s
fixtures are intentionally small and content-checkable by hand, which is
what makes `expected_keywords` meaningful. Larger/real doc sets need their
own golden-set file with `--golden-set`.
