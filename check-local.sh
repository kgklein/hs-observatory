#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo "== Ruff lint =="
python -m ruff check .

echo
echo "== Ruff formatting =="
python -m ruff format --check .

echo
echo "== Documentation architecture tests =="
python -m unittest discover -s tests -p 'test_architecture_graph.py'

echo
echo "== MkDocs strict build =="
NO_MKDOCS_2_WARNING=1 mkdocs build --strict

echo
echo "All local checks passed."
