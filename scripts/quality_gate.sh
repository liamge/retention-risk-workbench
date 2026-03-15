#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running Python bytecode compilation (syntax/import sanity)..."
python -m compileall src tests

if ! python -m ruff --version >/dev/null 2>&1; then
  echo "Ruff is not installed. Install dev tools with: pip install -r requirements-dev.txt"
  exit 1
fi

echo "Running Ruff (fast lint: unused imports, undefined names)..."
python -m ruff check src tests

echo "Scanning for junk/untracked artifacts (pyc, cache, OS cruft, logs)..."
JUNK_PATTERNS='(\.pyc$|__pycache__|\.DS_Store$|\.log$|\.ipynb_checkpoints|\.coverage$|^mlruns/|^artifacts/)' 
JUNK_FILES=$(git ls-files --others --exclude-standard | grep -E "$JUNK_PATTERNS" || true)

if [[ -n "$JUNK_FILES" ]]; then
  echo "Found junk/untracked artifacts. Remove or gitignore them:"
  echo "$JUNK_FILES"
  exit 1
fi

echo "Quality gate passed."
