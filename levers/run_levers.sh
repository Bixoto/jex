#!/usr/bin/env bash
# run_levers.sh -- reproduce the adopted lever ladder with the packaged copies.
# Requires the three banked inputs listed in LEVERS.md to be restored first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/tmp/opencode/submission03/venv03/bin/python}"
cd "$HERE"

for f in inputs/wave1-v1x-results.jsonl.gz bundle60/table.pkl fold60/results-folded.jsonl.gz; do
  [ -f "$f" ] || { echo "missing input: $f (see LEVERS.md)" >&2; exit 1; }
done

echo "== build canonical combined capture =="
"$PY" build_combined.py

echo "== re-score all variants (kit03) =="
"$PY" run_variants.py

echo "== per-catalog diff (full vs base) =="
"$PY" diff.py

echo "expected: full 60.29 / base 59.48 (see LADDER.md)"
