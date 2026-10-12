#!/usr/bin/env bash
# build_combined.sh -- build only the canonical combined capture (full adopted
# stack) from the frozen v1.x capture. Same prerequisites as run_levers.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/tmp/opencode/submission03/venv03/bin/python}"
cd "$HERE"
for f in inputs/wave1-v1x-results.jsonl.gz bundle60/table.pkl fold60/results-folded.jsonl.gz; do
  [ -f "$f" ] || { echo "missing input: $f (see LEVERS.md)" >&2; exit 1; }
done
"$PY" build_combined.py
echo "wrote $HERE/results-combined.jsonl.gz"
sha256sum "$HERE/results-combined.jsonl.gz"
