#!/usr/bin/env bash
# fetch_inputs.sh -- restore the reproduction inputs from the published results
# dataset into the layout the packaged lever scripts expect (see LEVERS.md).
#
# Requires: `hf` CLI authenticated (or anonymous read on a public dataset).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-J-I-B/decision-index-results}"
REV="${REV:-}"   # optional revision pin, e.g. REV=<sha>

rev_arg=()
[ -n "$REV" ] && rev_arg=(--revision "$REV")

mkdir -p "$HERE/inputs" "$HERE/bundle60" "$HERE/fold60"

hf download "$REPO" runs/jex-1.0/inputs/wave1-v1x-results.jsonl.gz \
  --repo-type dataset "${rev_arg[@]}" --local-dir "$HERE/inputs"
mv "$HERE/inputs/runs/jex-1.0/inputs/wave1-v1x-results.jsonl.gz" "$HERE/inputs/wave1-v1x-results.jsonl.gz"
rm -rf "$HERE/inputs/runs"

hf download "$REPO" runs/jex-1.0/inputs/table.pkl \
  --repo-type dataset "${rev_arg[@]}" --local-dir "$HERE/bundle60"
mv "$HERE/bundle60/runs/jex-1.0/inputs/table.pkl" "$HERE/bundle60/table.pkl"
rm -rf "$HERE/bundle60/runs"

hf download "$REPO" runs/jex-1.0/inputs/results-folded.jsonl.gz \
  --repo-type dataset "${rev_arg[@]}" --local-dir "$HERE/fold60"
mv "$HERE/fold60/runs/jex-1.0/inputs/results-folded.jsonl.gz" "$HERE/fold60/results-folded.jsonl.gz"
rm -rf "$HERE/fold60/runs"

echo "restored:"
ls -la "$HERE/inputs/wave1-v1x-results.jsonl.gz" \
       "$HERE/bundle60/table.pkl" \
       "$HERE/fold60/results-folded.jsonl.gz"
echo "expected sha256:"
echo "  59f27834802f5cab7aff6fd6fea2dfc43fca34fac48f22d0a0ec71de6807dee1  inputs/wave1-v1x-results.jsonl.gz"
echo "  cf241d59f220c6fd65c6ec5ca6b2562e788e50e3fa2b94998bd3e0d014577107  bundle60/table.pkl"
echo "  5c9b96ae45a9f7b0ad810cab7e704fdb82de6203ea60e7ea12122ecd971a2165  fold60/results-folded.jsonl.gz"
