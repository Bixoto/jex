# REPRO.md — `runs/wave1-v1x-plus` (v1.x+, Decision Index 0.3)

## Re-score the submitted combined results (dogfood)

From a checkout of `apolinario/decision-index@62d2f51` (the canonical venv):

```bash
PYTHONPATH=<kit03> python -m decision_index.cli score \
  --edition 0.3 \
  --suite-dir <suite03> \
  --results runs/wave1-v1x-plus/results-combined.jsonl.gz \
  --out runs/wave1-v1x-plus/score
```

Expected (verified here bit-exact, see `../../VERIFY.md`):

```
decision_index        60.29
raw_index             69.88
balanced_skill        60.29
balanced_raw          69.88
breadth_skill         59.25
complete              true
completed             140178
```

## Rebuild the combined file from the frozen capture

Restore the three banked inputs (`levers/LEVERS.md`) then:

```bash
bash levers/build_combined.sh     # -> results-combined.jsonl.gz
```

The overlay is deterministic; a rebuild reproduces the same decisions (probabilities
untouched). The packaged canonical file's sha256 is

```
5003231c9757d430ff9439c17a0e20f166b8d3dacef1dcde66e49056708ab9c7  results-combined.jsonl.gz
```

(matches the banked `finalstack/results-combined.jsonl.gz` sha256 exactly).

## Frozen suite hashes

| file | sha256 |
|---|---|
| selected-rows | `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` |
| added-rows | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| gsm8k-rows | `759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a` |

The results file must not be edited after scoring.
