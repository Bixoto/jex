# REPRO.md — full reproduction chain (jex-1.0, Decision Index 0.3)

Three steps: **fetch inputs → run levers → re-score 60.29**. Inputs all ship in
`J-I-B/decision-index-results` → `runs/jex-1.0/inputs/` (see `REPRO-INPUTS.md`,
`INPUTS.md`). Maintainers supply the official kit + suite.

## 0. Prerequisites

- Python venv with the official kit on `PYTHONPATH`
  (`apolinario/decision-index@62d2f51de34a2de64906345b6bc3e98e27ff55c7`, MIT).
- Suite: `suite03/` (edition 0.3) from the same checkout.
- `hf` CLI (anonymous read is enough; public dataset).

## 1. Fetch the shipped inputs

```bash
bash levers/fetch_inputs.sh
# -> levers/inputs/wave1-v1x-results.jsonl.gz
# -> levers/bundle60/table.pkl
# -> levers/fold60/results-folded.jsonl.gz
```

Pinned hashes (verify after download):

| file | sha256 |
|---|---|
| `inputs/wave1-v1x-results.jsonl.gz` | `59f27834802f5cab7aff6fd6fea2dfc43fca34fac48f22d0a0ec71de6807dee1` |
| `bundle60/table.pkl` | `cf241d59f220c6fd65c6ec5ca6b2562e788e50e3fa2b94998bd3e0d014577107` |
| `fold60/results-folded.jsonl.gz` | `5c9b96ae45a9f7b0ad810cab7e704fdb82de6203ea60e7ea12122ecd971a2165` |

## 2. Run the levers → combined file

```bash
bash levers/build_combined.sh     # -> levers/results-combined.jsonl.gz
```

Expected single-pass counters (must equal
`runs/jex-1.0/combined-fold-receipt.json`):
`fold_changed 476, c59_changed 2700, r42_changed 427, r20_changed 186`,
`no_probability_changes: true`.

The rebuild reproduces byte-identical **decompressed** content:

```bash
gunzip -c levers/results-combined.jsonl.gz | sha256sum
# 24d4b2be21839043690b6281ef7c23f12538938bff344495990bac8403af7678  -
```

The shipped canonical file `runs/jex-1.0/results-combined.jsonl.gz` has container
sha256 `5003231c9757d430ff9439c17a0e20f166b8d3dacef1dcde66e49056708ab9c7`
(140,178 rows). Its outer gzip sha depends on the gzip timestamp, so verify by
decompressed hash.

## 3. Re-score → expect 60.29

```bash
PYTHONPATH=<kit03> python -m decision_index.cli score \
  --edition 0.3 \
  --suite-dir <suite03> \
  --results runs/jex-1.0/results-combined.jsonl.gz \
  --out runs/jex-1.0/score
```

Expected (verified here bit-exact via the CLI):

```
decision_index        60.29
raw_index             69.88
balanced_skill        60.29
balanced_raw          69.88
breadth_skill         59.25
complete              true
completed             140178
```

## 4. Full ladder (optional)

```bash
bash levers/run_levers.sh         # base / fold_only / full / extras / noR20 / noC05 / noR42
```

Base capture → `59.48`; +fold `59.96`; +c05best `60.09`; +R42 `60.21`; +R20
**`60.29`**. Extras R56/R21 → 60.36 (not adopted). Exactly four catalogs move
(62, 59, 42, 20); all others bit-identical.

## Frozen suite hashes

| file | sha256 |
|---|---|
| `selected-rows.jsonl.gz` | `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` |
| `added-rows.jsonl.gz` | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| `gsm8k-rows.jsonl.gz` | `759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a` |

The results file must not be edited after scoring.
