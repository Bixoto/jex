# INPUTS.md — exact construction & provenance of the reproduction inputs

This file documents **how each shipped input was produced**, so the chain can be
re-derived and audited. All paths below are the frozen, read-only source work
tree; nothing here was modified.

## 1. `wave1-v1x-results.jsonl.gz` — v1.x base capture

Produced by the `jex` engine package (`engine/` + `harness/`) on the frozen
`google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`,
vLLM 0.30.0, batch-invariant (`VLLM_BATCH_INVARIANT=1`), serial, greedy, no
thinking. One JSON object per request (140,178 rows); fields are the engine's
native capture schema (`run_id, catalog_id, payload, response, raw_output, …`).
Decision Index 0.3 re-score of this exact file is **59.48** (raw 69.35).

Source: `/tmp/opencode/phase8/v1x-capture/fetched/wave1-v1x/results.jsonl.gz`
(sha256 `59f27834802f5cab7aff6fd6fea2dfc43fca34fac48f22d0a0ec71de6807dee1`).
Environment/status companions: `environment.json`, `status.json` in the same dir.

## 2. `table.pkl` — fitted calibration table (derived panel)

Python `pickle` of a list of 312,320 records, each an 11-tuple:

```
(cid, group_id, run_id, field, qtype, option_keys, probabilities, noul, gold, st, winner)
```

Built by `bundle60/build_table.py` from the base capture + suite gold (0.3),
over all catalogs. Used by `final_common.c59_override()` → `calib_core.collect(recs, 59)`
→ `run_noul.oof_fit_predict(items, "iso_shrink", nf=10, seed=7, lam=0.75)` to
derive the cat59 `noul` overlay.

Source: `/tmp/opencode/phase8/v10/bundle60/table.pkl`
(sha256 `cf241d59f220c6fd65c6ec5ca6b2562e788e50e3fa2b94998bd3e0d014577107`).

## 3. `results-folded.jsonl.gz` — cat62 folded capture

Base capture with the cat62 When2Call fold applied post-hoc by
`fold60/fold_apply.py` (rule: signature `choice|A,B,C,D`, instructions contain
"When2Call phrase", emit `D` if `logit(p_work[D]) ≥ −13.63`). Same schema as the
base capture; probabilities untouched. This is the exact file
`scorekit.load_a1()` reads (`fold60/results-folded.jsonl.gz`).

Source: `/tmp/opencode/phase8/v10/fold60/results-folded.jsonl.gz`
(sha256 `5c9b96ae45a9f7b0ad810cab7e704fdb82de6203ea60e7ea12122ecd971a2165`).

## 4. Rebuild → combined file

From the three inputs above (plus the external kit/suite for scoring):

```bash
bash levers/build_combined.sh     # writes results-combined.jsonl.gz (full adopted stack)
bash levers/run_levers.sh         # builds + re-scores every ladder variant
```

Single-pass counters (must match `runs/jex-1.0/combined-fold-receipt.json`):

```
fold_changed 476   c59_changed 2700   r42_changed 427   r20_changed 186
no_probability_changes: true
```

### Verification detail (gzip container vs content)

The shipped canonical `results-combined.jsonl.gz` has container sha256
`5003231c9757d430ff9439c17a0e20f166b8d3dacef1dcde66e49056708ab9c7`. A rebuild
reproduces **byte-identical decompressed content** — decompressed sha256
`24d4b2be21839043690b6281ef7c23f12538938bff344495990bac8403af7678`, 140,178
rows — and re-scores to **60.29**. The outer gzip sha256 depends on the gzip
header timestamp, so it is not byte-stable across rebuilds; use the decompressed
hash (or the re-score) for verification:

```bash
gunzip -c runs/jex-1.0/results-combined.jsonl.gz | sha256sum
# 24d4b2be21839043690b6281ef7c23f12538938bff344495990bac8403af7678  -
```

## 5. Frozen suite hashes (from the official kit)

| file | sha256 |
|---|---|
| `selected-rows.jsonl.gz` | `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` |
| `added-rows.jsonl.gz` | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| `gsm8k-rows.jsonl.gz` | `759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a` |
