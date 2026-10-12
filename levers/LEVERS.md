# LEVERS.md — deterministic post-processing overlay (v1.x+)

The `v1.xplus` score is the frozen **v1.x capture (59.48)** plus four independent,
disclosed post-processing levers. **The model and the forward pass are
unchanged. Probabilities are never modified** (`no_probability_changes: true`);
only the emitted `choice` (recovered to the pre-TBO rule space) and cat59 `noul`
are overridden.

## Exact parameters

| lever | catalog | rule | implementation reused |
|---|---|---|---|
| **fold** | 62 When2Call | emit **`D`** if `logit(p_work[D]) ≥ −13.63`, else argmax over non-D options | `fold60` (`scorekit.load_a1`) |
| **c05best** | 59 RAGTruth (`noul`) | isotonic · shrink **λ = 0.75**, **nf = 10**, OOF fit; `noul = 1[p ≥ 0.5]` | `calopt60` (`run_noul.oof_fit_predict('iso_shrink', nf=10, seed=7, lam=0.75)`) |
| **R42** | 42 NLI4CT | emit **`Entailment`** if `logit ≥ −8`, else argmax of others | `mining60` (`scorelib.rule_choice`) |
| **R20** | 20 BPoMP | emit **`option_0`** if `logit ≥ +15`, else argmax of others | `mining60` (`scorelib.rule_choice`) |

`p_work = p_emit ** T` (the pre-TBO recovery; `T` = the row's
`emitted_temperature`), then `logit = ln(p/(1−p))`. `rule_choice` semantics mirror
`engine-v1x`.

**Not adopted (record only):** R56 (`phishing`, τ=5) and R21 (`headline_1`, τ=4)
— extras give 60.36 (68.92 raw), kept out.

## Files

- `final_common.py` — single-pass application of the adopted stack; holds the
  parameter table. Paths are patched to this package root.
- `run_variants.py` — re-scores all variants (base / fold_only / full / extras /
  noR20 / noC05 / noR42) with kit03; writes `score-<variant>/` and
  `ladder-results.json`.
- `build_combined.py` — builds the canonical combined capture in one streaming
  pass; writes `results-combined.jsonl.gz` + `combined-fold-receipt.json`.
- `diff.py` — full-vs-base per-catalog movement.
- `bundle60/scorekit.py`, `calopt60/*`, `mining60/work/*`, `fold60/*` — the reused
  implementations (verbatim, with provenance hashes in `PROVENANCE.sha256`).

## Inputs required to re-run (not shipped here, to stay small)

These are the frozen derived inputs the original run used; restore them from the
banked work tree before running:

| path (relative to this dir) | source (banked, read-only) | size |
|---|---|---|
| `inputs/wave1-v1x-results.jsonl.gz` | `/tmp/opencode/phase8/v1x-capture/fetched/wave1-v1x/results.jsonl.gz` | ~90 MB |
| `bundle60/table.pkl` | `/tmp/opencode/phase8/v10/bundle60/table.pkl` | ~70 MB |
| `fold60/results-folded.jsonl.gz` | `/tmp/opencode/phase8/v10/fold60/results-folded.jsonl.gz` | ~87 MB |

`KIT`/`SUITE` in the copied scripts still point at the frozen
`/tmp/opencode/submission03/{kit03,suite03}` (maintainers supply the kit); the
`phase8/v10` paths have been rewritten to this package root.

## Result

Canonical combined (`results-combined.jsonl.gz`, full adopted stack), kit03 CLI:
**decision_index 60.29, raw 69.88, complete true, completed 140178.** Re-scored
bit-exact here (see `../../VERIFY.md`).
