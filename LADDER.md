# LADDER.md — v1.x+ adopted stack (Decision Index 0.3)

All numbers are exact kit03 (`62d2f51`, edition 0.3) on the same 140,178-row
in-edition set. Source: `ladder-results.json` (copied from
`finalstack/results.json`), `combined-fold-receipt.json`, `receipt_diff.json`,
`VERIFY-rescore/scores.json`, `VERIFY-rescore-inprocess/scores.json`.

## Cumulative ladder

| step | stack | decision index | raw |
|---|---|---:|---:|
| base | frozen v1.x capture | **59.48** | 69.35 |
| + fold | + cat62 fold (D, τ=−13.63) | **59.96** | 69.71 |
| + c05best | + cat59 isotonic·shrink λ=0.75 nf=10 | **60.09** | — |
| + R42 | + cat42 Entailment τ=−8 | **60.21** | 69.84 |
| + R20 | + cat20 option_0 τ=+15 | **60.29** | 69.88 |
| extras | full + R56/R21 | 60.36 | 69.92 |

## Leave-one-out gate checks

| variant | decision index |
|---|---:|
| full (adopted) | **60.29** |
| no-R20 (fold + c05best + R42) | 60.21 |
| no-c05best (fold + R42 + R20) | 60.16 |
| no-R42 (fold + c05best + R20) | 60.17 |
| extras (+R56 +R21) | 60.36 — **NOT adopted** |

## Per-catalog movement (full vs base) — exactly 4 catalogs

| cat | base skill | full skill | Δ |
|---|---:|---:|---:|
| 62 When2Call | 0.6904 | 0.8324 | +0.1420 |
| 59 RAGTruth noul | 0.5343 | 0.5808 | +0.0465 |
| 42 NLI4CT | 0.6282 | 0.6743 | +0.0461 |
| 20 BPoMP | 0.8613 | 0.9094 | +0.0481 |

All other catalogs are bit-identical (`receipt_diff.json`, `n_moved = 4`).

## Rollback

Drop any lever by re-running `../levers/run_variants.py` with that flag `False`
(equivalently `build_combined.py` restricted to the remaining catalogs). Dropping
the full stack returns to **59.48**. Probabilities are never modified; only
`choice` and cat59 `noul` are overridden. The frozen capture is never modified.

## Receipt (canonical combined)

`combined-fold-receipt.json`: source `.../wave1-v1x/results.jsonl.gz`,
`no_probability_changes: true`, counters r20 186 · r42 427 · fold 476 ·
c59 2700.

> Note on counters: the streaming pass (`combined-fold-receipt.json`) applied the
> fold to all **3,652** cat62 rows and the c05 override to all **2,700** cat59
> `noul` rows, while the rule levers actually *changed* **427** cat42 and **186**
> cat20 choices. The base-vs-full per-catalog deltas (§above) are the auditable
> movement; all other catalogs are bit-identical.
