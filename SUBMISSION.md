# SUBMISSION.md — v1.x+ (superseding), Decision Index 0.3

| item | value |
|---|---|
| public index | **60.29** |
| balanced raw | 69.88 |
| breadth skill | 59.25 |
| rows | 140,178 complete (`unsupported 0`, `error 0`) |
| kit | `62d2f51` (edition 0.3) |
| engine pin | revision `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| base capture | v1.x, 59.48 (frozen; post-processed, never edited) |
| hardware | 1× RTX PRO 6000 96 GB |
| latency (S1) | median 116.4 / mean 369.6 / p80 319.9 ms — PASS |

This package **supersedes** `v1.0/`: same frozen model and forward pass, plus four
disclosed deterministic post-processing levers (`../DISCLOSURES.md`).

## Checklist (runbook §e/§f/§g)

- [x] `results-dataset/runs/wave1-v1x-plus/results-combined.jsonl.gz` (canonical, 60.29).
- [x] `scores.json`, `index.json`, `benchmark-summary.json` (kit03 CLI, 60.29).
- [x] `environment.json` (v1.x write path), `status.json`, `combined-fold-receipt.json`.
- [x] `ladder-results.json`, `receipt_diff.json`, `LADDER.md`, `REPRO.md`.
- [x] `levers/` code + exact parameters + `PROVENANCE.sha256` + rollback.
- [x] `latency/LATENCY-CERTIFICATE.md`.
- [x] `VERIFY-rescore/` + `VERIFY-rescore-inprocess/` — bit-exact 60.29.
- [x] `MANIFEST.sha256`.
- [ ] Publish dataset + engine package; open PR (PR-v1.xplus.md).
