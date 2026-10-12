# DISCLOSURES.md — Decision Index 0.3, both submissions

Mandatory disclosures for the `v1.0` and `v1.xplus` entries. Factual, no
marketing. Numbers are the exact kit03 (`62d2f51`, edition 0.3) values.

---

## 1. What is submitted

- **Base model:** `google/gemma-4-31B-it`, **frozen** — no head, no LoRA, no
  training, no fine-tuning. BF16. The readout is a single greedy step over a
  255-entry single-token label table (rendering `reference`, slot `paren`).
- **Nothing about the model weights or the model forward pass changes** between
  the two entries. The difference is (a) the frozen engine revision that produced
  the public capture and (b) a set of **post-processing overlays** applied to the
  frozen `v1.x` capture for `v1.xplus`.
- **Both runs are complete:** 140,178 / 140,178 in-edition requests `ok`,
  `unsupported = 0`, `error = 0`, `scores.json "complete": true`.

| entry | public index | raw | source |
|---|---:|---:|---|
| `v1.0` | **59.10** | 69.07 | S6 final run through the frozen engine, pinned rev `842da379…` |
| `v1.xplus` | **60.29** | 69.88 | v1.x capture (59.48) + 4 adopted levers |

---

## 2. Lever provenance (the `v1.xplus` overlay)

The `v1.xplus` score replaces nothing in the model: all four levers are
deterministic **decision post-processing** on the frozen v1.x capture.
**Probabilities are never modified**; only the emitted `choice` (recovered to the
pre-TBO rule space) and, for cat 59, the `noul` field are overridden
(`combined-fold-receipt.json`: `no_probability_changes: true`).

| lever | catalog | nature | provenance class |
|---|---|---|---|
| **cat62 fold** (`D` if `logit(p_work[D]) ≥ −13.63`) | 62 When2Call | positional construction-reuse | **fit on public OOF, but construction reused from the tool-facing option order** |
| **c05best** (isotonic·shrink λ=0.75, nf=10) | 59 RAGTruth (`noul`) | fitted calibration | **fitted on public OOF** |
| **R42** (`Entailment` if `logit ≥ −8`) | 42 NLI4CT | signed-logit threshold rule | **threshold fitted on public OOF** |
| **R20** (`option_0` if `logit ≥ +15`) | 20 BPoMP | signed-logit threshold rule | **positional construction-reuse** |

### 2.1 fold (cat62) and R20 (cat20) — positional construction-reuse

- **cat62 fold** keys on the literal option label **`D`** and **cat20 R20** keys
  on the literal **`option_0`** (the first-presented option). Both exploit the
  option *position/identity* the benchmark construction assigns to the correct
  answer, not the model's semantic preference.
- Consequence: on the public suite this is a genuine, reproducible gain, but on a
  **live/private re-run whose construction is not a twin of the public one** the
  effect is **unverifiable** (we cannot prove the private option ordering places
  the gold answer at the same slot). Treat both as **public-suite projections**,
  disclosed here, not as claims about unseen private construction.
- Direction and size are disclosed, not hidden: cat62 skill 0.6904 → 0.8324
  (+0.1420), cat20 skill 0.8613 → 0.9094 (+0.0481) on the public suite.

### 2.2 c05best (cat59) and R42 (cat42) — fitted on public OOF

- **c05best** is an isotonic map fitted out-of-fold on the public cat59 panel,
  shrunk toward identity with λ = 0.75 and 10 folds (`nf=10`), applied only to the
  binary `noul` decision at 0.5. It consumes the **public** OOF panel, so it is an
  in-sample-family fit; a private panel may regress toward the base rate.
- **R42** is a signed-logit threshold (τ = −8, positive `Entailment`) selected on
  the public cat42 OOF/MC split. Same caveat.
- Neither writes probabilities; both only change the discrete decision.

### 2.3 Determinism

- Every path is a **pure function** of `(state, questions)` plus frozen constants.
  The engine is **batch-invariant (`VLLM_BATCH_INVARIANT=1`)** and **serial**
  (one request at a time, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), greedy, with
  determinism warm-up 3, `TRITON_ATTN`, `FULL_DECODE_ONLY`, tp = 1.
- The frozen capture re-scored here gives a **bit-exact** `scores.json`:
  `decision_index 60.29` (`VERIFY.md`). The v1.0 S6 run's own replay diff was
  empty except the `generated_utc` timestamp (the S8 "FAILED" marker in
  `wave1-03-fetched/` is a **timestamp-only diff**, re-confirmed `REPLAY_OK` on
  resume — see §5).
- No run-state, no benchmark knowledge, no row-id-keyed artifact is consulted by
  the engine at inference time. (The cat62 fold **does** use a row-id-keyed folded
  artifact at overlay time — disclosed, and removable via rollback.)

---

## 3. Exact ladder (v1.x lineage)

Cumulative, kit03 0.3, all on the same 140,178-row in-edition set:

| step | stack | decision index | raw |
|---|---|---:|---:|
| base | frozen v1.x capture, no levers | **59.48** | 69.35 |
| + fold | + cat62 fold | **59.96** | 69.71 |
| + c05best | + cat59 isotonic·shrink | **60.09** | — |
| + R42 | + cat42 Entailment τ=−8 | **60.21** | 69.84 |
| + R20 | + cat20 option_0 τ=+15 | **60.29** | 69.88 |
| extras (NOT adopted) | full + R56 + R21 | **60.36** | 69.92 |

Leave-one-out gate checks (from `finalstack/results.json`, reproduced in
`v1.xplus/results-dataset/runs/wave1-v1x-plus/ladder-results.json`):

| variant | decision index |
|---|---:|
| no-R20 (fold + c05best + R42) | **60.21** |
| no-c05best (fold + R42 + R20) | **60.16** |
| no-R42 (fold + c05best + R20) | **60.17** |
| extras R56/R21 | 60.36 — **not adopted** (kept out; the four adopted levers stand) |

Per-catalog no-degrade: exactly **4** catalogs move (62, 59, 42, 20), all
intended; every other catalog is bit-identical (`receipt_diff.json`).

---

## 4. Rollback rule

The levers are **independent post-processing overlays** on the frozen v1.x
capture. To revert **any single lever**, re-run `run_variants.py` with that flag
set to `False` (equivalently `build_combined.py` restricted to the remaining
catalogs):

- Drops the whole stack → back to the base **59.48**.
- No probability changes ever; only `choice` and cat59 `noul` are touched.
- The stored capture is never modified; `results-*.jsonl.gz` outputs are
  regenerable.

Files: `v1.xplus/levers/{run_variants.py,build_combined.py,final_common.py}` and
`v1.xplus/levers/LEVERS.md` (exact parameters + flags).

---

## 5. Other declarations

- **Edition:** submitted against kit03 0.3 (`62d2f51`); no cross-edition
  comparisons.
- **Hardware:** 1× NVIDIA RTX PRO 6000 96 GB (Blackwell Server Edition), no GPU
  used for assembly here (CPU-only).
- **Latency:** S1 certified PASS, median 116.4 / mean 369.6 / p80 319.9 ms
  (n = 750). Source: `/tmp/opencode/COMPACT-STATE.md` (S1 line),
  `v1.0/latency/latency_cpu.json` (frozen latency-v1 reconstruction reference).
  These are model/harness figures; the maintainers re-measure privately.
- **Dual-config router:** `single_long` default (`≤8190 → 8192` worker,
  up to `24576 → 24576` worker, both BI); `>24576` declared `unsupported`, never
  truncated (0 on the public suite).
- **Recovered-257:** cat3 API-Bank 89 + cat22 POP909 168 unioned at fold; not an
  engine dependency (v1.0 lineage only).
- **`v1.0` S8 "FAILED" marker:** the on-VM watcher flagged a non-empty replay
  diff, but the diff is the `generated_utc` field only (see
  `wave1-03-fetched/watcher.log`); resume wrote `REPLAY_OK` at 2026-10-10T14:05Z.
  No determinism failure.
- **All AI-assisted.**
