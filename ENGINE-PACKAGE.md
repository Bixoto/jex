# Engine package — maintainer runnability

Requirement (kit03 README §Submitting, step 3): after review, the maintainers
run the submitted model themselves on the two private parts, "exactly as you
did", so the model must be runnable by them: weights on the Hub, the inference
code you used (or a `/v1/systemone` server they can start), and its settings.

Our engine is **not** a `/v1/systemone` server; it is the `harness/` + `engine/`
vLLM pipeline. This directory stages exactly that.

## Contents

```
engine/        label-slot readout engine (frozen decision rules included)
  engine.py  config.py  labels.py  render.py  readout.py  decision_rules.json
harness/       the run pipeline
  vllm_pipeline.py        fast pass -> gate -> reason -> fold (+rules/TBO/JCD)
  vllm_fastpass.py        vLLM V1 BI fast pass (render + label readout)
  vllm_reason.py          reasoning runner (thought/re-read, cap/format)
  vllm_reason_resume.py   reasoning-only resume over an existing fast pass
  reason_pilot.py         gate + eligibility helpers
  pilot_analyze.py        weighting / delta / chance math
  _batching.py            fixed-composition batch helpers
  decision_rules_apply.py applied at fold (rules from engine/decision_rules.json)
  jcd_c48.py              catalog-48 coherence projection (no-op for 0.3: c48 retired)
  vllm_production_bootstrap.sh   the exact production launcher (pins, preflight, run)
requirements.txt   vllm==0.30.0, transformers==5.14.1 (+ torch via wheel)
```

`harness/vllm_fastpass.py` puts `engine/` on `sys.path` (or uses the installed
`di_engine`); it reads `engine/decision_rules.json` itself. `harness/*.py`
expect the official kit at `$REPO/kit` for scoring/weights (`pilot_analyze`),
which maintainers already have.

## What the maintainers need to run a private request set

1. **Weights**: `google/gemma-4-31B-it` (public). Frozen — no head, no LoRA.
   Revision was not pinned in the recorded bootstrap
   (`[verify]`; the reference HF run used `842da3794eaa0b77d5f08bae87a17459d91ff475`).
   If exactness matters, pin the revision they used for the public run.
2. **This package** at the repo root, alongside `kit/`.
3. **Env**: vLLM 0.30.0 with `VLLM_BATCH_INVARIANT=1`, `TRITON_ATTN`,
   `FULL_DECODE_ONLY`, bf16, tp=1, max-model-len 8192, util 0.9, max-num-seqs 32.
4. **Rows**: the private request rows in kit schema (`_evaluation.run_id`,
   `questions`, `expected`, `state`).
5. **Command** (same shape as `run_gsm8k.sh`):

```bash
VLLM_BATCH_INVARIANT=1 python3 harness/vllm_pipeline.py \
  --rows <private-rows.jsonl.gz> --reference <any non-empty results.jsonl.gz> \
  --model google/gemma-4-31B-it --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.9 --max-num-seqs 32 \
  --attention-backend TRITON_ATTN --cudagraph-mode FULL_DECODE_ONLY \
  --rendering reference --slot paren --trust-remote-code \
  --coverage <config> --cap 512 --fast-batch 8 --reason-batch 32 \
  --temperature-by-options "10.124 -1.633 0.05" \
  --kit-root <repo>/kit --out <merged.jsonl.gz> --fast-out <fast.jsonl.gz>
```

`--reference` is only used to print summary deltas; any non-empty results file
works.

## Gaps / risks to flag in the PR

- **No single-server entrypoint.** The maintainers prefer a `/v1/systemone`
  server they can start. We ship a Python pipeline instead; it is deterministic
  and runnable, but it is a script, not a server. *Option: wrap
  `vllm_pipeline.run_fast_pass` behind a tiny `/v1/systemone` FastAPI shim.*
- **Dual-config router.** The public 0.3 run used two workers: ≤8190 tokens →
  max-model-len 8192; longer → 24576 (each BI). The staged single-config
  `vllm_pipeline.py` uses one `--max-model-len`. For the private sets, either run
  both configs with the router rule or declare the 8192-only capacity limit
  (longer prompts → `unsupported`, no truncation).
- **Recovered-257 rows** (cat 3 API-Bank 89, cat 22 POP909 168) are unioned from
  `recovered-257.jsonl.gz` at fold; they are part of the declared run. Include
  that file if a private set touches those catalogs.
- **Latency bar.** The engine reasons ~19–30% of requests at 2,061 ms; main-board
  admission needs median/mean/p80 ≤ 1000 ms. Submit at a P80-safe coverage (see
  `SUBMISSION-RUNBOOK.md` §d) or accept the reasoning-board placement.
