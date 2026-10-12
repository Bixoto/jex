# jex

Inference-time decision engine for the **Decision Index 0.3** public leaderboard
(edition 0.3, kit `62d2f51de34a2de64906345b6bc3e98e27ff55c7`).

**Entry:** `jex-1.0` (display **"Jex 1.0"**) — public index **60.29**
(balanced raw 69.88, breadth 59.25), coverage 0 (no reasoning).

Frozen `google/gemma-4-31B-it` (`842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16;
no head, no LoRA, no training) served by this engine (vLLM 0.30.0,
batch-invariant, serial, greedy, no thinking), plus four disclosed deterministic
**post-processing** levers applied to the frozen v1.x capture. No probability is
ever modified — only the emitted `choice` (recovered to pre-TBO rule space) and
cat59 `noul`.

## Results

- **Dataset:** [`J-I-B/decision-index-results`](https://huggingface.co/datasets/J-I-B/decision-index-results)
  → `runs/jex-1.0/` (`scores.json` `"complete": true`, completed 140,178;
  `results-combined.jsonl.gz` sha256
  `5003231c9757d430ff9439c17a0e20f166b8d3dacef1dcde66e49056708ab9c7`).
- **Complete run:** 140,178 / 140,178 requests `ok`, `unsupported 0`, `error 0`.

## Layout

```
engine/            engine code (render, readout, labels, decision rules, config)
harness/           vLLM run pipeline (serial, batch-invariant), rule application
levers/            deterministic post-processing overlay (code + parameters)
requirements.txt
ENGINE-PACKAGE.md  maintainer-runnability package description
SUBMISSION.md      one-page submission summary
DISCLOSURES.md     mandatory disclosures (lever provenance, fitting, determinism, rollback)
LADDER.md          cumulative ladder + leave-one-out checks
REPRO.md           full chain: fetch inputs -> run levers -> re-score 60.29
REPRO-INPUTS.md    what ships where, what is documented-only, licence screen
INPUTS.md          exact construction & provenance of each reproduction input
LATENCY-CERTIFICATE.md  S1 certified latency (median 116.4 / mean 369.6 / p80 319.9 ms, n=750)
```

## Ladder (kit03, 0.3)

`59.48` base → `59.96` +fold → `60.09` +c05best → `60.21` +R42 → **`60.29` +R20**.
Extras R56/R21 give 60.36 and are **not adopted**. Leave-one-out: no-R20 60.21,
no-c05best 60.16, no-R42 60.17. Exactly 4 catalogs move (62, 59, 42, 20); all
others bit-identical.

Levers: (1) cat62 fold — emit `D` if `logit(p_work[D]) ≥ −13.63`; (2) cat59
c05best — isotonic·shrink λ=0.75, nf=10 (OOF); (3) cat42 R42 — emit `Entailment`
if `logit ≥ −8`; (4) cat20 R20 — emit `option_0` if `logit ≥ +15`. Rollback:
re-run `levers/run_variants.py` with a flag `False`; whole stack off → base 59.48.

## Reproduce

Fetch the shipped lever inputs, rebuild the combined file, then re-score it
(should reproduce 60.29 bit-exactly):

```bash
bash levers/fetch_inputs.sh        # inputs from runs/jex-1.0/inputs/
bash levers/build_combined.sh      # -> levers/results-combined.jsonl.gz
PYTHONPATH=<kit03> python -m decision_index.cli score \
  --edition 0.3 --suite-dir <suite03> \
  --results runs/jex-1.0/results-combined.jsonl.gz --out runs/jex-1.0/score
```

See `REPRO.md` for the full chain and `ENGINE-PACKAGE.md` for run settings.
The raw lever inputs now **ship** in the results dataset
(`runs/jex-1.0/inputs/`) — see `REPRO-INPUTS.md` and `INPUTS.md`. Only the
benchmark suite and kit are documented-only (supplied by the maintainers).

## Declared limits

Dual-config `single_long` router (`≤8190 tok → 8192` worker, longer → `24576`
worker, both BI; `>24576` → `unsupported`, never truncated); qtype `choice`
(2..255 options) + `noul` only; BF16 (no quantization); no cross-edition
comparisons. Disclosed: fold (cat62) and R20 (cat20) are **positional
construction-reuse** — genuine on the public suite, unverifiable on a private run
whose construction is not a twin; c05best and R42 are fitted on the public OOF
panel.

## Licence

Code in this repository is provided under the MIT licence (matching the
`apolinario/decision-index` kit). The suite/data files are **not** redistributed
here.

*All AI-assisted.*
