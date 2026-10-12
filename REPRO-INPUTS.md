# REPRO-INPUTS.md — reproduction inputs: what ships where, what is documented-only

**Revised decision D3.** The raw lever inputs **are** shipped now (previously
doc-only). Everything needed to go
`base capture → run levers → combined file → re-score 60.29` is published. The
frozen ground-truth **suite** and the evaluation **kit** are *not* ours to
redistribute; they are obtained from the official benchmark (documented-only).

## 1. Shipped inputs (results dataset)

Repo: `J-I-B/decision-index-results` (public) → `runs/jex-1.0/inputs/`

| dataset path | size (bytes) | sha256 | what it is |
|---|---:|---|---|
| `runs/jex-1.0/inputs/wave1-v1x-results.jsonl.gz` | 93,623,397 | `59f27834802f5cab7aff6fd6fea2dfc43fca34fac48f22d0a0ec71de6807dee1` | frozen **v1.x base capture** (pre-lever engine output, 140,178 rows, scored 59.48) |
| `runs/jex-1.0/inputs/table.pkl` | 73,658,027 | `cf241d59f220c6fd65c6ec5ca6b2562e788e50e3fa2b94998bd3e0d014577107` | fitted **calibration table** (derived probability panel; 312,320 per-item records) |
| `runs/jex-1.0/inputs/results-folded.jsonl.gz` | 91,550,294 | `5c9b96ae45a9f7b0ad810cab7e704fdb82de6203ea60e7ea12122ecd971a2165` | **cat62 folded capture** (base + fold overlay only); `scorekit.load_a1()` reads this |

Restore them into the layout the packaged scripts expect (see
`levers/LEVERS.md`) with:

```bash
bash levers/fetch_inputs.sh        # hf download into levers/{inputs,bundle60,fold60}
```

The canonical result file `runs/jex-1.0/results-combined.jsonl.gz` is **not**
touched by this addition.

## 2. Documented-only (not redistributed)

These are the benchmark's, not ours; obtain from the official kit at
`apolinario/decision-index@62d2f51`:

- `suite03/` — edition-0.3 suite (`selected-rows.jsonl.gz`,
  `added-rows.jsonl.gz`, `gsm8k-rows.jsonl.gz`, `excluded-questions.json`,
  `manifest.json`). Ground truth / test set → **restricted**.
- `kit03/` — the official scoring kit (`decision_index` package, licence MIT).
  Supplied by the maintainers; pinned by commit `62d2f51`.

`levers/REPRO.md` walks the full chain; the only external inputs are the kit and
the suite.

## 3. Licence screen (per input)

| input | embeds restricted suite text/rows? | verdict |
|---|---|---|
| base capture | Yes — contains the `payload` (question text/options) — **but this is already public** in the shipped `results-combined.jsonl.gz` (same schema/fields); no gold labels | **ship** (nothing new exposed) |
| `table.pkl` | No question text; only ids, option labels, model probabilities and gold flags — a derived panel | **ship** (own derived table) |
| folded capture | Same schema as base capture; cat62 `choice` overlay only | **ship** (nothing new exposed) |
| `suite03`, `kit03` | The restricted suite / official kit themselves | **do not ship** (documented-only above) |

No uploaded input adds restricted information beyond what the already-published
canonical results file contains.

## 4. Optional, for the record

`R56`/`R21` extra levers (`phishing` τ=5, `headline_1` τ=4) are not adopted;
their parameters are in `levers/LEVERS.md` and can be reproduced with the same
inputs via `run_variants.py`.
