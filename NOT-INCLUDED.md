# NOT-INCLUDED.md — deliberately omitted reproduction inputs

Per the publish decision **D3 = doc-only**, the raw lever inputs are **not**
shipped. The overlay is instead certified by (a) the bit-exact re-score of the
packaged combined file (`60.29`, kit03 CLI **and** in-process `score_run_v02`),
(b) `levers/PROVENANCE.sha256` on the verbatim code, and (c) the banked
`combined-fold-receipt.json` / `ladder-results.json` uploaded in the results
dataset (`runs/jex-1.0/`).

Not included (source files remain in the read-only staging package):

- the frozen v1.x base capture (`results.jsonl.gz`, ~90 MB),
- the fitted calibration artifact `table.pkl` (~70 MB),
- the folded capture `results-folded.jsonl.gz` (~87 MB).

Consequence: `levers/build_combined.py` cannot be run from this repo alone; it
needs those three inputs. `levers/run_levers.sh` documents the prerequisites.

Everything else (engine, harness, lever code, exact parameters, provenance
hashes) **is** included and is sufficient for the maintainers to re-score the
submitted results and to run the model on the private parts.
