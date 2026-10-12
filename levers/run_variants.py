#!/usr/bin/env python3
"""Apply adopted+gated levers in one pass; rescore all variants with kit03.

Outputs: finalstack/score-<variant>/{scores.json,index.json,benchmark-summary.json},
finalstack/results.json (summary), finalstack/receipts.json.
"""
import sys, os, json, time, collections
sys.path.insert(0, "/tmp/opencode/submission04/v1.xplus/levers/finalstack")
from final_common import *

t0 = time.time()
edition, rows = C.load_suite_rows()
a1 = K.load_a1()
T = load_T(CAP)
noul = c59_override()
print("prep done %.1fs  noul_ov=%d" % (time.time() - t0, len(noul)), flush=True)

summary = {"baseline_capture": CAP, "suite": SUITE, "edition": "0.3",
           "levers": {
               "fold_cat62": {"tau": -13.63, "positive": "D"},
               "c05best": {"cat": 59, "mode": "isotonic*shrink", "lam": 0.75, "nf": 10},
               "R42": {"cat": 42, "positive": "Entailment", "tau": -8.0},
               "R20": {"cat": 20, "positive": "option_0", "tau": 15.0},
               "R56": {"cat": 56, "positive": "phishing", "tau": 5.0},
               "R21": {"cat": 21, "positive": "headline_1", "tau": 4.0}},
           "variants": {}}

def cats_of(score_dir):
    s = json.load(open(os.path.join(score_dir, "scores.json")))
    out = {}
    for n, b in s["benchmarks"].items():
        out[n] = (b.get("index_raw"), b.get("index_skill"), b.get("in_index"))
    return out, s["decision_index"]

for name, cfg in VARIANTS.items():
    res = load_results(CAP)
    touched = {}
    if cfg["fold"]:
        touched["fold"] = apply_fold(res, a1)
    if cfg["c59"]:
        touched["c59"] = apply_c59(res, noul)
    for cid in (42, 20, 56, 21):
        key = "r%d" % cid
        if cfg.get(key, False):
            P, tau = RULES[cid]
            touched[key] = apply_rule(res, T, cid, P, tau)
    out = os.path.join(FINAL, "score-" + name)
    os.makedirs(out, exist_ok=True)
    s = K.score(res, edition, rows, out)
    summary["variants"][name] = {"config": cfg, "touched": touched,
                                 "decision_index": s["decision_index"],
                                 "raw_index": s["raw_index"],
                                 "scores": s["scores"]}
    print("%-10s index=%.4f raw=%.4f  touched=%s  %.1fs" %
          (name, s["decision_index"], s["raw_index"], touched, time.time() - t0), flush=True)

# per-catalog no-degrade check: full vs fold_only (and full vs base)
sc_full, _ = cats_of(os.path.join(FINAL, "score-full"))
sc_fold, _ = cats_of(os.path.join(FINAL, "score-fold_only"))
sc_base, _ = cats_of(os.path.join(FINAL, "score-base"))
moved = {}
for n, v in sc_full.items():
    fv = sc_fold.get(n)
    if fv and (abs((v[1] or 0) - (fv[1] or 0)) > 1e-9 or abs((v[0] or 0) - (fv[0] or 0)) > 1e-9):
        moved[n] = {"fold_only_skill": fv[1], "full_skill": v[1],
                    "d_skill": round((v[1] or 0) - (fv[1] or 0), 5)}
summary["no_degrade_full_vs_fold_only"] = moved
summary["expected_moved_cats"] = ["62", "59", "42", "20"]
print("MOVED (full vs fold_only):", json.dumps(moved), flush=True)

json.dump(summary, open(os.path.join(FINAL, "results.json"), "w"), indent=1)
print("done %.1fs" % (time.time() - t0))
