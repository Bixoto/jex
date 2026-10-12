#!/usr/bin/env python3
"""Per-catalog diff full vs base and full vs fold_only."""
import json, os
F = "/tmp/opencode/submission04/v1.xplus/levers/finalstack"
def cats(name):
    s = json.load(open(os.path.join(F, "score-" + name, "scores.json")))
    return {n: (b.get("index_raw"), b.get("index_skill")) for n, b in s["benchmarks"].items()}, s["decision_index"]
base, ib = cats("base")
full, if_ = cats("full")
fold, ifo = cats("fold_only")
moved_vs_base = {}
for n, v in full.items():
    bv = base.get(n)
    if bv and (abs((v[1] or 0) - (bv[1] or 0)) > 1e-9 or abs((v[0] or 0) - (bv[0] or 0)) > 1e-9):
        moved_vs_base[n] = {"base_skill": bv[1], "full_skill": v[1], "d": round((v[1] or 0) - (bv[1] or 0), 5)}
print("FULL vs BASE index", if_, "vs", ib)
print("moved:", json.dumps(moved_vs_base))
print("n_moved", len(moved_vs_base))
json.dump({"full_vs_base_index": if_, "base_index": ib, "moved": moved_vs_base},
          open(os.path.join(F, "receipt_diff.json"), "w"), indent=1)
