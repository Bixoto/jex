#!/usr/bin/env python3
"""Build the canonical combined capture (full adopted stack) as
finalstack/results-combined.jsonl.gz, applied in ONE streaming pass.

Stack = cat62 fold + c05best(cat59) + R42(cat42) + R20(cat20).
Probabilities untouched; only `choice` and cat59 `noul` are overridden.
"""
import sys, os, json, gzip, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_common import *

OUT = os.path.join(FINAL, "results-combined.jsonl.gz")
a1 = K.load_a1()
noul = c59_override()
counters = collections.Counter()
per_cat = collections.defaultdict(collections.Counter)


def rule_row(r, cid, P, tau):
    n = 0
    for key, a in (r.get("response", {}).get("answers", {}) or {}).items():
        if a.get("type") != "choice":
            continue
        probs = a.get("probabilities") or {}
        if P not in probs:
            continue
        keys = list(probs.keys())
        T = (r.get("raw_output") or {}).get(key, {}).get("emitted_temperature")
        work = inv_tbo(probs, keys, T)
        nc = rule_choice(work, keys, P, tau)
        if nc != a.get("choice"):
            a["choice"] = nc
            n += 1
    return n


with gzip.open(CAP, "rt", encoding="utf-8") as fin, \
     gzip.open(OUT, "wt", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        rid = r.get("run_id")
        ans = (r.get("response") or {}).get("answers", {}) or {}
        for field, a in ans.items():
            if (rid, field) in a1 and a.get("type") == "choice":
                if a.get("choice") != a1[(rid, field)]:
                    counters["fold_changed"] += 1
                    per_cat["62"]["changed"] += 1
                a["choice"] = a1[(rid, field)]
            if (rid, field) in noul and a.get("type") == "noul":
                if a.get("noul") != noul[(rid, field)]:
                    counters["c59_changed"] += 1
                    per_cat["59"]["changed"] += 1
                a["noul"] = noul[(rid, field)]
        cid = r.get("catalog_id")
        if cid in (42, 20):
            P, tau = RULES[cid]
            n = rule_row(r, cid, P, tau)
            if n:
                counters["r%d_changed" % cid] += n
                per_cat[str(cid)]["changed"] += n
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")

receipt = {"source": CAP, "output": OUT,
           "stack": {"fold_cat62": True, "c05best_cat59": True, "R42_cat42": True, "R20_cat20": True},
           "counters": dict(counters),
           "per_catalog_changed": {c: dict(v) for c, v in per_cat.items()},
           "no_probability_changes": True}
json.dump(receipt, open(os.path.join(FINAL, "combined-fold-receipt.json"), "w"), indent=1)
print(json.dumps(receipt, indent=1))
