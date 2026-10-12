#!/usr/bin/env python3
"""finalstack common: single-pass application of the adopted+gated stack.

Levers (reused implementations, no re-implementation):
  * cat62 fold          -> fold60 (via scorekit.load_a1 / fold60/fold_apply.py)
  * c05best             -> calopt60 run_noul.oof_fit_predict(iso_shrink, lam=.75, nf=10)
  * R42 cat42 Entailment tau -8  -> mining60 (scorelib rule semantics)
  * R20 cat20 option_0  tau 15   -> mining60 (scorelib rule semantics)
  * extras R56/R21      -> mining60

Everything is applied to the stored v1.x capture in ONE pass. Probabilities are
never modified; only `choice` (rule space) and cat59 `noul` are overridden.
"""
import sys, os, json, gzip, math, pickle, collections

BASE = "/tmp/opencode/submission04/v1.xplus/levers"
FINAL = BASE
sys.path.insert(0, os.path.join(BASE, "calopt60"))
sys.path.insert(0, os.path.join(BASE, "bundle60"))
sys.path.insert(0, os.path.join(BASE, "mining60", "work"))

import calib_core as C            # noqa: E402
import run_noul as RN            # noqa: E402
import scorekit as K             # noqa: E402
from base import load_results, CachedSuite  # noqa: E402
from decision_index.suite.io import Suite   # noqa: E402
from decision_index.pipeline import score_run_v02  # noqa: E402
from scorelib import inv_tbo, rule_choice   # noqa: E402

CAP = "/tmp/opencode/submission04/v1.xplus/levers/inputs/wave1-v1x-results.jsonl.gz"
SUITE = "/tmp/opencode/submission03/suite03"

VARIANTS = {
    "base":     dict(fold=False, c59=False, r42=False, r20=False, r56=False, r21=False),
    "fold_only": dict(fold=True, c59=False, r42=False, r20=False, r56=False, r21=False),
    "full":     dict(fold=True, c59=True, r42=True, r20=True, r56=False, r21=False),
    "extras":   dict(fold=True, c59=True, r42=True, r20=True, r56=True, r21=True),
    "noR20":    dict(fold=True, c59=True, r42=True, r20=False, r56=False, r21=False),
    "noC05":    dict(fold=True, c59=False, r42=True, r20=True, r56=False, r21=False),
    "noR42":    dict(fold=True, c59=True, r42=False, r20=True, r56=False, r21=False),
}

RULES = {  # catalog -> (positive, tau)
    42: ("Entailment", -8.0),
    20: ("option_0", 15.0),
    56: ("phishing", 5.0),
    21: ("headline_1", 4.0),
}


def c59_override():
    """c05best = cat59 isotonic*shrink lam=0.75 nf=10 (reused calopt60 code)."""
    recs = pickle.load(open(C.TABLE, "rb"))
    items = C.collect(recs, 59)
    for x in items:
        x["pred"] = x["p"] >= 0.5
    pr = RN.oof_fit_predict(items, "iso_shrink", nf=10, seed=7, lam=0.75)
    return {k: (1.0 if p >= 0.5 else 0.0) for k, p in pr.items()}


def load_T(path):
    T = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for qk, v in (r.get("raw_output") or {}).items():
                T[(r["run_id"], qk)] = v.get("emitted_temperature")
    return T


def apply_fold(res, a1):
    n = 0
    for (rid, field), ch in a1.items():
        if ch is None or rid not in res:
            continue
        ans = res[rid]["response"]["answers"].get(field)
        if ans is not None and ans.get("type") == "choice":
            ans["choice"] = ch
            n += 1
    return n


def apply_rule(res, T, n, P, tau):
    touched = 0
    for rid, rec in res.items():
        if rec.get("catalog_id") != n:
            continue
        for key, a in rec["response"]["answers"].items():
            if a.get("type") != "choice":
                continue
            probs = a.get("probabilities") or {}
            if P not in probs:
                continue
            keys = list(probs.keys())
            work = inv_tbo(probs, keys, T.get((rid, key)))
            nc = rule_choice(work, keys, P, tau)
            if nc != a.get("choice"):
                a["choice"] = nc
                touched += 1
    return touched


def apply_c59(res, ov):
    n = 0
    for (rid, field), v in ov.items():
        rec = res.get(rid)
        if not rec:
            continue
        a = rec["response"]["answers"].get(field)
        if a is not None and a.get("type") == "noul":
            a["noul"] = v
            n += 1
    return n
