#!/usr/bin/env python3
"""In-process scoring harness: apply bundle folds to the banked logits and
score with kit03's own score_run_v02, reusing one cached Suite (no large
intermediate results files, no re-parsing the suite per variant)."""
import gzip, json, math, pickle, sys, os, collections, hashlib, tempfile, shutil

KIT = "/tmp/opencode/submission03/kit03"
sys.path.insert(0, KIT)

from decision_index.suite.io import Suite
from decision_index.pipeline import score_run_v02

HERE = "/tmp/opencode/submission04/v1.xplus/levers/bundle60"
TABLE = f"{HERE}/table.pkl"
A1 = f"{HERE}/../fold60/results-folded.jsonl.gz"
SUITE = "/tmp/opencode/submission03/suite03"


def fold_of(gid, nf=5):
    return hashlib.md5(gid.encode()).digest()[0] % nf


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def raw_argmax(keys, vals):
    return keys[max(range(len(vals)), key=lambda i: vals[i])]


class FakeSuite:
    def __init__(self, edition, rows):
        self.edition = edition
        self._rows = rows

    def rows(self, apply_exclusions=True):
        return self._rows

    def verify(self, strict=True):
        return {"match": True}

    @property
    def in_edition(self):
        return lambda e: True


def load_suite_rows():
    suite = Suite(SUITE, "0.3")
    rows = list(suite.rows(apply_exclusions=True))
    return suite.edition, rows


def build_results(recs, transform):
    """transform(rec) -> new choice-prob list (choice) or new noul scalar."""
    results = {}
    for r in recs:
        cid, gid, rid, field, qt, keys, vals, noul, gold, st, winner = r
        if qt == "noul":
            v2 = transform(r)
            ans = {"type": "noul", "noul": v2}
        else:
            p2 = transform(r)
            ra = raw_argmax(keys, list(vals))
            na = raw_argmax(keys, p2)
            choice = na if (winner == ra or winner is None) else winner
            ans = {"type": "choice", "choice": choice,
                   "probabilities": {keys[i]: p2[i] for i in range(len(keys))}}
        slot = results.get(rid)
        if slot is None:
            slot = {"run_id": rid, "catalog_id": cid, "status": "ok",
                    "response": {"answers": {}}, "total_wall_ms": 0.0}
            results[rid] = slot
        slot["response"]["answers"][field] = ans
    return results


def apply_a1(results, a1):
    """Overlay Lane A1's cat62 choices (rules take precedence)."""
    for (rid, field), ch in a1.items():
        if ch is None or rid not in results:
            continue
        ans = results[rid]["response"]["answers"].get(field)
        if ans is not None and ans.get("type") == "choice":
            ans["choice"] = ch
    return results


def load_a1():
    out = {}
    try:
        with gzip.open(A1, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r.get("catalog_id") != 62:
                    continue
                for k, a in (r.get("response") or {}).get("answers", {}).items():
                    out[(r["run_id"], k)] = a.get("choice")
    except FileNotFoundError:
        return None
    return out


def score(results, edition, rows, out):
    from pathlib import Path
    return score_run_v02(FakeSuite(edition, rows), results, "bundle60", Path(out))
