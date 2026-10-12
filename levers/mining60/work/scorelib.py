#!/usr/bin/env python3
"""scorelib: exact per-catalog metric on the v1.x capture + deterministic rule application.

Metric path mirrors kit03 index02.benchmark_value:
  * track_scored catalogs -> index.static_score -> index.headline (cat40 track A)
  * 'added' catalogs      -> scoring.added.report (56,57,58,59,61,62,64)
  * everything else       -> report.benchmark_summary native score (metrics override for 38)

Rule path mirrors engine-v1x: recover the working (pre-TBO) distribution from the
emitted probabilities + emitted_temperature, then
  choice = P  if logit(p_work[P]) >= tau  else argmax over non-P keys.
"""
import gzip, json, math, sys, os, collections
from pathlib import Path

KIT = os.environ.get("KIT", "/tmp/opencode/submission03/kit03")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from decision_index.suite.io import Suite
from decision_index.scoring import index as I
from decision_index.scoring import report as R
from decision_index.scoring import added as A
from decision_index.scoring import index02

CAPTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "inputs", "wave1-v1x-results.jsonl.gz")
SUITE = os.environ.get("SUITE", "/tmp/opencode/submission03/suite03")


def sig_of(keys):
    return "choice|" + ",".join(sorted({str(k) for k in keys}))


def logit(p):
    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return math.log(p) - math.log1p(-p)


def inv_tbo(emitted, keys, T):
    T = float(T) if T else 1.0
    w = {k: (max(float(emitted.get(k, 0.0)), 0.0)) ** T for k in keys}
    s = sum(w.values())
    return {k: (v / s if s > 0 else v) for k, v in w.items()}


def rule_choice(work, keys, P, tau):
    if logit(work.get(P, 0.0)) >= float(tau):
        return P
    others = [k for k in keys if k != P]
    if not others:
        return P
    return max(others, key=lambda k: (work.get(k, 0.0), str(k)))


class Lib:
    def __init__(self, capture=CAPTURE, suite_dir=SUITE):
        self.suite = Suite(suite_dir, "0.3")
        self.rows = list(self.suite.rows(apply_exclusions=True))
        self.s = index02.spec("0.3")
        self.baselines = I.chance_baselines()
        self.metrics = {int(n): (m["name"], m["key"]) for n, m in self.s.get("metrics", {}).items()}
        self.bycat = collections.defaultdict(list)
        for r in self.rows:
            self.bycat[r["_evaluation"]["catalog_id"]].append(r)
        # capture: compact records + per-question emitted_temperature
        self.results = {}
        self.T = {}
        with gzip.open(capture, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["status"] != "ok":
                    continue
                rid = r["run_id"]
                self.results[rid] = {"run_id": rid, "catalog_id": r["catalog_id"],
                                     "status": "ok", "response": r["response"],
                                     "total_wall_ms": r.get("total_wall_ms")}
                ro = r.get("raw_output") or {}
                for qk, v in ro.items():
                    self.T[(rid, qk)] = v.get("emitted_temperature")

    # ---- metric ----
    def _res_sub(self, rows, results):
        out = {}
        for r in rows:
            rec = results.get(r["_evaluation"]["run_id"])
            if rec is None:
                continue
            out[r["_evaluation"]["run_id"]] = {
                "status": rec.get("status"),
                "answers": (rec.get("response") or {}).get("answers", {}),
            }
        return out

    def metric_raw(self, n, rows=None, results=None):
        rows = self.bycat[n] if rows is None else rows
        results = self.results if results is None else results
        if n in self.s["track_scored"]:
            res = self._res_sub(rows, results)
            sc = I.static_score(n, rows, res, self.baselines)
            return I.headline(sc)["raw"]
        if str(n) in self.s["added"]:
            return A.report(n, rows, results)["score"]
        summ = R.benchmark_summary(None, results, "x", rows=rows, metrics=self.metrics)
        return summ["benchmarks"][0]["score"]

    def metric_skill(self, n, rows=None, results=None):
        rows = self.bycat[n] if rows is None else rows
        results = self.results if results is None else results
        if n in self.s["track_scored"]:
            res = self._res_sub(rows, results)
            sc = I.static_score(n, rows, res, self.baselines)
            return I.headline(sc)["skill"]
        # native path: value = display_value / added_value; reconstruct skill vs chance
        raw = self.metric_raw(n, rows, results)
        ch = index02.chance_of(n, self.s)
        return min(1.0, max(0.0, (raw - ch) / (1 - ch))) if ch < 1 else raw

    # ---- rule evaluation ----
    def apply_sub(self, n, P, tau, scope=None):
        """Results sub-dict for catalog n only, with the rule applied."""
        out = {}
        for r in self.bycat[n]:
            rid = r["_evaluation"]["run_id"]
            rec = self.results.get(rid)
            if rec is None:
                continue
            newans = None
            for key, q in r["questions"].items():
                if q.get("type") != "choice":
                    continue
                if scope is not None and not scope(q):
                    continue
                keys = list((q.get("criteria") or {}).keys())
                if P not in keys:
                    continue
                a = rec["response"]["answers"].get(key)
                if a is None:
                    continue
                T = self.T.get((rid, key))
                work = inv_tbo(a.get("probabilities", {}), keys, T)
                nc = rule_choice(work, keys, P, tau)
                if newans is None:
                    newans = {k: dict(v) for k, v in rec["response"]["answers"].items()}
                newans[key] = dict(a)
                newans[key]["choice"] = nc
            if newans is not None:
                nr = dict(rec)
                nr["response"] = dict(rec["response"])
                nr["response"]["answers"] = newans
                out[rid] = nr
            else:
                out[rid] = rec
        return out

    def apply(self, n, P, tau, scope=None):
        """Full results dict copy with the rule applied to catalog n."""
        res2 = dict(self.results)
        res2.update(self.apply_sub(n, P, tau, scope))
        return res2


def main():
    lib = Lib()
    for n in (62, 61, 33, 25, 21):
        print(n, "base raw", round(lib.metric_raw(n), 5), "skill", round(lib.metric_skill(n), 5))


if __name__ == "__main__":
    main()
