#!/usr/bin/env python3
"""Base harness: load suite03 + v1.x capture, rescore in-memory exactly like kit03.

score_run_v02(suite, results, engine, out) takes a results dict, so we can
perturb choices in memory and rescore without rewriting the 90MB capture.
"""
import gzip, json, os, sys, time, copy
from pathlib import Path

KIT = "/tmp/opencode/submission03/kit03"
SUITE = "/tmp/opencode/submission03/suite03"
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from decision_index.suite.io import Suite            # noqa: E402
from decision_index.pipeline import score_run_v02    # noqa: E402
from decision_index.scoring.report import compact    # noqa: E402

CAPTURE = "/tmp/opencode/submission04/v1.xplus/levers/inputs/wave1-v1x-results.jsonl.gz"


class CachedSuite:
    """Wrap a Suite so .rows() is served from an in-memory list (fast rescoring)."""
    def __init__(self, suite):
        self.edition = suite.edition
        self._rows = list(suite.rows(apply_exclusions=True))

    def rows(self, apply_exclusions=False):
        return iter(self._rows)

    @property
    def n_rows(self):
        return len(self._rows)


def load_results(path=CAPTURE):
    out = {}
    t0 = time.time()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["run_id"]] = compact(r)
    return out


def compute(suite, results, outdir, engine="mine", keep_files=False):
    out = Path(outdir)
    if not keep_files:
        os.makedirs(out, exist_ok=True)
    t0 = time.time()
    scores = score_run_v02(suite, results, engine, out)
    return scores, time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    suite = CachedSuite(Suite(SUITE, "0.3"))
    print(f"suite rows: {suite.n_rows}  ({time.time()-t0:.1f}s)")
    results = load_results()
    print(f"results: {len(results)}  ({time.time()-t0:.1f}s)")
    scores, dt = compute(suite, results, "/tmp/opencode/submission04/v1.xplus/levers/mining60/work/_base_out")
    print(f"score time {dt:.1f}s")
    print("index", scores["decision_index"], "scores", scores["scores"])
