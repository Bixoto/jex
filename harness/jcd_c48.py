#!/usr/bin/env python3
"""Catalog-48 coherence projection (JCD) for the production fold.

Implements Kotawala's *Joint-Coherent Decoding* post-process for ForecastBench
(DI catalog 48), CPU-only and label-free:

* build the conjunction relation graph from each joint row's
  ``state.component_events`` (matched to standalone c48 rows by exact
  ``(round, question)``), and
* project the elicited Bernoulli probabilities onto the coherent polytope with an
  L2 minimum-distance projection: Frechet conjunction box clauses solved with
  Boyle-Dykstra per connected clique (marginals are free variables, as the
  ``argmin ||p_hat - r||`` definition requires).

The production order is full-cal (temperature) **then** JCD.  ``apply_jcd``
therefore projects whatever probabilities the records already carry; pass
``temperature_by_options`` only for standalone use where calibration has not run.

Gold is never read.  ``apply_jcd(rows, records)`` mutates kit-schema ``records``
in place (c48 only) and returns a stats dict.
"""

from __future__ import annotations

import collections
import json
import math
import time

CATALOG_ID = 48
TWO_PI = 2.0 * math.pi  # unused placeholder kept out of the hot path


# --------------------------------------------------------------------- helpers
def _catalog_id(row):
    ev = row.get("_evaluation") or {}
    cid = ev.get("catalog_id")
    if cid is None:
        cid = row.get("catalog_id")
    return cid


def _run_id(row):
    ev = row.get("_evaluation") or {}
    return ev.get("run_id") or row.get("id")


def _round(row):
    return (row.get("metadata") or {}).get("round")


def _is_joint(row):
    return (row.get("metadata") or {}).get("direction") is not None


def _question(row):
    return (row.get("state") or {}).get("event", {}).get("question", "")


def _components(row):
    comps = (row.get("state") or {}).get("component_events") or []
    out = []
    for c in comps:
        q = (c.get("event") or {}).get("question", "")
        out.append((q, c.get("required_outcome") == "yes"))
    return out


def _yes_prob(record):
    """First binary 'yes' probability in a kit-schema record, else None."""
    answers = (record.get("response") or {}).get("answers") or {}
    for ans in answers.values():
        if not isinstance(ans, dict):
            continue
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and "yes" in probs:
            try:
                return float(probs["yes"])
            except (TypeError, ValueError):
                return None
    return None


def _set_yes(record, value):
    answers = (record.get("response") or {}).get("answers") or {}
    for ans in answers.values():
        if not isinstance(ans, dict):
            continue
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and "yes" in probs:
            probs["yes"] = value
            if "no" in probs:
                probs["no"] = 1.0 - value
            return True
    return False


def _sharpen_yes(p, temperature):
    p = min(max(p, 1e-300), 1 - 1e-16)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) / temperature)))


def _emitted_temperature(n, tbyo):
    a, b, lo = tbyo
    if n <= 1:
        return max(lo, a)
    return max(lo, a + b * math.log(n))


# ------------------------------------------------------- structure + projection
def _build_structure(rows, rec_by_rid):
    """Return (by_q, resolved, skipped, comp_total, comp_matched).

    by_q: (round, question) -> [run_id, ...] for standalone c48 rows that have a
    scorable record.  resolved: [(joint_run_id, [(latent_key, required_bool)])].
    """
    by_q = collections.defaultdict(list)
    joint_rows = []
    for row in rows:
        if _catalog_id(row) != CATALOG_ID:
            continue
        rid = _run_id(row)
        if rid not in rec_by_rid:
            continue
        if _is_joint(row):
            joint_rows.append((rid, row))
        else:
            by_q[(_round(row), _question(row))].append(rid)

    resolved, skipped = [], []
    comp_total = comp_matched = 0
    for rid, row in joint_rows:
        rnd = _round(row)
        comps, ok = [], True
        for q, req in _components(row):
            key = (rnd, q)
            comp_total += 1
            members = by_q.get(key) or []
            if members:
                comp_matched += 1
                comps.append((key, req))
            else:
                ok = False
        if ok:
            resolved.append((rid, comps))
        else:
            skipped.append(rid)
    return by_q, resolved, skipped, comp_total, comp_matched


def _build_constraints(resolved):
    var_key, order = {}, []

    def ensure(kind, key):
        k = (kind, key)
        if k not in var_key:
            var_key[k] = len(order)
            order.append(k)
        return var_key[k]

    comp_sets = []
    for rid, comps in resolved:
        lat = [(ensure("latent", key), req) for key, req in comps]
        J = ensure("joint", rid)
        comp_sets.append({"joint": J, "comps": lat, "k": len(lat),
                          "n_no": sum(1 for _, req in lat if not req), "rid": rid})
    return order, comp_sets


def _components_union(n, comp_sets):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for cs in comp_sets:
        rj = find(cs["joint"])
        for i, _ in cs["comps"]:
            ri = find(i)
            if ri != rj:
                parent[ri] = rj
    groups = collections.defaultdict(list)
    for cs in comp_sets:
        groups[find(cs["joint"])].append(cs)
    root_vars = collections.defaultdict(set)
    for v in range(n):
        root_vars[find(v)].add(v)
    return groups, root_vars


def _project_poly(order, comp_sets, x0, maxit=3000, tol=1e-10):
    groups, root_vars = _components_union(len(order), comp_sets)
    x = list(x0)
    for root, css in groups.items():
        idx = sorted(root_vars[root])
        pos = {v: p for p, v in enumerate(idx)}
        nv = len(idx)
        cons = []
        for cs in css:
            J = pos[cs["joint"]]
            for i, req in cs["comps"]:
                ii = pos[i]
                if req:
                    cons.append(((J, ii), (1.0, -1.0), 0.0, 2.0))
                else:
                    cons.append(((J, ii), (1.0, 1.0), 1.0, 2.0))
            inds = [J]
            coefs = [-1.0]
            for i, req in cs["comps"]:
                inds.append(pos[i])
                coefs.append(1.0 if req else -1.0)
            b = (cs["k"] - 1) - cs["n_no"]
            norm = sum(c * c for c in coefs)
            cons.append((tuple(inds), tuple(coefs), b, norm))
        sub = [x0[v] for v in idx]
        nsets = len(cons)
        z = [[0.0] * nv for _ in range(nsets)]
        zb = [0.0] * nv
        for _ in range(maxit):
            maxchg = 0.0
            for ci, (inds, coefs, b, norm) in enumerate(cons):
                zc = z[ci]
                viol = -b
                for t, i in enumerate(inds):
                    viol += coefs[t] * (sub[i] + zc[i])
                f = viol / norm if viol > 0 else 0.0
                for t, i in enumerate(inds):
                    yi = sub[i] + zc[i]
                    xn = yi - f * coefs[t]
                    d = abs(xn - sub[i])
                    if d > maxchg:
                        maxchg = d
                    zc[i] = yi - xn
                    sub[i] = xn
            for i in range(nv):
                yi = sub[i] + zb[i]
                xn = 1.0 if yi > 1.0 else (0.0 if yi < 0.0 else yi)
                d = abs(xn - sub[i])
                if d > maxchg:
                    maxchg = d
                zb[i] = yi - xn
                sub[i] = xn
            if maxchg < tol:
                break
        for p, v in enumerate(idx):
            x[v] = sub[p]
    return x


# -------------------------------------------------------------------- public
def apply_jcd(rows, records, temperature_by_options=None):
    """Project catalog-48 probabilities in place.  Returns a stats dict.

    ``rows``    -- kit request rows (as prepared for the run).
    ``records`` -- kit result records (mutated in place, c48 answers only).
    ``temperature_by_options`` -- optional ``(a, b, min)``; when given, the
    full-cal transform is applied to c48 records before projection (standalone
    use only; the production pipeline already calibrated).
    """
    t0 = time.perf_counter()
    records = [r for r in records if isinstance(r, dict)]
    rec_by_rid = {}
    for r in records:
        rid = r.get("run_id")
        if rid is not None and rid not in rec_by_rid:
            rec_by_rid[rid] = r

    by_q, resolved, skipped, comp_total, comp_matched = _build_structure(rows, rec_by_rid)

    stats = {
        "catalog_id": CATALOG_ID,
        "c48_records": sum(1 for r in records if r.get("catalog_id") == CATALOG_ID),
        "joint_rows": len(resolved) + sum(1 for _ in skipped),
        "joints_constrained": len(resolved),
        "joints_skipped": len(skipped),
        "components_total": comp_total,
        "components_matched": comp_matched,
        "records_changed": 0,
        "max_abs_change": 0.0,
        "temperature_applied": False,
    }
    if not resolved:
        stats["latent_vars"] = 0
        stats["joint_vars"] = 0
        stats["linked_records"] = 0
        stats["elapsed_s"] = time.perf_counter() - t0
        return stats

    if temperature_by_options:
        for rid in rec_by_rid:
            rec = rec_by_rid[rid]
            if rec.get("catalog_id") != CATALOG_ID:
                continue
            p = _yes_prob(rec)
            if p is None:
                continue
            answers = (rec.get("response") or {}).get("answers") or {}
            for ans in answers.values():
                if isinstance(ans, dict) and isinstance(ans.get("probabilities"), dict):
                    n = len(ans["probabilities"])
                    _set_yes(rec, _sharpen_yes(p, _emitted_temperature(n, tuple(temperature_by_options))))
                    break
        stats["temperature_applied"] = True

    order, comp_sets = _build_constraints(resolved)

    def cur(rid):
        return _yes_prob(rec_by_rid[rid])

    x0 = []
    for kind, key in order:
        if kind == "joint":
            x0.append(cur(key))
        else:
            members = by_q[key]
            vals = [cur(m) for m in members]
            x0.append(sum(vals) / len(vals))

    proj = _project_poly(order, comp_sets, x0)

    # map projected values back onto records (latents shared across duplicates)
    prev = {}
    changed = 0
    maxd = 0.0
    for p, (kind, key) in enumerate(order):
        rids = [key] if kind == "joint" else list(by_q[key])
        for rid in rids:
            old = cur(rid)
            if old is None:
                continue
            prev[rid] = old
            if abs(proj[p] - old) > 1e-12:
                changed += 1
                maxd = max(maxd, abs(proj[p] - old))
            _set_yes(rec_by_rid[rid], proj[p])

    stats.update({
        "latent_vars": sum(1 for k, _ in order if k == "latent"),
        "joint_vars": sum(1 for k, _ in order if k == "joint"),
        "linked_records": len(prev),
        "records_changed": changed,
        "max_abs_change": maxd,
        "method": "JCD L2 coherence projection (Frechet conjunction box, Boyle-Dykstra)",
    })
    stats["elapsed_s"] = time.perf_counter() - t0
    return stats


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import gzip

    def _load(path):
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    import sys
    rows = _load(sys.argv[1] if len(sys.argv) > 1 else "subsets/full_edition.jsonl.gz")
    recs = _load(sys.argv[2] if len(sys.argv) > 2 else "runs/full-rules/results.jsonl.gz")
    print(json.dumps(apply_jcd(rows, recs), indent=2))
