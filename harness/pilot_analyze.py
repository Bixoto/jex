#!/usr/bin/env python3
"""Post-pilot analysis for the reasoning pilot (REASON_PILOT.md §4).

Stdlib-only and importable without GPU/kit: reuses ``harness/analysis.py``'s
paired cluster bootstrap when present, else a seeded percentile fallback.

Policy definition (§4, primary endpoint)
----------------------------------------
For an arm, ``policy = arm.post_choice`` on the questions the arm gated, and the
reference (B0) choice everywhere else. Questions are paired by
``(run_id, question_key)``. Per-benchmark chance-corrected skill is aggregated
with the kit's own area weights + gold-star within-area weights, restricted to
the panel catalogs. Two numbers are reported:

* **panel-scale delta** -- the weighted skill delta renormalised over the panel
  (i.e. "as if the panel were the whole index"), in points;
* **full-index delta** -- the same weighted sum using each panel benchmark's
  share of the FULL index, in points. ``full = panel_scale * panel_share`` where
  ``panel_share`` (verified ~0.153) is the six catalogs' fraction of the index.

The pre-registered rule is applied to the full-index delta: ADOPT if
``delta >= +0.20`` and CI lower > 0; INCONCLUSIVE if CI upper >= +0.20; else
REFUSE.

Leakage (REASON_PILOT §3)
-------------------------
A rationale leaks if it contains an ``answer: X`` / ``the answer is`` pattern or
a verbatim option description of at least 40 characters (option text comes from
``--rows``). The report gives the leak rate per benchmark and the full-index
policy delta recomputed with leaking gated questions reverted to B0 (the
leak-excluded sensitivity). Records written before ``rationale_text`` existed are
reported ``leak_classifier: unavailable`` rather than silently skipped.

Usage::

    python3 harness/pilot_analyze.py \
        --b0 subsets/.../reference_panel_results.jsonl.gz \
        --rows subsets/.../panel.jsonl.gz \
        --scores runs/full-rules/scores.json \
        --arms "b1_cov0.05=/tmp/runs/reason-pilot-b1-c05.jsonl \
                b1_cov0.10=/tmp/runs/reason-pilot-b1-c10.jsonl \
                b2_cov0.10=/tmp/runs/reason-pilot-b2-c10.jsonl \
                w=/tmp/runs/reason-pilot-w.jsonl" \
        --out /tmp/reason-pilot-analysis.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import shlex
import statistics
import sys

PANEL_CATALOGS = (25, 43, 32, 44, 30, 58)
EXPECTED_PANEL_SHARE = 0.153
INDEX_GATE = 0.20
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 20260929


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def open_jsonl(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_jsonl(path):
    with open_jsonl(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def catalog_id(record):
    evaluation = record.get("_evaluation")
    if isinstance(evaluation, dict) and evaluation.get("catalog_id") is not None:
        return int(evaluation["catalog_id"])
    if record.get("catalog_id") is not None:
        return int(record["catalog_id"])
    return None


# ---------------------------------------------------------------------------
# Weights (mirrors decision_index.scoring.index02; pure and testable)
# ---------------------------------------------------------------------------


def benchmark_weights(spec):
    """Full-index weight per benchmark from the kit's own rule.

    ``area_weight * gold-star-within-area weight``; area weights use the kit's
    sqrt sizing with ``area_sizing.fixed`` (arts = 0.10) and are renormalised to
    sum to 1 exactly as ``index02.aggregate`` does.
    """
    areas = spec["areas"]
    gold = spec.get("gold") or {}
    bench = {}
    for area in areas:
        ids = list(area["benchmarks"])
        raw = {n: float(gold.get(str(n), 1.0)) for n in ids}
        total = sum(raw.values()) or 1.0
        for n in ids:
            bench[n] = raw[n] / total
    sizing = spec.get("area_sizing")
    if not sizing:
        area_w = {a["id"]: 1.0 / len(areas) for a in areas}
    else:
        fixed = dict(sizing.get("fixed") or {})
        size = {a["id"]: len(a["benchmarks"]) ** 0.5 for a in areas if a["id"] not in fixed}
        rest = 1.0 - sum(fixed.values())
        total = sum(size.values()) or 1.0
        area_w = {
            a["id"]: (fixed[a["id"]] if a["id"] in fixed else rest * size[a["id"]] / total)
            for a in areas
        }
    norm = sum(area_w.values()) or 1.0
    area_w = {k: v / norm for k, v in area_w.items()}
    full = {n: area_w[a["id"]] * bench[n] for a in areas for n in a["benchmarks"]}
    return {
        "area_weights": area_w,
        "bench_weights": bench,
        "full_weights": full,
    }


def panel_weighting(weights, panel_catalogs=PANEL_CATALOGS):
    """``(panel_share, panel_norm)`` over the panel catalogs present in weights."""
    full = weights["full_weights"]
    present = [n for n in panel_catalogs if n in full]
    share = sum(full[n] for n in present)
    norm = {n: (full[n] / share if share else 0.0) for n in present}
    return share, norm


def chance_table(spec, scores=None):
    """catalog_id -> chance value.

    The kit spec is authoritative (``index02.chance_of`` reads it), and it can
    differ from a run's recorded ``scores.json`` (e.g. GSM8K catalog 30 is 0.175
    in ``index-0.2.1.json`` but 0.25 in ``runs/full-rules/scores.json``). The
    scores fallback is used only for benchmarks the spec does not carry.
    """
    out = {}
    for n, entry in (spec.get("chance") or {}).items():
        if isinstance(entry, dict) and "chance" in entry:
            out[int(n)] = float(entry["chance"])
    for n, entry in ((scores or {}).get("index_benchmarks") or {}).items():
        key = int(n)
        if key not in out and isinstance(entry, dict) and entry.get("random") is not None:
            out[key] = float(entry["random"])
    return out


def partial_spec_from_scores(scores):
    """A best-effort spec when the kit is unavailable (gold table missing).

    Area sizing is the kit's fixed-arts rule; the gold-star table is unavailable
    from scores.json, so within-area weights are uniform and the panel share is
    approximate. Flagged ``_partial`` so the report can say so.
    """
    areas = [
        {"id": a["id"], "label": a.get("label"), "benchmarks": list(a["benchmarks"])}
        for a in scores.get("areas", [])
    ]
    return {
        "areas": areas,
        "area_sizing": {"rule": "sqrt", "fixed": {"arts": 0.1}},
        "gold": {},
        "chance": {
            str(n): {"chance": e["random"]}
            for n, e in (scores.get("index_benchmarks") or {}).items()
            if isinstance(e, dict) and e.get("random") is not None
        },
        "_partial": True,
    }


def load_index_spec(scores, index_spec_path=None, kit_root=None, edition=None):
    """Load the kit's index spec: explicit file, then the kit, then scores.json."""
    if index_spec_path:
        with open(index_spec_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    root = kit_root or os.environ.get("DI_KIT_ROOT")
    if not root:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit")
    try:
        if root and os.path.isdir(root) and root not in sys.path:
            sys.path.insert(0, root)
        from decision_index.scoring import index02  # noqa: E402

        return index02.spec(edition or scores.get("edition") or "0.2.1")
    except Exception:
        return partial_spec_from_scores(scores)


# ---------------------------------------------------------------------------
# Loading rows / results / arms
# ---------------------------------------------------------------------------


def option_descriptions(question):
    """The non-empty option description strings of a choice question.

    These feed the leakage classifier's "verbatim option description" test. A
    non-string description is JSON-encoded the way the renderer does, so a
    structured option is still comparable.
    """
    if not isinstance(question, dict) or question.get("type") != "choice":
        return []
    out = []
    for value in (question.get("criteria") or {}).values():
        if isinstance(value, str):
            text = value
        elif value is None:
            continue
        else:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if text.strip():
            out.append(text)
    return out


def load_rows(path):
    """``{(run_id, question_key): {"catalog_id", "gold", "options"}}`` from rows."""
    out = {}
    for row in iter_jsonl(path):
        evaluation = row.get("_evaluation") or {}
        run_id = evaluation.get("run_id") or row.get("id")
        cid = catalog_id(row)
        expected = row.get("expected") or {}
        questions = row.get("questions") or {}
        for question_key, gold in expected.items():
            out[(run_id, question_key)] = {
                "catalog_id": cid,
                "gold": gold,
                "options": option_descriptions(questions.get(question_key)),
            }
    return out


def load_results(path):
    """``{run_id: {"catalog_id", "answers": {qk: choice}}}`` from results."""
    out = {}
    for record in iter_jsonl(path):
        if record.get("status") not in (None, "ok"):
            continue
        answers = (record.get("response") or {}).get("answers") or {}
        out[record.get("run_id")] = {
            "catalog_id": record.get("catalog_id"),
            "answers": {k: (v or {}).get("choice") for k, v in answers.items()
                        if isinstance(v, dict)},
        }
    return out


def load_arm(path):
    """``{(run_id, question_key): record}`` from a reason_pilot JSONL output."""
    out = {}
    for record in iter_jsonl(path):
        out[(record.get("run_id"), record.get("question_key"))] = record
    return out


def build_questions(rows, results, panel_catalogs=PANEL_CATALOGS):
    """Panel questions with gold and the B0 choice, in a stable order."""
    wanted = {int(c) for c in panel_catalogs}
    questions = []
    for (run_id, question_key), meta in rows.items():
        cid = meta["catalog_id"]
        if cid is None or int(cid) not in wanted:
            continue
        b0 = results.get(run_id) or {}
        questions.append(
            {
                "run_id": run_id,
                "question_key": question_key,
                "catalog_id": int(cid),
                "gold": meta["gold"],
                "options": meta.get("options") or [],
                "b0_choice": (b0.get("answers") or {}).get(question_key),
            }
        )
    questions.sort(key=lambda q: (int(q["catalog_id"]), str(q["run_id"]), str(q["question_key"])))
    return questions


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def policy_outcomes(questions, arm_map):
    """Per question: baseline/policy correctness and whether the arm gated it."""
    outcomes = []
    for question in questions:
        key = (question["run_id"], question["question_key"])
        arm = arm_map.get(key)
        gated = arm is not None
        policy_choice = arm.get("post_choice") if gated else question["b0_choice"]
        baseline_choice = question["b0_choice"]
        gold = question["gold"]
        outcomes.append(
            {
                "question": question,
                "gated": gated,
                "baseline_correct": (baseline_choice is not None and baseline_choice == gold),
                "policy_choice": policy_choice,
                "policy_correct": (policy_choice is not None and policy_choice == gold),
            }
        )
    return outcomes


def per_catalog_counts(outcomes):
    """Fix/flip/unchanged counts per catalog over all panel questions."""
    counts = {}
    for outcome in outcomes:
        cid = outcome["question"]["catalog_id"]
        entry = counts.setdefault(
            cid, {"n": 0, "gated": 0, "baseline_correct": 0, "policy_correct": 0,
                  "fixed": 0, "flipped": 0, "unchanged_right": 0, "unchanged_wrong": 0}
        )
        base, pol = outcome["baseline_correct"], outcome["policy_correct"]
        entry["n"] += 1
        entry["gated"] += int(outcome["gated"])
        entry["baseline_correct"] += int(base)
        entry["policy_correct"] += int(pol)
        if not base and pol:
            entry["fixed"] += 1
        elif base and not pol:
            entry["flipped"] += 1
        elif base and pol:
            entry["unchanged_right"] += 1
        else:
            entry["unchanged_wrong"] += 1
    for cid, entry in counts.items():
        denom = entry["n"]
        entry["baseline_accuracy"] = (entry["baseline_correct"] / denom) if denom else None
        entry["policy_accuracy"] = (entry["policy_correct"] / denom) if denom else None
    return counts


def clip01(value):
    return min(1.0, max(0.0, value))


def skill(correct, requests, chance):
    if not requests:
        return None
    raw = correct / requests
    if chance >= 1.0:
        return raw
    return clip01((raw - chance) / (1.0 - chance))


def evaluate_arm(questions, arm_map, weights, chances, panel_catalogs=PANEL_CATALOGS):
    """Panel-scale and full-index deltas + per-catalog counts for one arm."""
    outcomes = policy_outcomes(questions, arm_map)
    counts = per_catalog_counts(outcomes)
    share, panel_norm = panel_weighting(weights, panel_catalogs)
    full = weights["full_weights"]

    per_catalog = {}
    panel_scale_delta = 0.0
    full_delta = 0.0
    for cid, entry in sorted(counts.items()):
        chance = chances.get(cid, 0.0)
        base_skill = skill(entry["baseline_correct"], entry["n"], chance)
        pol_skill = skill(entry["policy_correct"], entry["n"], chance)
        delta = (pol_skill - base_skill) if (base_skill is not None and pol_skill is not None) else 0.0
        weight = panel_norm.get(cid, 0.0)
        per_catalog[cid] = {
            "n": entry["n"],
            "gated": entry["gated"],
            "baseline_accuracy": entry["baseline_accuracy"],
            "policy_accuracy": entry["policy_accuracy"],
            "baseline_skill": base_skill,
            "policy_skill": pol_skill,
            "delta_skill": delta,
            "panel_weight": weight,
            "full_weight": full.get(cid, 0.0),
        }
        panel_scale_delta += weight * delta
        full_delta += full.get(cid, 0.0) * delta

    gated = [o for o in outcomes if o["gated"]]
    fixed = sum(1 for o in gated if not o["baseline_correct"] and o["policy_correct"])
    flipped = sum(1 for o in gated if o["baseline_correct"] and not o["policy_correct"])
    baseline_wrong = sum(1 for o in gated if not o["baseline_correct"])
    return {
        "questions": len(outcomes),
        "gated": len(gated),
        "panel_scale_delta": 100.0 * panel_scale_delta,
        "full_index_delta": 100.0 * full_delta,
        "panel_share": share,
        "counts": {
            "fixed": fixed,
            "flipped": flipped,
            "baseline_wrong_gated": baseline_wrong,
            "fix_rate": (fixed / baseline_wrong) if baseline_wrong else None,
            "unchanged": len(gated) - fixed - flipped,
        },
        "per_catalog": per_catalog,
        "outcomes": outcomes,
    }


# ---------------------------------------------------------------------------
# Bootstrap + rule
# ---------------------------------------------------------------------------


def _percentile(sorted_values, p):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * p) - 1))
    return sorted_values[idx]


def cluster_bootstrap_delta(clusters, base_stat, arm_stat, n_boot=DEFAULT_N_BOOT,
                            seed=DEFAULT_SEED, alpha=0.05):
    """Seeded paired percentile bootstrap clustered by question id.

    Uses ``harness/analysis.py::cluster_bootstrap_delta`` when importable (same
    algorithm, same return shape); otherwise a local fallback so the tool never
    depends on a sibling module.
    """
    try:
        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import analysis  # noqa: E402

        return analysis.cluster_bootstrap_delta(
            clusters, base_stat, arm_stat, n_boot=n_boot, seed=seed, alpha=alpha
        )
    except Exception:
        return fallback_bootstrap_delta(
            clusters, base_stat, arm_stat, n_boot=n_boot, seed=seed, alpha=alpha
        )


def fallback_bootstrap_delta(clusters, base_stat, arm_stat, n_boot=DEFAULT_N_BOOT,
                             seed=DEFAULT_SEED, alpha=0.05):
    """Local deterministic fallback (see :func:`cluster_bootstrap_delta`)."""
    keys = sorted(clusters)
    all_idx = [i for k in keys for i in clusters[k]]
    obs_base, obs_arm = base_stat(all_idx), arm_stat(all_idx)
    observed = None if (obs_base is None or obs_arm is None) else obs_arm - obs_base
    rng = random.Random(seed)
    deltas = []
    for _ in range(int(n_boot)):
        idx = []
        for _ in range(len(keys)):
            idx.extend(clusters[rng.choice(keys)])
        base, arm = base_stat(idx), arm_stat(idx)
        if base is None or arm is None:
            continue
        deltas.append(arm - base)
    deltas.sort()
    lo = _percentile(deltas, alpha / 2)
    hi = _percentile(deltas, 1 - alpha / 2)
    gt0 = sum(1 for d in deltas if d > 0)
    lt0 = sum(1 for d in deltas if d < 0)
    total_nonzero = gt0 + lt0
    return {
        "observed_delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_boot": len(deltas),
        "n_clusters": len(keys),
        "excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
        "boot_p": (2.0 * min(gt0, lt0) / total_nonzero) if total_nonzero else 1.0,
        "boot_gt0": gt0,
        "boot_lt0": lt0,
        "boot_eq0": len(deltas) - gt0 - lt0,
    }


def classify(delta, ci_low, ci_high, gate=INDEX_GATE):
    """REASON_PILOT §4 pre-registered rule on the full-index delta."""
    if delta is None or ci_low is None or ci_high is None:
        return "UNDEFINED"
    if delta >= gate and ci_low > 0:
        return "ADOPT"
    if ci_high >= gate:
        return "INCONCLUSIVE"
    return "REFUSE"


def bootstrap_arm(questions, arm_map, weights, chances,
                  n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, alpha=0.05,
                  panel_catalogs=PANEL_CATALOGS):
    """Paired cluster bootstrap of the full-index policy delta."""
    outcomes = policy_outcomes(questions, arm_map)
    share, panel_norm = panel_weighting(weights, panel_catalogs)
    chances_local = dict(chances)

    def index_stat(correct_flags, indices):
        per = {}
        for i in indices:
            cid = questions[i]["catalog_id"]
            entry = per.setdefault(cid, [0, 0])
            entry[0] += 1
            entry[1] += int(correct_flags[i])
        total = 0.0
        for cid, (n, correct) in per.items():
            value = skill(correct, n, chances_local.get(cid, 0.0))
            total += panel_norm.get(cid, 0.0) * (value or 0.0)
        return 100.0 * total if per else None

    base_flags = [o["baseline_correct"] for o in outcomes]
    policy_flags = [o["policy_correct"] for o in outcomes]
    clusters = {}
    for i, question in enumerate(questions):
        clusters.setdefault(question["run_id"], []).append(i)
    result = cluster_bootstrap_delta(
        clusters,
        lambda idx: index_stat(base_flags, idx),
        lambda idx: index_stat(policy_flags, idx),
        n_boot=n_boot, seed=seed, alpha=alpha,
    )
    result["panel_share"] = share
    result["full_index_delta"] = result.get("observed_delta")
    result["panel_scale_delta"] = (
        (result["observed_delta"] / share) if (result["observed_delta"] is not None and share) else None
    )
    result["decision"] = classify(
        result.get("observed_delta"), result.get("ci_low"), result.get("ci_high")
    )
    return result


# ---------------------------------------------------------------------------
# Truncation / rationale / leakage
# ---------------------------------------------------------------------------


def truncation_split(outcomes, arm_map):
    """Fix counts split by whether the arm's rationale was truncated at cap."""
    split = {
        "truncated": {"gated": 0, "fixed": 0, "flipped": 0, "baseline_wrong": 0},
        "untruncated": {"gated": 0, "fixed": 0, "flipped": 0, "baseline_wrong": 0},
    }
    for outcome in outcomes:
        if not outcome["gated"]:
            continue
        key = (outcome["question"]["run_id"], outcome["question"]["question_key"])
        record = arm_map.get(key) or {}
        bucket = split["truncated" if record.get("truncated") else "untruncated"]
        bucket["gated"] += 1
        if not outcome["baseline_correct"]:
            bucket["baseline_wrong"] += 1
        if not outcome["baseline_correct"] and outcome["policy_correct"]:
            bucket["fixed"] += 1
        elif outcome["baseline_correct"] and not outcome["policy_correct"]:
            bucket["flipped"] += 1
    for bucket in split.values():
        bucket["fix_rate"] = (
            bucket["fixed"] / bucket["baseline_wrong"] if bucket["baseline_wrong"] else None
        )
    return split


def rationale_stats(arm_map):
    tokens = [int(r.get("rationale_tokens") or 0) for r in arm_map.values()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return {"n": 0}
    tokens.sort()
    return {
        "n": len(tokens),
        "mean": statistics.fmean(tokens),
        "median": statistics.median(tokens),
        "p90": _percentile(tokens, 0.9),
        "max": tokens[-1],
    }


#: REASON_PILOT.md §3 leakage patterns (case-insensitive).
ANSWER_PATTERNS = (
    re.compile(r"answer\s*:\s*\S", re.IGNORECASE),
    re.compile(r"\bthe answer is\b", re.IGNORECASE),
)
#: A verbatim option description at least this long counts as a leak.
OPTION_DESCRIPTION_MIN = 40


def rationale_leaks(text, options, min_description=OPTION_DESCRIPTION_MIN):
    """True if the rationale leaks the answer (§3); None when unclassifiable.

    Leaks: an ``answer: X`` / ``the answer is`` pattern (case-insensitive), or a
    verbatim option description of at least ``min_description`` characters.
    ``None`` means the record carries no ``rationale_text`` (legacy output), so
    it cannot be classified either way.
    """
    if text is None:
        return None
    if not text:
        return False
    for pattern in ANSWER_PATTERNS:
        if pattern.search(text):
            return True
    lowered = text.lower()
    for description in options or ():
        if len(description) >= min_description and description.lower() in lowered:
            return True
    return False


def leak_analysis(questions, arm_map, weights, chances,
                  n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED, panel_catalogs=PANEL_CATALOGS):
    """Leak rate per benchmark + the policy delta restricted to clean rows.

    Gated records without ``rationale_text`` are ``unavailable``. If no gated
    record carries text at all, the whole classifier is reported
    ``leak_classifier: unavailable``. Otherwise leaking gated questions are
    reverted to the B0 choice (the policy no longer applies to them) and the
    full-index delta + CI are recomputed, which is the leak-excluded sensitivity.
    """
    question_by_key = {(q["run_id"], q["question_key"]): q for q in questions}
    per_catalog = {}
    leaking, clean, unknown = 0, 0, 0
    clean_map = {}
    for key, record in arm_map.items():
        question = question_by_key.get(key)
        if question is None:
            continue
        verdict = rationale_leaks(record.get("rationale_text"), question.get("options"))
        cid = question["catalog_id"]
        entry = per_catalog.setdefault(cid, {"gated": 0, "leaking": 0, "unknown": 0})
        entry["gated"] += 1
        if verdict is None:
            unknown += 1
            entry["unknown"] += 1
            clean_map[key] = record          # unclassified: keep in the clean set
        elif verdict:
            leaking += 1
            entry["leaking"] += 1
        else:
            clean += 1
            clean_map[key] = record

    classified = leaking + clean
    report = {
        "leak_classifier": "ok" if classified else "unavailable",
        "reason": None if classified else (
            "no gated record carries rationale_text; the leakage classifier "
            "cannot run on pre-rationale_text outputs"
        ),
        "pattern": {
            "answer_regex": [p.pattern for p in ANSWER_PATTERNS],
            "option_description_min_chars": OPTION_DESCRIPTION_MIN,
            "case_insensitive": True,
        },
        "gated": len(arm_map),
        "classified": classified,
        "leaking": leaking,
        "clean": clean,
        "unknown": unknown,
        "leak_rate": (leaking / classified) if classified else None,
        "per_catalog": {},
    }
    for cid, entry in sorted(per_catalog.items()):
        entry["leak_rate"] = (entry["leaking"] / entry["gated"]) if entry["gated"] else None
        report["per_catalog"][str(cid)] = entry

    if not classified:
        return report

    excluded = evaluate_arm(questions, clean_map, weights, chances, panel_catalogs)
    excluded_boot = bootstrap_arm(
        questions, clean_map, weights, chances,
        n_boot=n_boot, seed=seed, panel_catalogs=panel_catalogs,
    )
    report.update(
        {
            "excluded_full_index_delta": excluded["full_index_delta"],
            "excluded_panel_scale_delta": excluded["panel_scale_delta"],
            "excluded_ci_low": excluded_boot.get("ci_low"),
            "excluded_ci_high": excluded_boot.get("ci_high"),
            "excluded_decision": excluded_boot.get("decision"),
        }
    )
    return report


# ---------------------------------------------------------------------------
# B1 - B2 contrast
# ---------------------------------------------------------------------------


def coverage_of(name):
    match = re.search(r"cov([0-9]*\.?[0-9]+)", name or "")
    return float(match.group(1)) if match else None


def b1_minus_b2(arm_reports):
    """Match ``b1_covX``/``b2_covX`` arms and report the content contrast."""
    b1 = {coverage_of(name): name for name in arm_reports if name.startswith("b1")}
    b2 = {coverage_of(name): name for name in arm_reports if name.startswith("b2")}
    out = {}
    for cov in sorted(set(b1) & set(b2)):
        one, two = arm_reports[b1[cov]], arm_reports[b2[cov]]
        out[f"cov{cov:g}"] = {
            "b1": b1[cov],
            "b2": b2[cov],
            "full_index_delta_b1": one["full_index_delta"],
            "full_index_delta_b2": two["full_index_delta"],
            "content_delta": (
                (one["full_index_delta"] - two["full_index_delta"])
                if one["full_index_delta"] is not None and two["full_index_delta"] is not None
                else None
            ),
            "fixed_b1": one["counts"]["fixed"],
            "fixed_b2": two["counts"]["fixed"],
            "fix_rate_b1": one["counts"]["fix_rate"],
            "fix_rate_b2": two["counts"]["fix_rate"],
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_arms(spec):
    arms = []
    for token in shlex.split(spec or ""):
        if "=" not in token:
            raise SystemExit(f"--arms entry is not name=path: {token!r}")
        name, path = token.split("=", 1)
        arms.append((name, path))
    return arms


def analyse(b0_path, rows_path, arms, scores_path=None, index_spec_path=None,
            kit_root=None, panel_catalogs=PANEL_CATALOGS, n_boot=DEFAULT_N_BOOT,
            seed=DEFAULT_SEED, edition=None):
    rows = load_rows(rows_path)
    b0 = load_results(b0_path)
    scores = {}
    if scores_path:
        with open(scores_path, "r", encoding="utf-8") as handle:
            scores = json.load(handle)
    spec = load_index_spec(scores, index_spec_path, kit_root, edition)
    weights = benchmark_weights(spec)
    chances = chance_table(spec, scores)
    share, _norm = panel_weighting(weights, panel_catalogs)
    questions = build_questions(rows, b0, panel_catalogs)

    arm_reports = {}
    for name, path in arms:
        arm_map = load_arm(path)
        report = evaluate_arm(questions, arm_map, weights, chances, panel_catalogs)
        boot = bootstrap_arm(
            questions, arm_map, weights, chances,
            n_boot=n_boot, seed=seed, panel_catalogs=panel_catalogs,
        )
        report.update(
            {
                "ci_low": boot.get("ci_low"),
                "ci_high": boot.get("ci_high"),
                "excludes_zero": boot.get("excludes_zero"),
                "decision": boot.get("decision"),
                "boot_p": boot.get("boot_p"),
                "n_boot": boot.get("n_boot"),
                "n_clusters": boot.get("n_clusters"),
                "truncation_split": truncation_split(report["outcomes"], arm_map),
                "rationale": rationale_stats(arm_map),
                "leakage": leak_analysis(
                    questions, arm_map, weights, chances,
                    n_boot=n_boot, seed=seed, panel_catalogs=panel_catalogs,
                ),
                "coverage": coverage_of(name),
            }
        )
        report.pop("outcomes", None)
        arm_reports[name] = report

    return {
        "panel_catalogs": [int(c) for c in panel_catalogs],
        "panel_share": share,
        "expected_panel_share": EXPECTED_PANEL_SHARE,
        "panel_share_ok": abs(share - EXPECTED_PANEL_SHARE) <= 0.01,
        "gate_full_index": INDEX_GATE,
        "n_boot": n_boot,
        "seed": seed,
        "spec_partial": bool(spec.get("_partial")),
        "chance": {str(k): v for k, v in sorted(chances.items())},
        "weights": {
            "area_weights": weights["area_weights"],
            "full_weights": {str(k): v for k, v in sorted(weights["full_weights"].items())},
        },
        "arms": arm_reports,
        "b1_minus_b2": b1_minus_b2(arm_reports),
    }


def format_table(report):
    def num(x, nd=3):
        return "-" if x is None else f"{x:.{nd}f}"

    lines = [
        f"panel_share={report['panel_share']:.4f} (expected ~{report['expected_panel_share']:.3f}) "
        f"ok={report['panel_share_ok']}  spec_partial={report['spec_partial']}",
        "",
        f"{'arm':<16}{'cov':>5}{'gated':>7}{'full_d':>9}{'panel_d':>9}"
        f"{'CI_low':>9}{'CI_high':>9}{'decision':>14}{'fixed':>7}{'flips':>7}"
        f"{'fix_rate':>9}{'rat_med':>9}",
        "-" * 110,
    ]
    for name, r in report["arms"].items():
        cov = "-" if r["coverage"] is None else f"{r['coverage']:g}"
        lines.append(
            f"{name:<16}{cov:>5}{r['gated']:>7}{num(r['full_index_delta']):>9}"
            f"{num(r['panel_scale_delta']):>9}{num(r['ci_low']):>9}{num(r['ci_high']):>9}"
            f"{str(r['decision']):>14}{r['counts']['fixed']:>7}{r['counts']['flipped']:>7}"
            f"{num(r['counts']['fix_rate']):>9}{num(r['rationale'].get('median', 0), 0):>9}"
        )
    if report["b1_minus_b2"]:
        lines.append("")
        lines.append("B1 - B2 (content contrast, matched coverage):")
        for cov, entry in report["b1_minus_b2"].items():
            lines.append(
                f"  {cov}: full_delta b1={entry['full_index_delta_b1']:.3f} "
                f"b2={entry['full_index_delta_b2']:.3f} "
                f"content={entry['content_delta']:.3f} "
                f"fix b1={entry['fixed_b1']} b2={entry['fixed_b2']}"
            )
    lines.append("")
    lines.append("leakage (REASON_PILOT §3) + leak-excluded full-index delta:")
    for name, r in report["arms"].items():
        leak = r.get("leakage") or {}
        if leak.get("leak_classifier") != "ok":
            lines.append(f"  {name}: leak_classifier=unavailable ({leak.get('reason')})")
            continue
        excluded = leak.get("excluded_full_index_delta")
        lines.append(
            f"  {name}: leak_rate={num(leak.get('leak_rate'))} "
            f"({leak.get('leaking')}/{leak.get('classified')}) "
            f"excluded_full_delta={num(excluded)} "
            f"excluded_decision={leak.get('excluded_decision')}"
        )
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description="Post-pilot analysis (REASON_PILOT.md §4).")
    p.add_argument("--b0", required=True, help="reference panel results (full-rules schema)")
    p.add_argument("--rows", required=True, help="panel rows with expected gold")
    p.add_argument("--arms", required=True,
                   help='space-separated name=path entries, e.g. "b1_cov0.10=p b2_cov0.10=q"')
    p.add_argument("--scores", default=None, help="runs/full-rules/scores.json (weights/chance)")
    p.add_argument("--index-spec", default=None, help="explicit index-<edition>.json spec")
    p.add_argument("--kit-root", default=None, help="kit root (default: repo/kit or DI_KIT_ROOT)")
    p.add_argument("--edition", default=None)
    p.add_argument("--panel", type=int, nargs="+", default=list(PANEL_CATALOGS))
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", default=None, help="write the JSON report here")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    arms = parse_arms(args.arms)
    report = analyse(
        args.b0, args.rows, arms, scores_path=args.scores,
        index_spec_path=args.index_spec, kit_root=args.kit_root,
        panel_catalogs=tuple(args.panel), n_boot=args.n_boot, seed=args.seed,
        edition=args.edition,
    )
    print(format_table(report))
    if args.out:
        write_json(args.out, report)
        print(f"\nwrote {args.out}")
    print("PILOT_ANALYZE " + json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
