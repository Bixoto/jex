#!/usr/bin/env python3
"""Integrated Route-B dry-run pipeline: fast pass -> margin gate -> reasoning.

One vLLM V1 engine (``VLLM_BATCH_INVARIANT=1``, ``TRITON_ATTN``,
``FULL_DECODE_ONLY``) runs the whole suite, composing the two validated halves:

* **Phase 1 (fast pass)** -- ``harness/vllm_fastpass``'s label-slot readout over
  every panel row, producing kit-schema ``response.answers[q].probabilities``.
* **Phase 2 (gate)** -- the pre-registered coverage quantile over the fast-pass
  margins, via ``reason_pilot``'s own gate helpers (exactly the pilot inputs:
  the fast-pass results + the panel rows).
* **Phase 3 (reasoning)** -- ``harness/vllm_reason``'s greedy thought + option
  re-read on the gated rows, its ``post_*`` folded over the fast-pass answers;
  non-gated rows keep the fast-pass distribution.

Optional ``--temperature-by-options "a b min"`` applies the engine's output-only
full-cal transform ``T(n) = max(min, a + b*ln n)`` post-hoc
(``engine/config.py:_parse_temperature_by_options`` +
``engine/engine.py`` + ``engine/readout.sharpen``); the decision path is
untouched, as in the engine.

Nothing is reimplemented: the fast pass calls ``vllm_fastpass`` internals, the
gate calls ``reason_pilot`` helpers, the reasoning calls ``vllm_reason``
internals, and the summary reuses ``pilot_analyze``'s weighting math. vLLM is
imported lazily and every phase failure is recorded, never raised out of
:func:`run`.

Usage::

    python harness/vllm_pipeline.py \
        --rows /tmp/reason-pilot/panel.jsonl.gz \
        --reference /tmp/reason-pilot/reference_panel_results.jsonl.gz \
        --coverage 0.05 --cap 512 --fast-batch 8 --reason-batch 16 \
        --scores runs/full-rules/scores.json \
        --out /tmp/runs/vllm-pipeline/results.jsonl.gz \
        --fast-out /tmp/runs/vllm-pipeline/fast-results.jsonl.gz \
        --summary /tmp/runs/vllm-pipeline/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (str(_HERE), str(_REPO_ROOT), str(_REPO_ROOT / "kit")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pilot_analyze as pa  # noqa: E402  (weighting / skill / chance math)
import reason_pilot as rp  # noqa: E402  (gate + thought/re-read helpers)
import vllm_fastpass as vf  # noqa: E402  (fast-pass internals)
import vllm_reason as vr  # noqa: E402  (reasoning internals)

# ---------------------------------------------------------------------------
# Fixed configuration of record.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = vf.DEFAULT_MODEL
DEFAULT_DTYPE = vf.DEFAULT_DTYPE
DEFAULT_MAX_MODEL_LEN = vf.DEFAULT_MAX_MODEL_LEN
DEFAULT_GPU_MEMORY_UTILIZATION = vf.DEFAULT_GPU_MEMORY_UTILIZATION
DEFAULT_ATTENTION_BACKEND = vf.DEFAULT_ATTENTION_BACKEND
DEFAULT_CUDAGRAPH_MODE = vf.DEFAULT_CUDAGRAPH_MODE
DEFAULT_TP = vf.DEFAULT_TP
DEFAULT_MAX_NUM_SEQS = vf.DEFAULT_MAX_NUM_SEQS
DEFAULT_RENDERING = vf.DEFAULT_RENDERING
DEFAULT_SLOT = vf.DEFAULT_SLOT
DEFAULT_LABELS = vf.DEFAULT_LABELS
DEFAULT_TEMPERATURE = vf.DEFAULT_TEMPERATURE
DEFAULT_TRUST_REMOTE_CODE = vf.DEFAULT_TRUST_REMOTE_CODE

DEFAULT_COVERAGE = 0.05
DEFAULT_CAP = 512
DEFAULT_FAST_BATCH = 8
DEFAULT_REASON_BATCH = 16

#: The summary document's keys (tests assert this exact set).
SUMMARY_KEYS = frozenset({
    "kind", "model", "rows", "questions", "gated", "coverage", "cap",
    "fast_batch", "reason_batch", "panel_catalogs", "panel_share",
    "panel_scale_delta", "full_index_delta", "per_catalog", "counts",
    "temperature_by_options", "fast_questions", "merged_questions",
    "reasoned", "folded", "fold_missing", "join_stats", "gate", "errors",
    "counters", "provenance", "weights_available", "wall_seconds_total",
    "fatal_error",
})


# ---------------------------------------------------------------------------
# Full-cal transform: exact engine semantics (engine/config.py, engine/readout.py)
# ---------------------------------------------------------------------------


def parse_temperature_by_options(value):
    """``None`` (off) or ``"a,b,min"`` -> ``(a, b, min)`` floats.

    Mirrors ``engine/config.py:_parse_temperature_by_options`` (comma or
    semicolon separated; the CLI additionally accepts whitespace). ``min`` must
    be > 0. Output-only calibration: emitted choice probabilities are
    re-normalised at ``T(n) = max(min, a + b*ln n)``.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    text = str(value)
    for sep in (";", ","):
        text = text.replace(sep, " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 3:
        raise ValueError(f"temperature_by_options: expected 'a,b,min', got {value!r}")
    try:
        a, b, tmin = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"temperature_by_options: non-numeric entry in {value!r}")
    if not tmin > 0.0:
        raise ValueError(f"temperature_by_options: min must be > 0, got {tmin}")
    return (a, b, tmin)


def emitted_temperature(n_options, temperature_by_options):
    """``T(n) = max(min, a + b*ln n)`` (engine/engine.py:1614)."""
    a, b, tmin = temperature_by_options
    return max(float(tmin), float(a) + float(b) * math.log(int(n_options)))


def _normalise_temperature(scores, keys, temperature=1.0):
    """``engine/readout.normalise``: stable softmax over ``score / T``."""
    n = len(keys)
    if n == 0:
        raise ValueError("cannot normalise over zero options")
    x = [float(s) / float(temperature) for s in scores]
    if not all(math.isfinite(v) for v in x):
        x = [0.0] * n
    top = max(x)
    w = [math.exp(v - top) for v in x]
    z = sum(w)
    if not math.isfinite(z) or z <= 0.0:
        w = [1.0] * n
        z = float(n)
    p = [max(0.0, v / z) for v in w]
    s = sum(p)
    if s <= 0.0:
        p = [1.0 / n] * n
    else:
        p = [v / s for v in p]
    return {k: v for k, v in zip(keys, p)}


def sharpen(probs, keys, temperature=1.0):
    """``engine/readout.sharpen``: ``p^(1/T) / sum(p^(1/T))``."""
    x = [math.log(max(float(probs.get(k, 0.0)), 1e-300)) for k in keys]
    return _normalise_temperature(x, keys, temperature)


def apply_temperature_by_options(probs, temperature_by_options):
    """Sharpen a complete distribution at ``T(n)`` for its own option count."""
    keys = list(probs)
    t_emit = emitted_temperature(len(keys), temperature_by_options)
    return sharpen(probs, keys, t_emit)


def apply_temperature_to_records(records, temperature_by_options):
    """Apply the engine's output-only calibration to every choice answer."""
    changed = 0
    for record in records:
        answers = (record.get("response") or {}).get("answers") or {}
        for answer in answers.values():
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                continue
            probabilities = answer.get("probabilities")
            if not probabilities:
                continue
            answer["probabilities"] = apply_temperature_by_options(
                probabilities, temperature_by_options
            )
            answer["emitted_temperature"] = emitted_temperature(
                len(probabilities), temperature_by_options
            )
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Gate (pure; exactly the pilot's helpers)
# ---------------------------------------------------------------------------


def select_gate(fast_results_path, rows_path, coverage, limit=None):
    """Gate the fast-pass results with ``reason_pilot``'s own helpers.

    Returns ``(selected_candidates, gate_summary, join_stats)``; ordering is
    ``(margin, run_id, question_key)`` so ``limit`` takes the same deterministic
    prefix the pilot would.
    """
    candidates, stats = rp.build_candidates(fast_results_path, rows_path)
    selected, summary = rp.gate_summary(candidates, coverage)
    if limit is not None:
        selected = selected[: int(limit)]
    return selected, summary, stats


# ---------------------------------------------------------------------------
# Fold (pure; tested without vLLM)
# ---------------------------------------------------------------------------


def _fast_record_copy(record):
    """A fresh kit-schema record (so folding never mutates the fast pass)."""
    response = record.get("response") or {}
    return {
        "run_id": record.get("run_id"),
        "catalog_id": record.get("catalog_id"),
        "engine": record.get("engine", "vllm_fastpass"),
        "status": record.get("status", "ok"),
        "response": {
            "model": response.get("model"),
            "answers": dict(response.get("answers") or {}),
        },
        "total_wall_ms": record.get("total_wall_ms"),
        "meta": dict(record.get("meta") or {}),
    }


def fold_records(fast_records, reasoned_records):
    """Fold reasoned ``post_*`` over the fast-pass answers.

    Returns ``(merged_records, folded, missing, gated_keys)``. A reasoned record
    with no ``post_probabilities`` (a readout error) is skipped; a reasoned
    question whose fast-pass row is absent is counted in ``missing`` and does not
    crash the run. Non-gated answers keep the fast-pass distribution.
    """
    merged = [_fast_record_copy(record) for record in fast_records]
    index = {record["run_id"]: record for record in merged}
    folded = 0
    missing = 0
    gated_keys = set()
    for record in reasoned_records:
        probabilities = record.get("post_probabilities")
        if not probabilities:
            continue
        run_id = record.get("run_id")
        question_key = record.get("question_key")
        target = index.get(run_id)
        if target is None:
            missing += 1
            continue
        target["response"]["answers"][question_key] = {
            "type": "choice",
            "choice": record.get("post_choice"),
            "probabilities": probabilities,
        }
        target["engine"] = "vllm_pipeline"
        folded += 1
        gated_keys.add((run_id, question_key))
    return merged, folded, missing, gated_keys


# ---------------------------------------------------------------------------
# Summary (pure; reuses pilot_analyze's weighting math)
# ---------------------------------------------------------------------------


def _results_index(records):
    """``{run_id: {"catalog_id", "answers": {qk: choice}}}`` (pa.load_results shape)."""
    out = {}
    for record in records or []:
        if record.get("status") not in (None, "ok"):
            continue
        answers = (record.get("response") or {}).get("answers") or {}
        out[record.get("run_id")] = {
            "catalog_id": record.get("catalog_id"),
            "answers": {
                key: (value or {}).get("choice")
                for key, value in answers.items()
                if isinstance(value, dict)
            },
        }
    return out


def _rows_index(rows):
    """``pa.load_rows`` shape from in-memory panel rows."""
    out = {}
    for row in rows or []:
        evaluation = row.get("_evaluation") or {}
        run_id = evaluation.get("run_id") or row.get("id")
        catalog_id = pa.catalog_id(row)
        expected = row.get("expected") or {}
        questions = row.get("questions") or {}
        for question_key, gold in expected.items():
            out[(run_id, question_key)] = {
                "catalog_id": catalog_id,
                "gold": gold,
                "options": pa.option_descriptions(questions.get(question_key)),
            }
    return out


def summarize_pipeline(panel_rows, reference_records, merged_records,
                       reasoned_records, *, scores_path=None,
                       index_spec_path=None, kit_root=None, edition=None,
                       panel_catalogs=pa.PANEL_CATALOGS):
    """Baseline (reference) vs merged per-catalog skill and full-index delta.

    Reuses ``pilot_analyze``'s weighting math (``benchmark_weights``,
    ``panel_weighting``, ``chance_table``, ``skill``). The policy is the *merged*
    route: reasoned ``post_choice`` on gated questions, the fast pass elsewhere.
    Per-catalog and weighted deltas follow ``pa.evaluate_arm`` exactly.
    """
    rows = _rows_index(panel_rows)
    reference = _results_index(reference_records)
    merged = _results_index(merged_records)
    gated_keys = {
        (record.get("run_id"), record.get("question_key"))
        for record in reasoned_records
        if record.get("post_probabilities")
    }

    scores = {}
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as handle:
            scores = json.load(handle)
    spec = pa.load_index_spec(scores, index_spec_path, kit_root, edition)
    weights = pa.benchmark_weights(spec)
    chances = pa.chance_table(spec, scores)
    share, panel_norm = pa.panel_weighting(weights, panel_catalogs)
    full = weights["full_weights"]
    questions = pa.build_questions(rows, reference, panel_catalogs)

    per_catalog = {}
    panel_delta = 0.0
    full_delta = 0.0
    for question in questions:
        catalog_id = question["catalog_id"]
        gold = question["gold"]
        key = (question["run_id"], question["question_key"])
        base_choice = question["b0_choice"]
        policy_choice = (merged.get(question["run_id"], {}).get("answers") or {}).get(
            question["question_key"]
        )
        entry = per_catalog.setdefault(catalog_id, {
            "n": 0, "gated": 0, "baseline_correct": 0, "policy_correct": 0,
            "fixed": 0, "flipped": 0, "unchanged_right": 0, "unchanged_wrong": 0,
        })
        base_ok = base_choice is not None and base_choice == gold
        policy_ok = policy_choice is not None and policy_choice == gold
        entry["n"] += 1
        entry["gated"] += int(key in gated_keys)
        entry["baseline_correct"] += int(base_ok)
        entry["policy_correct"] += int(policy_ok)
        if not base_ok and policy_ok:
            entry["fixed"] += 1
        elif base_ok and not policy_ok:
            entry["flipped"] += 1
        elif base_ok and policy_ok:
            entry["unchanged_right"] += 1
        else:
            entry["unchanged_wrong"] += 1

    weights_available = bool(full) and share > 0.0
    for catalog_id, entry in sorted(per_catalog.items()):
        chance = chances.get(catalog_id, 0.0)
        base_skill = pa.skill(entry["baseline_correct"], entry["n"], chance)
        policy_skill = pa.skill(entry["policy_correct"], entry["n"], chance)
        delta = (
            (policy_skill - base_skill)
            if (base_skill is not None and policy_skill is not None) else 0.0
        )
        weight = panel_norm.get(catalog_id, 0.0)
        full_weight = full.get(catalog_id, 0.0)
        entry.update({
            "baseline_accuracy": (entry["baseline_correct"] / entry["n"]) if entry["n"] else None,
            "policy_accuracy": (entry["policy_correct"] / entry["n"]) if entry["n"] else None,
            "baseline_skill": base_skill,
            "policy_skill": policy_skill,
            "delta_skill": delta,
            "panel_weight": weight,
            "full_weight": full_weight,
        })
        panel_delta += weight * delta
        full_delta += full_weight * delta

    total_n = sum(entry["n"] for entry in per_catalog.values())
    total_gated = sum(entry["gated"] for entry in per_catalog.values())
    fixed = sum(entry["fixed"] for entry in per_catalog.values())
    flipped = sum(entry["flipped"] for entry in per_catalog.values())
    baseline_wrong_gated = sum(
        1 for question in questions
        if (question["run_id"], question["question_key"]) in gated_keys
        and not (question["b0_choice"] is not None and question["b0_choice"] == question["gold"])
    )
    return {
        "panel_catalogs": [int(c) for c in panel_catalogs],
        "panel_share": share,
        "weights_available": weights_available,
        "panel_scale_delta": (100.0 * panel_delta) if weights_available else None,
        "full_index_delta": (100.0 * full_delta) if weights_available else None,
        "per_catalog": {str(k): v for k, v in sorted(per_catalog.items())},
        "counts": {
            "n": total_n,
            "gated": total_gated,
            "fixed": fixed,
            "flipped": flipped,
            "baseline_wrong_gated": baseline_wrong_gated,
            "fix_rate": (fixed / baseline_wrong_gated) if baseline_wrong_gated else None,
            "unchanged": total_gated - fixed - flipped,
        },
    }


# ---------------------------------------------------------------------------
# Fast pass (vllm_fastpass internals, over a shared engine)
# ---------------------------------------------------------------------------


def run_fast_pass(llm, SamplingParams, modules, tokenizer, label_table, rows, args,
                  mechanism, max_label_ids, counters, errors):
    """Phase 1: score every question of the panel rows; returns kit records."""
    out_records = []
    for start in range(0, len(rows), int(args.fast_batch)):
        chunk = rows[start:start + int(args.fast_batch)]
        requests = []
        for row in chunk:
            state = row.get("state")
            run_id = (row.get("_evaluation") or {}).get("run_id") or row.get("id")
            catalog_id = (row.get("_evaluation") or {}).get("catalog_id")
            for question_key, question in (row.get("questions") or {}).items():
                request, error = vf.build_question_request(
                    modules, tokenizer, label_table, state, question,
                    rendering=args.rendering, slot=args.slot,
                    system_prompt=args.system_prompt,
                    neutralize=args.neutralize_options,
                    max_model_len=args.max_model_len,
                )
                if error is not None:
                    if error.startswith("context_refusal"):
                        counters["context_refusals"] += 1
                    elif error.startswith("key_fallback"):
                        counters["key_fallback_questions"] += 1
                    else:
                        counters["unsupported_questions"] += 1
                    continue
                requests.append((row, run_id, catalog_id, question_key, question, request))

        if not requests:
            continue
        id_lists = [request["ids"] for *_, request in requests]
        # Only as many label ids as the widest question in this batch needs;
        # ``score_batch`` chunks anything beyond the engine's per-request cap.
        batch_max = max(len(request["label_ids"]) for *_, request in requests)
        rows_scores, batch_error = vf.score_batch(
            llm, SamplingParams, mechanism, id_lists, max_label_ids[:batch_max]
        )
        if batch_error:
            counters["batch_errors"] += 1
            errors.append(f"fast batch {start}: {batch_error}")
            continue
        if len(rows_scores) != len(requests):
            counters["batch_errors"] += 1
            errors.append(
                f"fast batch {start}: mechanism returned {len(rows_scores)} rows "
                f"for {len(requests)} prompts"
            )
            continue

        by_row = {}
        for (row, run_id, catalog_id, question_key, question, request), score_row in zip(
                requests, rows_scores):
            entry = by_row.setdefault(run_id, {
                "row": row, "catalog_id": catalog_id, "answers": {}})
            label_scores = score_row[:len(request["label_ids"])]
            if not label_scores or any(value is None for value in label_scores):
                counters["readout_errors"] += 1
                continue
            probabilities = vf.normalise_scores(
                label_scores, request["keys"], args.temperature)
            if request["question_type"] == "noul":
                entry["answers"][question_key] = {
                    "type": "noul",
                    "noul": probabilities.get(request["noul_true_key"], 0.5),
                }
            else:
                entry["answers"][question_key] = {
                    "type": "choice",
                    "choice": vf.argmax_distribution(probabilities),
                    "probabilities": probabilities,
                }
                counters["scored"] += 1

        for run_id, entry in by_row.items():
            if not entry["answers"]:
                continue
            out_records.append(vf.make_output_record(
                run_id=run_id, catalog_id=entry["catalog_id"], model=args.model,
                answers=entry["answers"],
                meta={"phase": "fast", "rendering": args.rendering, "slot": args.slot,
                      "mechanism": mechanism},
            ))
    return out_records


# ---------------------------------------------------------------------------
# Reasoning (vllm_reason internals, over the same engine)
# ---------------------------------------------------------------------------


def run_reasoning(llm, tokenizer, selected, args, cap, batch_size):
    """Phase 3: greedy thought + option re-read on the gated candidates.

    Items whose worst-case re-read prompt (thought prompt + a cap-sized
    rationale + the repeated question body) cannot fit the model context are
    skipped up front and keep their fast-pass answers: a single over-long
    prompt would otherwise reject its whole vLLM batch and abort the phase.
    """
    max_model_len = int(getattr(args, "max_model_len", 8192) or 8192)
    cap_i = int(cap)
    worst_rationale = "a " * (cap_i + 128)  # >= cap tokens, safe upper bound
    kept, thought_prompts, thought_ids = [], [], []
    skipped_context = 0
    exact_checks = 0
    for candidate in selected:
        prompt = rp.build_thought_prompt(
            tokenizer, candidate["state"], candidate["question"], args.rendering)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) + 2 > max_model_len:
            skipped_context += 1
            continue
        # Cheap sufficient bound: re-read <= prompt + cap rationale + a second
        # body (<= prompt) + markers. Exact tokenization only when inconclusive.
        if 2 * len(ids) + cap_i + 512 > max_model_len:
            exact_checks += 1
            worst_text = rp.build_reread_text(
                prompt, worst_rationale, candidate["state"],
                candidate["question"], args.rendering)
            worst_ids = tokenizer(worst_text, add_special_tokens=False)["input_ids"]
            if len(worst_ids) + 1 > max_model_len:
                skipped_context += 1
                continue
        kept.append(candidate)
        thought_prompts.append(prompt)
        thought_ids.append(ids)
    if skipped_context or exact_checks:
        print(f"reasoning: {len(selected)} gated items, skipped {skipped_context} "
              f"over the context budget (max_model_len={max_model_len}; "
              f"{exact_checks} exact checks)")

    generated = vr.generate_thoughts(llm, tokenizer, thought_ids, cap, batch_size)

    reread_ids, labels_per_item, label_ids_per_item, rationales = [], [], [], []
    for index, (candidate, gen) in enumerate(zip(kept, generated)):
        keys = rp.question_keys(candidate["question"])
        rationale = gen["rationale"]
        rationale_tokens = len(tokenizer.encode(rationale, add_special_tokens=False))
        labels, label_ids = rp.option_label_token_ids(tokenizer, keys)
        reread_text = rp.build_reread_text(
            thought_prompts[index], rationale, candidate["state"],
            candidate["question"], args.rendering,
        )
        reread_ids.append(tokenizer(reread_text, add_special_tokens=False)["input_ids"])
        labels_per_item.append(labels)
        label_ids_per_item.append(label_ids)
        rationales.append((rationale, rationale_tokens))

    readouts = vr.readout_probs(
        llm, tokenizer, reread_ids, labels_per_item, label_ids_per_item, batch_size
    )

    records = []
    for index, candidate in enumerate(kept):
        gen = generated[index]
        rationale, rationale_tokens = rationales[index]
        readout = readouts[index] or {"error": "no readout produced"}
        post_probabilities = readout.get("probabilities")
        post_choice = (
            max(post_probabilities, key=lambda key: post_probabilities[key])
            if post_probabilities else None
        )
        records.append(vr.build_record(
            candidate, "b1", post_choice, post_probabilities, rationale_tokens,
            gen["truncated"], gen["batch_seconds"] / max(gen["batch_size"], 1),
            gen["generated_tokens"],
            batch_size=gen["batch_size"], batch_id=gen["batch_id"],
            batch_seconds=gen["batch_seconds"],
            rationale_text=rp.bound_rationale_text(rationale),
            readout_fresh_seconds=readout.get("seconds"),
            engine="vllm", bi=True, cap=cap, readout_error=readout.get("error"),
        ))
    return records


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _panel_catalogs(args):
    """The catalogs the summary is restricted to (default: the pilot six).

    ``--panel-catalogs`` lets a full-edition production run score every catalog
    present in the panel instead of the Phase-A pilot's six; without it the
    pilot behaviour is unchanged.
    """
    requested = getattr(args, "panel_catalogs", None)
    if requested:
        return tuple(int(c) for c in requested)
    return tuple(pa.PANEL_CATALOGS)


def _fatal(args, message, **extra):
    summary = {
        "kind": "vllm_pipeline",
        "model": getattr(args, "model", None),
        "rows": 0, "questions": 0, "gated": 0,
        "coverage": getattr(args, "coverage", None),
        "cap": getattr(args, "cap", None),
        "fast_batch": getattr(args, "fast_batch", None),
        "reason_batch": getattr(args, "reason_batch", None),
        "panel_catalogs": [int(c) for c in _panel_catalogs(args)],
        "panel_share": None, "panel_scale_delta": None, "full_index_delta": None,
        "per_catalog": {}, "counts": {}, "temperature_by_options": None,
        "fast_questions": 0, "merged_questions": 0, "reasoned": 0, "folded": 0,
        "fold_missing": 0, "errors": [message], "counters": {},
        "provenance": {}, "weights_available": False,
        "wall_seconds_total": None, "fatal_error": message,
    }
    summary.update(extra)
    if getattr(args, "summary", None):
        vf._write_json(args.summary, summary)
    print(f"FATAL: {message}")
    return summary


def run(args):
    """Execute the pipeline; always return a summary (fatal errors recorded)."""
    started = time.perf_counter()
    errors = []
    counters = vf._empty_counters()

    if not args.rows:
        return _fatal(args, "no --rows panel supplied")
    if not args.reference:
        return _fatal(args, "no --reference fast-pass reference supplied")

    try:
        rows = list(vf.load_jsonl(args.rows))
    except Exception as exc:  # noqa: BLE001
        return _fatal(args, f"could not read rows {args.rows!r}: "
                            f"{type(exc).__name__}: {exc}")
    if not rows:
        return _fatal(args, f"rows file is empty or unreadable: {args.rows!r}")
    reference_records = list(vf.load_jsonl(args.reference))
    if not reference_records:
        return _fatal(args, f"reference file is empty or unreadable: "
                            f"{args.reference!r}")
    if args.limit:
        rows = rows[: int(args.limit)]
    counters["rows"] = len(rows)
    counters["questions"] = sum(len(r.get("questions") or {}) for r in rows)

    # Env must be set before vLLM is imported.
    env_used = vf.apply_env(args)
    try:
        import vllm
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return _fatal(args, f"could not import vllm: {type(exc).__name__}: {exc}",
                      counters=counters, errors=errors + [str(exc)])

    try:
        modules = vf._engine_modules()
        api_probe = vf.introspect_vllm(vllm)
        versions = vf._versions()
        requested = vf._requested_kwargs(args, api_probe)
        llm, accepted, dropped, notes, err = vf._construct_llm(LLM, requested)
        if llm is None:
            raise RuntimeError(err or "LLM construction failed")
        tokenizer = vf._resolve_tokenizer(llm, args.model, args.trust_remote_code)
        label_table = vf.build_label_table(modules, tokenizer, args.labels, args.slot)
        labels_valid = len(label_table)
    except Exception as exc:  # noqa: BLE001
        return _fatal(args, f"engine setup failed: {type(exc).__name__}: {exc}",
                      counters=counters, errors=errors + [str(exc)])

    provenance = {
        "versions": versions,
        "bi": True,
        "attention_backend": args.attention_backend,
        "cudagraph_mode": args.cudagraph_mode,
        "env": env_used,
        "llm_kwargs": accepted,
        "llm_kwargs_dropped": dropped,
        "construct_notes": notes,
        "api_probe": api_probe,
    }

    max_options = max(
        (len(q.get("criteria") or {}) for r in rows
         for q in (r.get("questions") or {}).values()), default=1)
    max_label_ids = [
        label_table.token_id(i) for i in range(min(max_options, labels_valid))
    ]
    if not max_label_ids:
        return _fatal(args, "no valid label ids", counters=counters, errors=errors,
                      provenance=provenance)

    # The probe must stay inside the engine's per-request logprob cap; wider
    # label sets are handled by ``score_batch`` chunking at scoring time.
    probe_label_ids = max_label_ids[:int(vf.MAX_LOGPROB_IDS)]
    probe_prompts = vf._probe_prompts(modules, tokenizer, probe_label_ids)
    try:
        mechanism, attempts = vf.probe_mechanism(
            llm, SamplingParams, tokenizer, probe_label_ids, probe_prompts,
            forced=args.mechanism)
    except Exception as exc:  # noqa: BLE001
        mechanism, attempts = None, []
        errors.append(f"mechanism probe failed: {type(exc).__name__}: {exc}")
    provenance["mechanism"] = mechanism
    provenance["mechanism_attempts"] = attempts
    if mechanism is None:
        counters["mechanism_failures"] += 1

    # ---------------------------------------------------------------- phase 1
    try:
        fast_records = run_fast_pass(
            llm, SamplingParams, modules, tokenizer, label_table, rows, args,
            mechanism, max_label_ids, counters, errors)
    except Exception as exc:  # noqa: BLE001
        return _fatal(args, f"fast pass failed: {type(exc).__name__}: {exc}",
                      counters=counters, errors=errors + [str(exc)],
                      provenance=provenance)
    if args.fast_out:
        vf._write_jsonl(args.fast_out, fast_records)
    fast_path = args.fast_out or "/tmp/vllm_pipeline_fast_results.jsonl.gz"
    if not args.fast_out:
        vf._write_jsonl(fast_path, fast_records)

    # ---------------------------------------------------------------- phase 2
    try:
        selected, gate, join_stats = select_gate(
            fast_path, args.rows, args.coverage)
    except Exception as exc:  # noqa: BLE001
        return _fatal(args, f"gate selection failed: {type(exc).__name__}: {exc}",
                      counters=counters, errors=errors + [str(exc)],
                      provenance=provenance)
    counters["gated"] = len(selected)

    # ---------------------------------------------------------------- phase 3
    reasoned = []
    if selected:
        try:
            reasoned = run_reasoning(
                llm, tokenizer, selected, args, int(args.cap),
                int(args.reason_batch))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reasoning failed: {type(exc).__name__}: {exc}")

    merged, folded, fold_missing, _gated_keys = fold_records(fast_records, reasoned)
    # The production reference ran with the frozen decision rules enabled; the
    # readout omits that step, so the fold applies it here (ACOS / iSarcasm /
    # FinEntity), before the full-cal transform. Deterministic per question.
    decision_rule_stats = None
    if not getattr(args, "no_decision_rules", False):
        try:
            import decision_rules_apply
            decision_rule_stats = decision_rules_apply.apply_rules_to_records(
                rows, merged)
        except Exception as exc:  # noqa: BLE001 - non-fatal, recorded
            errors.append(f"decision_rules failed: {type(exc).__name__}: {exc}")
    temperature_by_options = None
    if args.temperature_by_options:
        try:
            temperature_by_options = parse_temperature_by_options(
                args.temperature_by_options)
            apply_temperature_to_records(merged, temperature_by_options)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"temperature_by_options failed: "
                          f"{type(exc).__name__}: {exc}")
            temperature_by_options = None

    # Catalog-48 coherence projection (cal -> JCD poly). Deterministic, CPU-only,
    # label-free; runs after full-cal and only touches ForecastBench answers.
    jcd_stats = None
    if not getattr(args, "no_jcd", False):
        try:
            import jcd_c48
            jcd_stats = jcd_c48.apply_jcd(rows, merged)
        except Exception as exc:  # noqa: BLE001 - non-fatal, recorded
            errors.append(f"jcd failed: {type(exc).__name__}: {exc}")

    if args.out:
        vf._write_jsonl(args.out, merged)

    # ---------------------------------------------------------------- summary
    panel_catalogs = _panel_catalogs(args)
    try:
        summary = summarize_pipeline(
            rows, reference_records, merged, reasoned,
            scores_path=args.scores, index_spec_path=args.index_spec,
            kit_root=args.kit_root, edition=args.edition,
            panel_catalogs=panel_catalogs)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"summary failed: {type(exc).__name__}: {exc}")
        summary = {"panel_catalogs": [int(c) for c in panel_catalogs],
                   "panel_share": None, "panel_scale_delta": None,
                   "full_index_delta": None, "per_catalog": {}, "counts": {},
                   "weights_available": False}

    summary.update({
        "kind": "vllm_pipeline",
        "decision_rules": decision_rule_stats,
        "jcd": jcd_stats,
        "model": args.model,
        "rows": len(rows),
        "questions": counters["questions"],
        "gated": len(selected),
        "coverage": float(args.coverage),
        "cap": int(args.cap),
        "fast_batch": int(args.fast_batch),
        "reason_batch": int(args.reason_batch),
        "temperature_by_options": (
            list(temperature_by_options) if temperature_by_options else None),
        "fast_questions": len(fast_records),
        "merged_questions": len(merged),
        "reasoned": len(reasoned),
        "folded": folded,
        "fold_missing": fold_missing,
        "join_stats": join_stats,
        "gate": gate,
        "errors": errors,
        "counters": counters,
        "provenance": provenance,
        "wall_seconds_total": time.perf_counter() - started,
        "fatal_error": None,
    })
    if args.summary:
        vf._write_json(args.summary, summary)

    print(f"pipeline rows={len(rows)} fast_questions={len(fast_records)} "
          f"gated={len(selected)} reasoned={len(reasoned)} folded={folded} "
          f"merged={len(merged)}")
    if summary.get("full_index_delta") is not None:
        print(f"VLLM_PIPELINE_DELTA full_index={summary['full_index_delta']:.4f} "
              f"panel_scale={summary['panel_scale_delta']:.4f} "
              f"fixed={summary['counts']['fixed']} flips={summary['counts']['flipped']}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Integrated Route-B dry-run pipeline (vLLM fast pass + "
                    "margin gate + vLLM reasoning) on the BI engine.")
    p.add_argument("--rows", default=None, help="panel JSONL(.gz) (required)")
    p.add_argument("--reference", default=None,
                   help="stored HF reference panel results JSONL(.gz) (baseline)")
    p.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE,
                   help=f"pre-registered gate coverage (default {DEFAULT_COVERAGE})")
    p.add_argument("--no-decision-rules", action="store_true",
                   help="skip the frozen decision-rule application in the fold")
    p.add_argument("--no-jcd", action="store_true",
                   help="skip the catalog-48 JCD coherence projection (default: on)")
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help=f"max new thought tokens (default {DEFAULT_CAP})")
    p.add_argument("--fast-batch", dest="fast_batch", type=int,
                   default=DEFAULT_FAST_BATCH,
                   help=f"fast-pass rows per vLLM call (default {DEFAULT_FAST_BATCH})")
    p.add_argument("--reason-batch", dest="reason_batch", type=int,
                   default=DEFAULT_REASON_BATCH,
                   help=f"reasoning batch width (default {DEFAULT_REASON_BATCH})")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of panel rows scored (smoke)")
    p.add_argument("--out", default=None,
                   help="merged kit-schema results JSONL(.gz)")
    p.add_argument("--fast-out", dest="fast_out", default=None,
                   help="fast-pass-only kit-schema results JSONL(.gz)")
    p.add_argument("--summary", default=None, help="summary JSON path")
    p.add_argument("--temperature-by-options", dest="temperature_by_options",
                   default=None,
                   help='full-cal output transform "a b min" (e.g. "10.124 -1.633 0.05")')

    # weights / baseline
    p.add_argument("--scores", default=None,
                   help="runs/full-rules/scores.json (weights + chance)")
    p.add_argument("--index-spec", dest="index_spec", default=None,
                   help="explicit index-<edition>.json spec")
    p.add_argument("--kit-root", dest="kit_root", default=None,
                   help="kit root (default: repo/kit)")
    p.add_argument("--edition", default=None)
    p.add_argument("--panel-catalogs", dest="panel_catalogs", type=int,
                   nargs="+", default=None,
                   help="catalogs the summary is restricted to (default: the "
                        "pilot six). Full-edition production passes every "
                        "catalog in the panel.")

    # engine (mirrors vllm_fastpass)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default=DEFAULT_DTYPE)
    p.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--gpu-memory-utilization", type=float,
                   default=DEFAULT_GPU_MEMORY_UTILIZATION)
    p.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    p.add_argument("--attention-backend", default=DEFAULT_ATTENTION_BACKEND)
    p.add_argument("--cudagraph-mode", default=DEFAULT_CUDAGRAPH_MODE)
    p.add_argument("--trust-remote-code", dest="trust_remote_code",
                   action="store_true", default=DEFAULT_TRUST_REMOTE_CODE)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code",
                   action="store_false")
    p.add_argument("--rendering", default=DEFAULT_RENDERING)
    p.add_argument("--slot", default=DEFAULT_SLOT)
    p.add_argument("--labels", type=int, default=DEFAULT_LABELS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--system-prompt", dest="system_prompt", action="store_true",
                   default=False)
    p.add_argument("--neutralize-options", dest="neutralize_options",
                   action="store_true", default=False)
    p.add_argument("--mechanism", default=None, choices=list(vf.MECHANISM_ORDER))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.kit_root is None:
        args.kit_root = str(_REPO_ROOT / "kit")
    if args.scores is None:
        default_scores = _REPO_ROOT / "runs" / "full-rules" / "scores.json"
        if default_scores.exists():
            args.scores = str(default_scores)
    summary = run(args)
    # Never fatal: a failed run is evidence, not a crash. The bootstrap decides.
    return 0


if __name__ == "__main__":
    sys.exit(main())
