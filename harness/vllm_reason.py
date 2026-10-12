#!/usr/bin/env python3
"""vLLM V1 reasoning runner: the BI-engine port of the Stage-1 pilot mechanism.

Route B is the production engine: vLLM V1 with ``VLLM_BATCH_INVARIANT=1``
(token-identical greedy decode at batch 1/8/16/32; ``TRITON_ATTN``;
``FULL_DECODE_ONLY``; vLLM 0.30.0). This module runs the *same* science as
``harness/reason_pilot.py`` -- margin-quantile gate, Rune-pattern thought prompt,
greedy rationale, generate-then-re-read, positional-label readout -- but on that
engine. It does **not** reimplement the mechanism: every pure helper (gate,
prompt construction, label mapping, fold-back, record schema) is imported from
``reason_pilot`` so the two arms are provably the same computation.

Pipeline (``--results`` fast pass + ``--rows`` panel in -> pilot-schema JSONL out)
--------------------------------------------------------------------------------
1. **Join + gate** via :func:`reason_pilot.build_candidates` /
   :func:`reason_pilot.gate_summary`: the pre-registered coverage is a
   nearest-rank quantile of the eligible fast-pass margins. ``--w-rows`` swaps
   the row set for the W breadth cut and defaults coverage to ``1.0``.
2. **Thought prompt** via :func:`reason_pilot.build_thought_prompt` (reference
   rendering, reasoning system prompt, ``enable_thinking=True``).
3. **Greedy generation** through vLLM at a fixed batch width, stopping at the
   close marker ``<channel|>`` and retaining it in the output
   (``include_stop_str_in_output=True``) so :func:`reason_pilot.parse_rationale`
   sees the same stream as the HF harness. ``truncated`` is True iff the close is
   absent or ``finish_reason == "length"``.
4. **Re-read** via :func:`reason_pilot.build_reread_text`; the option labels are
   scored with vLLM's ``logprob_token_ids`` in a one-token greedy step and
   softmaxed exactly like :func:`reason_pilot.probs_from_last_logits` (a softmax
   over log-probs equals a softmax over the underlying logits), then mapped back
   to option keys.
5. **Fold-back** through :func:`reason_pilot.make_record`: ``post_choice`` is the
   argmax of the re-read distribution and ``post_probabilities`` is that
   distribution -- byte-for-byte the harness's arm semantics.

Prompts are passed to vLLM as ``prompt_token_ids`` (tokenised here with
``add_special_tokens=False``) so there is no retokenisation drift between the
harness and the engine.

CLI::

    python harness/vllm_reason.py \
        --results <fastpass>/results.jsonl.gz --rows panel.jsonl.gz \
        --coverage 0.05 --limit 120 --cap 512 --batch-size 16 --arm b1 \
        --out /tmp/runs/vllm-reason-b1.jsonl

    # W breadth arm (all 25+43 + 200 of 30), coverage forced to 1.0:
    python harness/vllm_reason.py --results <fastpass> --w-rows w_rows.jsonl.gz \
        --arm w --batch-size 16 --out w.jsonl

    # Compare a produced arm record-for-record against a stored HF arm:
    python harness/vllm_reason.py --records vllm-b1.jsonl --compare hf-b1.jsonl \
        --compare-out compare.json

vLLM is imported lazily and only inside the GPU-side functions, so importing
this module never requires vLLM (the tests run on any Python 3).

Uncertainties (see the report at the end of the run): label logprobs at the
slot, batch composition vs the engine's own scheduler, and generation stop
handling are recorded in the output/report rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import reason_pilot as rp  # noqa: E402  (pure gate/prompt/readout/fold helpers)
import vllm_bakeoff as bk  # noqa: E402  (LLM construction, tolerant of API drift)
import vllm_probe as vp  # noqa: E402  (env contract + kwargs introspection)
from _batching import grouped_batches  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed configuration of record.
# ---------------------------------------------------------------------------

DEFAULT_MODEL = vp.DEFAULT_MODEL
DEFAULT_DTYPE = vp.DEFAULT_DTYPE
DEFAULT_MAX_MODEL_LEN = vp.DEFAULT_MAX_MODEL_LEN
DEFAULT_GPU_MEMORY_UTILIZATION = vp.DEFAULT_GPU_MEMORY_UTILIZATION
DEFAULT_ATTENTION_BACKEND = vp.DEFAULT_ATTENTION_BACKEND
DEFAULT_CUDAGRAPH_MODE = vp.DEFAULT_CUDAGRAPH_MODE
DEFAULT_TRUST_REMOTE_CODE = vp.DEFAULT_TRUST_REMOTE_CODE
DEFAULT_TP = vp.DEFAULT_TP

#: Route-B defaults. The pilot's decode smoke runs b1 at 5% coverage, so 5% is
#: the runner default; the BI-validated batch ladder is 1/8/16/32.
DEFAULT_COVERAGE = 0.05
DEFAULT_CAP = 512
DEFAULT_BATCH_SIZE = 16

#: n-gram speculative-decoding prompt-lookup length. The default matches the
#: bake-off's (:data:`vllm_bakeoff.DEFAULT_SPEC_NGRAM_TOKENS`) so the A/B uses
#: the exact arm that won the ``--spec-only`` bake-off.
DEFAULT_SPEC_NGRAM_TOKENS = bk.DEFAULT_SPEC_NGRAM_TOKENS

ARMS = ("b1", "b2", "w")
DEFAULT_RENDERING = rp.DEFAULT_RENDERING

#: The BI-validated fixed batch widths (documentation only; any width >= 1 is
#: accepted, and batch composition is always fixed via grouped_batches).
BI_BATCH_WIDTHS = (1, 8, 16, 32)

#: Record keys the runner always emits (pilot schema + the runner's provenance).
RECORD_KEYS = frozenset({
    "run_id", "catalog_id", "question_key", "arm", "gold", "pre_choice",
    "pre_probabilities", "post_choice", "post_probabilities", "rationale_tokens",
    "rationale_text", "truncated", "gen_seconds", "generated_tokens",
    "batch_size", "batch_id", "batch_seconds", "readout_fresh_seconds",
    "cont_choice", "cont_probabilities", "readout_cont_seconds",
    "continuation_ok", "continuation_note",
})

#: Provenance keys written on top of the pilot schema (``spec`` only when the
#: optional n-gram arm is enabled).
RUNNER_KEYS = ("engine", "bi", "cap", "spec")


# ---------------------------------------------------------------------------
# Config resolution (pure)
# ---------------------------------------------------------------------------


def resolve_config(args):
    """Resolve row set / coverage / arm from the CLI, with the W-arm rule.

    ``--w-rows`` selects the W breadth cut and defaults coverage to ``1.0``
    (the pilot's W arm reasons on every row). ``--arm w`` requires ``--w-rows``.
    The arm tag defaults to ``w`` for W rows, ``b2`` under ``--filler`` and
    ``b1`` otherwise.
    """
    coverage = getattr(args, "coverage", None)
    rows_path = getattr(args, "rows", None)
    w_rows = getattr(args, "w_rows", None)
    arm = getattr(args, "arm", None)

    if arm == "w" and not w_rows:
        raise SystemExit("--arm w requires --w-rows (the W breadth row cut)")

    if w_rows:
        rows_path = w_rows
        if coverage is None:
            coverage = 1.0
        if arm is None:
            arm = "w"
    else:
        if coverage is None:
            coverage = DEFAULT_COVERAGE
        if arm is None:
            arm = "b2" if getattr(args, "filler", False) else "b1"

    if coverage is None or not (0.0 < float(coverage) <= 1.0):
        raise SystemExit(f"--coverage must be in (0, 1], got {coverage!r}")
    return {
        "rows": rows_path,
        "w_rows": bool(w_rows),
        "coverage": float(coverage),
        "arm": arm,
    }


def resolve_spec(args):
    """Resolve the optional n-gram spec arm to ``(tokens, candidates)``.

    Disabled (no ``--spec-ngram``) returns ``(None, [])`` and leaves every other
    code path untouched. Enabled returns the prompt-lookup token count and the
    ordered constructor-kwarg candidates from the bake-off
    (:func:`vllm_bakeoff.spec_kwarg_candidates`): the newer flat
    ``spec_method``/``spec_tokens`` signature first, the older
    ``speculative_config`` dict after, so a different vLLM signature still
    constructs.
    """
    if not getattr(args, "spec_ngram", False):
        return None, []
    raw = getattr(args, "spec_ngram_tokens", None)
    tokens = DEFAULT_SPEC_NGRAM_TOKENS if raw is None else int(raw)
    if tokens < 1:
        raise SystemExit(f"--spec-ngram-tokens must be >= 1, got {tokens}")
    return tokens, bk.spec_kwarg_candidates("ngram", ngram_tokens=tokens)


def select_gated(results_path, rows_path, coverage, limit=None):
    """Gate selection with the harness's own pure functions (parity by import).

    Returns ``(selected, summary, join_stats)``. ``selected`` is ordered by
    ``(margin, run_id, question_key)`` exactly as
    :func:`reason_pilot.select_by_coverage`, so ``--limit`` takes the same
    deterministic prefix the harness would.
    """
    candidates, stats = rp.build_candidates(results_path, rows_path)
    selected, summary = rp.gate_summary(candidates, coverage)
    if limit is not None:
        selected = selected[: int(limit)]
    return selected, summary, stats


# ---------------------------------------------------------------------------
# Readout scoring (pure; tested without vLLM)
# ---------------------------------------------------------------------------


def probs_from_scores(scores, labels):
    """Softmax over the allowed label *log-probs* and map back to option keys.

    A softmax over log-probs is identical to the harness's softmax over logits
    (:func:`reason_pilot.probs_from_last_logits`): the log-sum-exp subtracted by
    ``log_softmax`` is a constant. ``labels`` is ``[(label, option_key), ...]``
    from :func:`reason_pilot.option_labels`.
    """
    probs = rp.softmax([float(score) for score in scores])
    return {key: p for (_label, key), p in zip(labels, probs)}


def _logprob_value(entry):
    """A vLLM ``Logprob`` object or a bare float -> float."""
    if entry is None:
        raise ValueError("missing logprob entry")
    if isinstance(entry, (int, float)):
        return float(entry)
    return float(getattr(entry, "logprob"))


def label_scores_from_step(step, token_ids):
    """Extract the requested label log-probs from one vLLM logprob step.

    ``step`` is the per-position ``{token_id: Logprob}`` mapping vLLM returns.
    A missing label id is a hard error (never a silent misread), matching the
    harness's single-token verification.
    """
    if step is None:
        raise ValueError("no logprobs returned by vLLM")
    scores = []
    for token_id in token_ids:
        entry = step.get(int(token_id))
        if entry is None:
            entry = step.get(str(int(token_id)))
        scores.append(_logprob_value(entry))
    return scores


# ---------------------------------------------------------------------------
# Record construction (pure)
# ---------------------------------------------------------------------------


def build_record(
    candidate,
    arm,
    post_choice,
    post_probabilities,
    rationale_tokens,
    truncated,
    gen_seconds,
    generated_tokens,
    *,
    batch_size=1,
    batch_id=None,
    batch_seconds=None,
    rationale_text=None,
    readout_fresh_seconds=None,
    engine="vllm",
    bi=True,
    cap=None,
    readout_error=None,
    spec=None,
):
    """The pilot record via :func:`reason_pilot.make_record` plus provenance.

    Identical schema to the HF harness (so ``reason_determinism_compare`` and the
    analyses work unchanged); ``engine``/``bi``/``cap`` are additive provenance.
    ``readout_error`` is present only when a re-read failed for that question.
    ``spec`` (the winning n-gram spec kwargs) is present only when the optional
    spec arm is enabled, so a non-spec artifact is byte-for-byte unchanged.
    """
    record = rp.make_record(
        candidate,
        arm,
        post_choice,
        post_probabilities,
        rationale_tokens,
        truncated,
        gen_seconds,
        generated_tokens,
        batch_size=batch_size,
        batch_id=batch_id,
        batch_seconds=batch_seconds,
        rationale_text=(
            rp.bound_rationale_text(rationale_text)
            if rationale_text is not None else None
        ),
        readout_fresh_seconds=readout_fresh_seconds,
    )
    record["engine"] = engine
    record["bi"] = bool(bi)
    record["cap"] = int(cap) if cap is not None else None
    if readout_error is not None:
        record["readout_error"] = readout_error
    if spec is not None:
        record["spec"] = spec
    return record


# ---------------------------------------------------------------------------
# Compare mode (pure; tested without vLLM)
# ---------------------------------------------------------------------------


def _record_margin(record, which="pre"):
    """``log p1 - log p2`` for a record's pre/post distribution, or ``None``."""
    if which == "pre":
        probs = record.get("pre_probabilities")
    else:
        probs = record.get("post_probabilities")
    if not probs:
        return None
    try:
        return rp.margin_from_probabilities(probs)
    except (ValueError, TypeError):
        return None


def _correct(record, choice):
    gold = record.get("gold")
    if gold is None or choice is None:
        return None
    return choice == gold


def compare_arms(reference_records, produced_records, max_examples=50):
    """Record-for-record ``post_choice`` agreement of one arm against another.

    ``reference_records`` is the stored HF arm; ``produced_records`` is the vLLM
    arm. A produced question absent from the reference is a failure; a reference
    question absent from the produced run is expected when the produced run is a
    ``--limit``ed prefix of the reference, and is reported as ``reference_only``
    without failing ``ok``. Mismatch examples carry both choices and the
    pre/post margins the task asks for. Pure and deterministic.
    """
    reference = {
        (r.get("run_id"), r.get("question_key")): r for r in reference_records
    }
    produced = {
        (r.get("run_id"), r.get("question_key")): r for r in produced_records
    }
    keys = sorted(set(reference) | set(produced), key=lambda k: (str(k[0]), str(k[1])))

    compared = matched = 0
    mismatches, missing_in_reference, reference_only = [], [], []
    for key in keys:
        if key not in produced:
            reference_only.append(key)
            continue
        if key not in reference:
            missing_in_reference.append(key)
            continue
        ref, prod = reference[key], produced[key]
        if ref.get("post_choice") is None or prod.get("post_choice") is None:
            continue
        compared += 1
        if ref.get("post_choice") == prod.get("post_choice"):
            matched += 1
            continue
        mismatches.append({
            "run_id": key[0],
            "question_key": key[1],
            "hf_choice": ref.get("post_choice"),
            "vllm_choice": prod.get("post_choice"),
            "gold": prod.get("gold"),
            "pre_choice": prod.get("pre_choice"),
            "pre_margin": _record_margin(prod, "pre"),
            "hf_post_margin": _record_margin(ref, "post"),
            "vllm_post_margin": _record_margin(prod, "post"),
            "hf_correct": _correct(ref, ref.get("post_choice")),
            "vllm_correct": _correct(prod, prod.get("post_choice")),
        })

    return {
        "kind": "vllm_reason_compare",
        "reference_questions": len(reference),
        "produced_questions": len(produced),
        "compared": compared,
        "agreement": matched,
        "agreement_rate": (matched / compared) if compared else 0.0,
        "mismatches": len(mismatches),
        "missing_in_reference": len(missing_in_reference),
        "reference_only": len(reference_only),
        "mismatch_examples": mismatches[:max_examples],
        "ok": bool(compared) and not mismatches and not missing_in_reference,
    }


def format_compare_report(report):
    """The human-readable comparison, with per-mismatch margins."""
    lines = [
        f"compared={report['compared']} agreement={report['agreement']} "
        f"({report['agreement_rate']:.4f}) mismatches={report['mismatches']} "
        f"missing_in_reference={report['missing_in_reference']} "
        f"reference_only={report['reference_only']}",
    ]
    for ex in report["mismatch_examples"]:
        lines.append(
            "  DIFF {run_id} {question_key}: hf={hf!r} vllm={vllm!r} "
            "gold={gold!r} pre_margin={pm} hf_post_margin={hm} "
            "vllm_post_margin={vm}".format(
                run_id=ex["run_id"], question_key=ex["question_key"],
                hf=ex["hf_choice"], vllm=ex["vllm_choice"], gold=ex["gold"],
                pm=_fmt(ex["pre_margin"]), hm=_fmt(ex["hf_post_margin"]),
                vm=_fmt(ex["vllm_post_margin"]),
            )
        )
    return "\n".join(lines)


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "None"


def compare_and_write(reference_path, produced_records, out_path=None):
    """Load the stored arm, compare, print, and optionally write the report."""
    reference_records = list(rp.load_jsonl(reference_path))
    report = compare_arms(reference_records, produced_records)
    report["reference"] = str(reference_path)
    print("== compare ==")
    print(format_compare_report(report))
    if out_path:
        _write_json(out_path, report)
    status = "OK" if report["ok"] else "FAILED"
    print(
        f"VLLM_REASON_COMPARE {status} compared={report['compared']} "
        f"agreement={report['agreement']} mismatches={report['mismatches']} "
        f"missing_in_reference={report['missing_in_reference']} "
        f"reference_only={report['reference_only']} out={out_path}"
    )
    return report


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# GPU-side functions (lazy vLLM import; not exercised on a tokenizer-only box)
# ---------------------------------------------------------------------------


def apply_engine_env(args):
    """Set the Route-B engine env **before** vLLM is imported; return them.

    ``VLLM_BATCH_INVARIANT=1`` is the whole point of Route B. ``--no-bi`` is the
    optional positive control. Mirrors :func:`vllm_probe.apply_env` and reuses
    its pins so the two never drift.
    """
    env = {
        "VLLM_BATCH_INVARIANT": "0" if getattr(args, "no_bi", False) else "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_DETERMINISM_WARMUP_ITERATIONS": vp.DETERMINISM_WARMUP_ITERATIONS,
        "VLLM_USE_V1": "1",
    }
    if getattr(args, "attention_backend", None):
        env["VLLM_ATTENTION_BACKEND"] = str(args.attention_backend)
    os.environ.update(env)
    return env


def _requested_kwargs(args, api_probe, batch_size):
    requested = {
        "model": args.model,
        "dtype": getattr(args, "dtype", DEFAULT_DTYPE),
        "tensor_parallel_size": int(getattr(args, "tensor_parallel_size", DEFAULT_TP)),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_prefix_caching": False,
        "max_num_seqs": int(max(int(getattr(args, "max_num_seqs", 0) or 0),
                               int(batch_size), 1)),
        "trust_remote_code": bool(getattr(args, "trust_remote_code",
                                          DEFAULT_TRUST_REMOTE_CODE)),
    }
    if getattr(args, "revision", None):
        requested["revision"] = args.revision
    if args.attention_backend and api_probe.get("llm_accepts_attention_backend"):
        requested["attention_backend"] = str(args.attention_backend)
    compilation_config = vp.make_compilation_config(args.cudagraph_mode)
    if compilation_config is not None:
        requested["compilation_config"] = compilation_config
    return requested


def load_llm(args, api_probe, batch_size):
    """Construct the vLLM engine tolerantly; raises on total failure.

    Without ``--spec-ngram`` this is exactly today's behaviour: one
    :func:`vllm_bakeoff.construct_llm` call with the resolved engine kwargs.
    With it, the ordered ``spec_kwarg_candidates("ngram")`` list is tried (each
    merged onto the base kwargs) until one constructs, and the winning kwargs
    plus every attempt are returned for self-describing run metadata.
    """
    from vllm import LLM

    base = _requested_kwargs(args, api_probe, batch_size)
    _tokens, candidates = resolve_spec(args)
    if not candidates:
        llm, accepted, dropped, notes, err = bk.construct_llm(LLM, base)
        if llm is None:
            raise RuntimeError(f"LLM construction failed: {err}")
        return llm, {"requested": vp.to_jsonable(base),
                     "accepted": vp.to_jsonable(accepted),
                     "dropped": dropped, "notes": notes, "spec": None}

    attempts = []
    llm = None
    merged = dict(base)
    accepted, dropped, notes, err = merged, [], [], "no candidates"
    for candidate in candidates:
        merged = dict(base)
        merged.update(candidate)
        llm, accepted, dropped, notes, err = bk.construct_llm(LLM, merged)
        attempts.append({
            "spec_kwargs": vp.to_jsonable(candidate),
            "ok": llm is not None,
            "error": err,
            "accepted": vp.to_jsonable(accepted),
            "dropped": dropped,
        })
        if llm is not None:
            break
    if llm is None:
        raise RuntimeError(f"spec LLM construction failed: {err}")
    return llm, {"requested": vp.to_jsonable(merged),
                 "accepted": vp.to_jsonable(accepted),
                 "dropped": dropped, "notes": notes,
                 "spec": vp.to_jsonable(candidate),
                 "spec_attempts": attempts}


def _generate(llm, inputs, sampling_params):
    """``LLM.generate`` tolerating the ``use_tqdm`` keyword's availability."""
    try:
        return llm.generate(inputs, sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(inputs, sampling_params)


def _token_prompt(ids):
    """vLLM input for pre-tokenised prompts (no retokenisation drift)."""
    return {"prompt_token_ids": [int(i) for i in ids]}


def _make_sampling_params(SamplingParams, **kwargs):
    """Build ``SamplingParams``, dropping any field this vLLM lacks.

    The resolved vLLM's field set is not guessed: unsupported keywords (e.g. if
    a version lacks ``logprob_token_ids`` or ``include_stop_str_in_output``) are
    dropped and logged rather than crashing the run. A dropped
    ``logprob_token_ids`` surfaces as a per-question ``readout_error``.
    """
    accepted, dropped = vp.filter_kwargs(SamplingParams, kwargs)
    if dropped:
        print(f"  [sampling] dropped unsupported SamplingParams fields: {dropped}")
    return SamplingParams(**accepted)


def generate_thoughts(llm, tokenizer, prompt_ids, cap, batch_size):
    """Greedy-decode fixed batches of thought prompts; one result per prompt.

    Returns a list of dicts (prompt order) with ``rationale``, ``truncated``,
    ``generated_tokens``, ``batch_seconds``, ``batch_id`` and ``batch_size``.
    Generation stops at ``<channel|>`` and the stop string is retained so the
    parsed stream matches the harness.
    """
    from vllm import SamplingParams

    sampling = _make_sampling_params(
        SamplingParams,
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(cap),
        stop=[rp.THINK_CLOSE],
        include_stop_str_in_output=True,
        ignore_eos=False,
    )
    results = []
    for batch_id, batch in enumerate(grouped_batches(len(prompt_ids), batch_size)):
        inputs = [_token_prompt(prompt_ids[index]) for index in batch]
        t0 = time.perf_counter()
        outputs = _generate(llm, inputs, sampling)
        seconds = time.perf_counter() - t0
        for output in outputs:
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            text = tokenizer.decode(ids, skip_special_tokens=False)
            rationale, parse_truncated = rp.parse_rationale(text)
            truncated = bool(
                parse_truncated or getattr(completion, "finish_reason", None) == "length"
            )
            results.append({
                "rationale": rationale,
                "generated_tokens": len(ids),
                "truncated": truncated,
                "batch_seconds": seconds,
                "batch_id": batch_id,
                "batch_size": len(batch),
            })
    return results


def readout_probs(llm, tokenizer, reread_ids, labels_per_item, label_ids_per_item,
                  batch_size):
    """Score option labels at the slot with vLLM ``logprob_token_ids``.

    ``labels_per_item`` / ``label_ids_per_item`` are the per-question positional
    labels and their verified single-token ids. The shared ``SamplingParams``
    requests the union of all needed ids (one greedy token per re-read prompt);
    each item's scores are extracted from its own step. Returns one dict per
    prompt: ``{"probabilities": {...}}`` or ``{"error": "..."}`` (never raises).
    """
    from vllm import SamplingParams

    union_ids = sorted({int(i) for ids in label_ids_per_item for i in ids})
    # Gated questions are capped at 26 options (section 10.1), so the union is
    # always inside the 128-id limit of ``logprob_token_ids``. ``logprobs`` is
    # deliberately left unset: vLLM 0.30 rejects sample logprobs > 20 while the
    # id-restricted readout returns exactly the requested entries (same pattern
    # as the fast pass's primary mechanism at scale).
    sampling = _make_sampling_params(
        SamplingParams,
        temperature=0.0,
        max_tokens=1,
        logprob_token_ids=union_ids,
    )
    results = [None] * len(reread_ids)
    for _batch_id, batch in enumerate(grouped_batches(len(reread_ids), batch_size)):
        inputs = [_token_prompt(reread_ids[index]) for index in batch]
        t0 = time.perf_counter()
        outputs = _generate(llm, inputs, sampling)
        seconds = time.perf_counter() - t0
        for offset, output in enumerate(outputs):
            index = batch[offset]
            try:
                completion = output.outputs[0]
                step = completion.logprobs[0] if completion.logprobs else None
                scores = label_scores_from_step(step, label_ids_per_item[index])
                results[index] = {
                    "probabilities": probs_from_scores(scores, labels_per_item[index]),
                    "seconds": seconds / max(len(batch), 1),
                }
            except Exception as exc:  # noqa: BLE001 - never abort a run
                results[index] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _versions():
    return vp._versions()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args):
    """Drive one arm end to end. Returns the records written."""
    config = resolve_config(args)
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    if batch_size < 1:
        raise SystemExit(f"--batch-size must be >= 1, got {batch_size}")
    cap = int(getattr(args, "cap", DEFAULT_CAP) or DEFAULT_CAP)
    rendering = getattr(args, "rendering", DEFAULT_RENDERING) or DEFAULT_RENDERING

    if not args.results or not config["rows"]:
        raise SystemExit("--results and (--rows or --w-rows) are required")

    try:
        selected, summary, join_stats = select_gated(
            args.results, config["rows"], config["coverage"], limit=args.limit
        )
    except ValueError as exc:
        print(f"gate selection found no eligible questions ({exc})")
        return []
    print("== gate ==")
    print(json.dumps({**summary, "rows": str(config["rows"]),
                      "arm": config["arm"], "join_stats": join_stats}, indent=2))
    if not selected:
        print("nothing gated; no output written")
        return []
    env_used = apply_engine_env(args)

    try:
        import vllm
        from vllm import LLM  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"could not import vllm: {type(exc).__name__}: {exc}")

    api_probe = vp.introspect_vllm(vllm)
    versions = _versions()
    llm, construct = load_llm(args, api_probe, batch_size)
    tokenizer = llm.get_tokenizer()
    engine_info = {"env": env_used, "versions": versions,
                   "llm_kwargs": construct["accepted"],
                   "llm_kwargs_dropped": construct["dropped"],
                   "construct_notes": construct["notes"]}
    if construct.get("spec"):
        engine_info["spec"] = construct["spec"]
        engine_info["spec_attempts"] = construct.get("spec_attempts", [])
    print("== engine ==")
    print(json.dumps(engine_info, indent=2))
    spec_used = construct.get("spec")

    thought_prompts = [
        rp.build_thought_prompt(tokenizer, c["state"], c["question"], rendering)
        for c in selected
    ]
    thought_ids = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in thought_prompts
    ]
    generated = generate_thoughts(llm, tokenizer, thought_ids, cap, batch_size)

    # Build the re-read prompt for every question (filler is length-matched to
    # the *generated* rationale, exactly as the harness does).
    reread_ids, labels_per_item, label_ids_per_item, rationales = [], [], [], []
    for index, (candidate, gen) in enumerate(zip(selected, generated)):
        state, question = candidate["state"], candidate["question"]
        keys = rp.question_keys(question)
        if getattr(args, "filler", False):
            target = len(tokenizer.encode(gen["rationale"], add_special_tokens=False)) or 1
            rationale, rationale_tokens, _reps = rp.build_filler(target, tokenizer)
        else:
            rationale = gen["rationale"]
            rationale_tokens = len(
                tokenizer.encode(rationale, add_special_tokens=False)
            )
        labels, label_ids = rp.option_label_token_ids(tokenizer, keys)
        reread_text = rp.build_reread_text(
            thought_prompts[index], rationale, state, question, rendering,
        )
        reread_ids.append(
            tokenizer(reread_text, add_special_tokens=False)["input_ids"]
        )
        labels_per_item.append(labels)
        label_ids_per_item.append(label_ids)
        rationales.append((rationale, rationale_tokens))

    readouts = readout_probs(
        llm, tokenizer, reread_ids, labels_per_item, label_ids_per_item, batch_size
    )

    records = []
    out_path = args.out or "vllm_reason_out.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for index, candidate in enumerate(selected):
            gen = generated[index]
            rationale, rationale_tokens = rationales[index]
            readout = readouts[index] or {"error": "no readout produced"}
            post_probs = readout.get("probabilities")
            post_choice = max(post_probs, key=lambda k: post_probs[k]) if post_probs else None
            record = build_record(
                candidate,
                config["arm"],
                post_choice,
                post_probs,
                rationale_tokens,
                gen["truncated"],
                gen["batch_seconds"] / max(gen["batch_size"], 1),
                gen["generated_tokens"],
                batch_size=gen["batch_size"],
                batch_id=gen["batch_id"],
                batch_seconds=gen["batch_seconds"],
                rationale_text=rp.bound_rationale_text(rationale),
                readout_fresh_seconds=readout.get("seconds"),
                engine="vllm",
                bi=not getattr(args, "no_bi", False),
                cap=cap,
                readout_error=readout.get("error"),
                spec=spec_used,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            records.append(record)
            if len(records) % 10 == 0:
                print(f"  reasoned {len(records)}/{len(selected)}")

    print(f"wrote {len(records)} records to {out_path}")
    stats = rp.generation_stats(records)
    if stats is not None:
        print(rp.format_stats_line(stats))

    if args.compare:
        compare_out = args.compare_out or (out_path + ".compare.json")
        compare_and_write(args.compare, records, compare_out)
    return records


def records_main(args):
    """Offline mode: load produced records and optionally compare."""
    records = list(rp.load_jsonl(args.records))
    print(f"loaded {len(records)} records from {args.records}")
    if args.compare:
        compare_out = args.compare_out
        compare_and_write(args.compare, records, compare_out)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="vLLM V1 (BI) reasoning runner: gate + greedy thought + "
                    "re-read, pilot schema."
    )
    p.add_argument("--results", help="fast-pass results.jsonl.gz (kit schema)")
    p.add_argument("--rows", help="panel rows JSONL(.gz); state/questions/expected")
    p.add_argument("--w-rows", dest="w_rows", default=None,
                   help="W breadth row cut; selects rows and defaults coverage to 1.0")
    p.add_argument("--coverage", type=float, default=None,
                   help=f"pre-registered coverage target; default {DEFAULT_COVERAGE} "
                        "(1.0 with --w-rows)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of gated questions (smoke)")
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help=f"max new thought tokens; default {DEFAULT_CAP}")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"fixed greedy batch width (BI-validated {BI_BATCH_WIDTHS}); "
                        f"default {DEFAULT_BATCH_SIZE}")
    p.add_argument("--arm", choices=list(ARMS), default=None,
                   help="arm tag recorded per record (default b1, b2 under "
                        "--filler, w with --w-rows)")
    p.add_argument("--filler", action="store_true",
                   help="B2: length-matched neutral filler instead of the rationale")
    p.add_argument("--rendering", default=DEFAULT_RENDERING,
                   choices=list(rp.RENDERINGS),
                   help=f"user-body layout; default {DEFAULT_RENDERING}")
    p.add_argument("--out", default=None, help="output JSONL path")

    # compare mode
    p.add_argument("--compare", default=None, metavar="HF_ARM_JSONL",
                   help="compare record-for-record against a stored HF arm")
    p.add_argument("--compare-out", dest="compare_out", default=None,
                   help="comparison report JSON path")
    p.add_argument("--records", default=None, metavar="VLLM_RECORDS_JSONL",
                   help="offline mode: load produced records (skip generation)")

    # engine
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--dtype", default=DEFAULT_DTYPE)
    p.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TP)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--gpu-memory-utilization", type=float,
                   default=DEFAULT_GPU_MEMORY_UTILIZATION)
    p.add_argument("--max-num-seqs", type=int, default=0,
                   help="vLLM max_num_seqs; 0 = max(batch-size, 1)")
    p.add_argument("--attention-backend", default=DEFAULT_ATTENTION_BACKEND)
    p.add_argument("--cudagraph-mode", default=DEFAULT_CUDAGRAPH_MODE)
    p.add_argument("--trust-remote-code", dest="trust_remote_code",
                   action="store_true", default=DEFAULT_TRUST_REMOTE_CODE)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code",
                   action="store_false")
    p.add_argument("--no-bi", dest="no_bi", action="store_true",
                   help="positive control: do NOT set VLLM_BATCH_INVARIANT")

    # optional n-gram speculative decoding (A/B experiment; off by default)
    p.add_argument("--spec-ngram", dest="spec_ngram", action="store_true",
                   help="enable n-gram speculative decoding on the BI engine "
                        "(off by default; non-spec behaviour is unchanged)")
    p.add_argument("--spec-ngram-tokens", dest="spec_ngram_tokens", type=int,
                   default=DEFAULT_SPEC_NGRAM_TOKENS,
                   help="n-gram prompt-lookup tokens; "
                        f"default {DEFAULT_SPEC_NGRAM_TOKENS}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.records:
        records_main(args)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
