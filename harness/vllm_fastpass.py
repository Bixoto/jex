#!/usr/bin/env python3
"""Route-B vLLM fast pass: the label-slot readout on a vLLM V1 engine.

Why this exists
---------------
The current fast pass is the custom HF label-slot engine (``engine/`` on disk,
staged and installed as ``di_engine`` on a rented VM). It renders every option
of a question as a single verified label token at one fixed answer slot and
reads the option distribution from the model's next-token distribution at that
slot. Route B is a vLLM V1 engine (``VLLM_BATCH_INVARIANT=1``, ``TRITON_ATTN``,
``FULL_DECODE_ONLY``) that was shown to be token-identical across batch 1/8/16/32
on an RTX PRO 6000. This module reproduces the *same* scoring semantics on that
engine so one stack can run the whole suite.

How the semantics are reproduced
--------------------------------
The HF engine's ``mode="reference"`` renderer and its verified label table are
imported, not re-derived. For one question (single rotation, the fast pass):

1. Render the reference-layout body with the option keys addressed by the
   surviving single-token labels (``A``, ``B``, ...), ending at the answer slot
   (``Answer: (`` for ``paren``).
2. Wrap it in the model's chat template exactly as the HF engine does
   (``chat_wrap`` with ``split_slot=True`` for the reference layout, no system
   prompt) and tokenize with the model's own tokenizer.
3. Hand vLLM the token ids directly (``prompt_token_ids``) and read the
   log-probability of each label id at the first generated position. That
   position is conditioned on the whole prompt and is exactly the HF engine's
   block-diagonal row 0, which predicts every option's first token.
4. Softmax the single-token label log-probs over the option keys with the
   engine's own ``readout.normalise``. Nothing else is reimplemented.

The readout mechanism is *not guessed*. Several vLLM APIs can expose the
next-token log-probs (``logprob_token_ids`` at generation, ``prompt_logprobs``
on a sentinel continuation, top-k ``logprobs`` at generation). Each is attempted
in turn on a tiny probe prompt; the first that returns finite values for every
label id wins, and **every** attempt (success or failure) is recorded in the
report. A failure is never fatal.

Inputs / outputs
----------------
``--rows`` is a panel JSONL(.gz) (the kit's ``panel.jsonl.gz`` schema);
``--reference`` is the stored HF reference results JSONL(.gz) (the kit's
``results.jsonl``/``reference_panel_results.jsonl.gz`` schema). The runner emits
one JSONL record per panel row carrying the kit-schema
``response.answers[q].probabilities``, plus a validation report JSON against the
reference: per-question argmax agreement, mean/p95 |deltap|, per-catalog
breakdown, flip count and margin distribution.

Usage::

    python harness/vllm_fastpass.py \
        --rows /tmp/reason-fastpass/panel.jsonl.gz \
        --reference /tmp/reason-fastpass/reference_panel_results.jsonl.gz \
        --out /tmp/reason-fastpass/results.jsonl.gz \
        --report /tmp/reason-fastpass/validation.json

Merely importing this module never requires vLLM; the vLLM-dependent paths are
guarded and every failure is recorded, never raised out of :func:`run`.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import inspect
import json
import logging
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

log = logging.getLogger("vllm_fastpass")

# ---------------------------------------------------------------------------
# Fixed configuration of record (mirrors harness/vllm_bakeoff.py).
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "google/gemma-4-31B-it"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_ATTENTION_BACKEND = "TRITON_ATTN"
DEFAULT_CUDAGRAPH_MODE = "FULL_DECODE_ONLY"
DEFAULT_MAX_NUM_SEQS = 32
DEFAULT_BATCH_SIZE = 8
DEFAULT_TRUST_REMOTE_CODE = True
DEFAULT_TP = 1

#: The stored reference was produced with these arm options (see
#: runs/full-rules/environment.json); they are the defaults here so the
#: comparison is apples-to-apples. ``projection`` is not a knob on vLLM: the
#: engine's own bf16 ``lm_head`` is the only projection available offline, which
#: is exactly the reference's ``bf16_lmhead``.
DEFAULT_RENDERING = "reference"
DEFAULT_SLOT = "paren"
DEFAULT_LABELS = 255
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1

#: ``VLLM_BATCH_INVARIANT=1`` is the point of Route B, so it is always on.
BI_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_DETERMINISM_WARMUP_ITERATIONS": "3",
    "VLLM_USE_V1": "1",
}

#: The report schema (tests assert this exact set).
REPORT_KEYS = frozenset({
    "kind", "model", "dtype", "rendering", "slot", "labels_requested",
    "labels_valid", "max_model_len", "gpu_memory_utilization",
    "attention_backend", "cudagraph_mode", "enable_prefix_caching", "bi",
    "batch_size", "temperature", "env", "llm_kwargs", "llm_kwargs_dropped",
    "api_probe", "device", "versions", "mechanism", "mechanism_attempts",
    "n_rows_panel", "n_rows_scored", "n_questions_total",
    "n_questions_compared", "counters", "comparison", "errors",
    "wall_seconds_total", "fatal_error", "meta",
})

#: The per-row output record keys (tests assert this set).
RECORD_KEYS = frozenset({
    "run_id", "catalog_id", "engine", "status", "response",
    "total_wall_ms", "meta",
})


# ---------------------------------------------------------------------------
# Pure helpers (stdlib only; exercised by tests)
# ---------------------------------------------------------------------------


def load_jsonl(path):
    """Yield dicts from a JSONL file, transparently gzip-decoding ``.gz``.

    A missing file returns no rows (the caller records that as an error).
    """
    path = str(path)
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    except FileNotFoundError:
        log.warning("load_jsonl: no such file %s", path)
        return


def iter_jsonl(path):
    """List helper: materialise :func:`load_jsonl`."""
    return list(load_jsonl(path))


def argmax_distribution(probs):
    """The highest-probability key; ties break to the first in iteration order.

    This is the engine's own ``max(keys, key=lambda x: probs[x])``; both the
    output and the reference preserve criteria order, so ties resolve the same
    way on both sides.
    """
    if not probs:
        return None
    keys = list(probs)
    return max(keys, key=lambda k: probs[k])


def probability_margin(probs):
    """The top1 - top2 probability gap, or ``None`` for fewer than two options."""
    vals = sorted((float(v) for v in probs.values()), reverse=True)
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


def normalise_scores(scores, keys, temperature=DEFAULT_TEMPERATURE):
    """Softmax ``scores`` over ``keys`` using the engine's own normaliser.

    The engine builds every distribution through ``readout.normalise``; the
    import is lazy so this module loads with no engine/GPU present.
    """
    out = _engine_modules()
    return out["readout"].normalise(list(scores), list(keys), float(temperature))


def scores_to_probs(label_scores, keys, temperature=DEFAULT_TEMPERATURE):
    """Alias of :func:`normalise_scores`, named for the label-slot readout."""
    return normalise_scores(label_scores, keys, temperature)


def _quantile(sorted_vals, q):
    """Nearest-rank quantile of a pre-sorted list; ``None`` when empty."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def summarize_margins(margins):
    """Summary statistics for a list of top1-top2 margins (may contain None)."""
    vals = sorted(float(m) for m in margins if m is not None and math.isfinite(m))
    if not vals:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None,
                "lt_0.05": 0, "lt_0.10": 0, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p10": _quantile(vals, 0.10),
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "lt_0.05": sum(1 for v in vals if v < 0.05),
        "lt_0.10": sum(1 for v in vals if v < 0.10),
        "min": vals[0],
        "max": vals[-1],
    }


def _answer_probabilities(record):
    """Map ``(run_id, question_key) -> probability dict`` from one result row."""
    rid = record.get("run_id")
    resp = record.get("response") or {}
    answers = resp.get("answers") or {}
    out = {}
    for qkey, ans in answers.items():
        if not isinstance(ans, dict):
            continue
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and probs:
            out[(rid, qkey)] = probs
    return out


def compare_probabilities(out_records, ref_records):
    """Compare output distributions against a stored reference.

    Pure and model-free. Matches on ``(run_id, question_key)`` and reports, over
    the matched questions:

    * ``argmax_agreement`` -- fraction whose model argmax matches the reference;
    * ``flip_count`` -- questions whose argmax differs;
    * pooled ``|p_out - p_ref|`` (mean, p95, max) and the per-question max;
    * a per-catalog breakdown;
    * the margin distribution for both sides.

    Refuses to invent data: unmatched output/reference questions are counted,
    never scored.
    """
    ref_index = {}
    for rec in ref_records or []:
        ref_index.update(_answer_probabilities(rec))
    out_index = {}
    for rec in out_records or []:
        out_index.update(_answer_probabilities(rec))

    out_keys = set(out_index)
    ref_keys = set(ref_index)
    matched = out_keys & ref_keys

    catalog_of = {}
    for rec in out_records or []:
        catalog_of[rec.get("run_id")] = rec.get("catalog_id")
    for rec in ref_records or []:
        catalog_of.setdefault(rec.get("run_id"), rec.get("catalog_id"))

    deltas = []
    per_question_max = []
    matches = 0
    flips = 0
    margins_out = []
    margins_ref = []
    per_catalog = {}

    for key in sorted(matched):
        po = out_index[key]
        pr = ref_index[key]
        common = [k for k in pr if k in po]
        if not common:
            continue
        a_out = argmax_distribution(po)
        a_ref = argmax_distribution(pr)
        agree = a_out == a_ref
        matches += int(agree)
        flips += int(not agree)
        cat = catalog_of.get(key[0])
        bucket = per_catalog.setdefault(cat, {
            "n": 0, "agreement": 0, "flips": 0,
            "abs_delta_sum": 0.0, "abs_delta_n": 0,
        })
        bucket["n"] += 1
        bucket["agreement"] += int(agree)
        bucket["flips"] += int(not agree)
        q_max = 0.0
        for k in common:
            d = abs(float(po[k]) - float(pr[k]))
            deltas.append(d)
            bucket["abs_delta_sum"] += d
            bucket["abs_delta_n"] += 1
            if d > q_max:
                q_max = d
        per_question_max.append(q_max)
        mo, mr = probability_margin(po), probability_margin(pr)
        if mo is not None:
            margins_out.append(mo)
        if mr is not None:
            margins_ref.append(mr)

    n = len(matched)
    sorted_deltas = sorted(deltas)
    for bucket in per_catalog.values():
        bucket["agreement"] = (bucket["agreement"] / bucket["n"]) if bucket["n"] else None
        bucket["mean_abs_delta"] = (
            bucket["abs_delta_sum"] / bucket["abs_delta_n"]
            if bucket["abs_delta_n"] else None)
        bucket.pop("abs_delta_sum", None)
        bucket.pop("abs_delta_n", None)

    return {
        "n_output_questions": len(out_keys),
        "n_reference_questions": len(ref_keys),
        "n_compared": n,
        "n_output_only": len(out_keys - ref_keys),
        "n_reference_only": len(ref_keys - out_keys),
        "argmax_agreement": (matches / n) if n else None,
        "flip_count": flips,
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "p95_abs_delta": _quantile(sorted_deltas, 0.95) if sorted_deltas else None,
        "max_abs_delta": sorted_deltas[-1] if sorted_deltas else None,
        "mean_per_question_max_abs_delta": (
            sum(per_question_max) / len(per_question_max)
            if per_question_max else None),
        "per_catalog": {str(k): v for k, v in sorted(
            per_catalog.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "margin": {
            "vllm": summarize_margins(margins_out),
            "reference": summarize_margins(margins_ref),
        },
    }


def make_output_record(*, run_id, catalog_id, model, answers, total_wall_ms=None,
                       meta=None):
    """The per-row JSONL record (kit-schema ``response.answers``)."""
    return {
        "run_id": run_id,
        "catalog_id": catalog_id,
        "engine": "vllm_fastpass",
        "status": "ok",
        "response": {"model": model, "answers": answers},
        "total_wall_ms": total_wall_ms,
        "meta": meta or {},
    }


def make_report(*, model, dtype=DEFAULT_DTYPE, rendering=DEFAULT_RENDERING,
                slot=DEFAULT_SLOT, labels_requested=DEFAULT_LABELS,
                labels_valid=None, max_model_len=DEFAULT_MAX_MODEL_LEN,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                attention_backend=DEFAULT_ATTENTION_BACKEND,
                cudagraph_mode=DEFAULT_CUDAGRAPH_MODE,
                enable_prefix_caching=False, batch_size=DEFAULT_BATCH_SIZE,
                temperature=DEFAULT_TEMPERATURE, env=None, llm_kwargs=None,
                llm_kwargs_dropped=None, api_probe=None, device=None,
                versions=None, mechanism=None, mechanism_attempts=None,
                n_rows_panel=0, n_rows_scored=0, n_questions_total=0,
                n_questions_compared=0, counters=None, comparison=None,
                errors=None, wall_seconds_total=None, fatal_error=None,
                meta=None):
    """The validation report schema (pure; tests assert :data:`REPORT_KEYS`)."""
    return {
        "kind": "vllm_fastpass_validation",
        "model": model,
        "dtype": dtype,
        "rendering": rendering,
        "slot": slot,
        "labels_requested": int(labels_requested),
        "labels_valid": labels_valid,
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "attention_backend": attention_backend,
        "cudagraph_mode": cudagraph_mode,
        "enable_prefix_caching": bool(enable_prefix_caching),
        "bi": True,
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "env": dict(env or {}),
        "llm_kwargs": llm_kwargs or {},
        "llm_kwargs_dropped": list(llm_kwargs_dropped or []),
        "api_probe": api_probe or {},
        "device": device,
        "versions": versions or {},
        "mechanism": mechanism,
        "mechanism_attempts": list(mechanism_attempts or []),
        "n_rows_panel": int(n_rows_panel),
        "n_rows_scored": int(n_rows_scored),
        "n_questions_total": int(n_questions_total),
        "n_questions_compared": int(n_questions_compared),
        "counters": dict(counters or {}),
        "comparison": comparison or {},
        "errors": list(errors or []),
        "wall_seconds_total": wall_seconds_total,
        "fatal_error": fatal_error,
        "meta": meta or {},
    }


def _extract_logprob(value):
    """A vLLM ``Logprob`` (object, float or None) -> finite float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        val = getattr(value, "logprob", None)
        if val is None:
            if isinstance(value, dict):
                val = value.get("logprob")
            else:
                return None
        try:
            out = float(val)
        except (TypeError, ValueError):
            return None
    return out if math.isfinite(out) else None


def read_step_token_logprobs(step, label_ids):
    """Pull the label-id log-probs out of one vLLM log-prob step mapping.

    ``step`` may be ``None`` (mechanism unavailable), a ``{token_id: Logprob}``
    dict, or a list/dict keyed by token id. Missing ids yield ``None``; the
    caller treats any ``None`` as a failed readout.
    """
    if not step:
        return [None] * len(label_ids)
    get = step.get if hasattr(step, "get") else (lambda k, d=None: None)
    return [_extract_logprob(get(int(tid), None)) for tid in label_ids]


# ---------------------------------------------------------------------------
# Engine module loading (di_engine on the VM, engine in the repo)
# ---------------------------------------------------------------------------


def _engine_modules():
    """Import the label-slot engine's pure modules, preferring ``di_engine``.

    On the VM the engine is staged and installed as ``di_engine``; in the repo
    it lives at ``engine/``. Both use relative imports, so either package works.
    Raises ``ImportError`` only when neither is importable (which the caller
    records rather than crashing). Successful loads are cached.
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE
    last = None
    for pkg in ("di_engine", "engine"):
        try:
            _ENGINE_CACHE = {
                "pkg": pkg,
                "engine": importlib.import_module(pkg + ".engine"),
                "render": importlib.import_module(pkg + ".render"),
                "labels": importlib.import_module(pkg + ".labels"),
                "readout": importlib.import_module(pkg + ".readout"),
            }
            return _ENGINE_CACHE
        except Exception as exc:  # noqa: BLE001 - try the next package name
            last = exc
            continue
    raise ImportError(
        "could not import the label-slot engine as di_engine or engine: "
        f"{type(last).__name__}: {last}")


_ENGINE_CACHE = None


def build_label_table(modules, tokenizer, labels=DEFAULT_LABELS,
                      slot=DEFAULT_SLOT):
    """The engine's verified single-token label table for this tokenizer+slot."""
    return modules["labels"].build_label_table(
        tokenizer, int(labels), slot, log=log)


# ---------------------------------------------------------------------------
# Env + API introspection (vLLM imported lazily, inside run())
# ---------------------------------------------------------------------------


def apply_env(args):
    """Set the BI engine env vars **before** vLLM is imported; return them."""
    env = dict(BI_ENV)
    if args.attention_backend:
        env["VLLM_ATTENTION_BACKEND"] = str(args.attention_backend)
    os.environ.update(env)
    return env


def introspect_vllm(vllm_mod):
    """Delegate to the probe's real-API introspection."""
    try:
        import vllm_probe as vp
        return vp.introspect_vllm(vllm_mod)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}",
                "version": getattr(vllm_mod, "__version__", None)}


def filter_kwargs(cls, kwargs):
    """Split ``kwargs`` into the subset whose names ``cls.__init__`` accepts."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return dict(kwargs), []
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), []
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return accepted, sorted(k for k in kwargs if k not in accepted)


def make_compilation_config(mode):
    """Build the ``compilation_config`` for ``cudagraph_mode`` (dict or real)."""
    try:
        import vllm_probe as vp
        return vp.make_compilation_config(mode)
    except Exception:  # noqa: BLE001
        return {"cudagraph_mode": mode} if mode else None


def _versions():
    versions = {}
    for name in ("vllm", "torch", "transformers", "triton"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[name] = None
    try:
        import torch
        versions["cuda"] = getattr(torch.version, "cuda", None)
        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        pass
    return versions


# ---------------------------------------------------------------------------
# Prompt construction (reuses engine/render.py + engine/labels.py by import)
# ---------------------------------------------------------------------------


def build_question_request(modules, tokenizer, label_table, state, question,
                           *, rendering=DEFAULT_RENDERING, slot=DEFAULT_SLOT,
                           system_prompt=False, neutralize=False,
                           max_model_len=DEFAULT_MAX_MODEL_LEN):
    """Render one question into vLLM token ids plus its option labels.

    Returns ``(request, error)``. ``request`` is a dict with the prompt ids,
    the option keys, the per-option label ids and the probabilities key order;
    ``error`` is a short string when the question cannot be scored (too wide for
    the label table, or longer than the model context). Exactly one of the two
    is non-``None``.
    """
    qtype = question.get("type")
    if qtype == "choice":
        keys = list(question["criteria"])
        noul_true_key = None
    elif qtype == "noul":
        base = question.get("criteria") or {}
        false_desc = base.get("false", base.get(False))
        true_desc = base.get("true", base.get(True))
        question = {"type": "noul",
                    "instructions": question.get("instructions", ""),
                    "criteria": {"false": false_desc, "true": true_desc}}
        keys = ["false", "true"]
        noul_true_key = "true"
    else:
        return None, f"unsupported question type {qtype!r}"

    if not keys:
        return None, "question has no options"
    if len(keys) > len(label_table):
        return None, (f"key_fallback: {len(keys)} options exceed "
                      f"{len(label_table)} valid labels")
    if len(keys) < 1:
        return None, "question has no options"

    labels = [label_table.label(i) for i in range(len(keys))]
    label_ids = [label_table.token_id(i) for i in range(len(keys))]

    body = modules["render"].render_question(
        state, question, mode=rendering, keys=keys, addressing=labels,
        slot=slot, channel="key", prefill=None, annotate=False,
        neutralize=neutralize, repeat=False, spaced_instructions=False)
    text = modules["engine"].chat_wrap(
        tokenizer, body, system_prompt=system_prompt,
        split_slot=(rendering == "reference"))
    ids = modules["labels"].encode(tokenizer, text)
    # Reserve one token for the generated readout step and one for the sentinel
    # continuation used by the prompt_logprobs mechanism, so no readout
    # mechanism can overflow the context.
    if len(ids) + 2 > int(max_model_len):
        return None, (f"context_refusal: {len(ids)} prompt tokens + 2 exceed "
                      f"max_model_len={int(max_model_len)}")
    return {
        "ids": ids,
        "keys": keys,
        "labels": labels,
        "label_ids": label_ids,
        "noul_true_key": noul_true_key,
        "question_type": qtype,
    }, None


# ---------------------------------------------------------------------------
# vLLM readout mechanisms
# ---------------------------------------------------------------------------
#
# Each mechanism returns a list of log-probs aligned with ``label_ids`` or
# raises. The probe tries them in order and records every attempt.

MECHANISM_ORDER = (
    "generate_logprob_token_ids",
    "prompt_logprobs_sentinel",
    "generate_topk_logprobs",
)

# vLLM validates ``logprob_token_ids`` per request (max 128 on the 0.30
# engine); the full edition carries questions with up to 151 options, so wide
# label sets are read out in consecutive ``MAX_LOGPROB_IDS`` chunks and merged.
MAX_LOGPROB_IDS = 128


def _sampling_params(SamplingParams, **kwargs):
    """Construct ``SamplingParams``, preferring raw-logit log-probs.

    The HF reference distribution is a softmax over the model's raw logits at
    temperature 1.0. vLLM can compute log-probs from either the raw or the
    processed logits; asking for ``logprobs_mode="raw_logits"`` pins it to the
    reference. When the installed vLLM has no such field the request falls back
    to the default, and temperature is 1.0 regardless so a post-temperature
    normaliser still matches. Unsupported keywords raise (recorded by the
    caller).
    """
    try:
        return SamplingParams(logprobs_mode="raw_logits", **kwargs)
    except (TypeError, ValueError):
        return SamplingParams(**kwargs)


def _generate(llm, prompts, sampling_params):
    """Call ``LLM.generate`` tolerating the ``use_tqdm`` keyword's availability."""
    try:
        return llm.generate(prompts, sampling_params, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, sampling_params)


def _prompts_from_ids(id_lists):
    """``TokensPrompt`` dicts so the vLLM tokenizer never re-tokenizes."""
    return [{"prompt_token_ids": list(ids)} for ids in id_lists]


def mechanism_generate_logprob_token_ids(llm, SamplingParams, id_lists,
                                         label_ids):
    """Read label log-probs from generation step 0, restricted by token id."""
    sp = _sampling_params(
        SamplingParams, max_tokens=DEFAULT_MAX_TOKENS, temperature=1.0,
        logprob_token_ids=[int(t) for t in label_ids], ignore_eos=True)
    outputs = _generate(llm, _prompts_from_ids(id_lists), sp)
    return [_read_generated_step(o, label_ids) for o in outputs]


def mechanism_prompt_logprobs_sentinel(llm, SamplingParams, id_lists,
                                       label_ids, sentinel_id=0):
    """Append one sentinel token; the last prompt position predicts the labels.

    ``prompt_logprobs`` yields, for each prompt position, the distribution
    conditioned on the prefix. The sentinel's own entry is therefore the next
    token distribution *after* the real prompt, which is the answer slot.
    """
    sp = _sampling_params(
        SamplingParams, max_tokens=DEFAULT_MAX_TOKENS, temperature=1.0,
        prompt_logprobs=1, logprob_token_ids=[int(t) for t in label_ids],
        ignore_eos=True)
    extended = [list(ids) + [int(sentinel_id)] for ids in id_lists]
    outputs = _generate(llm, _prompts_from_ids(extended), sp)
    out = []
    for o in outputs:
        plp = getattr(o, "prompt_logprobs", None)
        step = plp[-1] if plp else None
        out.append(read_step_token_logprobs(step, label_ids))
    return out


def mechanism_generate_topk_logprobs(llm, SamplingParams, id_lists, label_ids):
    """Read label log-probs from the top-k distribution at generation step 0."""
    k = max(32, min(512, len(label_ids) + 64))
    sp = _sampling_params(
        SamplingParams, max_tokens=DEFAULT_MAX_TOKENS, temperature=1.0,
        logprobs=k, ignore_eos=True)
    outputs = _generate(llm, _prompts_from_ids(id_lists), sp)
    out = []
    for o in outputs:
        step = _generated_step(o)
        if not step:
            out.append([None] * len(label_ids))
            continue
        by_id = {}
        items = step.items() if hasattr(step, "items") else enumerate(step)
        for key, value in items:
            try:
                by_id[int(key)] = _extract_logprob(value)
            except (TypeError, ValueError):
                continue
        out.append([by_id.get(int(t)) for t in label_ids])
    return out


def _generated_step(out):
    """The step-0 log-prob mapping of a vLLM ``RequestOutput``, or ``None``."""
    outputs = getattr(out, "outputs", None)
    if not outputs:
        return None
    lp = getattr(outputs[0], "logprobs", None)
    if not lp:
        return None
    return lp[0]


def _read_generated_step(out, label_ids):
    return read_step_token_logprobs(_generated_step(out), label_ids)


def _mechanism_callable(name):
    return {
        "generate_logprob_token_ids":
            mechanism_generate_logprob_token_ids,
        "prompt_logprobs_sentinel":
            mechanism_prompt_logprobs_sentinel,
        "generate_topk_logprobs":
            mechanism_generate_topk_logprobs,
    }[name]


def probe_mechanism(llm, SamplingParams, tokenizer, label_ids, id_lists,
                    forced=None):
    """Find the readout mechanism that works; record *every* attempt.

    Returns ``(mechanism_name_or_None, attempts)``. A mechanism "works" when it
    returns a finite log-prob for every probed label id on every probed prompt.
    """
    attempts = []
    order = [forced] if forced else list(MECHANISM_ORDER)
    for name in order:
        attempt = {"mechanism": name, "ok": False, "error": None,
                   "already_forced": bool(forced)}
        try:
            fn = _mechanism_callable(name)
            results = fn(llm, SamplingParams, id_lists, label_ids)
            ok = bool(results) and all(
                row and all(v is not None for v in row) for row in results)
            attempt["ok"] = ok
            if ok:
                attempts.append(attempt)
                return name, attempts
        except Exception as exc:  # noqa: BLE001 - a failed probe is evidence
            attempt["error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(attempt)
    return None, attempts


def score_batch(llm, SamplingParams, mechanism, id_lists, label_ids):
    """Score a batch of prompts with the resolved mechanism.

    Returns ``(list_of_score_rows, error_or_None)``. ``label_ids`` wider than
    the engine's per-request cap (``MAX_LOGPROB_IDS``) is read out in
    consecutive chunks and each prompt's row is the concatenation of its chunk
    rows, in order; every mechanism therefore sees requests inside its limits.
    A batch failure is recorded and yields no rows.
    """
    if mechanism is None:
        return [], "no working readout mechanism"
    ids = [int(t) for t in label_ids]
    try:
        fn = _mechanism_callable(mechanism)
        if len(ids) <= MAX_LOGPROB_IDS:
            rows = fn(llm, SamplingParams, id_lists, ids)
            if any(len(row) != len(ids) for row in rows):
                return [], "readout rows misaligned with label ids"
            return rows, None
        merged = None
        for start in range(0, len(ids), MAX_LOGPROB_IDS):
            chunk = ids[start:start + MAX_LOGPROB_IDS]
            rows = fn(llm, SamplingParams, id_lists, chunk)
            if len(rows) != len(id_lists):
                return [], (f"chunked readout returned {len(rows)} rows for "
                            f"{len(id_lists)} prompts")
            if any(len(row) != len(chunk) for row in rows):
                return [], "chunked readout rows misaligned with chunk"
            if merged is None:
                merged = [list(row) for row in rows]
            else:
                for row, extra in zip(merged, rows):
                    row.extend(extra)
        return merged, None
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def _empty_counters():
    return {
        "rows": 0, "questions": 0, "scored": 0,
        "context_refusals": 0, "key_fallback_questions": 0,
        "unsupported_questions": 0, "readout_errors": 0,
        "batch_errors": 0, "mechanism_failures": 0,
    }


def _requested_kwargs(args, api_probe):
    requested = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enable_prefix_caching": False,
        "max_num_seqs": int(args.max_num_seqs),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.revision:
        requested["revision"] = args.revision
    if args.attention_backend and api_probe.get("llm_accepts_attention_backend"):
        requested["attention_backend"] = str(args.attention_backend)
    compilation_config = make_compilation_config(args.cudagraph_mode)
    if compilation_config is not None:
        requested["compilation_config"] = compilation_config
    return requested


def _construct_llm(LLM, requested):
    """Construct an ``LLM``, tolerating an unsupported ``compilation_config``.

    Returns ``(llm_or_None, accepted, dropped, notes, error_or_None)``.
    """
    accepted, dropped = filter_kwargs(LLM, requested)
    notes = []
    try:
        return LLM(**accepted), accepted, dropped, notes, None
    except Exception as exc:  # noqa: BLE001
        notes.append(f"first attempt failed: {type(exc).__name__}: {exc}")
        if "compilation_config" not in accepted:
            return None, accepted, dropped, notes, (
                f"LLM construction failed: {type(exc).__name__}: {exc}")
        retry = {k: v for k, v in accepted.items() if k != "compilation_config"}
        try:
            llm = LLM(**retry)
            notes.append("retried without compilation_config")
            return (llm, retry,
                    sorted(set(dropped) | {"compilation_config"}), notes, None)
        except Exception as exc2:  # noqa: BLE001
            return None, retry, dropped, notes, (
                f"LLM construction failed: {type(exc2).__name__}: {exc2}; "
                + " | ".join(notes))
    return None, accepted, dropped, notes, "unreachable"


def _resolve_tokenizer(llm, model, trust_remote_code):
    """The model's own tokenizer, via vLLM when possible, else transformers."""
    getter = getattr(llm, "get_tokenizer", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001
            pass
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model, trust_remote_code=bool(trust_remote_code))


def _probe_prompts(modules, tokenizer, label_ids):
    """Two tiny prompts that exercise the readout without touching the panel."""
    texts = [
        "Question: pick a colour.\nOptions:\nA: red\nB: blue\n\nAnswer: (",
        "Question: is it true?\nOptions:\nA: yes\nB: no\n\nAnswer: (",
    ]
    id_lists = [modules["labels"].encode(tokenizer, t) for t in texts]
    return id_lists


def run(args, env_used):
    """Execute the fast pass; always return a report (fatal errors recorded)."""
    started = time.perf_counter()
    versions = _versions()
    device = versions.get("device") or "unknown"
    counters = _empty_counters()
    errors = []
    common = dict(
        model=args.model, dtype=args.dtype, rendering=args.rendering,
        slot=args.slot, labels_requested=args.labels,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        cudagraph_mode=args.cudagraph_mode, batch_size=args.batch_size,
        temperature=args.temperature, env=env_used, device=device,
        versions=versions,
    )

    def _fail(message, **extra):
        kwargs = dict(common)
        kwargs.update(extra)
        kwargs["counters"] = counters
        kwargs["errors"] = errors
        kwargs["wall_seconds_total"] = time.perf_counter() - started
        kwargs["fatal_error"] = message
        report = make_report(**kwargs)
        # A failed run must still leave evidence for the bootstrap to upload.
        try:
            _write_json(args.report, report)
        except Exception:  # noqa: BLE001
            pass
        return report

    try:
        modules = _engine_modules()
    except Exception as exc:  # noqa: BLE001
        return _fail(f"engine import failed: {type(exc).__name__}: {exc}")

    panel_path = args.rows
    reference_path = args.reference
    if not panel_path:
        return _fail("no --rows panel supplied")
    try:
        panel = list(load_jsonl(panel_path))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"could not read panel {panel_path!r}: "
                     f"{type(exc).__name__}: {exc}")

    reference = []
    if reference_path:
        try:
            reference = list(load_jsonl(reference_path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reference read failed: {type(exc).__name__}: {exc}")

    rows = _filter_rows(panel, args)
    counters["rows"] = len(rows)
    counters["questions"] = sum(len(r.get("questions") or {}) for r in rows)

    try:
        import vllm
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        return _fail(f"could not import vllm: {type(exc).__name__}: {exc}",
                     n_rows_panel=len(panel))

    api_probe = introspect_vllm(vllm)
    requested = _requested_kwargs(args, api_probe)
    llm, accepted, dropped, notes, err = _construct_llm(LLM, requested)
    common.update(llm_kwargs=accepted, llm_kwargs_dropped=dropped,
                  api_probe=api_probe)
    if llm is None:
        return _fail(err or "LLM construction failed", n_rows_panel=len(panel),
                     meta={"construct_notes": notes})

    try:
        tokenizer = _resolve_tokenizer(
            llm, args.model, args.trust_remote_code)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"tokenizer resolution failed: {type(exc).__name__}: {exc}",
                     n_rows_panel=len(panel))

    try:
        label_table = build_label_table(
            modules, tokenizer, args.labels, args.slot)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"label table failed: {type(exc).__name__}: {exc}",
                     n_rows_panel=len(panel))
    labels_valid = len(label_table)
    try:
        label_desc = label_table.describe()
    except Exception:  # noqa: BLE001
        label_desc = {"labels_valid": labels_valid}

    max_options = max(
        (len(q.get("criteria") or {}) for r in rows
         for q in (r.get("questions") or {}).values()), default=1)
    max_label_ids = [label_table.token_id(i)
                     for i in range(min(max_options, labels_valid))]
    if not max_label_ids:
        return _fail("no valid label ids", n_rows_panel=len(panel),
                     labels_valid=labels_valid)

    probe_label_ids = max_label_ids[:int(MAX_LOGPROB_IDS)]
    probe_prompts = _probe_prompts(modules, tokenizer, probe_label_ids)
    mechanism, attempts = probe_mechanism(
        llm, SamplingParams, tokenizer, probe_label_ids, probe_prompts,
        forced=args.mechanism)
    if mechanism is None:
        counters["mechanism_failures"] += 1

    # ------------------------------------------------------------------ score
    out_records = []
    t_rows = time.perf_counter()
    for start in range(0, len(rows), int(args.batch_size)):
        chunk = rows[start:start + int(args.batch_size)]
        requests = []       # (row, run_id, catalog_id, qkey, question, request)
        for row in chunk:
            state = row.get("state")
            run_id = (row.get("_evaluation") or {}).get("run_id") or row.get("id")
            catalog_id = (row.get("_evaluation") or {}).get("catalog_id")
            for qkey, question in (row.get("questions") or {}).items():
                req, error = build_question_request(
                    modules, tokenizer, label_table, state, question,
                    rendering=args.rendering, slot=args.slot,
                    system_prompt=args.system_prompt,
                    neutralize=args.neutralize_options,
                    max_model_len=args.max_model_len)
                if error is not None:
                    if error.startswith("context_refusal"):
                        counters["context_refusals"] += 1
                    elif error.startswith("key_fallback"):
                        counters["key_fallback_questions"] += 1
                    else:
                        counters["unsupported_questions"] += 1
                    continue
                requests.append((row, run_id, catalog_id, qkey, question, req))

        if not requests:
            continue
        id_lists = [req["ids"] for *_, req in requests]
        rows_scores, batch_error = score_batch(
            llm, SamplingParams, mechanism, id_lists, max_label_ids)
        if batch_error:
            counters["batch_errors"] += 1
            errors.append(f"batch {start}: {batch_error}")
            continue
        if len(rows_scores) != len(requests):
            counters["batch_errors"] += 1
            errors.append(
                f"batch {start}: mechanism returned {len(rows_scores)} rows "
                f"for {len(requests)} prompts")
            continue

        # Assemble per-row answers.
        by_row = {}
        for (row, run_id, catalog_id, qkey, question, req), score_row in zip(
                requests, rows_scores):
            entry = by_row.setdefault(run_id, {
                "row": row, "catalog_id": catalog_id, "answers": {}})
            label_scores = score_row[:len(req["label_ids"])]
            if not label_scores or any(v is None for v in label_scores):
                counters["readout_errors"] += 1
                continue
            probs = normalise_scores(
                label_scores, req["keys"], args.temperature)
            if req["question_type"] == "noul":
                entry["answers"][qkey] = {
                    "type": "noul", "noul": probs.get(req["noul_true_key"], 0.5)}
            else:
                entry["answers"][qkey] = {
                    "type": "choice",
                    "choice": argmax_distribution(probs),
                    "probabilities": probs,
                }
                counters["scored"] += 1

        for run_id, entry in by_row.items():
            if not entry["answers"]:
                continue
            out_records.append(make_output_record(
                run_id=run_id, catalog_id=entry["catalog_id"],
                model=args.model, answers=entry["answers"],
                meta={"rendering": args.rendering, "slot": args.slot,
                      "labels_valid": labels_valid,
                      "mechanism": mechanism}))

    wall = time.perf_counter() - t_rows
    counters["rows"] = len(rows)
    counters["batching"] = {"batch_size": int(args.batch_size),
                            "wall_seconds": wall}

    _write_jsonl(args.out, out_records)

    comparison = compare_probabilities(out_records, reference) if reference \
        else {}
    n_compared = comparison.get("n_compared", 0) if comparison else 0
    report = make_report(
        model=args.model, dtype=args.dtype, rendering=args.rendering,
        slot=args.slot, labels_requested=args.labels, labels_valid=labels_valid,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        cudagraph_mode=args.cudagraph_mode, enable_prefix_caching=False,
        batch_size=args.batch_size, temperature=args.temperature, env=env_used,
        llm_kwargs=accepted, llm_kwargs_dropped=dropped, api_probe=api_probe,
        device=device, versions=versions, mechanism=mechanism,
        mechanism_attempts=attempts, n_rows_panel=len(panel),
        n_rows_scored=len(out_records), n_questions_total=counters["questions"],
        n_questions_compared=n_compared, counters=counters,
        comparison=comparison, errors=errors,
        wall_seconds_total=time.perf_counter() - started,
        meta={"construct_notes": notes, "labels": label_desc,
              "out": args.out})
    _write_json(args.report, report)
    return report


def _filter_rows(panel, args):
    """Apply ``--catalogs`` / ``--limit`` to the panel (order preserved)."""
    rows = list(panel)
    cats = getattr(args, "catalogs", None)
    if cats:
        want = {int(c) for c in cats}
        rows = [r for r in rows
                if (r.get("_evaluation") or {}).get("catalog_id") in want]
    limit = getattr(args, "limit", None)
    if limit:
        rows = rows[:int(limit)]
    return rows


# ---------------------------------------------------------------------------
# Reporting / IO
# ---------------------------------------------------------------------------


def _write_json(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    print(f"wrote {path}")


def _write_jsonl(path, records):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(records)} rows)")


def print_summary(report):
    if report.get("fatal_error"):
        print(f"FATAL: {report['fatal_error']}")
        return
    print(f"model={report['model']} rendering={report['rendering']} "
          f"slot={report['slot']} labels_valid={report['labels_valid']} "
          f"mechanism={report['mechanism']}")
    print("mechanism attempts:")
    for a in report.get("mechanism_attempts") or []:
        print(f"  {a['mechanism']}: ok={a['ok']} error={a.get('error')}")
    c = report.get("comparison") or {}
    if c:
        print(f"compared={c.get('n_compared')} "
              f"argmax_agreement={c.get('argmax_agreement')} "
              f"flips={c.get('flip_count')} "
              f"mean|dp|={c.get('mean_abs_delta')} p95={c.get('p95_abs_delta')}")
        for cat, b in (c.get("per_catalog") or {}).items():
            print(f"  catalog {cat}: n={b['n']} agreement={b['agreement']} "
                  f"flips={b['flips']} mean|dp|={b['mean_abs_delta']}")
    print(f"counters: {report.get('counters')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Route-B vLLM fast pass (label-slot readout) + validation "
                    "against a stored HF reference.")
    p.add_argument("--rows", default=None,
                   help="panel JSONL(.gz) to score (required to run)")
    p.add_argument("--reference", default=None,
                   help="stored HF reference results JSONL(.gz) to compare")
    p.add_argument("--out", default=None, help="output results JSONL(.gz)")
    p.add_argument("--report", default=None, help="validation report JSON")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="optional HF revision")
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
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="panel rows per vLLM generate call (default 8)")

    p.add_argument("--rendering", default=DEFAULT_RENDERING,
                   help="engine rendering mode (default reference)")
    p.add_argument("--slot", default=DEFAULT_SLOT,
                   help="answer slot: paren/space/prefill (default paren)")
    p.add_argument("--labels", type=int, default=DEFAULT_LABELS,
                   help="label candidates to verify (default 255)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--system-prompt", dest="system_prompt",
                   action="store_true", default=False,
                   help="prepend the engine system prompt (reference: off)")
    p.add_argument("--neutralize-options", action="store_true", default=False,
                   help="rewrite exact abstain option text to the neutral literal")
    p.add_argument("--mechanism", default=None, choices=list(MECHANISM_ORDER),
                   help="force a readout mechanism (default: probe in order)")

    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N panel rows")
    p.add_argument("--catalogs", type=int, nargs="*", default=None,
                   help="restrict to these catalog ids")
    return p


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    if not args.rows:
        args.rows = None
    env_used = apply_env(args)
    report = run(args, env_used)
    print_summary(report)
    # Never fatal: a failed run is evidence, not a crash. The bootstrap decides.
    return 0


if __name__ == "__main__":
    sys.exit(main())
