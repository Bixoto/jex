#!/usr/bin/env python3
"""Stage-1 reasoning pilot harness (REASON_PILOT.md v2 + section 10 v3 amendments).

This module is the *scratch* harness for the margin-gated reasoning second pass
on ``google/gemma-4-31B-it``. It is deliberately **standalone and importable
without ``transformers``**: every pure function at module scope uses only the
standard library, and every model/tokenizer dependency is imported lazily inside
the GPU-only functions near the bottom.

Pipeline (``--results`` fast pass + ``--rows`` in -> JSONL out)
--------------------------------------------------------------
1. **Join** the fast-pass ``results.jsonl.gz`` to the row file the way the kit
   runner does: a result record carries the row's ``_evaluation`` fields spread
   at top level, so ``result["run_id"]`` equals ``row["_evaluation"]["run_id"]``.
   A kit-style id fallback (``match_row``) is kept for robustness.
2. **Gate** on the per-question *margin* ``log p1 - log p2`` computed from the
   fast-pass ``response.answers[q]["probabilities"]``. Only *index* catalogs
   (all catalog ids except ``NON_INDEX_CATALOGS``) are eligible, with the
   mechanistic exclusions from section 10.1: catalog 48 (Brier), catalogs 56/59
   (``noul`` probability guard), and questions with more than 26 options (Rune
   rule). ``noul`` questions and questions with no scoreable gold are skipped.
   The gate is a **pre-registered coverage target**, not an absolute tau: the
   cutoff is the nearest-rank percentile of the eligible margins, so the
   selected set is scale-free and transports across editions.
3. **Thought prompt** (section 10.2, Rune pattern): the same user body the fast
   pass uses (by default the **reference** layout of ``runs/full-rules``; kit is
   available via ``--rendering kit``) under the reasoning system prompt, rendered
   through the model's own chat template with ``enable_thinking=True``. Gemma-4's
   template injects ``<|think|>`` at the top of the system turn and, because
   thinking is enabled, leaves the generation prompt open at ``<|turn>model\\n``
   so the model generates its own ``<|channel>thought\\n...<channel|>`` block.
4. **Generation**: greedy (``do_sample=False``, ``num_beams=1``), at most
   ``--cap`` new tokens, stopped exactly at the close marker ``<channel|>``.
   ``truncated`` is True when the cap is reached without a close.
5. **Re-read** (generate-then-re-read only; never CoT-prefix scoring): append the
   thought block ``<|channel>thought\\n{rationale}\\n<channel|>``, then a second
   copy of the question/options body, then the answer slot ``\\nAnswer: (``.
   Read the next-token logits for the *allowed option letters* -- the positional
   labels ``A``, ``B``, ``C`` ... from ``engine/labels.py``'s candidate order,
   one per option, in criteria order -- and softmax over those labels only, then
   map back to the option keys.

Why positional labels: the fast-pass readout in the shipping engine scores the
verified single-token positional label under the ``paren`` slot (bare letters)
and maps position i back to the i-th option key. Reusing that mapping keeps the
second readout on the same numeric surface as the engine, and it is the only
mapping that works for every option-key naming scheme in the suite (``A``..,
``option_0``.., ``yes``/``no``). The system prompt asks for "one option letter"
for the same reason.

Rendering: the pilot's B0 is ``runs/full-rules`` (REASON_PILOT.md section 2/3),
which ran with ``rendering="reference"``, ``system_prompt=false``,
``slot="paren"``, ``keys="label"`` and annotate/neutralize/repeat/
spaced_instructions off. ``--rendering reference`` (the default) reproduces that
exact user body -- ``Context:`` / inline ``Question:`` / ``(A) key: text`` option
lines / ``Answer: (`` -- so B1/B2 are conditioned on the same question rendering
as B0. ``--rendering kit`` is kept for cross-checks. In both layouts the options
carry positional letters, because the re-read scores the bare positional label
under the paren slot (the engine's label readout).

B2 filler mode (``--filler``) replaces the generated rationale with a
token-length-matched neutral filler before the re-read. It still generates the
rationale first so the target length is measured on this machine/tokenizer; a
future ``--b1-out`` hook can reuse B1 lengths to skip that generation. The
generated rationale is discarded either way.

Generation timing (for the decode smoke's kill criteria): generation is greedy
and, by default, per-question (batch 1), so each output record carries
``gen_seconds`` and ``generated_tokens`` (the raw token count produced, including
the close marker). ``--batch-size N`` (N > 1) decodes fixed batches of N gated
questions in one pass, left-padded to the longest prompt; each record then also
carries ``batch_size``, ``batch_id`` and ``batch_seconds``, and ``gen_seconds``
becomes that batch's wall time divided by N. Batching is the measured
non-destructive speed-up (batch-4: 56.8 tok/s vs 16.1, token agreement 1.000);
it never samples and stops each sequence at the same close token. At the end of a
run the harness prints one grep-able line::

    REASON_STATS questions=<n> mean_gen_seconds=<x> mean_tokens=<y> tokens_per_second=<z>

where ``tokens_per_second = total generated tokens / sum of unique batch
seconds`` -- each batch's decode is counted once, never N times
(``generation_stats``/``format_stats_line``). Timing is recorded only; it never
affects a choice.

Determinism: no sampling anywhere. Generation is greedy (the batch-1 path is the
manual KV-cache loop with an exact close-token stop; the batched path is
``model.generate(do_sample=False, num_beams=1, max_new_tokens=cap)`` with the
close token passed as ``eos_token_id``, so the close token is retained in the
returned sequence). The re-read is a single forward. Cross-host answer-level
validation is a separate gate (section 10.5).

Readout A/B diagnostic (``--readout-mode``, default ``fresh`` = shipping
behavior). ``fresh`` re-prefills the whole re-read prompt and is timed as
``readout_fresh_seconds``. ``both`` runs that unchanged as
``post_choice``/``post_probabilities`` AND records a KV-cache-continuation readout
(``cont_choice``, ``cont_probabilities``, ``readout_cont_seconds``,
``continuation_ok``, ``continuation_note``): the continuation appends only the
re-read suffix after the generated thought prefix, reusing the generation cache.
``continue`` promotes the continuation result to ``post_*``. ``continue``/``both``
require ``--batch-size 1``. The continuation never crashes a run: any failure is
caught and recorded with ``continuation_ok=False``.

Checkpoint probe (``--checkpoint-probe``, default off; logging only). At
checkpoints scheduled every ``--probe-every`` generated tokens (default 32) and
immediately after each newline token, the batch-1 decode records the free
close-token ``rank``/``margin`` from the live last-token logits (zero extra
forwards). ``--probe-readout`` additionally deep-copies the live cache at each
checkpoint and runs the short suffix readout on the copy, recording the option
distribution, top choice and probe seconds. Generation is never changed and never
stops early; any probe failure is caught and recorded on the entry. The per-row
``checkpoints`` field is present only when the flag is on (off runs stay
byte-identical), and ``harness/analyze_checkpoints.py`` turns the arm into the
offline exit-potential curve. Requires ``--batch-size 1``.

Not implemented here (marked hooks at the bottom): B4 two-stage escalation and
B5 P(True) self-check.

Run a dry prompt construction (tokenizer only, no weights)::

    python3 harness/reason_pilot.py --model <snapshot-dir> --dry-run

Run the full pass (GPU VM)::

    python3 harness/reason_pilot.py \
        --results <fast-pass>/results.jsonl.gz \
        --rows subsets/full_edition.jsonl.gz \
        --model google/gemma-4-31B-it \
        --coverage 0.10 --cap 256 --out harness/out/reason-b1.jsonl
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import sys
import time

from _batching import grouped_batches, padded_batch, trim_generated  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gemma-4 thinking tokens, verified locally against the cached chat template
# (chat_template.jinja, dated 2026-07-09) and tokenizer.json.
THINK_TAG = "<|think|>"
THINK_OPEN = "<|channel>thought"
THINK_CLOSE = "<channel|>"

# REASON_PILOT.md section 10.2: reasoning system prompt.
THOUGHT_SYSTEM_PROMPT = (
    "You are solving a multiple-choice question. "
    "Reason step by step, then reply with one option letter."
)

# Section 10.1 hard exclusions.
# Catalogs present in the edition but carrying zero index weight (manifest
# `not_in_index_benchmarks` / `not_in_index_families`). These never gate.
NON_INDEX_CATALOGS = frozenset({6, 10, 24, 26, 27, 34})
# Mechanistic guards: 48 is Brier-scored (ForecastBench), 56/59 are the
# `noul` probability-scored catalogs. Thought must never touch these.
PROBABILITY_GUARD_CATALOGS = frozenset({48, 56, 59})
# Rune's rule: no question with more than 26 options takes the thought path.
MAX_OPTIONS = 26

DEFAULT_COVERAGE = 0.10
DEFAULT_CAP = 256
DEFAULT_BATCH_SIZE = 1
#: Readout A/B diagnostic (perf-cont). ``fresh`` is the shipping behavior: a
#: fresh forward over the whole re-read prompt. ``both`` keeps ``fresh`` as
#: ``post_choice``/``post_probabilities`` (arm semantics unchanged) and ALSO
#: records the KV-cache-continuation readout. ``continue`` makes the
#: continuation readout the ``post_*`` values (for a later scored-path A/B).
#: ``continue``/``both`` require ``--batch-size 1``.
READOUT_MODES = ("fresh", "continue", "both")
DEFAULT_READOUT_MODE = "fresh"
#: Checkpoint-probe instrumentation (PERF_RESEARCH.md §9.3 B1). When
#: ``--checkpoint-probe`` is on, the greedy decode records a free close-token
#: rank/margin at checkpoints scheduled every ``--probe-every`` generated tokens
#: (default 32) and immediately after every newline token. It is a logging-only
#: arm: it never stops generation and never changes the decode. ``--probe-readout``
#: additionally deep-copies the live cache at each checkpoint and runs the
#: short suffix readout on the copy. Both require ``--batch-size 1``.
DEFAULT_PROBE_EVERY = 32
#: Cap on the persisted rationale text. The re-read uses the full string; only
#: the stored copy is bounded, and the leakage classifier runs on this prefix
#: (a leaked answer pattern almost always appears early in the rationale).
RATIONALE_TEXT_LIMIT = 4000
COVERAGES = (0.05, 0.10, 0.15)
CAPS = (128, 256, 512)

# B2 neutral filler. Content-free by construction; only its token count matters.
NEUTRAL_FILLER = (
    "This is neutral filler text with no information about the question or its options."
)
FILLER_TOLERANCE = 0.05

_EMPTY_STATES = ("", None, {}, [])

# The candidate label order from engine/labels.py: A-Z, AA-ZZ, a-z.
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# Engine body rendering (reference + kit), adapted from engine/render.py.
#
# The pilot's B0 is ``runs/full-rules``, which ran with ``rendering="reference"``,
# ``system_prompt=false``, ``slot="paren"``, ``keys="label"`` (the label
# readout), and annotate/neutralize/repeat/spaced_instructions all off
# (``runs/full-rules/environment.json``). The thought prompt must reuse the same
# user body, so **reference is the default here**. Kit mode is kept for
# cross-checks.
#
# Both layouts label the options positionally (A, B, C ...) because the engine's
# label readout scores the bare positional label under the paren slot; the kit
# package's own ``render_prompt`` instead prints the option key and has no answer
# slot, so it is not reproduced.
#
# test_reason_prompt.py asserts byte-identity of each body against
# ``engine.render.render_reference`` / ``render_question`` when the engine
# package imports.
# ---------------------------------------------------------------------------

RENDERINGS = ("reference", "kit")
DEFAULT_RENDERING = "reference"
# Newlines between the body and the answer slot, exactly as each engine layout
# emits them: reference appends "\n" + slot; kit appends "\n\n" + slot.
_ANSWER_NEWLINES = {"reference": "\n", "kit": "\n\n"}


def text(x):
    """Identical to ``decision_index.engines.base.text``."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def question_keys(question):
    """The option keys in readout order: criteria order, or false/true for noul."""
    if question.get("type") == "choice":
        return list(question.get("criteria") or {})
    return ["false", "true"]


def _reference_txt(v):
    """``engine.render._reference_txt`` (decider/systemone.py:15-16)."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def annotate_indices(x, min_len=8):
    """``engine.render.annotate_indices`` (decider/systemone.py:19-31).

    Every list of length >= 8 becomes a list of ``{"_index": i, ...}`` records.
    Reference rendering always applies it (unlike kit rendering, where it is an
    opt-in arm).
    """
    if isinstance(x, list):
        if len(x) >= min_len:
            return [
                (
                    {"_index": i, **annotate_indices(v, min_len)}
                    if isinstance(v, dict)
                    else {"_index": i, "value": annotate_indices(v, min_len)}
                )
                for i, v in enumerate(x)
            ]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def reference_state(state):
    """``engine.render.reference_state`` (systemone.render_state)."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def reference_option(key, desc):
    """``engine.render.reference_option`` (systemone option string)."""
    if desc is None or desc == "":
        return key
    return f"{key}: {_reference_txt(desc)}"


def render_body(state, question, rendering=DEFAULT_RENDERING):
    """Question/options body with positional option labels, *no* answer slot.

    Reproduces the engine's body up to (but not including) its trailing answer
    slot:

    * ``reference`` -- ``engine.render.render_reference(state, question,
      addressing=[labels], slot="paren", channel="key")`` minus ``"\\nAnswer: ("``.
      Layout ``Context:`` / inline ``Question:`` / ``(A) key: text`` option lines.
    * ``kit`` -- ``engine.render.render_question(..., mode="kit",
      addressing=[labels], slot="paren")`` minus ``"\\n\\nAnswer: ("``. Layout
      ``State:`` / ``Question:`` / ``A: text`` option lines.
    """
    keys = question_keys(question)
    labels = [label for label, _ in option_labels(keys)]
    criteria = question.get("criteria") or {}
    if rendering == "reference":
        out = [
            "Context:\n",
            reference_state(state),
            f"\n\nQuestion: {text(question.get('instructions', ''))}\nOptions:",
        ]
        for label, key in zip(labels, keys):
            out.append(f"\n({label}) {reference_option(key, criteria.get(key))}")
        return "".join(out)
    if rendering == "kit":
        lines = [
            "State:",
            text(state) if state not in _EMPTY_STATES else "(empty)",
            "",
            "Question:",
            text(question.get("instructions", "")),
            "",
            "Options:",
        ]
        for label, key in zip(labels, keys):
            desc = criteria.get(key)
            shown = key if desc is None else text(desc)
            lines.append(f"{label}: {shown}")
        return "\n".join(lines)
    raise ValueError(f"unknown rendering {rendering!r}; expected one of {RENDERINGS}")


def answer_slot_text(slot="paren"):
    """The answer slot the re-read logits are read at.

    Only ``paren`` is used by the pilot (mirrors the engine's default arm): the
    prompt ends in ``"("`` so the label is a bare single token.
    """
    if slot != "paren":
        raise ValueError(f"reason_pilot only supports the 'paren' slot, got {slot!r}")
    return "Answer: ("


def render_fast_body(state, question, rendering=DEFAULT_RENDERING):
    """The fast pass user body: question/options body + the engine answer slot.

    Byte-identical to the engine's rendered body for the chosen ``rendering``
    (``render_reference`` for reference, ``render_question(mode="kit")`` for
    kit).
    """
    return render_body(state, question, rendering) + _ANSWER_NEWLINES[rendering] + answer_slot_text()


# ---------------------------------------------------------------------------
# Gate: margins, eligibility, coverage selection
# ---------------------------------------------------------------------------


def margin_from_probabilities(probabilities):
    """``log p1 - log p2`` in nats, where p1 >= p2.

    Matches ``analysis_out/reason_margin/gate_analysis.py`` and the gate-signal
    script: probabilities are floored at 1e-300 so a zero never raises.
    """
    values = sorted((max(float(v), 1e-300) for v in probabilities.values()), reverse=True)
    if len(values) < 2:
        raise ValueError("margin needs at least two option probabilities")
    return math.log(values[0]) - math.log(values[1])


def is_index_catalog(catalog_id):
    return catalog_id not in NON_INDEX_CATALOGS


def is_mechanically_excluded(catalog_id, n_options):
    """True when section 10.1 forbids the thought path for this question."""
    return catalog_id in PROBABILITY_GUARD_CATALOGS or n_options > MAX_OPTIONS


def eligible_question(catalog_id, question, gold):
    """Eligibility for the gate and the thought pass.

    Excludes, in order: non-index catalogs, catalog 48 / 56 / 59, non-choice
    questions, more than 26 options, and questions without a scoreable gold.
    """
    if not is_index_catalog(catalog_id):
        return False
    keys = question_keys(question)
    if question.get("type") != "choice":
        return False
    if is_mechanically_excluded(catalog_id, len(keys)):
        return False
    return gold is not None


def coverage_cutoff(margins, coverage):
    """Nearest-rank percentile cutoff for a pre-registered coverage target.

    ``coverage`` is the fraction of eligible questions to reason on. The cutoff
    is the ``ceil(coverage * N)``-th smallest margin (1-indexed), so the
    selected set is ``{q : margin(q) <= cutoff}`` and contains at least
    ``ceil(coverage * N)`` questions -- ties at the cutoff may push it slightly
    above the nominal target, which is documented and intentional (a quantile
    cutoff never silently drops tied questions).
    """
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    if not margins:
        raise ValueError("cannot compute a coverage cutoff over zero margins")
    ordered = sorted(float(m) for m in margins)
    k = max(1, int(math.ceil(float(coverage) * len(ordered))))
    return ordered[k - 1]


def select_by_coverage(candidates, coverage, key="margin"):
    """Return ``(selected, cutoff)`` for the lowest-margin ``coverage`` fraction.

    Deterministic: selected candidates are sorted by ``(margin, run_id,
    question_key)`` so output order does not depend on file order.
    """
    cutoff = coverage_cutoff([c[key] for c in candidates], coverage)
    selected = [c for c in candidates if float(c[key]) <= cutoff]
    selected.sort(key=lambda c: (float(c[key]), c.get("run_id", ""), c.get("question_key", "")))
    return selected, cutoff


# ---------------------------------------------------------------------------
# Positional label mapping (engine/labels.py candidate order)
# ---------------------------------------------------------------------------


def candidate_labels():
    """``A``-``Z``, then ``AA``..``ZZ``, then ``a``-``z``."""
    out = list(_UPPER)
    out.extend(a + b for a in _UPPER for b in _UPPER)
    out.extend(_LOWER)
    return out


def option_labels(keys):
    """Map ordered option keys to positional labels: ``[("A", keys[0]), ...]``.

    Position ``i`` -> label ``i`` in ``candidate_labels()`` order. This is the
    same positional contract the shipping engine's label readout uses.
    """
    labels = candidate_labels()[: len(keys)]
    if len(labels) < len(keys):
        raise ValueError(f"more option keys ({len(keys)}) than labels")
    return list(zip(labels, keys))


def label_slot_text(label, slot="paren"):
    """The exact string whose single-token id the re-read scores."""
    if slot != "paren":
        raise ValueError(f"reason_pilot only supports the 'paren' slot, got {slot!r}")
    return label


def softmax(values):
    """Numerically stable softmax over a plain sequence of floats."""
    if not values:
        raise ValueError("cannot softmax over zero values")
    top = max(values)
    weights = [math.exp(float(v) - float(top)) for v in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def map_label_probs_to_keys(label_probs, keys):
    """``{label: p}`` (or a label-indexed list) -> ``{option_key: p}`` in order."""
    if isinstance(label_probs, dict):
        labels = list(label_probs)
        probs = [label_probs[l] for l in labels]
    else:
        probs = list(label_probs)
        labels = [lab for lab, _ in option_labels(keys)]
    table = dict(option_labels(keys))
    return {table[label]: float(p) for label, p in zip(labels, probs)}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def apply_thought_template(tokenizer, body, system_prompt=THOUGHT_SYSTEM_PROMPT):
    """Render the user body through the model's chat template with thinking on.

    ``enable_thinking=True`` is the whole point: Gemma-4 injects ``<|think|>`` at
    the top of the first system turn and leaves the generation prompt at
    ``<|turn>model\\n`` (no empty thought block), so the model generates its own
    ``<|channel>thought\\n...<channel|>``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": body},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def build_thought_prompt(
    tokenizer,
    state,
    question,
    rendering=DEFAULT_RENDERING,
    system_prompt=THOUGHT_SYSTEM_PROMPT,
):
    """The generation prompt for one question (the fast pass's user body + thinking).

    The user turn is the fast-pass body for ``rendering`` (reference by default;
    label-addressed options ending in the ``Answer: (`` slot); the reasoning
    system prompt and ``enable_thinking=True`` tell the model to emit its
    ``<|channel>thought`` block before answering.
    """
    return apply_thought_template(
        tokenizer, render_fast_body(state, question, rendering), system_prompt
    )


def build_reread_text(
    thought_prompt,
    rationale,
    state,
    question,
    rendering=DEFAULT_RENDERING,
    slot="paren",
):
    """The re-read prompt: thought block + a second body copy + the answer slot.

    Layout (section 10.2)::

        <thought prompt ending in "<|turn>model\\n">
        <|channel>thought
        {rationale}
        <channel|>
        <question/options body>
        Answer: (

    ``thought_prompt`` already carries the system+user body, so the second
    ``render_body`` copy *is* the repeated question/options body. Appending the
    engine's own ``_ANSWER_NEWLINES`` + slot makes the tail byte-identical to
    the fast-pass body for the chosen ``rendering`` (for reference:
    ``...\\nAnswer: (``). The generated rationale is inserted verbatim between
    the channel markers -- no scoring of a CoT prefix, only a forward read of the
    next token after ``"Answer: ("``.
    """
    body = render_body(state, question, rendering)
    return (
        thought_prompt
        + THINK_OPEN
        + "\n"
        + rationale
        + "\n"
        + THINK_CLOSE
        + "\n"
        + body
        + _ANSWER_NEWLINES[rendering]
        + answer_slot_text(slot)
    )


def parse_rationale(generated_text):
    """Extract ``(rationale, truncated)`` from a decoded generation.

    Handles the model emitting the channel open itself (the normal path) and the
    degenerate path where the open is missing. ``truncated`` is True iff the
    close marker ``<channel|>`` is absent, i.e. the cap was hit.
    """
    if THINK_OPEN in generated_text:
        rest = generated_text.split(THINK_OPEN, 1)[1]
        if rest.startswith("\n"):
            rest = rest[1:]
    else:
        rest = generated_text
    if THINK_CLOSE in rest:
        return rest.split(THINK_CLOSE, 1)[0].rstrip("\n"), False
    return rest.rstrip("\n"), True


def bound_rationale_text(text, limit=RATIONALE_TEXT_LIMIT):
    """The rationale text actually inserted into the re-read, capped for storage.

    The re-read uses the full string; only the persisted copy is truncated to
    ``limit`` characters. Empty/``None`` becomes ``""``. The bound is documented
    on the record: the leakage classifier (REASON_PILOT.md §3) runs on this
    prefix.
    """
    if not text:
        return ""
    if limit is None or limit < 0:
        return text
    return text[:limit]


# ---------------------------------------------------------------------------
# B2 filler (length matching)
# ---------------------------------------------------------------------------


def within_tolerance(got, target, tolerance=FILLER_TOLERANCE):
    """True when ``got`` is within ``tolerance`` (relative) of ``target``."""
    if target <= 0:
        return got == 0
    return abs(got - target) <= tolerance * target


def filler_repetitions(target_tokens, unit_tokens):
    """How many copies of a phrase to emit to approximate ``target_tokens``.

    Pure arithmetic (no tokenizer): the caller supplies the phrase's own token
    count. At least one copy is always emitted.
    """
    target_tokens = int(target_tokens)
    unit_tokens = int(unit_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be > 0")
    return max(1, int(round(target_tokens / unit_tokens)))


def build_filler(target_tokens, tokenizer, phrase=NEUTRAL_FILLER, tolerance=FILLER_TOLERANCE):
    """Build neutral filler within ``tolerance`` of ``target_tokens``.

    Returns ``(filler_text, actual_tokens, repetitions)``. Uses the real
    tokenizer so the match is machine-faithful; ``+/- 5%`` is the B2 contract
    (section 10.2 / v2 B2). Searches nearby repetition counts and keeps the
    closest that satisfies the tolerance, else the closest found.
    """
    unit = len(tokenizer.encode(phrase, add_special_tokens=False))
    if unit <= 0:
        raise ValueError("neutral filler phrase tokenises to nothing")
    base = filler_repetitions(target_tokens, unit)
    best = None
    for n in range(max(1, base - 3), base + 4):
        candidate = " ".join([phrase] * n)
        got = len(tokenizer.encode(candidate, add_special_tokens=False))
        if best is None or abs(got - target_tokens) < abs(best[1] - target_tokens):
            best = (candidate, got, n)
        if within_tolerance(got, target_tokens, tolerance):
            return candidate, got, n
    return best


# ---------------------------------------------------------------------------
# JSONL and row/result loading (stdlib only)
# ---------------------------------------------------------------------------


def open_jsonl(path):
    """Open a plain or gzip-compressed JSONL file for text reading."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_jsonl(path):
    """Yield parsed JSON objects from a plain or gzip JSONL file."""
    with open_jsonl(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_row_index(rows):
    """Index rows by ``_evaluation.run_id`` and by ``id``.

    Returns ``(by_run_id, by_id)``. ``by_run_id`` is the primary join; ``by_id``
    feeds the kit-style fallback in :func:`match_row`.
    """
    by_run_id, by_id = {}, {}
    for row in rows:
        by_id[row.get("id")] = row
        run_id = (row.get("_evaluation") or {}).get("run_id")
        if run_id is not None:
            by_run_id[run_id] = row
    return by_run_id, by_id


def match_row(run_id, by_run_id, by_id):
    """Resolve a result record's ``run_id`` to its row, kit-style.

    Primary: exact ``_evaluation.run_id``. Fallback: the id-match the margin
    analyses use -- try the raw id, then strip one ``catalog:`` segment, then
    two. Never guesses beyond that.
    """
    if run_id in by_run_id:
        return by_run_id[run_id]
    if run_id in by_id:
        return by_id[run_id]
    parts = run_id.split(":", 1)
    if len(parts) == 2 and parts[1] in by_id:
        return by_id[parts[1]]
    parts = run_id.split(":", 2)
    if len(parts) == 3 and parts[2] in by_id:
        return by_id[parts[2]]
    return None


def build_candidates(results_path, rows_path):
    """Join results to rows and return eligible gate candidates.

    Each candidate is a dict: ``run_id, catalog_id, dataset, question_key,
    row, question, state, gold, pre_choice, pre_probabilities, margin``. Only
    ``status == "ok"`` records are read; only eligible questions are returned
    (see :func:`eligible_question`).
    """
    by_run_id, by_id = build_row_index(load_jsonl(rows_path))
    candidates = []
    stats = {
        "results_records": 0,
        "status_not_ok": 0,
        "unmatched_rows": 0,
        "missing_question": 0,
        "skipped_noul": 0,
        "skipped_catalog": 0,
        "skipped_options": 0,
        "skipped_no_gold": 0,
        "skipped_bad_margin": 0,
    }
    for record in load_jsonl(results_path):
        stats["results_records"] += 1
        if record.get("status") != "ok":
            stats["status_not_ok"] += 1
            continue
        run_id = record.get("run_id")
        row = match_row(run_id, by_run_id, by_id)
        if row is None:
            stats["unmatched_rows"] += 1
            continue
        catalog_id = record.get("catalog_id")
        expected = row.get("expected") or {}
        answers = (record.get("response") or {}).get("answers") or {}
        for question_key, answer in answers.items():
            question = (row.get("questions") or {}).get(question_key)
            if question is None:
                stats["missing_question"] += 1
                continue
            gold = expected.get(question_key)
            qtype = question.get("type")
            n_options = len(question_keys(question))
            if not is_index_catalog(catalog_id):
                stats["skipped_catalog"] += 1
                continue
            if qtype != "choice":
                stats["skipped_noul"] += 1
                continue
            if is_mechanically_excluded(catalog_id, n_options):
                stats["skipped_catalog" if catalog_id in PROBABILITY_GUARD_CATALOGS else "skipped_options"] += 1
                continue
            if gold is None:
                stats["skipped_no_gold"] += 1
                continue
            if answer.get("type") != "choice":
                stats["skipped_noul"] += 1
                continue
            probabilities = answer.get("probabilities") or {}
            try:
                margin = margin_from_probabilities(probabilities)
            except ValueError:
                stats["skipped_bad_margin"] += 1
                continue
            candidates.append(
                {
                    "run_id": run_id,
                    "catalog_id": catalog_id,
                    "dataset": record.get("dataset"),
                    "question_key": question_key,
                    "row": row,
                    "question": question,
                    "state": row.get("state"),
                    "gold": gold,
                    "pre_choice": answer.get("choice"),
                    "pre_probabilities": dict(probabilities),
                    "margin": margin,
                }
            )
    return candidates, stats


def make_record(
    candidate,
    arm,
    post_choice,
    post_probabilities,
    rationale_tokens,
    truncated,
    gen_seconds,
    generated_tokens,
    batch_size=1,
    batch_id=None,
    batch_seconds=None,
    rationale_text=None,
    readout_fresh_seconds=None,
    cont_choice=None,
    cont_probabilities=None,
    readout_cont_seconds=None,
    continuation_ok=None,
    continuation_note=None,
    checkpoints=None,
):
    """The per-question output record (pilot schema).

    ``gen_seconds`` is this question's share of its generation work. At batch
    size 1 it is the exact wall time of the greedy decode (backward compatible);
    in a batch of N it is the batch's wall time divided by N (an amortised
    per-question attribution).

    Batching provenance (added for ``--batch-size``):

    * ``batch_size`` -- the fixed batch width this question was decoded in;
    * ``batch_id``   -- the index of the batch within the run (questions sharing
      one decode share a ``batch_id``);
    * ``batch_seconds`` -- the exact wall time of that batch's decode, stored on
      every member so the aggregate throughput can count it **once**
      (:func:`generation_stats`). ``None`` means "legacy record, use
      ``gen_seconds``".

    ``generated_tokens`` is the raw number of tokens the decode produced
    (including the closing ``<channel|>`` when emitted), used for the end-of-run
    throughput; ``rationale_tokens`` is the parsed/stored rationale length (or
    the B2 filler length).

    ``rationale_text`` is the text actually inserted into the re-read (the
    parsed thought rationale, or the length-matched filler under ``--filler``),
    bounded to the first :data:`RATIONALE_TEXT_LIMIT` characters so records stay
    small. ``None`` on records written before this field existed.

    Readout A/B diagnostic (``--readout-mode``): ``readout_fresh_seconds`` is the
    wall time of the fresh re-prefill readout; ``cont_choice`` /
    ``cont_probabilities`` / ``readout_cont_seconds`` are the KV-cache
    continuation readout's answer, distribution and wall time;
    ``continuation_ok`` is ``True``/``False`` when it was attempted and ``None``
    when the run was ``fresh`` (not attempted); ``continuation_note`` carries the
    failure reason (or ``None``). All default to ``None`` so the ``fresh``
    default is byte-for-byte the previous record.

    ``checkpoints`` is the ``--checkpoint-probe`` instrumentation (a list of
    :func:`checkpoint_entry` dicts). It is ``None`` (the key is then absent, so a
    row written without the flag stays byte-identical) unless the probe ran.
    """
    record = {
        "run_id": candidate["run_id"],
        "catalog_id": candidate["catalog_id"],
        "question_key": candidate["question_key"],
        "arm": arm,
        "gold": candidate["gold"],
        "pre_choice": candidate["pre_choice"],
        "pre_probabilities": candidate["pre_probabilities"],
        "post_choice": post_choice,
        "post_probabilities": post_probabilities,
        "rationale_tokens": rationale_tokens,
        "rationale_text": rationale_text,
        "truncated": bool(truncated),
        "gen_seconds": float(gen_seconds),
        "generated_tokens": int(generated_tokens),
        "batch_size": int(batch_size),
        "batch_id": batch_id,
        "batch_seconds": (float(batch_seconds) if batch_seconds is not None else None),
        "readout_fresh_seconds": (
            float(readout_fresh_seconds) if readout_fresh_seconds is not None else None
        ),
        "cont_choice": cont_choice,
        "cont_probabilities": cont_probabilities,
        "readout_cont_seconds": (
            float(readout_cont_seconds) if readout_cont_seconds is not None else None
        ),
        "continuation_ok": continuation_ok,
        "continuation_note": continuation_note,
    }
    if checkpoints is not None:
        record["checkpoints"] = checkpoints
    return record


# ---------------------------------------------------------------------------
# Checkpoint probe (PERF_RESEARCH.md §9.3 B1: free close-token rank gate)
# ---------------------------------------------------------------------------


def checkpoint_due(pos, last_checkpoint, probe_every, is_newline):
    """True when a checkpoint is taken at generated position ``pos``.

    A checkpoint is due after the decode has produced at least ``probe_every``
    new tokens since the previous checkpoint, or immediately after a newline
    token was generated (the natural segment boundary the free rank gate keys
    on). Position 0 -- before any token exists -- is never a checkpoint. Pure.
    """
    if pos <= 0:
        return False
    return bool(is_newline) or (int(pos) - int(last_checkpoint)) >= int(probe_every)


def free_stop_signal(last_logits, close_id):
    """``(rank, margin)`` of the close token in the live last-token logits.

    ``rank`` is 1-based over the full vocabulary (``1`` means the close marker is
    the argmax, i.e. the model would stop *now*). ``margin`` is the close token's
    logit minus the best competing (non-close) logit, so it is the overall
    ``top1 - top2`` gap exactly when the close token is the argmax and negative
    otherwise. Pure: accepts a plain sequence or a 1-D torch tensor (torch is
    imported lazily and never mutates the input).
    """
    try:
        import torch

        if isinstance(last_logits, torch.Tensor):
            flat = last_logits.reshape(-1)
            close_logit = flat[int(close_id)]
            rank = int((flat > close_logit).sum().item()) + 1
            masked = flat.clone()
            masked[int(close_id)] = float("-inf")
            return rank, float((close_logit - masked.max()).item())
    except ImportError:  # pragma: no cover - torch absent is the stdlib-only box
        pass
    values = [float(v) for v in last_logits]
    cid = int(close_id)
    close_logit = values[cid]
    rank = 1
    best_other = float("-inf")
    for idx, value in enumerate(values):
        if idx == cid:
            continue
        if value > close_logit:
            rank += 1
        if value > best_other:
            best_other = value
    if best_other == float("-inf"):
        best_other = close_logit
    return rank, close_logit - best_other


def build_probe_suffix(state, question, rendering=DEFAULT_RENDERING, slot="paren"):
    """The suffix the probe appends to the live prefix (the re-read tail).

    Exactly the tail of :func:`build_reread_text` from the close marker on:
    ``<channel|>`` + newline + the repeated question/options body + the answer
    slot. Built from the same parts, not sliced out of the fragile re-read
    string, so it stays correct when the generated prefix is mid-stream.
    """
    return (
        THINK_CLOSE
        + "\n"
        + render_body(state, question, rendering)
        + _ANSWER_NEWLINES[rendering]
        + answer_slot_text(slot)
    )


def checkpoint_entry(pos, rank, margin, choice=None, probabilities=None,
                     seconds=None, error=None):
    """One ``checkpoints`` entry (schema shared with ``analyze_checkpoints.py``).

    ``choice``/``probabilities``/``seconds`` are ``None`` for a free-only
    checkpoint (no ``--probe-readout``). ``error`` is added only when a probe was
    attempted and failed; it never aborts the run.
    """
    entry = {
        "pos": int(pos),
        "rank": int(rank),
        "margin": float(margin),
        "choice": choice,
        "probabilities": probabilities,
        "seconds": (float(seconds) if seconds is not None else None),
    }
    if error is not None:
        entry["error"] = error
    return entry


# ---------------------------------------------------------------------------
# GPU-side functions (lazy imports; not exercised on a tokenizer-only box)
# ---------------------------------------------------------------------------


def load_tokenizer(model, revision=None):
    """Load only the tokenizer from a HF id or snapshot directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision)


def load_model(model, revision=None, device=None, dtype=None, attn="sdpa"):
    """Load the causal LM for greedy generation (lazy; weights live on the VM)."""
    import torch
    from transformers import AutoModelForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = "bfloat16" if device == "cuda" else "float32"
    model_obj = (
        AutoModelForCausalLM.from_pretrained(
            model, revision=revision, dtype=getattr(torch, dtype), attn_implementation=attn
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    return model_obj, device


def greedy_generate(model, tokenizer, prompt_text, cap, device):
    """Greedy decode at most ``cap`` new tokens, stopping at ``<channel|>``.

    Manual loop on purpose: it stops *exactly* at the close token id (no
    version-dependent ``stop_strings``), never samples, and needs no generation
    config. Returns ``(generated_text, hit_cap, n_generated)`` where
    ``generated_text`` is the decoded new tokens (including ``<channel|>`` when it
    was produced) and ``n_generated`` is the raw count of tokens produced.
    """
    import torch

    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(f"{THINK_CLOSE!r} is not a single token for this tokenizer")
    close_id = close_ids[0]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated = []
    past = None
    current = input_ids
    hit_cap = True
    # inference_mode is load-bearing: without it, each step's autograd graph is
    # retained through ``past_key_values`` and GPU memory grows with every token
    # until OOM (the 2026-10-02 smoke caught exactly this at ~93 GiB).
    with torch.inference_mode():
        for _ in range(int(cap)):
            out = model(current, past_key_values=past, use_cache=True)
            next_id = int(out.logits[:, -1, :].argmax(dim=-1)[0])
            generated.append(next_id)
            if next_id == close_id or (eos_id is not None and next_id == eos_id):
                hit_cap = False
                break
            past = out.past_key_values
            current = torch.tensor([[next_id]], device=device)
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, hit_cap, len(generated)


def greedy_generate_with_cache(model, tokenizer, prompt_text, cap, device):
    """``greedy_generate`` that also returns the ids and the KV cache.

    Used only by the ``--readout-mode continue|both`` diagnostic (batch-size 1).
    Identical decoding contract to :func:`greedy_generate` (greedy, exact close
    stop, ``inference_mode``), but it keeps the cache advanced through the close
    token and returns the raw ids so the continuation readout can append the
    re-read suffix to the *exact* generated prefix.

    Returns ``(text, hit_cap, n_generated, generated_ids, prompt_ids, cache)``:

    * ``generated_ids`` -- the tokens produced, including the stopping close/eos.
    * ``prompt_ids`` -- the tokenized thought prompt (``add_special_tokens=False``).
    * ``cache`` -- the ``past_key_values`` covering ``prompt_ids + generated_ids``
      (the manual loop assigns the cache *before* the stop check, unlike
      :func:`greedy_generate`, which discards it).

    ``greedy_generate`` is left untouched so the default path is provably
    unchanged; this mirrors it deliberately.
    """
    import torch

    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    prompt_ids = input_ids[0].tolist()
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(f"{THINK_CLOSE!r} is not a single token for this tokenizer")
    close_id = close_ids[0]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated = []
    past = None
    current = input_ids
    hit_cap = True
    with torch.inference_mode():
        for _ in range(int(cap)):
            out = model(current, past_key_values=past, use_cache=True)
            next_id = int(out.logits[:, -1, :].argmax(dim=-1)[0])
            generated.append(next_id)
            # Advance the cache to include the token just processed *before* the
            # stop check, so the returned cache covers the full prefix (prompt +
            # every generated token, close included).
            past = out.past_key_values
            if next_id == close_id or (eos_id is not None and next_id == eos_id):
                hit_cap = False
                break
            current = torch.tensor([[next_id]], device=device)
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, hit_cap, len(generated), generated, prompt_ids, past


def greedy_generate_checkpointed(
    model,
    tokenizer,
    prompt_text,
    cap,
    device,
    probe_every=DEFAULT_PROBE_EVERY,
    probe_readout=False,
    state=None,
    question=None,
    rendering=DEFAULT_RENDERING,
    slot="paren",
):
    """Greedy decode with checkpoint instrumentation (logging-only; never stops).

    Identical decoding contract to :func:`greedy_generate_with_cache` (greedy,
    exact close stop, ``inference_mode``), so the produced tokens are unchanged.
    At each checkpoint (:func:`checkpoint_due`: every ``probe_every`` tokens or
    just after a newline) it records the free close-token ``rank``/``margin`` from
    the live last-token logits -- no extra forward. With ``probe_readout`` it
    additionally **deep-copies** the live cache and runs the suffix readout on the
    copy (:func:`build_probe_suffix` + the shared continuation readout), recording
    the option distribution, top choice and probe seconds. Any probe failure is
    caught and recorded on the entry; generation proceeds untouched.

    Returns ``(text, hit_cap, n_generated, generated_ids, prompt_ids, cache,
    checkpoints)``.
    """
    import torch

    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    prompt_ids = input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(f"{THINK_CLOSE!r} is not a single token for this tokenizer")
    close_id = close_ids[0]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))

    keys = question_keys(question) if question is not None else []
    suffix_ids = None
    if probe_readout:
        suffix_text = build_probe_suffix(state, question, rendering, slot)
        suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]

    generated = []
    checkpoints = []
    past = None
    current = input_ids
    hit_cap = True
    last_checkpoint = 0
    with torch.inference_mode():
        for _ in range(int(cap)):
            out = model(current, past_key_values=past, use_cache=True)
            logits = out.logits[0, -1, :]
            new_past = out.past_key_values
            pos = len(generated)
            is_newline = bool(generated) and generated[-1] in newline_ids
            if checkpoint_due(pos, last_checkpoint, probe_every, is_newline):
                rank, margin = free_stop_signal(logits, close_id)
                entry = checkpoint_entry(pos, rank, margin)
                if probe_readout:
                    t_probe = time.perf_counter()
                    try:
                        try:
                            prefix_len = int(new_past.get_seq_length())
                        except AttributeError:
                            prefix_len = prompt_len + pos
                        cache_copy = copy.deepcopy(new_past)
                        probe_probs = reread_option_probs_continue(
                            model, tokenizer, suffix_ids, cache_copy, prefix_len,
                            keys, device, slot,
                        )
                        entry["probabilities"] = probe_probs
                        entry["choice"] = max(probe_probs, key=lambda k: probe_probs[k])
                    except Exception as exc:  # noqa: BLE001 - never break a run
                        entry["error"] = f"{type(exc).__name__}: {exc}"
                    entry["seconds"] = time.perf_counter() - t_probe
                checkpoints.append(entry)
                last_checkpoint = pos
            next_id = int(logits.argmax(dim=-1))
            generated.append(next_id)
            past = new_past
            if next_id == close_id or (eos_id is not None and next_id == eos_id):
                hit_cap = False
                break
            current = torch.tensor([[next_id]], device=device)
    text = tokenizer.decode(generated, skip_special_tokens=False)
    return text, hit_cap, len(generated), generated, prompt_ids, past, checkpoints


def stop_token_ids(close_id, eos_id=None):
    """The stop token ids for the batched generate path.

    ``greedy_generate`` stops at the close token ``<channel|>`` OR the tokenizer's
    real ``eos``; the batched path reproduces both by passing this list as
    ``eos_token_id`` (close first). Hugging Face keeps the matched stop token in
    the returned sequence, so semantics match the manual loop.
    """
    ids = [int(close_id)]
    if eos_id is not None and int(eos_id) not in ids:
        ids.append(int(eos_id))
    return ids


def batched_greedy_generate(model, tokenizer, prompt_texts, cap, device):
    """Greedy-decode a *fixed* batch, returning one result per prompt.

    The batch is left-padded to its longest prompt (the convention validated by
    the speed matrix) and decoded in one ``model.generate(do_sample=False,
    num_beams=1, max_new_tokens=cap)`` call that stops per sequence at the close
    token by passing :func:`stop_token_ids` as ``eos_token_id``. Hugging Face
    right-pads rows that finish early with ``pad_token_id``; that is trimmed and
    each sequence is cut at its stop token so the close/eos token is retained --
    the same contract as :func:`greedy_generate`.

    Returns ``[(generated_text, hit_cap, n_generated), ...]`` in prompt order.
    """
    import torch

    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    if len(close_ids) != 1:
        raise ValueError(f"{THINK_CLOSE!r} is not a single token for this tokenizer")
    close_id = close_ids[0]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
    stop_ids = stop_token_ids(close_id, eos_id)

    id_lists = [
        tokenizer(text, add_special_tokens=False)["input_ids"] for text in prompt_texts
    ]
    input_ids, attention_mask = padded_batch(id_lists, pad_id, device)
    width = input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(cap),
            eos_token_id=stop_ids,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
        )
    generated = out.sequences[:, width:]
    results = []
    for row in range(len(prompt_texts)):
        kept, stopped = trim_generated(
            generated[row].tolist(), stop_ids, pad_id=pad_id
        )
        text = tokenizer.decode(kept, skip_special_tokens=False)
        results.append((text, not stopped, len(kept)))
    return results


def option_label_token_ids(tokenizer, keys, slot="paren"):
    """Verified single-token ids for the positional labels of ``keys``.

    Shared by the fresh and continuation readouts so both mask the *same* allowed
    label ids. Returns ``(labels, token_ids)`` where ``labels`` is
    ``[(label, option_key), ...]``. A label that does not encode to exactly one
    token and round-trip is a hard error, never a silent misread.
    """
    labels = option_labels(keys)
    token_ids = []
    for label, _key in labels:
        ids = tokenizer.encode(label_slot_text(label, slot), add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label_slot_text(label, slot):
            raise ValueError(f"label {label!r} is not a verified single token for this tokenizer")
        token_ids.append(ids[0])
    return labels, token_ids


def probs_from_last_logits(last_logits, token_ids, labels):
    """Softmax over the allowed label positions, mapped back to option keys."""
    allowed = [float(last_logits[tid]) for tid in token_ids]
    probs = softmax(allowed)
    return {key: p for (_label, key), p in zip(labels, probs)}


def continuation_suffix_ids(prefix_ids, full_ids):
    """The readout suffix to append to a KV cache, or ``None`` on a prefix mismatch.

    Pure: ``full_ids`` is the tokenized re-read prompt, ``prefix_ids`` is the
    thought prompt plus the tokens actually generated (through the close). The
    continuation forward is only valid when the re-read prompt begins with
    exactly that generated prefix; a detokenize/retokenize drift (or an empty
    suffix) returns ``None`` so the caller records ``continuation_ok=False``
    instead of scoring a misaligned context.
    """
    prefix_ids = list(prefix_ids)
    full_ids = list(full_ids)
    if len(prefix_ids) > len(full_ids):
        return None
    if prefix_ids != full_ids[: len(prefix_ids)]:
        return None
    return full_ids[len(prefix_ids):]


def reread_option_probs(model, tokenizer, reread_text, keys, device, slot="paren"):
    """Single forward on the re-read prompt; softmax over the allowed labels only.

    Returns ``{option_key: probability}``. The label token ids are verified to be
    single-token round-trips; a tokenizer that fails that is a hard error, not a
    silent misread.
    """
    import torch

    labels, token_ids = option_label_token_ids(tokenizer, keys, slot)
    input_ids = tokenizer(reread_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    with torch.inference_mode():
        logits = model(input_ids).logits[0, -1, :]
    return probs_from_last_logits(logits, token_ids, labels)


def reread_option_probs_continue(
    model, tokenizer, suffix_ids, cache, prefix_len, keys, device, slot="paren"
):
    """Continuation readout: append only the re-read suffix to a generation cache.

    ``suffix_ids`` are the tokens of ``build_reread_text(...)`` past the generated
    thought prefix (see :func:`continuation_suffix_ids`); ``cache`` is that
    prefix's ``past_key_values``; ``prefix_len`` is its length. Masking, softmax
    and key mapping are identical to :func:`reread_option_probs` (same verified
    label ids).

    Call-shape verification against the installed transformers 5.17.0 sources
    (``models/gemma4/modeling_gemma4.py``):

    * ``Gemma4TextModel.forward`` (``modeling_gemma4.py:1620``) declares
      ``input_ids, attention_mask, position_ids, past_key_values, ..., use_cache``
      and **not** ``cache_position``. With ``position_ids=None`` it derives them
      from ``past_key_values.get_seq_length()`` (``:1656-1659``), exactly the
      prefix offset we need, and builds the causal/sliding masks from
      ``past_key_values`` + ``attention_mask`` + ``position_ids`` (``:1662-1675``).
    * There is no ``cache_position`` argument to forward; a stray kwarg would be
      threaded through the decoder layers into the attention interface (typed
      ``FlashAttentionKwargs``), so we deliberately do **not** pass it. The 2-D
      ``attention_mask`` must cover ``prefix_len + len(suffix_ids)`` (the cache's
      length plus the new tokens), which is what we build.
    """
    import torch

    labels, token_ids = option_label_token_ids(tokenizer, keys, slot)
    suffix = torch.tensor([list(suffix_ids)], dtype=torch.long, device=device)
    total = int(prefix_len) + len(suffix_ids)
    attention_mask = torch.ones((1, total), dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(
            input_ids=suffix,
            past_key_values=cache,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return probs_from_last_logits(out.logits[0, -1, :], token_ids, labels)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def gate_summary(candidates, coverage):
    """Build the gated set and a printable summary dict."""
    selected, cutoff = select_by_coverage(candidates, coverage)
    by_cat = {}
    for c in selected:
        by_cat[c["catalog_id"]] = by_cat.get(c["catalog_id"], 0) + 1
    return selected, {
        "eligible": len(candidates),
        "coverage": float(coverage),
        "cutoff_margin": cutoff,
        "gated": len(selected),
        "gated_fraction": (len(selected) / len(candidates)) if candidates else 0.0,
        "gated_by_catalog": dict(sorted(by_cat.items())),
    }


def sample_question():
    """A tiny built-in question so ``--dry-run`` works without results/rows."""
    state = {"note": "dry-run sample"}
    question = {
        "type": "choice",
        "instructions": "What is 2 + 2?",
        "criteria": {"A": "3", "B": "4", "C": "5"},
    }
    return state, question


def dry_run_report(selected, tokenizer, cap, filler, rendering, limit=3, model=None):
    """Construct and validate prompts with only the tokenizer, no weights.

    Prints the gate summary, then for up to ``limit`` questions: the tokenized
    thought prompt, the verified label token ids, and the re-read prompt with a
    placeholder rationale. Returns a dict of validation results.
    """
    print("== reason_pilot dry run (tokenizer only, no weights) ==")
    print(f"model: {model}")
    print(f"rendering: {rendering}; cap: {cap} new tokens; "
          f"mode: {'B2 filler' if filler else 'B1 thought'}")
    if selected:
        first = selected[0]["state"], selected[0]["question"]
    else:
        first = sample_question()
        print("no gated candidates in scope; using the built-in sample question")

    checked = 0
    for item in (selected[:limit] if selected else [None]):
        state, question = (item["state"], item["question"]) if item else first
        keys = question_keys(question)
        labels = option_labels(keys)

        thought_prompt = build_thought_prompt(tokenizer, state, question, rendering)
        prompt_ids = tokenizer(thought_prompt, add_special_tokens=False)["input_ids"]
        assert THINK_TAG in thought_prompt, "template did not inject the thinking tag"
        assert thought_prompt.endswith("<|turn>model\n"), "generation prompt is not open"

        label_ids = []
        for label, _key in labels:
            ids = tokenizer.encode(label_slot_text(label), add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids) != label:
                raise ValueError(f"label {label!r} failed the single-token round-trip")
            label_ids.append(ids[0])

        placeholder = "This is a dry-run placeholder rationale."
        reread = build_reread_text(thought_prompt, placeholder, state, question, rendering)
        reread_ids = tokenizer(reread, add_special_tokens=False)["input_ids"]
        if filler:
            _filler_text, filler_tokens, reps = build_filler(40, tokenizer)
            print(f"  filler: {reps} reps, {filler_tokens} tokens (target 40)")
        print(f"- question_key={None if item is None else item['question_key']} "
              f"catalog={None if item is None else item['catalog_id']} n_options={len(keys)}")
        print(f"  thought prompt tokens: {len(prompt_ids)}")
        print(f"  label mapping: {labels}")
        print(f"  label token ids: {label_ids}")
        print(f"  re-read prompt tokens: {len(reread_ids)}")
        print(f"  re-read tail: {reread[-120:]!r}")
        checked += 1
    print(f"validated {checked} prompt(s)")
    return {"checked": checked, "thought_prompt_tokens": len(prompt_ids), "reread_prompt_tokens": len(reread_ids)}


def run(args):
    """Drive the pilot end to end. Returns the output records written."""
    if args.coverage is not None and not 0.0 < args.coverage <= 1.0:
        raise SystemExit(f"--coverage must be in (0, 1], got {args.coverage}")
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    if batch_size < 1:
        raise SystemExit(f"--batch-size must be >= 1, got {batch_size}")
    readout_mode = getattr(args, "readout_mode", DEFAULT_READOUT_MODE) or DEFAULT_READOUT_MODE
    if readout_mode not in READOUT_MODES:
        raise SystemExit(f"--readout-mode must be one of {READOUT_MODES}, got {readout_mode!r}")
    if readout_mode != "fresh" and batch_size != 1:
        raise SystemExit(
            "--readout-mode continue/both requires --batch-size 1 (the KV-cache "
            "continuation readout reuses the batch-1 generation cache)"
        )
    checkpoint_probe = bool(getattr(args, "checkpoint_probe", False))
    probe_readout = bool(getattr(args, "probe_readout", False))
    raw_probe_every = getattr(args, "probe_every", DEFAULT_PROBE_EVERY)
    probe_every = (
        DEFAULT_PROBE_EVERY if raw_probe_every is None else int(raw_probe_every)
    )
    if probe_every < 1:
        raise SystemExit(f"--probe-every must be >= 1, got {probe_every}")
    if checkpoint_probe and batch_size != 1:
        raise SystemExit(
            "--checkpoint-probe requires --batch-size 1 (the live cache copy and "
            "the per-token checkpoint schedule are batch-1 only)"
        )
    if probe_readout and not checkpoint_probe:
        raise SystemExit("--probe-readout requires --checkpoint-probe")

    if args.results and args.rows:
        candidates, stats = build_candidates(args.results, args.rows)
        selected, summary = gate_summary(candidates, args.coverage)
        print("== gate ==")
        print(json.dumps({**summary, "join_stats": stats}, indent=2))
        if args.limit:
            selected = selected[: args.limit]
    else:
        selected, summary = [], {"eligible": 0, "gated": 0}
        print("== gate == (no --results/--rows supplied)")

    arm = args.arm or ("B2" if args.filler else "B1")

    tokenizer = load_tokenizer(args.model, revision=args.revision)

    if args.dry_run:
        dry_run_report(
            selected, tokenizer, args.cap, args.filler, args.rendering,
            limit=args.limit or 3, model=args.model,
        )
        return []

    if not selected:
        print("nothing gated; no output written")
        return []

    model, device = load_model(args.model, revision=args.revision, device=args.device)
    if batch_size > 1:
        print(f"== batching == batch_size={batch_size} "
              f"(fixed composition, left-padded, stop at {THINK_CLOSE!r})")
    records = []
    out_path = args.out or "reason_pilot_out.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    batches = grouped_batches(len(selected), batch_size)
    with open(out_path, "w", encoding="utf-8") as out:
        done = 0
        for batch_id, batch in enumerate(batches):
            batch_candidates = [selected[index] for index in batch]
            thought_prompts = [
                build_thought_prompt(
                    tokenizer, c["state"], c["question"], args.rendering
                )
                for c in batch_candidates
            ]
            # One timed decode per fixed batch. At batch_size 1 this is exactly
            # the old per-question `greedy_generate` wall time; for N>1 it is the
            # batch's wall time, amortised per question below.
            t0 = time.perf_counter()
            cont_info = None
            checkpoint_entries = None
            if batch_size == 1:
                if checkpoint_probe:
                    (
                        text,
                        hit_cap,
                        n,
                        gen_ids,
                        prompt_ids,
                        cache,
                        checkpoint_entries,
                    ) = greedy_generate_checkpointed(
                        model,
                        tokenizer,
                        thought_prompts[0],
                        args.cap,
                        device,
                        probe_every=probe_every,
                        probe_readout=probe_readout,
                        state=batch_candidates[0]["state"],
                        question=batch_candidates[0]["question"],
                        rendering=args.rendering,
                    )
                    generated = [(text, hit_cap, n)]
                    cont_info = {
                        "generated_ids": gen_ids,
                        "prompt_ids": prompt_ids,
                        "cache": cache,
                    }
                elif readout_mode == "fresh":
                    generated = [
                        greedy_generate(model, tokenizer, thought_prompts[0], args.cap, device)
                    ]
                else:
                    text, hit_cap, n, gen_ids, prompt_ids, cache = greedy_generate_with_cache(
                        model, tokenizer, thought_prompts[0], args.cap, device
                    )
                    generated = [(text, hit_cap, n)]
                    cont_info = {
                        "generated_ids": gen_ids,
                        "prompt_ids": prompt_ids,
                        "cache": cache,
                    }
            else:
                generated = batched_greedy_generate(
                    model, tokenizer, thought_prompts, args.cap, device
                )
            batch_seconds = time.perf_counter() - t0
            gen_seconds = batch_seconds / len(batch_candidates)

            for candidate, thought_prompt, (generated_text, hit_cap, generated_tokens) in zip(
                batch_candidates, thought_prompts, generated
            ):
                state, question = candidate["state"], candidate["question"]
                keys = question_keys(question)
                rationale, truncated = parse_rationale(generated_text)
                truncated = bool(truncated or hit_cap)

                if args.filler:
                    target = len(tokenizer.encode(rationale, add_special_tokens=False)) or 1
                    rationale, rationale_tokens, _n = build_filler(target, tokenizer)
                else:
                    rationale_tokens = len(tokenizer.encode(rationale, add_special_tokens=False))

                # `rationale` is now exactly the text inserted into the re-read
                # (parsed thought, or the length-matched filler under --filler).
                rationale_text = bound_rationale_text(rationale)
                reread_text = build_reread_text(
                    thought_prompt, rationale, state, question, args.rendering
                )

                # Fresh readout (the shipping behavior) -- timed, and left as the
                # record's post_* for fresh and both. The continuation diagnostic
                # never changes the fresh arm's semantics.
                post_probs = None
                post_choice = None
                readout_fresh_seconds = None
                if readout_mode in ("fresh", "both"):
                    t_read = time.perf_counter()
                    post_probs = reread_option_probs(model, tokenizer, reread_text, keys, device)
                    readout_fresh_seconds = time.perf_counter() - t_read
                    post_choice = max(post_probs, key=lambda k: post_probs[k])

                # KV-cache-continuation readout (diagnostic; batch-size 1 only).
                cont_choice = None
                cont_probabilities = None
                readout_cont_seconds = None
                continuation_ok = None
                continuation_note = None
                if readout_mode in ("continue", "both"):
                    t_cont = time.perf_counter()
                    try:
                        if cont_info is None:  # unreachable: guarded above by batch_size 1
                            raise RuntimeError("continuation requires the batch-1 generation cache")
                        full_ids = tokenizer(reread_text, add_special_tokens=False)["input_ids"]
                        prefix_ids = list(cont_info["prompt_ids"]) + list(
                            cont_info["generated_ids"]
                        )
                        suffix_ids = continuation_suffix_ids(prefix_ids, full_ids)
                        if not suffix_ids:
                            continuation_ok = False
                            continuation_note = (
                                "prefix mismatch: reread tokens do not start with the "
                                "generated token sequence (detokenize/retokenize drift or "
                                "empty suffix)"
                            )
                        else:
                            cont_probabilities = reread_option_probs_continue(
                                model,
                                tokenizer,
                                suffix_ids,
                                cont_info["cache"],
                                len(prefix_ids),
                                keys,
                                device,
                            )
                            cont_choice = max(
                                cont_probabilities, key=lambda k: cont_probabilities[k]
                            )
                            continuation_ok = True
                    except Exception as exc:  # noqa: BLE001 - must never break a run
                        cont_choice = None
                        cont_probabilities = None
                        continuation_ok = False
                        continuation_note = f"{type(exc).__name__}: {exc}"
                    readout_cont_seconds = time.perf_counter() - t_cont

                # `continue` promotes the continuation readout to the arm's answer
                # (for a later scored-path A/B); `both` keeps fresh as post_*.
                if readout_mode == "continue":
                    post_choice = cont_choice
                    post_probs = cont_probabilities

                record = make_record(
                    candidate,
                    arm,
                    post_choice,
                    post_probs,
                    rationale_tokens,
                    truncated,
                    gen_seconds,
                    generated_tokens,
                    batch_size=batch_size,
                    batch_id=batch_id,
                    batch_seconds=batch_seconds,
                    rationale_text=rationale_text,
                    readout_fresh_seconds=readout_fresh_seconds,
                    cont_choice=cont_choice,
                    cont_probabilities=cont_probabilities,
                    readout_cont_seconds=readout_cont_seconds,
                    continuation_ok=continuation_ok,
                    continuation_note=continuation_note,
                    checkpoints=checkpoint_entries,
                )
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                records.append(record)
                done += 1
                if done % 10 == 0:
                    print(f"  reasoned {done}/{len(selected)}")
    print(f"wrote {len(records)} records to {out_path}")
    stats = generation_stats(records)
    if stats is not None:
        # grep-able decode-throughput summary (measured, not estimated)
        print(format_stats_line(stats))
    return records


def generation_stats(records):
    """Measured decode-throughput stats over the output records, or ``None``.

    ``tokens_per_second`` is total generated tokens divided by the **sum of
    unique batch seconds**, not the sum of per-record ``gen_seconds``. Under
    batching (``--batch-size N>1``) a group of N questions shares one decode, so
    every member of the batch carries the same ``batch_seconds``; summing
    ``gen_seconds`` would count the same wall time N times and overstate the
    throughput by ~N. Counting each ``batch_id`` once measures the true decode
    rate. Records without batching provenance (legacy output) are treated as one
    batch each, so N=1 results are unchanged: the sum of unique batch seconds
    equals the sum of ``gen_seconds``.

    ``mean_tokens`` is the mean raw generated token count (the thought length
    including the close marker); ``mean_gen_seconds`` is total unique batch
    seconds / questions (per-question average latency). Pure and deterministic
    given the records.
    """
    if not records:
        return None
    n = len(records)
    total_tokens = sum(int(r["generated_tokens"]) for r in records)
    seen = {}
    for index, record in enumerate(records):
        batch_id = record.get("batch_id")
        if batch_id is None:
            batch_id = ("record", index)  # legacy/batchless: one batch per record
        if batch_id in seen:
            continue
        seconds = record.get("batch_seconds")
        if seconds is None:
            seconds = record.get("gen_seconds", 0.0)
        seen[batch_id] = float(seconds)
    total_seconds = sum(seen.values())
    return {
        "questions": n,
        "batches": len(seen),
        "mean_gen_seconds": total_seconds / n,
        "mean_tokens": total_tokens / n,
        "tokens_per_second": (total_tokens / total_seconds) if total_seconds > 0 else 0.0,
    }


def format_stats_line(stats):
    """The exact grep-able end-of-run summary line."""
    return (
        "REASON_STATS "
        f"questions={stats['questions']} "
        f"mean_gen_seconds={stats['mean_gen_seconds']:.3f} "
        f"mean_tokens={stats['mean_tokens']:.1f} "
        f"tokens_per_second={stats['tokens_per_second']:.2f}"
    )


# ---------------------------------------------------------------------------
# Marked hooks (do NOT implement now)
# ---------------------------------------------------------------------------


def b4_escalation_hook(*_args, **_kwargs):
    """TODO(B4): re-gate on the post-thought margin; a second thought only for
    residual-uncertain questions (~1/3 of thinkers); report the incremental fix.
    """
    raise NotImplementedError("B4 escalation is not part of the Stage-1 harness")


def b5_ptrue_hook(*_args, **_kwargs):
    """TODO(B5): a single-token True/False readout on (question, rationale,
    proposed answer); accept/reject and fold the signal. Never free-form revision.
    """
    raise NotImplementedError("B5 P(True) self-check is not part of the Stage-1 harness")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Stage-1 margin-gated reasoning pilot harness (no GPU required for --dry-run)."
    )
    p.add_argument("--results", help="fast-pass results.jsonl.gz (kit schema)")
    p.add_argument("--rows", help="subsets/full_edition.jsonl.gz (state/questions/expected)")
    p.add_argument("--model", required=True, help="HF id or local snapshot directory")
    p.add_argument("--revision", default=None, help="optional HF revision for --model")
    p.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help=f"pre-registered coverage target (one of {COVERAGES}); default {DEFAULT_COVERAGE}",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help=f"max new thought tokens (ablate {CAPS}); default {DEFAULT_CAP}",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="fixed-composition greedy batch width; 1 = per-question (default), "
             "4 = the measured non-destructive speed-up",
    )
    p.add_argument(
        "--readout-mode",
        choices=READOUT_MODES,
        default=DEFAULT_READOUT_MODE,
        help="readout A/B diagnostic: 'fresh' (default, shipping behavior), "
             "'continue' (KV-cache continuation is post_*), or 'both' (fresh "
             "post_* plus cont_* fields); continue/both require --batch-size 1",
    )
    p.add_argument(
        "--checkpoint-probe",
        action="store_true",
        help="log a free close-token rank/margin at checkpoints during generation "
             "(every --probe-every tokens and after each newline); logging only, "
             "never stops; requires --batch-size 1",
    )
    p.add_argument(
        "--probe-every",
        type=int,
        default=DEFAULT_PROBE_EVERY,
        help=f"checkpoint spacing in generated tokens; default {DEFAULT_PROBE_EVERY}",
    )
    p.add_argument(
        "--probe-readout",
        action="store_true",
        help="with --checkpoint-probe, also run the suffix option readout on a "
             "deep copy of the live cache at each checkpoint (requires "
             "--checkpoint-probe and --batch-size 1)",
    )
    p.add_argument("--filler", action="store_true", help="B2 mode: length-matched neutral filler")
    p.add_argument(
        "--rendering",
        choices=RENDERINGS,
        default=DEFAULT_RENDERING,
        help="user-body layout; default reference (the runs/full-rules B0 layout)",
    )
    p.add_argument("--arm", default=None, help="arm tag recorded per record (default B1/B2)")
    p.add_argument("--out", default=None, help="output JSONL path")
    p.add_argument("--dry-run", action="store_true", help="tokenizer-only prompt construction/validation")
    p.add_argument("--limit", type=int, default=None, help="cap the number of gated questions (smoke)")
    p.add_argument("--device", default=None, help="torch device override (default auto)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
