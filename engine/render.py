"""State and question rendering.

The ``kit`` mode is a byte-for-byte reimplementation of
``decision_index.engines.transformers_engine.render_prompt`` -- ``test_local.py``
asserts that equality against the installed kit on real suite rows, so Arm A
really is the baseline and not an approximation of it.

The two alternative modes exist because every entrant inherits ``text(state)``,
which JSON-serialises a dict into one long line where a 10,000-element list is
addressable only by counting commas. Both alternatives address structure
explicitly. Neither of them changes which options are offered, and neither
truncates anything.

State always precedes the question. Reversing that costs measured index.
"""

import json
import re

from .config import DEFAULT_PREFILL_PREFIX

SYSTEM_PROMPT = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain."

# decider/infer.py:30-51. The reference rewrites an option whose text reads like
# an abstain signal to a neutral literal, because the model learned the exact
# "none of the above" string as an abstain cue. The task spec narrows the match
# to an exact, whole-string, case-insensitive equality with these three; the
# reference's own `startswith("none of the above")` and its extra "none of these"
# are intentionally NOT reproduced (see the report).
NEUTRAL_NONE = "not listed here"
NEUTRAL_MATCHES = ("none of the above", "none", "n/a")


def neutralize_text(value):
    """Rewrite an exact (whole-string, case-insensitive) abstain option.

    ``"And Then There Were None"`` is not a match: it is neither equal to
    "none" nor a prefix kept as exact equality.
    """
    if isinstance(value, str) and value.strip().lower() in NEUTRAL_MATCHES:
        return NEUTRAL_NONE
    return value


def display_option_text(key, desc):
    """The option's rendered word/text: its description, or the key when bare."""
    return key if desc is None else text(desc)


def answer_slot_text(slot, prefill=None):
    """The final line of the prompt, whose last token the label is read after.

    ``paren`` ends in "(" so the label is a bare letter token; ``space`` ends in
    ":" and the label carries the leading space. ``prefill`` appends the
    prefilled-slot arm's natural-language lead-in to "Answer: " and, like
    ``space``, leaves the label to carry the space -- so the prompt's last token
    stays the cue's ':' and the readout position is unchanged.
    """
    if slot == "paren":
        return "Answer: ("
    if slot == "space":
        return "Answer:"
    if slot == "prefill":
        return "Answer: " + (DEFAULT_PREFILL_PREFIX if prefill is None else prefill)
    raise ValueError(f"unknown slot {slot!r}")

EMPTY_STATES = ("", None, {}, [])
_JSONISH = re.compile(r"^\s*[\[{]")
_TOKENISH = re.compile(r"\w+|[^\w\s]")


def text(x):
    """Identical to ``decision_index.engines.base.text``."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def token_estimate(s):
    """Cheap, tokenizer-free token estimate, used only for the load-time warning."""
    return len(_TOKENISH.findall(s)) + s.count("\n")


# ---------------------------------------------------------------- kit rendering


def repeat_body(body):
    """Emit the rendered body twice before a single answer slot (prompt repetition).

    The final line is always the answer slot, so the boundary is its last newline.
    """
    i = body.rfind("\n")
    if i < 0:
        return body + "\n\n" + body
    pre, slot = body[:i], body[i + 1 :]
    return pre + "\n" + pre + "\n" + slot


def annotate_state(state):
    """The state with ``{"_index": i}`` written into every array of length >= 8."""
    if isinstance(state, str) or state in EMPTY_STATES:
        return state
    return annotate_indices(state)


def render_state(state, mode="kit", annotate=False):
    if annotate and not isinstance(state, str) and state not in EMPTY_STATES:
        state = annotate_indices(state)
    if mode == "kit":
        return text(state) if state not in EMPTY_STATES else "(empty)"
    if mode == "reference":
        # The Decision Index entry reaches the reference through its
        # ``/v1/systemone`` server (kit http.py:38-39 -> decider/serve.py:124),
        # which renders the state with ``systemone.render_state``
        # (decider/systemone.py:34-37): a string passes through untouched, any
        # other value is ``json.dumps(annotate_indices(state), ensure_ascii=False)``
        # -- default separators, and ``annotate_indices`` on by default.
        return reference_state(state)
    if mode == "keyaddressed":
        return "(empty)" if state in EMPTY_STATES else keyaddressed(state)
    if mode == "prose":
        return "Nothing." if state in EMPTY_STATES else prose(state)
    raise ValueError(f"unknown rendering mode {mode!r}")


def render_instructions(question, spaced=False):
    """The rendered instruction text.

    ``spaced`` reproduces ``systemone._txt``: a non-string value is serialised
    with stdlib default separators (``", "`` / ``": "``) rather than our compact
    ``text()``. Strings are unchanged either way.
    """
    v = question.get("instructions", "")
    if spaced and not isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return text(v)


def render_option_lines(question, keys, addressing, neutralize=False):
    """One line per option, in the kit's ``key: description`` shape.

    ``addressing`` supplies the left-hand side of each line: a label for the
    label readout, or the option key itself. ``neutralize`` rewrites an exact
    abstain option text to ``NEUTRAL_NONE`` (the reference transform); the
    option's key -- what the answer is reported as -- never changes.
    """
    criteria = question.get("criteria") or {}
    out = []
    for i, key in enumerate(keys):
        desc = criteria.get(key)
        shown = display_option_text(key, desc)
        if neutralize:
            shown = neutralize_text(shown)
        out.append(f"{addressing[i]}: {shown}")
    return out


def kit_prompt(state, question):
    """Byte-identical to the kit's ``render_prompt``."""
    lines = [
        "State:",
        render_state(state, "kit"),
        "",
        "Question:",
        render_instructions(question),
        "",
        "Options:",
    ]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    lines += render_option_lines(question, list(criteria), list(criteria))
    lines += ["", "Reply with exactly one option key."]
    return "\n".join(lines)


def kit_prefix(state, question):
    """The kit prompt up to (not including) its closing instruction.

    This is the part a label readout may substitute into without changing what
    the model has been told about the state or the question. The test asserts
    that our rendering starts with exactly this string.
    """
    return kit_prompt(state, question)[: -len("\n\nReply with exactly one option key.")]


def render_question(state, question, mode="kit", keys=None, addressing=None, slot="paren", channel="key", prefill=None,
                    annotate=False, neutralize=False, repeat=False, spaced_instructions=False):
    """The user-turn body, ending in the answer slot.

    ``channel`` selects what the slot asks for: ``key`` reads the option label
    (or the option key, in the overflow fallback); ``description`` reads the
    option description text. The description channel reuses the same option
    listing so the model can still see all the candidates, and only the slot
    changes.

    ``annotate`` writes ``_index`` into long state arrays; ``neutralize``
    rewrites an exact abstain option; ``repeat`` emits the body twice before the
    single answer slot. All three default off, so the default string is
    unchanged.
    """
    q = dict(question)
    if keys is not None:
        q["criteria"] = {k: (question.get("criteria") or {}).get(k) for k in keys}
    if addressing is None:
        addressing = list(q["criteria"])
    if annotate:
        state = annotate_state(state)
    if mode == "reference":
        body = render_reference(state, q, addressing, slot, channel, prefill, neutralize, spaced_instructions)
    else:
        lines = ["State:", render_state(state, mode), "", "Question:",
                 render_instructions(q, spaced_instructions), "", "Options:"]
        lines += render_option_lines(q, list(q["criteria"]), addressing, neutralize)
        if channel == "description":
            lines += ["", 'Reply with the text of exactly one option: "']
        else:
            lines += ["", answer_slot_text(slot, prefill)]
        body = "\n".join(lines)
    return repeat_body(body) if repeat else body


# ------------------------------------------------- Mapika/decider reference layout
#
# Verbatim from github.com/Mapika/decider, decider/prompt.py:128-144 (v1.6.0,
# commit 23579f7). The plain, single-question, state-first layout is assembled
# there as:
#
#     ctx_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]   # :128
#     head   = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"                  # :141
#     tail   = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("                                       # :142
#     # <= NARROW opts -> narrow path:  [head] + [f"\n({LETTERS[j]}) {q.options[oi]}", ...] + [tail]   # :143-145
#     # >  NARROW opts -> wide path:   "\n(" + <one label token> + ") text" per option                # :146-151
#
# NARROW = len(LETTERS) = len("ABCDEFGHIJ") = 10, so at most 10 options take the
# NARROW path; 11 or more take the WIDE path. The two branches differ only in
# *tokenization* -- narrow encodes the whole option block as one string using
# ``LETTERS[j]``; wide emits "\n(" and a single label token, then ") text", where
# the label comes from ``label_table`` (A..Z, then AA..). At the string level both
# therefore read "\n(<label>) <text>" and this renderer reproduces that text.
#
# The entry reaches the reference through its ``/v1/systemone`` server, so the
# state and the option strings are produced by the ``systemone`` adapter before
# prompt.py renders them (kit http.py:38-39 -> decider/serve.py:124-125 ->
# decider/systemone.py). That adapter is reproduced verbatim here:
#
#   render_state (systemone.py:34-37):
#       str -> the string; else json.dumps(annotate_indices(state), ensure_ascii=False)
#   annotate_indices (systemone.py:19-31):
#       every list of length >= 8 becomes a list of {"_index": i, ...} records
#   option string (systemone.py:58):
#       key if crit[key] in (None, "") else f"{key}: {_txt(crit[key])}"
#   _txt (systemone.py:15-16): str as-is, else json.dumps(v, ensure_ascii=False)
#
# ``json.dumps`` here is the stdlib default (spaced separators), not ``text()``.
# Only differences from our kit layout are represented: "Context:" in place of
# "State:", an inline "Question: <text>", "(<label>) <text>" option lines, and no
# blank lines between the question and the options (the reference concatenates
# head + option lines + tail with no separators). The answer slot stays ours
# (``slot``) and ``Answer: (`` is unchanged. The readout, label table and scoring
# are untouched: this is a text rendering only.

ANNOTATE_MIN = 8


def annotate_indices(x, min_len=ANNOTATE_MIN):
    """Verbatim equivalent of ``decider/systemone.py:19-31``."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [
                ({"_index": i, **annotate_indices(v, min_len)}
                 if isinstance(v, dict)
                 else {"_index": i, "value": annotate_indices(v, min_len)})
                for i, v in enumerate(x)
            ]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def reference_state(state):
    """``systemone.render_state`` (decider/systemone.py:34-37)."""
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state), ensure_ascii=False)


def _reference_txt(v):
    """``systemone._txt`` (decider/systemone.py:15-16)."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def reference_option(key, desc):
    """``systemone.render_question``'s choice option string (decider/systemone.py:58)."""
    if desc is None or desc == "":
        return key
    return f"{key}: {_reference_txt(desc)}"


def reference_option_text(key, desc, neutralize=False):
    """One reference choice option string, with the optional neutral rewrite."""
    if neutralize and desc is not None and desc != "":
        return f"{key}: {neutralize_text(_reference_txt(desc))}"
    if neutralize:
        return neutralize_text(key)
    return reference_option(key, desc)


def reference_pieces(state, question, addressing, slot="paren", channel="key", prefill=None, neutralize=False,
                     spaced_instructions=False):
    """The reference's *tokenisation pieces* for one question row.

    The reference assembles its ids from separately-encoded pieces
    (``prompt_fast.context_ids`` + ``question_piece`` / ``_chat_row``); encoding
    the concatenation can differ at the boundaries. Returns

        dict(ctx=str, head=str, options=[(label, text), ...], slot=str)

    ``slot`` is None for the description channel (which the reference has no
    analogue of; callers fall back to the whole-body path).
    """
    criteria = question.get("criteria") or {}
    ctx = "Context:\n" + render_state(state, "reference")
    head = f"\n\nQuestion: {render_instructions(question, spaced_instructions)}\nOptions:"
    options = [
        (addressing[i], reference_option_text(key, criteria.get(key), neutralize))
        for i, key in enumerate(criteria)
    ]
    slot_text = None if channel == "description" else answer_slot_text(slot, prefill)
    return {"ctx": ctx, "head": head, "options": options, "slot": slot_text}


def render_reference(state, question, addressing, slot="paren", channel="key", prefill=None, neutralize=False,
                     spaced_instructions=False):
    """The Mapika/decider layout as a string (see the source note above)."""
    criteria = question.get("criteria") or {}
    out = [
        "Context:\n",
        render_state(state, "reference"),
        f"\n\nQuestion: {render_instructions(question, spaced_instructions)}\nOptions:",
    ]
    for i, key in enumerate(criteria):
        out.append(f"\n({addressing[i]}) {reference_option_text(key, criteria.get(key), neutralize)}")
    if channel == "description":
        out.append('\n\nReply with the text of exactly one option: "')
        return "".join(out)
    out.append("\n" + answer_slot_text(slot, prefill))
    return "".join(out)


# ---------------------------------------------------------- alternative renderings


def _maybe_json(s):
    if isinstance(s, str) and _JSONISH.match(s):
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def keyaddressed(value, indent=0, pad="  "):
    """Explicit addresses for every level: dicts by key, lists by index.

    Embedded JSON inside a string (ToolRet candidates, instructions payloads)
    is parsed and re-rendered, because that is exactly where positional
    addressing hurts: a list of candidate documents arrives as one opaque
    string otherwise.
    """
    parsed = _maybe_json(value)
    if parsed is not None:
        return keyaddressed(parsed, indent, pad)
    if isinstance(value, dict):
        if not value:
            return pad * indent + "(no fields)"
        rows = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                rows.append(pad * indent + f"{k}:")
                rows.append(keyaddressed(v, indent + 1, pad))
            else:
                rows.append(pad * indent + f"{k}: {_scalar(v)}")
        return "\n".join(rows)
    if isinstance(value, list):
        if not value:
            return pad * indent + "(empty)"
        rows = []
        for i, v in enumerate(value):
            if isinstance(v, (dict, list)) and v:
                rows.append(pad * indent + f"[{i}]")
                rows.append(keyaddressed(v, indent + 1, pad))
            else:
                rows.append(pad * indent + f"[{i}] {_scalar(v)}")
        return "\n".join(rows)
    return pad * indent + _scalar(value)


def prose(value):
    """The same content as running text, with structure spoken in words."""
    parsed = _maybe_json(value)
    if parsed is not None:
        return prose(parsed)
    if isinstance(value, dict):
        if not value:
            return "nothing."
        parts = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{k} is as follows: {prose(v)}")
            else:
                parts.append(f"{k} is {_scalar(v)}")
        return " ".join(parts)
    if isinstance(value, list):
        if not value:
            return "nothing."
        parts = []
        for i, v in enumerate(value):
            ordinal = f"item {i + 1}"
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{ordinal} is as follows: {prose(v)}")
            else:
                parts.append(f"{ordinal} is {_scalar(v)}")
        return " ".join(parts)
    return _scalar(value)


def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return "(empty)"
    return str(v)
