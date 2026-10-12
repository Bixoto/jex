"""The 255-label table.

Candidate order is fixed by the spec: ``A``-``Z``, then ``AA``..``ZZ``, then
``a``-``z``. A candidate is **valid** for the active tokenizer and slot when

  1. it encodes to exactly one token id, and
  2. that id round-trips: decoding the single id reproduces the exact string
     the readout is asking the model to emit.

Condition 2 is the runtime contract, not a nicety. A one-token id whose decode
is ``" A"`` while we asked for ``"A"`` (or a SentencePiece id whose surface
form carries a stray marker) makes the readout silently score the wrong
continuation, and that failure is invisible in the resulting distribution.

Nothing here is tokenizer-specific. The count that survives is whatever the
active tokenizer allows; the caller logs it at load time.
"""

import logging

log = logging.getLogger("decision_index.engine")

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"

# Surface markers that mean "this token carries a leading space". Used only to
# warn when the paren slot is handed a token that is not actually bare.
_SPACE_MARKERS = ("Ġ", "▁", "Ċ")


class LabelError(RuntimeError):
    pass


def candidate_labels():
    """``A``-``Z``, then ``AA``..``ZZ``, then ``a``-``z`` -- 728 candidates."""
    out = list(UPPER)
    out.extend(a + b for a in UPPER for b in UPPER)
    out.extend(LOWER)
    return out


def slot_text(label, slot):
    """The exact string whose single-token id the readout scores.

    ``prefill`` scores the leading-space label, exactly as ``space`` does: the
    prompt's lead-in ends at the ``:`` and the space is carried by the label
    token, so the rendered answer reads "... my answer is: A".
    """
    if slot in ("space", "prefill"):
        return " " + label
    if slot == "paren":
        return label
    raise LabelError(f"unknown slot {slot!r}")


def encode(tokenizer, text):
    """Token ids for ``text``, no special tokens. Works with HF, tiktoken, stubs."""
    enc = getattr(tokenizer, "encode", None)
    if enc is not None:
        try:
            return list(enc(text, add_special_tokens=False))
        except TypeError:
            return list(enc(text))
    if callable(tokenizer):
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    raise LabelError(f"cannot tokenize with {type(tokenizer).__name__}: no usable encode()")


def decode(tokenizer, ids):
    dec = getattr(tokenizer, "decode", None)
    if dec is not None:
        return dec(list(ids))
    raise LabelError(f"cannot decode with {type(tokenizer).__name__}: no usable decode()")


def surface(tokenizer, ids):
    """Raw token string (byte-level BPE / SentencePiece form), for auditing only."""
    to_str = getattr(tokenizer, "token_str", None)
    if callable(to_str):
        return to_str(list(ids))
    to_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
    if to_ids is not None:
        try:
            return str(to_ids(list(ids))[0])
        except Exception:
            return None
    return None


class LabelTable:
    """A contiguous, ordered list of labels that are valid for one tokenizer+slot."""

    __slots__ = ("labels", "token_ids", "slot", "requested", "surfaces", "rejected")

    def __init__(self, labels, token_ids, slot, requested, surfaces=None, rejected=0):
        self.labels = list(labels)
        self.token_ids = list(token_ids)
        self.slot = slot
        self.requested = requested
        self.surfaces = list(surfaces) if surfaces else [None] * len(self.labels)
        self.rejected = rejected

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.labels[i], self.token_ids[i]

    def label(self, i):
        return self.labels[i]

    def token_id(self, i):
        return self.token_ids[i]

    def describe(self):
        head = ", ".join(f"{l}={s}" for l, s in zip(self.labels[:4], self.surfaces[:4]) if s)
        return {
            "slot": self.slot,
            "labels_requested": self.requested,
            "labels_valid": len(self.labels),
            "labels_rejected": self.rejected,
            "first_labels": self.labels[:6],
            "first_token_surfaces": head or None,
        }


def _is_single_roundtrip(tokenizer, text):
    ids = encode(tokenizer, text)
    if len(ids) != 1:
        return None, None
    try:
        back = decode(tokenizer, ids)
    except Exception:
        return None, None
    return ids, back


def build_label_table(tokenizer, limit=255, slot="paren", log=log):
    """Verify, drop, and return the label table for ``tokenizer`` under ``slot``.

    Drops every candidate that is not exactly one token or does not round-trip,
    then keeps what is left in candidate order. The result is contiguous by
    construction: it *is* the surviving prefix-ordered sequence, with no gaps
    in the ordering and no duplicated token id.
    """
    limit = int(limit)
    cands = candidate_labels()[:limit]
    labels, ids, surfaces, rejected = [], [], [], 0
    seen = set()
    for cand in cands:
        target = slot_text(cand, slot)
        got, back = _is_single_roundtrip(tokenizer, target)
        if got is None or back != target:
            rejected += 1
            continue
        if got[0] in seen:  # cannot happen once round-trip holds, but it is a correctness invariant
            rejected += 1
            continue
        seen.add(got[0])
        labels.append(cand)
        ids.append(got[0])
        surfaces.append(surface(tokenizer, got))

    table = LabelTable(labels, ids, slot, limit, surfaces, rejected)
    log.info(
        "label table: %d valid of %d requested (slot=%s, %d candidates dropped for failing "
        "single-token or round-trip); first labels %s",
        len(table),
        limit,
        slot,
        rejected,
        labels[:4],
    )
    if len(table) < 2:
        raise LabelError(
            f"only {len(table)} label(s) survive for slot={slot!r}: cannot build a distribution over "
            "two or more options without refusing them"
        )
    if len(table) < limit:
        log.warning(
            "label table is short: %d < %d requested. Options beyond %d fall back to key scoring "
            "(nothing is dropped or refused).",
            len(table),
            limit,
            len(table),
        )
    if slot == "paren":
        spaced = [l for l, s in zip(table.labels, table.surfaces) if s and s[0] in _SPACE_MARKERS]
        if spaced:
            log.warning(
                "paren slot: %d label(s) carry a leading-space surface form (e.g. %s=%s). This tokenizer "
                "does not expose a distinct bare-letter id; the bare and space slots coincide here.",
                len(spaced),
                spaced[0],
                surface(tokenizer, [table.token_id(table.labels.index(spaced[0]))]),
            )
    return table
