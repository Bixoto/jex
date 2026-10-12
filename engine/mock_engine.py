"""A model-free stand-in for the whole plumbing path.

``MockBackend`` is deterministic and needs nothing: a stub tokenizer, a fixed
synthetic unembedding matrix, and a hidden state that is a function of the
prompt and the option token. It goes through the *same* ``readout.project``
arithmetic the real path uses, so both the bf16 and the fp32 projection are
genuinely exercised by the plumbing tests, and it counts its forward passes so
the cost-multiple and gate assertions mean something.

It is a stand-in for testing, never a model: the numbers it produces are
meaningless as predictions and it must not be used for a scored run.
"""

import hashlib
import re

import numpy as np

from .labels import decode as decode_ids, encode as encode_text, surface
from . import readout as R
from .render import SYSTEM_PROMPT

_TOKEN = re.compile(r" ?\w+|[^\w\s]")

# A label is only a label if it encodes to one token and round-trips. A stub
# tokenizer that broke either rule would make every downstream assertion
# meaningless, so it enforces the real contract for a realistic alphabet.
_STUB_VOCAB = ["A", "B", "C", "D", "E", "F", "G", "H"] + [
    f"{a}{b}" for a in "ABCDEFGH" for b in "ABCDEFGH"
]

HIDDEN = 48
VOCAB = 4096


class StubTokenizer:
    """Word/punctuation splitter with a real single-token label table.

    A word keeps its leading space if it has one, the way a byte-level BPE does,
    so ``"A"`` and ``" A"`` are different ids and the ``space`` slot is testable.

    ``broken`` injects labels that fail the single-token test (they encode to
    two tokens) and ``mismatched`` injects labels that fail only the round-trip
    test (one token, but it decodes to something else). Between them the tests
    prove the table drops both, because a table built on either failure scores
    the wrong continuation silently.
    """

    def __init__(self, vocab=None, broken=(), mismatched=()):
        self.words = list(vocab) if vocab else list(_STUB_VOCAB)
        self.broken = {str(b).lstrip(" ") for b in broken}
        self.mismatched = {str(m).lstrip(" ") for m in mismatched}
        self._ids = {}
        self._decode_piece = {}
        self._raw_piece = {}
        self._nxt = 1000
        for w in self.words:
            self._add(w, w, w)
        for w in " ?.,:;()\"'[]{}":
            self._add(w, w, w)
        for w in self.words:
            # A byte-level leading-space variant: decodes to " A", raw form "ĠA".
            self._add(" " + w, " " + w, "Ġ" + w)
        for w in "abcdefghijklmnopqrstuvwxyz":
            self._add(w, w, w)
        for w in self.mismatched:
            # One token whose decode is a different string: the silent failure the
            # round-trip clause exists to catch.
            tid = self._ids[w][0]
            self._decode_piece[tid] = w.lower()

    def _add(self, key, decoded, raw):
        if key in self._ids:
            return
        tid = self._nxt
        self._nxt += 1
        self._ids[key] = [tid]
        self._decode_piece[tid] = decoded
        self._raw_piece[tid] = raw

    def _word_id(self, piece):
        stable = int.from_bytes(hashlib.blake2b(piece.encode(), digest_size=4).digest(), "big")
        return 200000 + (stable % 100000)

    def encode(self, text, add_special_tokens=False):
        out = []
        for m in _TOKEN.finditer(text):
            piece = m.group(0)
            ids = self._ids.get(piece) or self._ids.get(" " + piece) or [self._word_id(piece)]
            if piece in self.broken:
                ids = [ids[0], self._word_id("\x00" + piece)]
            out.extend(ids)
        return out

    def decode(self, ids):
        return "".join(self._decode_piece.get(int(i), "?") for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self._raw_piece.get(int(i), "?") for i in ids]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens)}


def _hash_vec(*parts, size):
    seed = "|".join(str(p) for p in parts).encode()
    chunks = [hashlib.blake2b(seed + bytes([i]), digest_size=32).digest() for i in range((size + 7) // 8)]
    raw = b"".join(chunks)[: 4 * size]
    return np.frombuffer(raw, dtype="<u4").astype(np.float32) / 2**31 - 1.0


class MockBackend:
    """Deterministic stand-in with the same surface the engine needs."""

    def __init__(self, model_id="mock/no-model", chat_template=True, cache_prefix=True, system_prompt=True,
                 context_limit=1 << 30):
        self.tokenizer = StubTokenizer()
        self.model_id = model_id
        self.chat_template = chat_template
        self.cache_prefix = cache_prefix
        self.system_prompt = bool(system_prompt)
        self.context_limit = context_limit
        self.revision = "mock"
        self.device = "none"
        self.dtype = "float32"
        rng = np.random.default_rng(20260919)
        self.weight = (rng.standard_normal((VOCAB, HIDDEN)) / np.sqrt(HIDDEN)).astype(np.float32)
        self._all_ids = np.arange(VOCAB)
        self.cached_ids = []
        self.reused_prefix = 0
        self.forwards = 0

    # -- tokenizer / template ------------------------------------------------

    def encode(self, s):
        return encode_text(self.tokenizer, s)

    def decode(self, ids):
        return decode_ids(self.tokenizer, ids)

    def token_str(self, ids):
        return surface(self.tokenizer, ids)

    def wrap(self, body, split_slot=False):
        # system_prompt=False mirrors the reference: a single user turn, no
        # system block, and the bare body in the plain layout. split_slot moves
        # the final answer line into "assistant" position, as the reference's
        # chat layout does (see engine.chat_wrap).
        if self.chat_template:
            head = (f"<system>\n{SYSTEM_PROMPT}\n</system>\n" if self.system_prompt else "") + "<user>\n"
            tail = "\n</user>\n"
            if not split_slot:
                return f"{head}{body}{tail}"
            from .engine import _split_answer_slot

            pre, slot = _split_answer_slot(body)
            return f"{head}{pre}{tail}assistant>\n{slot}"
        return body if not self.system_prompt else SYSTEM_PROMPT + "\n\n" + body

    def synchronize(self):
        pass

    def reset_cache(self):
        self.cached_ids = []

    def _common_prefix(self, ids):
        n = min(len(ids), len(self.cached_ids), len(ids) - 1)
        p = 0
        while p < n and ids[p] == self.cached_ids[p]:
            p += 1
        return p

    # -- the scoring pass ----------------------------------------------------

    def score(self, p, ids, options, precision="bf16_lmhead"):
        """Same contract as ``TorchBackend.score``: one summed log-prob per option.

        ``ids`` is the full prompt and ``p`` the prefix already cached; the mock
        has no real cache, so it scores the uncached suffix. The hidden state is
        a deterministic function of the prompt and of the option token being
        predicted, so both the multi-token option path and the precision switch
        behave like the real thing. The softmax runs over the whole synthetic
        vocabulary, as it does on a real model, so a one-token option does not
        trivially get log-prob 0.
        """
        q_ids = ids[p:]
        if not q_ids:
            raise RuntimeError("mock backend: empty prompt")
        self.reused_prefix = p
        self.forwards += 1
        prompt_vec = _hash_vec(*q_ids[-24:], size=HIDDEN)
        scores = []
        for opt in options:
            total = 0.0
            for j, tok in enumerate(opt):
                h = np.tanh(prompt_vec * (1.0 + 0.25 * j) + _hash_vec("opt", tok, size=HIDDEN))
                logits = R.project(h[None, :], self.weight, self._all_ids, precision)[0]
                total += float(R.log_softmax(logits)[tok % VOCAB])
            scores.append(total)
        return scores

    def runtime(self):
        return {"mock": True, "model": self.model_id, "vocab": VOCAB, "hidden": HIDDEN, "forwards": self.forwards}


class MockEngine:
    """``engine.engine.LabelSlotEngine`` with the mock backend already injected.

    A factory, not a subclass, so tests and the harness lane construct it with
    exactly the ``**options`` a real run would use.
    """

    @staticmethod
    def create(backend=None, **options):
        from .engine import LabelSlotEngine

        options.pop("backend", None)
        if backend is None:
            # Honour the engine-level flag on the mock backend too, so
            # MockEngine.create(system_prompt=False) really sends no system prompt.
            backend = MockBackend(system_prompt=options.get("system_prompt", True))
        return LabelSlotEngine(backend=backend, **options)

    @staticmethod
    def backend(**kwargs):
        return MockBackend(**kwargs)
