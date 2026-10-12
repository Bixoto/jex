"""The slot readout: precision of the projection, rotation composition, gating,
the PriDe accumulator, and the probability algebra.

Everything here is plain Python + numpy and takes plain lists/floats, so every
arm can be exercised with no model and no GPU. Torch lives in ``engine.py``,
behind the backend protocol, and only where it has to be.

Precision
---------
``bf16_lmhead`` is the reference path: the model's own output projection, as
the model returns it, then ``log_softmax`` in fp32 -- exactly what the kit's
``transformers`` engine does. ``fp32_hidden`` takes the final hidden state,
gathers only the candidate rows of the unembedding matrix, and does the
projection in float32.

Gathering before the matmul is not an approximation *for the logits*: a matmul
only ever reads the rows it needs, so ``h @ W[ids].T`` in fp32 equals the fp32
full-vocab matmul restricted to ``ids``, bit for bit, and it costs 255 rows
instead of the whole vocabulary -- which matters, because an fp32 copy of a
256k x 5k unembedding is over 5 GB and there is no room for it next to a 62 GB
model.

A **log-probability**, however, is not just the gathered logits: it needs the
log-sum-exp over the *whole* vocabulary. ``engine.py`` therefore computes that
normaliser in chunks over the full unembedding (never materialising an fp32
copy) and subtracts it; renormalising over the candidate rows alone would
change the answer. The model's ``final_logit_softcapping`` applies to every
logit before the normaliser, so both projections and the normaliser apply it,
and ``project`` takes it as an explicit ``softcap`` argument.
"""

import math

import numpy as np

# Numerical guard for the prior division. This is a guard, not a parameter: it
# is never fitted and never tuned against data.
PRIOR_FLOOR_FRAC = 0.1


# ------------------------------------------------------------------- precision


def bf16_round(x):
    """Round a float32 array to bfloat16 precision (8-bit significand, ties-to-even)."""
    a = np.asarray(x, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(np.isfinite(a) & (a != 0), np.floor(np.log2(np.abs(a))), 0.0)
    scale = np.exp2(e - 7).astype(np.float32)
    out = np.round(np.nan_to_num(a / scale, nan=0.0, posinf=0.0, neginf=0.0)) * scale
    return out.astype(np.float32)


def project(hidden, weight, ids, precision="bf16_lmhead", softcap=None):
    """Unembed ``hidden`` (n, h) against ``weight`` (V, h) rows ``ids``, in the named precision.

    Returns (n, len(ids)) float32 logits. ``softcap`` is the model's
    ``final_logit_softcapping``: when set, the model applies
    ``tanh(logits / c) * c`` to the logits it returns, and both projections must
    reproduce that transform or the two arms read different numbers.
    """
    h = np.asarray(hidden, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    rows = w[list(ids)]
    if precision == "bf16_lmhead":
        # Inputs are already bf16 in memory; the kernel accumulates in fp32 and
        # returns bf16. Emulate by rounding both inputs and the output.
        h = bf16_round(h)
        rows = bf16_round(rows)
        out = bf16_round(h @ rows.T)
    elif precision == "fp32_hidden":
        out = h @ rows.T.astype(np.float32)
    else:
        raise ValueError(f"unknown projection {precision!r}")
    if softcap:
        out = np.tanh(out / float(softcap)) * float(softcap)
    return out


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float32)
    m = z.max(axis=-1, keepdims=True)
    lse = m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))
    return z - lse


# ------------------------------------------------------------ probability algebra


def normalise(scores, keys, temperature=1.0):
    """Softmax over ``scores`` -> a complete, finite, normalised distribution.

    This is the only place a probability dict is built, and it is written to
    satisfy ``decision_index.engines.base.validate`` unconditionally: every key
    present, every value finite and in [0, 1], sum to 1 within 0.01.
    """
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


def uniform(keys):
    n = len(keys)
    return {k: 1.0 / n for k in keys}


def sharpen(probs, keys, temperature=1.0):
    """Re-normalise an already-complete distribution at a temperature.

    ``p^(1/T) / sum(p^(1/T))``. Routing it through :func:`normalise` means there
    is exactly one place where a probability dict is built, so the guarantee
    ``validate`` checks -- every key present, every value finite in [0, 1],
    sum to 1 within 0.01 -- is made once and holds for every arm.
    """
    n = len(keys)
    if n == 0:
        raise ValueError("cannot normalise over zero options")
    x = [math.log(max(float(probs.get(k, 0.0)), 1e-300)) for k in keys]
    return normalise(x, keys, temperature)


# ------------------------------------------------------------------- rotations


def rotate(keys, shift):
    """Cyclic option order for rotation ``shift``; position i shows original key ``(i+shift) % n``."""
    n = len(keys)
    if n == 0:
        return []
    s = shift % n
    return keys[s:] + keys[:s]


def map_back(order, position_probs):
    """Position-space probabilities -> original-key space."""
    return {key: position_probs[i] for i, key in enumerate(order)}


def compose_rotations(passes, keys):
    """Average several complete distributions that are all in ORIGINAL-key space.

    The output key set is ``keys`` -- the original options, never the rotated
    ones. Every pass must already cover exactly ``keys``; anything else is a bug
    in the caller and is caught here rather than silently shipping a partial
    distribution to ``validate``.
    """
    want = set(keys)
    total = dict.fromkeys(keys, 0.0)
    for i, p in enumerate(passes):
        if set(p) != want:
            raise ValueError(f"rotation pass {i} covers {sorted(set(p))[:4]}... not the original option keys")
        for k in keys:
            total[k] += p[k]
    m = float(len(passes)) or 1.0
    return {k: total[k] / m for k in keys}


def gate_passes(max_p, tau, available):
    """How many passes to run. Pass 0 always runs; the rest only if ungated.

    ``tau is None`` -> ungated, run everything.
    ``tau`` set     -> the extra rotations are spent only on this question if
                       its first-pass max probability is below ``tau``.
    """
    if available <= 0:
        return 0
    if tau is None:
        return available
    return available if max_p < tau else 0


# ----------------------------------------------------------------- PriDe prior


class PriorAccumulator:
    """Streaming per-position label prior, estimated from permuted readouts.

    Tier A by construction: the prior is a function of the engine's own
    observed readouts on the run's first ``pride_alpha`` fraction of requests
    and of nothing else. It is discarded between runs.
    """

    def __init__(self, floor_frac=PRIOR_FLOOR_FRAC):
        self.floor_frac = floor_frac
        self.sums = {}
        self.counts = {}

    def observe(self, probs):
        for i, p in enumerate(probs):
            self.sums[i] = self.sums.get(i, 0.0) + float(p)
            self.counts[i] = self.counts.get(i, 0) + 1

    @property
    def seen(self):
        return sum(self.counts.values())

    def prior(self, n):
        """Floored prior over positions ``0..n-1``. Never zero, always finite.

        A position with no observations falls back to uniform, so an unseen
        position is corrected by nothing rather than boosted by a division
        through a near-zero.
        """
        out = []
        for i in range(n):
            obs = self.sums.get(i)
            if obs is None or not self.counts.get(i):
                out.append(1.0 / n)
            else:
                out.append(obs / self.counts[i])
        if not out:
            return out
        floor = self.floor_frac / n
        out = [v if (math.isfinite(v) and v > floor) else floor for v in out]
        s = sum(out)
        if not math.isfinite(s) or s <= 0.0:
            out = [1.0 / n] * n
        else:
            out = [v / s for v in out]
        return out


class FrozenPrior:
    """A fixed, order-independent additive correction over option positions.

    Unlike :class:`PriorAccumulator` (the streaming PriDe prior, which depends on
    the order and set of rows seen), a ``FrozenPrior`` is loaded once from an
    artifact measured *before* the run on a neutral prompt, and applied as a pure
    function of each row's option position. The same row scores identically
    whatever order the rows arrive in, which is the determinism property the
    streaming prior fails.

    ``offsets[i]`` is the log-score measured for the i-th option position on the
    neutral prompt; :meth:`apply` subtracts it. A row with more options than the
    artifact has offsets leaves the extra positions uncorrected.
    """

    def __init__(self, offsets):
        self.offsets = [float(x) for x in offsets]

    @classmethod
    def load(cls, path):
        import json

        with open(path) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("offsets", data.get("label_offsets"))
        if not isinstance(data, (list, tuple)):
            raise ValueError(f"frozen prior artifact {path!r} has no 'offsets' list")
        return cls(data)

    def apply(self, scores):
        """Subtract the per-position offset from a list of log-scores."""
        out = []
        for i, s in enumerate(scores):
            off = self.offsets[i] if i < len(self.offsets) else 0.0
            out.append(float(s) - off)
        return out


def apply_prior(probs, prior, floor_frac=PRIOR_FLOOR_FRAC):
    """Divide the prior out of ``probs`` (position order) and renormalise.

    ``probs`` and ``prior`` are both indexed by label position, so
    ``probs[i]`` and ``prior[i]`` must refer to the same option.
    """
    n = len(probs)
    if n == 0 or not prior or len(prior) < n:
        return list(probs)
    floor = floor_frac / n
    out = []
    for i in range(n):
        p = float(probs[i])
        pr = float(prior[i])
        if not math.isfinite(p) or p < 0.0:
            p = 0.0
        if not math.isfinite(pr) or pr < floor:
            pr = floor
        out.append(p / pr)
    s = sum(out)
    if not math.isfinite(s) or s <= 0.0:
        return [1.0 / n] * n
    out = [v / s for v in out]
    return out if all(math.isfinite(v) and v >= 0.0 for v in out) else [1.0 / n] * n


# ------------------------------------------------------------------ channel mix


def combine_channels(key_scores, desc_scores, weight):
    """Fixed-weight, unfitted log-space mix of the key channel and the description channel.

    ``weight`` is the description channel's share. It is a constant declared in
    ``config.py``; there is no code path that fits it, per-benchmark or
    otherwise. Log space (a weighted geometric mean of the two distributions)
    is used because the result must still be a calibrated distribution -- a
    rank aggregation cannot be.
    """
    w = float(weight)
    if w <= 0.0:
        return list(key_scores)
    if w >= 1.0:
        return list(desc_scores)
    return [(1.0 - w) * a + w * b for a, b in zip(key_scores, desc_scores)]
