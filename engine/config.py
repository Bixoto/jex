"""Arm flags, defaults, and validation for the label-slot engine.

Every flag here is settable through the kit's engine option mechanism
(``--option name=value``), so all values arrive as strings. Nothing in this
module carries a number derived from the benchmark suite: the only value that
is ever fitted is ``temperature``, and it is a single global scalar (see
``README.md``).

Defaults describe Arm A -- the baseline replication arm: kit rendering, paren
slot, the model's own bf16 ``lm_head``, a single forward pass, no gate, no
prior division, no second channel, no temperature scaling.

``max_prompt_tokens`` (default 65,536) is a defensive guard against a
pathological row, not a hardware or model limit. It was raised from 32,768
after the number was re-derived from the model's own ``config.json`` for
``google/gemma-4-31B-it``:

* ``layer_types`` = 50 sliding-attention + 10 full-attention layers, with
  ``sliding_window = 1024``. The model's *own* cache is
  ``DynamicCache(config=text_config)``: it keeps the last
  ``sliding_window - 1 = 1023`` tokens for each of the 50 sliding layers and
  grows with context only for the 10 full layers. A bare ``DynamicCache()``
  replicates a full-length cache for all 60 layers and costs ~9.5x more.
* the 10 full-attention layers use ``global_head_dim = 512`` and
  ``num_global_key_value_heads = 4`` (applied through ``per_layer_config``);
  the 50 sliding layers use the base ``head_dim = 256`` and
  ``num_key_value_heads = 16``. ``attention_k_eq_v = True`` removes the *weight*
  ``v_proj`` on the full layers (value states reuse the key states), but the
  cache still stores K and V as two separate tensors, so it does **not** halve
  cache memory.

Measured on the real config, bfloat16, with the model's own cache: each sliding
layer holds ``2 * 16 * 256 * 1023 * 2 B = 15.98 MiB`` (~0.78 GiB for all 50),
and each full layer holds ``2 * 4 * 512 * 2 B = 8 KiB`` per token (~80 KiB per
token for all 10). Totals: **~3.28 GiB at ctx 32,768**, **~5.78 GiB at ctx
65,536** (4.87 GiB at the panel's longest row, 53,653 tokens) and ~10.78 GiB at
ctx 131,072. The same model under a bare ``DynamicCache()`` would need ~27.5 GiB,
~55 GiB and ~110 GiB. Add ~62 GB of weights on a 96 GB RTX PRO 6000: 65,536
fits with ~28 GB to spare. The longest row in the 124,971-row panel is
**53,653 tokens**, so the 65,536 default answers 100% of it. The old 32,768 was
self-imposed and left 247 panel rows (0.20%) over the cap, measured at 0.784
index points of covered weight; do not re-derive it as a model limit, because
it never was one. The cap is kept -- ``Unsupported`` is the only legal response
once it is genuinely exceeded, never truncation -- and ``context_refusals``
stays instrumented.
"""

import logging

log = logging.getLogger("decision_index.engine")

SLOTS = ("paren", "space", "prefill")

# The prefilled-slot arm's natural-language lead-in, emitted immediately before
# the answer position (the label remains the next token). Literal from
# arXiv:2505.15323v2, section 4.1 "Prompt Format": the default template used in
# the main experiments is
#     'Given the question and the possible options, my answer is: '
# (a trailing space that we supply via the leading-space label token instead, so
# the prompt's last token is the ':'). The abstract's shorter example is
# 'The correct option is:'; the supplementary lists 10 alternatives. Overridable
# per arm with --option prefill_prefix='...'.
DEFAULT_PREFILL_PREFIX = "Given the question and the possible options, my answer is:"
PROJECTIONS = ("bf16_lmhead", "fp32_hidden")
# "reference" is the byte-for-byte Mapika/decider plain layout (Context: /
# inline Question: / "(A) text" option lines / no system prompt). It is a
# variant for the Phase 2b validation, never the default.
RENDERINGS = ("kit", "keyaddressed", "prose", "reference")
# The readout-factorial axis: the positional label token ("label", the
# historical behaviour) versus the option's own word/text token ("word").
KEYS_AXIS = ("label", "word")

# Largest number of option labels the candidate sequence can supply.
# A-Z (26) + AA..ZZ (676) + a-z (26).
MAX_LABELS = 728


class ConfigError(ValueError):
    pass


def as_bool(value, name):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off", "", "none"):
        return False
    raise ConfigError(f"{name}: expected a boolean, got {value!r}")


def _parse_temperature_by_options(value):
    """None (off) or "a,b,min" -> (a, b, min) floats. Output-only calibration:
    the emitted choice probabilities are re-normalised at
    T(n) = max(min, a + b*ln n) while every decision path stays untouched."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    parts = [p.strip() for p in str(value).replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ConfigError(f"temperature_by_options: expected 'a,b,min', got {value!r}")
    try:
        a, b, tmin = (float(p) for p in parts)
    except ValueError:
        raise ConfigError(f"temperature_by_options: non-numeric entry in {value!r}")
    if not tmin > 0.0:
        raise ConfigError(f"temperature_by_options: min must be > 0, got {tmin}")
    return (a, b, tmin)


def as_float_or_none(value, name):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected a float or None, got {value!r}")


def as_int(value, name):
    if isinstance(value, bool):
        raise ConfigError(f"{name}: expected an int, got {value!r}")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ConfigError(f"{name}: expected an int, got {value!r}")


def as_int_or_none(value, name):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return as_int(value, name)


def as_bool_or_none(value, name):
    """Like :func:`as_bool`, but ``None``/absent is preserved as ``None``.

    Used for ``rules_enabled``: the engine option, when set, overrides the
    master switch shipped inside ``decision_rules.json``; when unset (``None``)
    the file's value is authoritative. ``as_bool`` would collapse the unset case
    into ``False`` and silently make a ``rules_enabled: true`` file un-opt-in-able.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return as_bool(value, name)


def as_path_or_none(value, name):
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("none", "null", ""):
        return None
    return s


class ArmConfig:
    """The nine arm flags plus the two guards (``max_prompt_tokens``, ``stream_length``)."""

    __slots__ = (
        "slot",
        "projection",
        "labels",
        "rendering",
        "rotations",
        "gate_tau",
        "pride_alpha",
        "description_channel",
        "channel_weight",
        "temperature",
        "temperature_by_options",
        "system_prompt",
        "prefill_prefix",
        "max_prompt_tokens",
        "stream_length",
        # The six reference/experiment levers. Every one defaults to the
        # historical behaviour, so an arm that sets none of them is byte-for-byte
        # the validated baseline.
        "annotate_indices",
        "neutralize_options",
        "keys",
        "mid_depth",
        "prompt_repeat",
        "frozen_prior",
        # The three renderer-parity flags: each reproduces a reference prompt
        # behaviour that is a byproduct of the reference's code structure, never
        # a silent default. All default off.
        "piecewise_tokenize",
        "spaced_instructions",
        "noul_reference_labels",
        # The frozen decision-rule options. All default off / absent, so the
        # baseline is unchanged unless a run opts in.
        "rules_enabled",
        "decision_rules",
        "catalog_id",
        "noul_threshold",
        "trust_remote_code",
    )

    def __init__(
        self,
        slot="paren",
        projection="bf16_lmhead",
        labels=255,
        rendering="kit",
        rotations=1,
        gate_tau=None,
        pride_alpha=0.0,
        description_channel=False,
        channel_weight=0.5,
        temperature=1.0,
        # temperature_by_options: OUTPUT calibration only, None (default) means
        # off. When set to "a,b,min", the *emitted* probabilities of choice
        # questions are re-normalised at T(n) = max(min, a + b*ln n), mirroring
        # the reference's public decider_config.json. The working distribution -
        # and therefore every choice, every decision rule and every other
        # output - is untouched; only the probability channel changes.
        temperature_by_options=None,
        # Whether a system prompt is sent at all. True is every previous
        # behaviour; False reproduces the reference, which sends no system
        # prompt (Mapika/decider prompt.py ChatTemplate builds one user turn
        # only, and its plain layout is the bare body).
        system_prompt=True,
        # The prefilled-slot arm's lead-in (slot="prefill" only). See
        # DEFAULT_PREFILL_PREFIX for the source of the literal.
        prefill_prefix=DEFAULT_PREFILL_PREFIX,
        # Defensive prompt guard, NOT a model/hardware limit (see the module
        # docstring). 65,536 tokens is safe on gemma-4-31B because the model's
        # own cache bounds 50 of 60 layers to a 1,023-token sliding window
        # (~5.78 GiB for the whole cache at 65,536); it answers the whole panel
        # (longest row 53,653 tokens). 32,768 was self-imposed and cost covered
        # weight.
        max_prompt_tokens=65536,
        stream_length=None,
        # ---- reference transforms / experiment levers (all default off) ----
        # annotate_indices: write {"_index": i, ...} into state arrays of length
        # >= 8, reproducing decider/systemone.py:19-34 (documented 0.49 -> 0.57).
        annotate_indices=False,
        # neutralize_options: rewrite an option's exact "none of the above" /
        # "none" / "n/a" text to "not listed here" (decider/infer.py:30-51),
        # while the emitted answer stays keyed by the original option.
        neutralize_options=False,
        # keys: which token the readout scores -- "label" (the positional A/B/C
        # label, the default) or "word" (the option's own word/text token).
        keys="label",
        # mid_depth: read the answer from an intermediate hidden layer (0-based
        # layer index) instead of the final one; None is the final logits.
        mid_depth=None,
        # prompt_repeat: emit the rendered prompt body twice before the answer
        # slot (prompt repetition).
        prompt_repeat=False,
        # frozen_prior: path to a JSON artifact of per-position log-score
        # offsets measured on a neutral prompt. Order-independent: loaded once
        # and applied identically regardless of row order.
        frozen_prior=None,
        # piecewise_tokenize: reproduce the reference's tokenisation boundaries.
        # The reference encodes "Context:\n<state>" and the "\n\nQuestion:..."
        # piece separately and concatenates the ids; we encode the whole wrapped
        # body in one call. When the state is empty or ends in a newline the
        # "\n\n" merges with it (one newline token instead of two).
        piecewise_tokenize=False,
        # spaced_instructions: render a non-string `instructions` value with
        # stdlib default separators (", " / ": "), as systemone._txt does,
        # instead of our compact text() separators.
        spaced_instructions=False,
        # noul_reference_labels: render a noul question's options as "no"/"yes"
        # (with "no: <false desc>" / "yes: <true desc>" when descriptions are
        # present), as systemone.render_question does, instead of our
        # "false"/"true".
        noul_reference_labels=False,
        # ---- decision rules: the frozen far-tail threshold (default off) ----
        # rules_enabled: master switch for the frozen decision rules. None means
        # "use the rules_enabled value shipped in decision_rules.json" (false);
        # True/False set explicitly. Default off, so an arm that sets none of
        # these is bit-identical to the validated baseline.
        rules_enabled=None,
        # decision_rules: path to the rules JSON. None -> the file shipped next
        # to the engine source (engine/decision_rules.json). The rules JSON is
        # part of the submitted source, so the rule travels with the engine.
        decision_rules=None,
        # catalog_id: the benchmark catalog this run covers, so a rule keyed
        # "<catalog_id>:<field>" can be matched. The kit's runner calls the
        # engine with only (state, questions) -- the catalog id lives in
        # row["_evaluation"], which never reaches the engine -- so it is supplied
        # as a run option (recorded in environment.json alongside the arm).
        catalog_id=None,
        # noul_threshold: its OWN switch, independent of rules_enabled and of
        # the choice rule, default False. When True the engine emits a noul
        # question's answer as a hard 0.0/1.0 scalar instead of the probability:
        # 1.0 iff P(true) >= 0.5. This is the kit's own cut for noul, so the
        # *decision* is unchanged -- it is a representation act, not a choice
        # override, and a reviewer may reasonably read a binarised probability as
        # gaming the metric rather than making a decision. Hence: default off,
        # never enabled automatically anywhere. See decision_rules.json.
        noul_threshold=False,
        # trust_remote_code: pass-through to transformers' from_pretrained for
        # checkpoints that ship custom architecture code (e.g. Spark-X2.5).
        # Default False: the validated Gemma path never sets it.
        trust_remote_code=False,
    ):
        self.slot = self._one_of(slot, SLOTS, "slot")
        self.projection = self._one_of(projection, PROJECTIONS, "projection")
        self.rendering = self._one_of(rendering, RENDERINGS, "rendering")

        self.labels = as_int(labels, "labels")
        if not 2 <= self.labels <= MAX_LABELS:
            raise ConfigError(f"labels: must be between 2 and {MAX_LABELS}, got {self.labels}")

        self.rotations = as_int(rotations, "rotations")
        if self.rotations < 1:
            raise ConfigError(f"rotations: must be >= 1 (1 means off), got {self.rotations}")

        self.gate_tau = as_float_or_none(gate_tau, "gate_tau")
        if self.gate_tau is not None and not 0.0 < self.gate_tau <= 1.0:
            raise ConfigError(f"gate_tau: must be in (0, 1] or None, got {self.gate_tau}")

        self.pride_alpha = float(as_float_or_none(pride_alpha, "pride_alpha") or 0.0)
        if not 0.0 <= self.pride_alpha < 1.0:
            raise ConfigError(f"pride_alpha: must be in [0, 1), got {self.pride_alpha}")

        self.description_channel = as_bool(description_channel, "description_channel")

        self.channel_weight = float(channel_weight)
        if not 0.0 <= self.channel_weight <= 1.0:
            raise ConfigError(f"channel_weight: must be in [0, 1], got {self.channel_weight}")

        self.temperature = float(temperature)
        if not self.temperature > 0.0:
            raise ConfigError(f"temperature: must be > 0, got {self.temperature}")
        self.temperature_by_options = _parse_temperature_by_options(temperature_by_options)

        self.system_prompt = as_bool(system_prompt, "system_prompt")
        self.prefill_prefix = str(prefill_prefix)

        self.max_prompt_tokens = as_int(max_prompt_tokens, "max_prompt_tokens")
        if self.max_prompt_tokens < 1:
            raise ConfigError(f"max_prompt_tokens: must be >= 1, got {self.max_prompt_tokens}")

        self.stream_length = None if stream_length is None else as_int(stream_length, "stream_length")

        self.annotate_indices = as_bool(annotate_indices, "annotate_indices")
        self.neutralize_options = as_bool(neutralize_options, "neutralize_options")
        self.keys = self._one_of(keys, KEYS_AXIS, "keys")
        self.mid_depth = as_int_or_none(mid_depth, "mid_depth")
        if self.mid_depth is not None and self.mid_depth < 0:
            raise ConfigError(f"mid_depth: must be >= 0 or None, got {self.mid_depth}")
        self.prompt_repeat = as_bool(prompt_repeat, "prompt_repeat")
        self.frozen_prior = as_path_or_none(frozen_prior, "frozen_prior")
        self.piecewise_tokenize = as_bool(piecewise_tokenize, "piecewise_tokenize")
        self.spaced_instructions = as_bool(spaced_instructions, "spaced_instructions")
        self.noul_reference_labels = as_bool(noul_reference_labels, "noul_reference_labels")

        self.rules_enabled = as_bool_or_none(rules_enabled, "rules_enabled")
        self.trust_remote_code = as_bool(trust_remote_code, "trust_remote_code")
        self.decision_rules = as_path_or_none(decision_rules, "decision_rules")
        self.catalog_id = as_int_or_none(catalog_id, "catalog_id")
        self.noul_threshold = as_bool(noul_threshold, "noul_threshold")

        if self.pride_alpha > 0.0 and not self.stream_length:
            raise ConfigError(
                "pride_alpha > 0 needs stream_length: the calibration window is pride_alpha * stream_length "
                "requests and the engine refuses to guess a suite size"
            )

    @staticmethod
    def _one_of(value, allowed, name):
        s = str(value).strip()
        if s not in allowed:
            raise ConfigError(f"{name}: expected one of {allowed}, got {value!r}")
        return s

    @classmethod
    def from_options(cls, options):
        """Build a config from engine ``**options``; unknown keys are the caller's business."""
        return cls(**{k: options[k] for k in cls.__slots__ if k in options})

    @property
    def extra_rotations(self):
        return max(0, self.rotations - 1)

    @property
    def calibration_requests(self):
        """Number of requests whose readouts feed the PriDe accumulator."""
        if not self.pride_alpha:
            return 0
        return max(1, int(round(self.pride_alpha * self.stream_length)))

    def provenance(self):
        return {
            "slot": self.slot,
            "projection": self.projection,
            "labels_requested": self.labels,
            "rendering": self.rendering,
            "rotations": self.rotations,
            "gate_tau": self.gate_tau,
            "pride_alpha": self.pride_alpha,
            "description_channel": self.description_channel,
            "channel_weight": self.channel_weight,
            "temperature": self.temperature,
            "temperature_by_options": (list(self.temperature_by_options)
                                       if self.temperature_by_options is not None else None),
            "system_prompt": self.system_prompt,
            "prefill_prefix": self.prefill_prefix,
            "max_prompt_tokens": self.max_prompt_tokens,
            "annotate_indices": self.annotate_indices,
            "neutralize_options": self.neutralize_options,
            "keys": self.keys,
            "mid_depth": self.mid_depth,
            "prompt_repeat": self.prompt_repeat,
            "frozen_prior": self.frozen_prior,
            "piecewise_tokenize": self.piecewise_tokenize,
            "spaced_instructions": self.spaced_instructions,
            "noul_reference_labels": self.noul_reference_labels,
            "rules_enabled": self.rules_enabled,
            "decision_rules": self.decision_rules,
            "catalog_id": self.catalog_id,
            "noul_threshold": self.noul_threshold,
            "trust_remote_code": self.trust_remote_code,
            "fitted_parameters": ["temperature"] if self.temperature != 1.0 else [],
        }
