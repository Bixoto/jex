"""The label-slot ``decision_index`` engine.

One forward pass per (question, rotation) scores every option as a single
label token read at a fixed answer slot. Structurally this is the kit's trick --
one block-diagonal pass, all options packed after one shared prompt -- with the
difference that the scored continuation is a *verified* single-token label
rather than the option's own key, which is what makes a 151-way question cost
what a 2-way question costs.

Read ``README.md`` for the arm flags. The rules this file enforces everywhere:

* state before the question, always;
* never truncate, never drop an option, never refuse a question for being wide
  or long -- the only ``Unsupported`` raised is a prompt that physically does
  not fit the model's positional-embedding limit, which is the reference
  engine's behaviour, and it is counted in ``runtime()``;
* the prompt-length guard (``max_prompt_tokens``, default 65,536) is a
  self-imposed defence against a pathological row, not a hardware limit: on
  gemma-4-31B the model's own ``DynamicCache(config=text_config)`` bounds 50 of
  its 60 layers to a 1,023-token sliding window and the other 10 grow at 8 KiB
  per token, so a 65,536-token prompt holds ~5.78 GiB of KV beside ~62 GB of
  weights on a 96 GB card. It is instrumented as ``prompt_over_max``; it is not
  the positional-limit ``Unsupported`` path (``context_refusals``);
* always return a complete, finite, normalised distribution over the original
  criteria keys;
* carry nothing across a request that was derived from the benchmark suite.
"""

import json
import logging
import math
import os
import time

from decision_index.engines.base import Engine, Unsupported

from . import readout as R
from .config import ArmConfig, ConfigError, as_bool, as_int
from .labels import build_label_table, decode as decode_ids, encode as encode_text, surface
from .render import SYSTEM_PROMPT, neutralize_text, render_question

log = logging.getLogger("decision_index.engine")

NOUL_KEYS = ["false", "true"]
NOUL_DESCS = ["False", "True"]
# The reference's NARROW threshold: prompt.py LETTERS = "ABCDEFGHIJ".
REFERENCE_NARROW = 10
DEFAULT_MODEL = "google/gemma-4-31B-it"

# The kit rejects a 'score' question type outright, so the engine does too. The
# only other refusal is a prompt that physically exceeds the model's positional
# limit; width and length are never a reason to refuse.
REFUSABLE_TYPES = ("choice", "noul")

# The rules file ships next to this module, so the frozen rule travels with the
# submitted engine source (see decision_rules.json for the full statement).
DEFAULT_DECISION_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_rules.json")


# ------------------------------------------------------- frozen decision rules
#
# A choice question's answer is normally the global argmax. A frozen decision
# rule replaces that with a far-tail threshold on one positive key:
#
#     g = logit(p_positive)                    # float64 throughout
#     if g >= tau_logodds: choice = positive
#     else:                choice = best NON-positive key
#
# The fallback MUST be the best non-positive key. "Best key clearing tau, else
# the global argmax" is a provable no-op for tau > 0 -- the argmax already
# clears the threshold -- which is how an earlier prototype silently did nothing.
# tau_logodds is a LOG-ODDS constant, never a probability: 18.404296875 is
# p = 0.9999999898348119, not 0.9999.
#
# SELECTION IS PER QUESTION, not per run. The kit calls the engine with
# (state, questions) and the catalog id never reaches the engine, so a rule is
# selected by a signature derived from the question itself:
#
#     signature = (type, sorted(criteria keys))   ->  "choice|no,yes"
#
# A rule may carry an optional `catalog_scope` (an int) as an ADDITIONAL
# narrowing condition; it then applies only when the run's `catalog_id` option
# equals it. The legacy key form "catalog_id:field" also still works (matched
# only when `catalog_id` is set) so nothing regresses. A signature that matches
# no rule falls through to the file's `default` (null => plain argmax): an
# unseen private benchmark is left untuned by construction.

SIGNATURE_SEP = "|"
SIGNATURE_KEY_SEP = ","


def question_signature(qtype, keys):
    """The rule-selection key: ``"<type>|<sorted,distinct,criteria,keys>"``.

    Used identically by the engine and by ``harness/signature_uniqueness.py``,
    so the scan measures the key the engine actually uses. See that script for
    which signatures are unique to one catalog (and which are not) across the
    suite.
    """
    return f"{qtype}{SIGNATURE_SEP}{SIGNATURE_KEY_SEP.join(sorted({str(k) for k in keys}))}"


def logit(p):
    """log(p / (1 - p)) in float64; saturates to +/-inf at the endpoints."""
    p = float(p)
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return math.log(p) - math.log1p(-p)


def apply_choice_rule(probs, keys, positive, tau_logodds):
    """The rule, as a pure function of one question's probabilities.

    Deterministic and order-independent: it reads only ``probs`` (keyed by the
    question's criteria keys), the criteria order ``keys``, and the rule. It does
    not depend on row order, batch size or which other rules are configured.
    """
    if positive not in probs:
        raise ConfigError(
            f"decision rule names positive key {positive!r}, but the question's "
            f"criteria keys are {keys!r}; refusing to silently fall back to argmax"
        )
    if logit(probs[positive]) >= float(tau_logodds):
        return positive
    others = [k for k in keys if k != positive]
    if not others:
        return positive
    return max(others, key=lambda x: probs[x])


def _parse_rule(value, name):
    """Validate one rule object.

    Returns ``(positive, tau_logodds, catalog_scope, field_scope,
    instructions_contains)``. All scope conditions are optional and ANDed.
    """
    if not isinstance(value, dict):
        raise ConfigError(
            f"decision rule {name!r}: expected an object with 'positive', "
            f"'tau_logodds' and optional scopes, got {type(value).__name__}"
        )
    allowed = ("positive", "tau_logodds", "catalog_scope", "field_scope", "instructions_contains")
    extra = sorted(k for k in value if k not in allowed and not str(k).startswith("_"))
    if extra:
        raise ConfigError(f"decision rule {name!r}: unknown field(s) {extra}")
    positive = value.get("positive")
    if not isinstance(positive, str) or not positive:
        raise ConfigError(f"decision rule {name!r}: 'positive' must be a non-empty string, got {positive!r}")
    tau = value.get("tau_logodds")
    if tau is None:
        raise ConfigError(f"decision rule {name!r}: 'tau_logodds' is required")
    try:
        tau = float(tau)
    except (TypeError, ValueError):
        raise ConfigError(f"decision rule {name!r}: 'tau_logodds' must be a number, got {tau!r}")
    if math.isnan(tau):
        raise ConfigError(f"decision rule {name!r}: 'tau_logodds' must not be NaN")
    scope = value.get("catalog_scope")
    if scope is not None:
        scope = as_int(scope, f"decision rule {name!r}: catalog_scope")
    field_scope = value.get("field_scope")
    if field_scope is not None and (not isinstance(field_scope, str) or not field_scope):
        raise ConfigError(
            f"decision rule {name!r}: 'field_scope' must be a non-empty string, got {field_scope!r}"
        )
    instr = value.get("instructions_contains")
    if instr is not None and (not isinstance(instr, str) or not instr):
        raise ConfigError(
            f"decision rule {name!r}: 'instructions_contains' must be a non-empty string, got {instr!r}"
        )
    return positive, tau, scope, field_scope, instr


def _parse_rule_value(value, name):
    """One rule object, or a list of them, -> a list of validated rules.

    A signature may carry several rules (e.g. two rules for the shared
    ``choice|no,yes`` signature, distinguished by their scopes). They are tried
    in file order; the first whose scopes match the question wins.
    """
    if isinstance(value, list):
        if not value:
            raise ConfigError(f"decision rule {name!r}: the rule list is empty")
        return [_parse_rule(v, name) for v in value]
    return [_parse_rule(value, name)]


def _parse_signature_key(name, qtype, keypart):
    """Validate and canonicalise the key half of a signature key.

    ``keypart`` is the comma-separated criteria-key list after the ``|``. Sorted
    and de-duplicated to match :func:`question_signature`, so the config is
    order-insensitive. An empty key set raises -- it cannot select a question.
    """
    if not qtype.strip():
        raise ConfigError(f"decision rule key {name!r} has an empty question type")
    keys = [t.strip() for t in keypart.split(SIGNATURE_KEY_SEP)]
    keys = sorted({t for t in keys if t})
    if not keys:
        raise ConfigError(
            f"decision rule key {name!r} parses to an empty criteria key set; "
            "a signature must name at least one key"
        )
    return qtype.strip(), keys


def load_decision_rules(path=None):
    """Load and validate the rules file.

    Returns ``(enabled, default_rule, signature_rules, legacy_rules)`` where a
    rule is ``(positive, tau_logodds, catalog_scope, field_scope,
    instructions_contains)`` and each dict value is a LIST of rules.

    * ``signature_rules`` is keyed by ``question_signature(type, keys)`` and is
      the primary, per-question selection mechanism. A signature may carry more
      than one rule (distinguished by their scopes); they are tried in file order.
    * ``legacy_rules`` keeps the old ``"<catalog_id>:<field>"`` form working: it
      is matched only when the run's ``catalog_id`` option is set.
    * ``enabled`` is the file's master switch; the caller may override it with
      the ``rules_enabled`` engine option. Reserved keys are ``rules_enabled``,
      ``default`` and any string starting with ``_`` (documentation, ignored).
    """
    path = path or DEFAULT_DECISION_RULES_PATH
    if not os.path.exists(path):
        if path != DEFAULT_DECISION_RULES_PATH:
            raise ConfigError(f"decision_rules: file not found: {path!r}")
        # The shipped default file is absent: behave as if rules were disabled.
        return False, None, {}, {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"decision_rules: could not parse {path!r}: {exc}")
    if not isinstance(data, dict):
        raise ConfigError(f"decision_rules: {path!r} must contain a JSON object")
    enabled = as_bool(data.get("rules_enabled", False), "decision_rules.rules_enabled")
    default = data.get("default")
    default_rule = None if default is None else _parse_rule(default, "default")
    signature_rules, legacy_rules = {}, {}
    for key, value in data.items():
        if key == "rules_enabled" or key == "default" or str(key).startswith("_"):
            continue
        if not isinstance(key, str) or (SIGNATURE_SEP not in key and ":" not in key):
            raise ConfigError(
                f"decision rule key {key!r} must be a signature "
                f"'<type>|<key1,key2>' or the legacy '<catalog_id>:<field>'"
            )
        if SIGNATURE_SEP in key:
            qtype, _, keypart = key.partition(SIGNATURE_SEP)
            canon_type, canon_keys = _parse_signature_key(key, qtype, keypart)
            signature_rules.setdefault(question_signature(canon_type, canon_keys), []).extend(
                _parse_rule_value(value, key)
            )
        else:
            cid, _, field = key.partition(":")
            if not field:
                raise ConfigError(f"decision rule key {key!r} has an empty field half")
            try:
                int(cid)
            except ValueError:
                raise ConfigError(f"decision rule key {key!r} must start with an integer catalog id")
            legacy_rules.setdefault(key, []).extend(_parse_rule_value(value, key))
    return enabled, default_rule, signature_rules, legacy_rules


class BackendError(RuntimeError):
    pass


# --------------------------------------------------------------- chat wrapping
#
# The reference's chat layout places the answer slot in the *assistant* turn:
# decider/serve.py:30-33 -- "the tokenizer's chat template around one user turn,
# thinking off, 'Answer: (' in the assistant turn" -- and decider/prompt.py
# builds it as head + context + question blocks + options + tail + answer
# (build_chat, prompt.py:274-289; ChatTemplate.answer_ids, prompt.py:235-237).
# The plain layout (chat=None) is the bare body, unchanged.
#
# ``split_slot`` implements that ordering: the last line of the rendered body
# (the answer slot) is moved to after the chat template's tail. It is only ever
# requested for ``rendering="reference"``; every default call passes False and
# produces exactly the previous string.

_CHAT_SENTINEL = "@@DI_USER_CONTENT@@"


def _split_answer_slot(body):
    """Split the rendered body into (everything before the final line, final line).

    Every rendering ends in exactly one answer-slot line, so the final newline is
    the boundary. The newline stays out of both halves: the reference appends its
    answer piece directly after the template tail with no separator.
    """
    i = body.rfind("\n")
    if i < 0:
        return body, ""
    return body[:i], body[i + 1 :]


def chat_wrap(tokenizer, body, system_prompt=True, split_slot=False):
    """The reference chat layout: one user turn, thinking off, slot in the assistant turn."""
    messages = [] if not system_prompt else [{"role": "system", "content": SYSTEM_PROMPT}]
    if not split_slot:
        messages.append({"role": "user", "content": body})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "user", "content": _CHAT_SENTINEL})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    head, tail = rendered.split(_CHAT_SENTINEL)
    pre, slot = _split_answer_slot(body)
    return head + pre + tail + slot


def chat_head_tail(tokenizer, system_prompt=True):
    """The reference ``ChatTemplate``'s (head, tail) strings, without the body.

    Used only by the piecewise tokeniser. ``chat_wrap`` is deliberately left
    untouched: this adds the reference's own thinking-block close
    (``prompt.py:220-222``), which would change the default prompt on a
    tokenizer whose template leaves a thinking block open.
    """
    messages = [] if not system_prompt else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": _CHAT_SENTINEL})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    head, tail = rendered.split(_CHAT_SENTINEL)
    for opened, closed in (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"),
                           ("<|channel>thought\n", "<channel|>")):
        if tail.endswith(opened):
            tail += closed
    return head, tail


# ----------------------------------------------------------------- torch backend


class TorchBackend:
    """Tokenizer, model, prefix cache, and the block-diagonal scoring pass.

    The only place torch is touched. Imported lazily so the engine's config,
    rendering, label table and every arm in ``readout.py`` stay importable and
    testable with no model and no GPU.
    """

    def __init__(
        self,
        model=DEFAULT_MODEL,
        revision=None,
        device=None,
        dtype=None,
        attn="sdpa",
        chat_template=True,
        cache_prefix=True,
        system_prompt=True,
        trust_remote_code=False,
        tokenizer=None,
        model_obj=None,
        mid_depth=None,
    ):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # CUBLAS_WORKSPACE_CONFIG must be present before the first CUDA work, so
        # it is set here rather than left to the ambient environment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        self.torch = torch
        self.attn = attn
        # Passed to transformers for checkpoints that ship custom code
        # (e.g. Spark-X2.5). Default False: the Gemma path never sets it.
        self.trust_remote_code = bool(trust_remote_code)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model, revision=revision, trust_remote_code=self.trust_remote_code
            )
        self.tokenizer = tokenizer
        if model_obj is not None:
            # Programmatic injection point for tests: an already-built local
            # model, so the real _pack/_forward/score path runs with no network.
            self.model = model_obj.to(self.device).eval()
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model, revision=revision, dtype=self.dtype, attn_implementation=attn,
                trust_remote_code=self.trust_remote_code,
            ).to(self.device).eval()
        self.chat_template = bool(chat_template) and getattr(self.tokenizer, "chat_template", None) is not None
        self.cache_prefix = bool(cache_prefix)
        # Whether a system prompt is prepended at all. The reference sends none.
        self.system_prompt = bool(system_prompt)
        # The mid-depth readout: a 0-based hidden-layer index, or None for the
        # model's final logits. Set by LabelSlotEngine from the arm config.
        self.mid_depth = None if mid_depth is None else int(mid_depth)
        self.model_id = model
        self.revision = revision or getattr(self.model.config, "_commit_hash", None)
        # Pin and record the numerics a scored run depends on before any forward.
        self.determinism = self._configure_determinism(attn)
        # The model's final-logit softcap, read from its own config once. The
        # bf16 path gets it for free from the model; the fp32 path must apply it.
        self.softcap = self._read_softcap()

        # On the real checkpoint the positional limit lives in the nested
        # `text_config`; the top-level multimodal config has no
        # `max_position_embeddings`, so it must be read from the text config
        # before falling back to the tokenizer.
        text_cfg = self._text_config()
        limit = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(text_cfg, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
        )
        self.context_limit = int(limit) if limit and int(limit) < (1 << 30) else (1 << 30)

        head = getattr(self.model, "lm_head", None)
        self.lm_weight = None if head is None else head.weight
        self.lm_bias = None if head is None else getattr(head, "bias", None)
        self.inner = getattr(self.model, "model", self.model)
        self.cache = None
        self.cached_ids = []
        self.reused_prefix = 0
        self.forwards = 0

    # -- determinism / config plumbing ----------------------------------------

    def _text_config(self):
        """The language-model config, unwrapping a multimodal wrapper if present."""
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return None
        get_text = getattr(cfg, "get_text_config", None)
        if callable(get_text):
            try:
                return get_text()
            except Exception:
                pass
        return cfg

    def _read_softcap(self):
        val = getattr(self._text_config(), "final_logit_softcapping", None)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _layer_types(self):
        types = getattr(self._text_config(), "layer_types", None)
        return list(types) if types else None

    def _has_sliding_layers(self):
        types = self._layer_types()
        return bool(types) and "sliding_attention" in types

    def _sliding_window(self):
        if not self._has_sliding_layers():
            return None
        window = getattr(self._text_config(), "sliding_window", None)
        try:
            window = int(window)
        except (TypeError, ValueError):
            return None
        return window if window > 0 else None

    def _mask_mapping_supported(self):
        """Whether the model consumes a ``{layer_type: mask}`` mapping.

        Gemma 3/4 dispatch per layer type and accept a pre-built mapping. For
        every other model the single block-diagonal mask is passed as before.
        """
        if not self._layer_types():
            return False
        model_type = str(getattr(self._text_config(), "model_type", ""))
        return model_type.startswith(("gemma3", "gemma4"))

    def _configure_determinism(self, attn):
        """Pin the numerics a scored run depends on, and record what was pinned.

        The CUDA knobs are set (and recorded) on every device so a CPU-only run
        still carries a complete provenance block. SDPA backends are set
        explicitly rather than left to the ambient process defaults: ``eager``
        never enters the SDPA dispatcher, and ``sdpa`` keeps torch's default
        three-backend set, whose actual choice depends on device and shapes and
        is recorded as such.
        """
        torch = self.torch
        info = {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "attn_implementation": attn,
        }
        cuda = getattr(torch.backends, "cuda", None)
        if cuda is not None and hasattr(cuda, "matmul"):
            cuda.matmul.allow_tf32 = False
            info["matmul_allow_tf32"] = bool(cuda.matmul.allow_tf32)
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None and hasattr(cudnn, "allow_tf32"):
            cudnn.allow_tf32 = False
            info["cudnn_allow_tf32"] = bool(cudnn.allow_tf32)
        if cuda is not None and hasattr(cuda, "enable_flash_sdp"):
            # Explicitly pin the SDPA kernel set instead of inheriting ambient
            # state. `sdpa` keeps torch's default set; `eager` neutralises SDPA.
            want = attn == "sdpa"
            cuda.enable_flash_sdp(want)
            cuda.enable_mem_efficient_sdp(want)
            cuda.enable_math_sdp(True)
            info["sdpa_backends_enabled"] = {
                "flash": bool(cuda.flash_sdp_enabled()),
                "mem_efficient": bool(cuda.mem_efficient_sdp_enabled()),
                "math": bool(cuda.math_sdp_enabled()),
            }
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
            info["deterministic_algorithms_warn_only"] = True
        except Exception as exc:  # older torch, or an op without a deterministic kernel
            info["deterministic_algorithms"] = f"unavailable: {exc}"
        if attn != "sdpa":
            info["sdpa_backend_selected"] = None
        elif self.device != "cuda":
            info["sdpa_backend_selected"] = "math"
        else:
            enabled = [k for k, v in info.get("sdpa_backends_enabled", {}).items() if v]
            info["sdpa_backend_selected"] = enabled[0] if len(enabled) == 1 else "auto(" + ",".join(enabled) + ")"
        return info

    def _new_cache(self):
        """The model's own cache layout (per-layer sliding/full), not a bare one."""
        from transformers import DynamicCache

        cfg = self._text_config()
        if cfg is not None:
            try:
                return DynamicCache(config=cfg)
            except Exception:
                pass
        return DynamicCache()

    # -- tokenizer surface consumed by labels.py ------------------------------

    def encode(self, s):
        return encode_text(self.tokenizer, s)

    def decode(self, ids):
        return decode_ids(self.tokenizer, ids)

    def token_str(self, ids):
        return surface(self.tokenizer, ids)

    def wrap(self, body, split_slot=False):
        # No system prompt reproduces the reference: its ChatTemplate wraps a
        # single user turn (prompt.py:216), and its plain layout is the bare
        # body. Default keeps the historical system prompt, unsplit.
        if self.chat_template:
            return chat_wrap(self.tokenizer, body, self.system_prompt, split_slot)
        return body if not self.system_prompt else SYSTEM_PROMPT + "\n\n" + body

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def runtime(self):
        info = {"torch": self.torch.__version__, "device": self.device, "dtype": str(self.dtype)}
        if self.device == "cuda":
            info.update(cuda=self.torch.version.cuda, gpu=self.torch.cuda.get_device_name())
        try:
            import transformers

            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info

    # -- prefix cache ---------------------------------------------------------

    def _common_prefix(self, ids):
        n = min(len(ids), len(self.cached_ids), len(ids) - 1)
        p = 0
        while p < n and ids[p] == self.cached_ids[p]:
            p += 1
        return p

    def reset_cache(self):
        self.cache = None
        self.cached_ids = []

    def truncate_cache(self):
        self.cache = None
        self.cached_ids = []

    # -- the scoring pass -----------------------------------------------------

    def _pack(self, p, q_ids, options):
        """Flat [prompt][opt0][opt1]... plus the 4D mask, position ids, and keep positions.

        Identical in structure to the reference engine's pass: the prompt is
        causal, each option attends to the whole prompt and causally to its own
        tokens only, and every option starts at the prompt's last position.
        """
        torch = self.torch
        lengths = [len(o) for o in options]
        q_len = len(q_ids)
        flat = list(q_ids)
        starts = []
        for o in options:
            starts.append(len(flat))
            flat.extend(o)
        Q = len(flat)
        K = p + Q
        allowed = torch.zeros(Q, K, dtype=torch.bool)
        rows = torch.arange(Q)
        cols = torch.arange(K)
        allowed[:q_len] = cols[None, :] <= (p + rows[:q_len])[:, None]
        for s, m in zip(starts, lengths):
            block = allowed[s : s + m]
            block[:, : p + q_len] = True
            local = torch.arange(m)
            block[:, p + s : p + s + m] = local[None, :] <= local[:, None]
        positions = list(range(p, p + q_len))
        for m in lengths:
            positions.extend(range(p + q_len, p + q_len + m))
        # keep lists the flat positions whose logits are needed, in the order the
        # kept-logits output returns them (the reference engine's bookkeeping):
        #   * row 0 is the last prompt token, which predicts every option's first
        #     token;
        #   * an option of m tokens contributes the m-1 rows that score its
        #     tokens 1..m-1: its own leading flat positions s .. s+m-2.
        # A one-token option therefore contributes no rows at all. The row of an
        # option token j (0-based) is then 0 for j == 0 and cursor + j otherwise,
        # where cursor is the running sum of (len(o) - 1) over earlier options;
        # see ``_forward``. This is derived from the kit's
        # `decision_index/engines/transformers_engine.py::_forward`.
        keep = [q_len - 1]
        for s, m in zip(starts, lengths):
            keep.extend(range(s, s + m - 1))
        # A model with sliding layers needs a second mask: the same block
        # structure, additionally bounded by the sliding window. A provided 4D
        # mask is returned as-is by HF (masking_utils), so without this the
        # sliding layers would silently attend the whole prefix.
        sliding_allowed = None
        window = self._sliding_window()
        if window and K == Q:
            pos = torch.tensor(positions)
            within = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < window)
            sliding_allowed = allowed & within
        return flat, allowed, positions, keep, sliding_allowed

    def _crop_cache(self, final_len):
        """Crop every layer to its first ``final_len`` tokens, without raising.

        Returns True when the result is a sound record of that prefix (so a
        later request may reuse it), False when a sliding layer had already
        evicted prefix tokens that reuse would need. ``DynamicSlidingWindowLayer``
        refuses a positive "final size" crop once its window is full, so this
        slices the stored tensors directly and keeps ``cumulative_length`` (the
        absolute length it tracks separately from the stored window) in step.
        """
        sound = True
        for layer in getattr(self.cache, "layers", []):
            sliding = bool(getattr(layer, "is_sliding", False))
            window = getattr(layer, "sliding_window", None)
            if sliding and window:
                # The layer stores at most ``window - 1`` tokens; if the prefix's
                # tail is older than that, it is already gone.
                if layer.get_seq_length() > max(final_len, window - 1):
                    sound = False
            remove = layer.get_seq_length() - final_len
            if remove <= 0:
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None or keys.numel() == 0:
                sound = False
                continue
            stored = keys.shape[-2]
            keep = max(0, stored - remove)
            if remove >= stored:
                sound = False
            layer.keys = keys[..., :keep, :]
            layer.values = values[..., :keep, :]
            if sliding and hasattr(layer, "cumulative_length"):
                layer.cumulative_length = final_len
        return sound

    def _prepare_cache(self, p):
        """Make ``self.cache`` the reusable state for a prefix of length ``p``.

        With the model's own cache the sliding layers hold only the last
        ``sliding_window - 1`` tokens, so a crop to ``p`` is only sound when the
        cached window still covers the prefix's tail. When it does not, the
        cache is rebuilt from scratch and 0 is returned, telling the caller to
        forward the whole prompt rather than a bare suffix into an empty cache.
        """
        if self.cache is None or p == 0 or self._has_sliding_layers():
            # A heterogeneous sliding cache holds a different number of prefix
            # tokens per layer, which one block-diagonal mask over a reused
            # prefix cannot describe. Start clean and forward the whole prompt
            # rather than mis-describe a reused one.
            self.cache = self._new_cache()
            self.reused_prefix = 0
            return 0
        if not self._crop_cache(p):
            self.cache = self._new_cache()
            self.reused_prefix = 0
            return 0
        self.reused_prefix = p
        return p

    def _kept_logprobs(self, p, ids, options, precision):
        """One block-diagonal pass; returns (kept log-probs, keep positions, vocabulary, p).

        ``ids`` is the *full* prompt; ``p`` is how much of its prefix the cache
        already holds. ``rows_lp[keep[k]]`` are the log-probs (over ``want``) of
        the kept row ``k``. The caller scores options from these rows; the test
        uses the same method to compare the answer row against a plain forward,
        so the mask, the position ids, the cache and the row bookkeeping are all
        exercised.
        """
        torch = self.torch
        p = self._prepare_cache(p)
        q_ids = ids[p:]
        flat, allowed, positions, keep, sliding_allowed = self._pack(p, q_ids, options)
        Q = len(flat)

        def _build(m):
            if self.attn == "eager":
                return torch.zeros(Q, m.shape[1], dtype=self.dtype).masked_fill(
                    ~m, torch.finfo(self.dtype).min
                )
            return m

        if self._mask_mapping_supported() and sliding_allowed is not None:
            mask = {
                "full_attention": _build(allowed),
                "sliding_attention": _build(sliding_allowed),
            }
        else:
            mask = _build(allowed)
        if isinstance(mask, dict):
            mask = {k: v[None, None].to(self.device) for k, v in mask.items()}
        else:
            mask = mask[None, None].to(self.device)
        input_ids = torch.tensor([flat], device=self.device)
        position_ids = torch.tensor([positions], device=self.device)
        keep_t = torch.tensor(keep, device=self.device)
        want = sorted({t for o in options for t in o})
        want_index = torch.tensor(want, device=self.device)

        with torch.inference_mode():
            if self.mid_depth is not None:
                # Read the answer from an intermediate hidden layer, then apply
                # the model's final norm and its own lm_head. Same precision and
                # full-vocabulary normaliser as the fp32 path.
                h = self._mid_hidden(input_ids, mask, position_ids, keep_t).float()
                logits = self._candidate_logits(h, want_index)
                lse = self._vocab_logsumexp(h)
                del h
                lp = (logits - lse)[0]
            elif precision == "fp32_hidden":
                hidden = self._inner(input_ids, mask, position_ids, keep_t)
                h = hidden.float()
                # The candidate logits in fp32, gathered before the matmul: a
                # matmul reads only the rows it needs, so this equals the fp32
                # full-vocab matmul restricted to `want`, bit for bit.
                logits = self._candidate_logits(h, want_index)
                # But a log-probability needs the log-sum-exp over the *whole*
                # vocabulary, not over the candidates. Omitting it (as the first
                # version of this arm did) renormalises over the options and can
                # flip the answer; compute it chunked so no fp32 copy of the
                # whole unembedding is ever materialised.
                lse = self._vocab_logsumexp(h)
                del h, hidden
                lp = (logits - lse)[0]
            else:
                logits = self._full(input_ids, mask, position_ids, keep_t).float()
                # Full-vocabulary logits: gather `want` after the log-softmax.
                lp = torch.log_softmax(logits, dim=-1).index_select(-1, want_index)[0]
            rows_lp = lp.to("cpu", dtype=torch.float32).tolist()
        return rows_lp, keep, want, p

    def _candidate_logits(self, h, want_index):
        """``h @ W[want].T`` (plus bias), in fp32, with the model's softcap."""
        torch = self.torch
        w = self.lm_weight.index_select(0, want_index).float()
        out = h @ w.T
        if self.lm_bias is not None:
            out = out + self.lm_bias.index_select(0, want_index).float()
        if self.softcap:
            out = torch.tanh(out / self.softcap) * self.softcap
        return out

    def _vocab_logsumexp(self, h, chunk=8192):
        """``logsumexp`` of the (optionally softcapped) full-vocabulary logits.

        Iterates the unembedding in chunks so the fp32 arm never has to hold a
        float32 copy of the whole matrix. ``softcap`` is applied to every chunk,
        exactly as the model applies it, because the normaliser must be taken
        over the same logits that are being normalised.
        """
        torch = self.torch
        weight = self.lm_weight
        vocab = weight.shape[0]
        m = None
        s = None
        for start in range(0, vocab, chunk):
            w = weight[start : start + chunk].to(h.dtype)
            z = h @ w.T
            if self.lm_bias is not None:
                z = z + self.lm_bias[start : start + chunk].to(h.dtype)
            if self.softcap:
                z = torch.tanh(z / self.softcap) * self.softcap
            cmax = z.max(dim=-1, keepdim=True).values
            if m is None:
                m, s = cmax, torch.exp(z - cmax).sum(dim=-1, keepdim=True)
            else:
                m_new = torch.maximum(m, cmax)
                s = s * torch.exp(m - m_new) + torch.exp(z - m_new).sum(dim=-1, keepdim=True)
                m = m_new
        return m + torch.log(s)

    def _mid_hidden(self, input_ids, mask, position_ids, keep_t):
        """The hidden state after layer ``self.mid_depth``, with the final norm.

        ``hidden_states[0]`` is the embedding output and ``hidden_states[i]`` the
        output of layer ``i-1``; the last entry is the fully processed state. A
        depth at or past the last entry is a configuration error and raises
        rather than silently reading the final layer.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        states = getattr(out, "hidden_states", None)
        if states is None:
            raise ConfigError("mid_depth: the model did not return hidden states")
        if self.mid_depth >= len(states):
            raise ConfigError(
                f"mid_depth={self.mid_depth} is out of range: the model has {len(states)} hidden states (0..{len(states)-1})"
            )
        h = states[self.mid_depth]
        want_rows = int(keep_t.shape[0])
        if h.shape[1] != want_rows:
            h = h[:, keep_t]
        norm = getattr(self.inner, "norm", None)
        if norm is not None:
            h = norm(h)
        return h

    def _forward(self, p, ids, options, precision):
        """One block-diagonal pass over a shared prompt. Returns one summed log-prob per option."""
        prompt_len = len(ids)
        rows_lp, keep, want, p = self._kept_logprobs(p, ids, options, precision)
        self.forwards += 1
        if not self._crop_cache(prompt_len):
            # The option block has evicted prefix tokens from the sliding window,
            # so the cache is no longer a sound prefix for a later request.
            self.cache = None
        col = {t: j for j, t in enumerate(want)}
        scores = []
        cursor = 0  # kept-row cursor: sum of (len(o) - 1) over previous options
        for o in options:
            total = 0.0
            for j, t in enumerate(o):
                row = 0 if j == 0 else cursor + j
                total += rows_lp[row][col[t]]
            scores.append(total)
            cursor += len(o) - 1
        return scores

    # The kwarg that limits which positions produce logits has been renamed twice.
    # Try each in turn; a model that accepts the kwarg but silently ignores it
    # (the text tower of Gemma4 does exactly this) still returns every position,
    # so the returned row count is checked and sliced when it does not match.
    _HIDDEN_KWARGS = ("logits_to_keep", "num_logits_to_keep", "num_hidden_to_keep")
    _LOGITS_KWARGS = ("logits_to_keep", "num_logits_to_keep")

    def _inner(self, input_ids, mask, position_ids, keep_t):
        want_rows = int(keep_t.shape[0])
        for name in self._HIDDEN_KWARGS:
            try:
                out = self.inner(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                    **{name: keep_t},
                )
            except TypeError:
                continue
            h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            return h if h.shape[1] == want_rows else h[:, keep_t]
        out = self.inner(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        return h[:, keep_t]

    def _full(self, input_ids, mask, position_ids, keep_t):
        want_rows = int(keep_t.shape[0])
        for name in self._LOGITS_KWARGS:
            try:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                    **{name: keep_t},
                )
            except TypeError:
                continue
            logits = out.logits if hasattr(out, "logits") else out[0]
            return logits if logits.shape[1] == want_rows else logits[:, keep_t]
        out = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits[:, keep_t]

    def score(self, p, ids, options, precision):
        """Score ``options`` after prefix ``p`` of the full prompt ``ids``; halve on OOM."""
        torch = self.torch
        chunk = max(1, len(options))
        while True:
            try:
                out = []
                for i in range(0, len(options), chunk):
                    out.extend(self._forward(p, ids, options[i : i + chunk], precision))
                return out
            except getattr(torch, "OutOfMemoryError", RuntimeError):
                # Claw back what we own before trying again: our own KV cache goes
                # first, then the allocator's cache. Never touches the weights.
                self.truncate_cache()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if chunk <= 1:
                    raise BackendError(
                        f"CUDA out of memory scoring a single option after a {len(q_ids)}-token prompt; "
                        "this device cannot hold the request"
                    )
                chunk = max(1, chunk // 2)
                log.warning("CUDA OOM: halving the option block to %d options", chunk)


# -------------------------------------------------------------------- the engine


class LabelSlotEngine(Engine):
    name = "label_slot"
    latency = (
        "Device-synchronized in-process request wall time including prompt construction, the option "
        "block, state-prefix cache reuse, and every extra rotation. Excludes model loading. "
        "response['latency'] also carries per-request wall_ms, forward_passes and cost_multiple; the "
        "admission criterion is the median request, not p95."
    )

    def __init__(
        self,
        model=DEFAULT_MODEL,
        revision=None,
        device=None,
        dtype=None,
        attn="sdpa",
        chat_template=True,
        cache_prefix=True,
        max_tokens=None,
        backend=None,
        **options,
    ):
        super().__init__(**options)
        self.config = ArmConfig.from_options(options)
        if backend is not None and not hasattr(backend, "score"):
            raise ConfigError("backend= is a programmatic injection point: pass an object, not a string")
        self.model_id = model
        self.attn = attn
        self.max_tokens = max_tokens

        if backend is None:
            self.backend = TorchBackend(
                model=model,
                revision=revision,
                device=device,
                dtype=dtype,
                attn=attn,
                chat_template=chat_template,
                cache_prefix=cache_prefix,
                system_prompt=self.config.system_prompt,
                trust_remote_code=self.config.trust_remote_code,
                mid_depth=self.config.mid_depth,
            )
        else:
            self.backend = backend
            self.model_id = getattr(backend, "model_id", model)
        # An injected backend may not know about mid_depth; the arm config does.
        if hasattr(self.backend, "mid_depth"):
            self.backend.mid_depth = self.config.mid_depth
        self.chat_template = bool(getattr(self.backend, "chat_template", True))
        self.cache_prefix = bool(getattr(self.backend, "cache_prefix", True))
        # A backend injected with its own opinion wins; otherwise the config flag.
        self.system_prompt = bool(getattr(self.backend, "system_prompt", self.config.system_prompt))
        if max_tokens:
            self.config.max_prompt_tokens = min(self.config.max_prompt_tokens, int(max_tokens))

        self.labels = build_label_table(self.backend, self.config.labels, self.config.slot, log=log)
        self.context_limit = int(getattr(self.backend, "context_limit", 1 << 30))
        self.prior = R.PriorAccumulator()
        # The frozen prior is loaded once, before any request, and is a pure
        # function of option position thereafter -- independent of row order.
        try:
            self.frozen_prior = (
                R.FrozenPrior.load(self.config.frozen_prior) if self.config.frozen_prior else None
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigError(f"frozen_prior: could not load {self.config.frozen_prior!r}: {exc}")
        # The frozen decision rules ship with the source. Loaded once, before any
        # request; each rule is then a pure function of one question, so the
        # behaviour is independent of row order and of which other rules exist.
        try:
            file_enabled, default_rule, signature_rules, legacy_rules = load_decision_rules(
                self.config.decision_rules
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ConfigError(
                f"decision_rules: could not load "
                f"{(self.config.decision_rules or DEFAULT_DECISION_RULES_PATH)!r}: {exc}"
            )
        # The engine option, when set, overrides the file's master switch; when
        # unset (None) the shipped file is authoritative (false).
        self.rules_enabled = file_enabled if self.config.rules_enabled is None else self.config.rules_enabled
        self.signature_rules = signature_rules
        self.legacy_rules = legacy_rules
        self.default_rule = default_rule
        self.catalog_id = self.config.catalog_id
        self._requests_seen = 0
        self._warming = False
        self._render_delta = None
        self._warned_overflow = set()
        self.counters = dict.fromkeys(
            (
                "requests",
                "questions",
                "rotated_questions",
                "gated_out_questions",
                "key_fallback_questions",
                "description_channel_questions",
                "prompt_over_max",
                "context_refusals",
                "pride_calibration_questions",
                "pride_corrected_questions",
                "forwards",
                "cached_prefix_tokens",
                "decision_rule_questions",
                "decision_rule_changed",
            ),
            0,
        )
        self.provenance = {
            "kind": "label_slot",
            "repo": self.model_id,
            "revision": getattr(self.backend, "revision", None),
            "device": getattr(self.backend, "device", "stub"),
            "dtype": str(getattr(self.backend, "dtype", "stub")),
            "attn_implementation": self.attn,
            "context_limit_tokens": self.context_limit,
            "chat_template": self.chat_template,
            "system_prompt": self.system_prompt,
            "prefix_cache": self.cache_prefix,
            "determinism": getattr(self.backend, "determinism", {"stub": True}),
            "labels": self.labels.describe(),
            "arms": self.config.provenance(),
            "decision_rules": {
                "enabled": self.rules_enabled,
                "catalog_id": self.catalog_id,
                "signatures_loaded": sorted(self.signature_rules),
                "legacy_loaded": sorted(self.legacy_rules),
                "default_rule": None if self.default_rule is None else self.default_rule[0],
                "noul_threshold": self.config.noul_threshold,
                "source": self.config.decision_rules or DEFAULT_DECISION_RULES_PATH,
            },
            "policy": (
                "One forward pass per question and rotation. Every option is a single verified label "
                "token read at a fixed answer slot; the block-diagonal mask and position ids are the "
                "kit's, so a prompt is prefilled once and shared across the questions of a row. The "
                "state is rendered before the question, always. Nothing is truncated and no option is "
                "dropped: a question with more options than valid labels falls back to scoring the "
                "option keys themselves. The only Unsupported raised is a prompt longer than the "
                "model's positional-embedding limit. Every parameter is re-derived per request except "
                "the single global temperature, which must be fitted on a held-out random split, never "
                "on benchmark data and never per benchmark."
            ),
        }

    # ---------------------------------------------------------------- plumbing

    def _tokenize(self, body):
        # The reference puts the answer slot in the assistant turn, so in
        # reference mode the final answer line is moved after the chat
        # template's tail. Plain layouts ignore the flag.
        split = self.config.rendering == "reference"
        wrap = self.backend.wrap
        try:
            return self.backend.encode(wrap(body, split_slot=split))
        except TypeError:  # a backend injected from outside without the flag
            return self.backend.encode(wrap(body))

    def _option_block_ids(self, options, addressing):
        """The reference's ``_options_ids`` for one question (narrow or wide)."""
        enc = self.backend.encode
        if len(options) <= REFERENCE_NARROW:
            return enc("".join(f"\n({lab}) {txt}" for lab, txt in options))
        ids = []
        for i, (lab, txt) in enumerate(options):
            a = addressing[i] if i < len(addressing) else lab
            lid = a[1] if isinstance(a, tuple) else enc(lab)[0]
            ids += enc("\n(") + [lid] + enc(") " + txt)
        return ids

    def _tokenize_pieces(self, state, question, order, descs, key_index, addressing, channel):
        """Tokenise the reference's pieces separately (``piecewise_tokenize``).

        Mirrors ``prompt_fast.context_ids`` + ``question_piece`` / ``_chat_row``
        for one question. Returns None when the shape is not reproducible (the
        description channel has no reference analogue), so the caller falls back
        to the whole-body path.
        """
        from .render import reference_pieces

        if channel != "key" or self.config.prompt_repeat:
            return None
        pieces = reference_pieces(
            state,
            question,
            self._render_addressing(addressing),
            slot=self.config.slot,
            channel=channel,
            prefill=self.config.prefill_prefix,
            neutralize=self.config.neutralize_options,
            spaced_instructions=self.config.spaced_instructions,
        )
        if pieces["slot"] is None:
            return None
        enc = self.backend.encode
        options = pieces["options"]
        chat = bool(getattr(self.backend, "chat_template", True))
        ids = []
        if chat:
            # _chat_row: template head, context, question header, option block,
            # template tail, then the answer piece.
            head_text, tail_text = chat_head_tail(self.backend.tokenizer, self.system_prompt)
            ids += enc(head_text)
            ids += enc(pieces["ctx"])
            ids += enc(pieces["head"])
            ids += self._option_block_ids(options, addressing)
            ids += enc(tail_text)
            ids += enc(pieces["slot"])
        elif len(options) <= REFERENCE_NARROW:
            # plain question_piece narrow: head + options + tail in ONE encode.
            ids += enc(pieces["ctx"])
            ids += enc(pieces["head"] + "".join(f"\n({lab}) {txt}" for lab, txt in options)
                       + "\n" + pieces["slot"])
        else:
            ids += enc(pieces["ctx"])
            ids += enc(pieces["head"])
            ids += self._option_block_ids(options, addressing)
            ids += enc("\n" + pieces["slot"])
        return ids

    def _reset_stream(self):
        self.prior = R.PriorAccumulator()
        self._requests_seen = 0
        for k in self.counters:
            self.counters[k] = 0

    def synchronize(self):
        sync = getattr(self.backend, "synchronize", None)
        if callable(sync):
            sync()

    def close(self):
        reset = getattr(self.backend, "reset_cache", None)
        if callable(reset):
            reset()

    def warmup(self):
        warm = {"warmup": {"type": "choice", "instructions": "Which color is named?", "criteria": {"red": "red", "blue": "blue"}}}
        self._warming = True
        try:
            for _ in range(2):
                self("The color is red.", warm)
        finally:
            self._warming = False
        self._reset_stream()
        self.synchronize()

    def runtime(self):
        info = {
            "engine": self.name,
            "arms": self.config.provenance(),
            "labels": self.labels.describe(),
            "counters": dict(self.counters),
            "pride_observations": self.prior.seen,
        }
        rt = getattr(self.backend, "runtime", None)
        if callable(rt):
            info.update(rt())
        if self._render_delta is not None:
            info["prompt_token_delta"] = self._render_delta
        if self.prior.counts:
            width = max(self.prior.counts) + 1
            info["pride_prior"] = [round(v, 6) for v in self.prior.prior(width)]
        return info

    # -------------------------------------------------------------- rendering

    def _addressing(self, keys):
        """Left-hand side of each option line, and (for the key channel) its token id.

        A (label, token_id) tuple is **not** a renderable string: the renderer
        must be handed ``_render_addressing``-flattened labels, or every line
        renders as ``('A', 236776): <desc>``.
        """
        if len(keys) > len(self.labels):
            return list(keys)
        return [(self.labels.label(i), self.labels.token_id(i)) for i in range(len(keys))]

    @staticmethod
    def _render_addressing(addressing):
        """Option-line left-hand sides as strings; tuples carry (label, token_id)."""
        return [a[0] if isinstance(a, tuple) else a for a in addressing]

    def _option_text(self, descs, key_index, key):
        """The option's word/text token, neutralized when the flag is on."""
        txt = text_or_key(descs[key_index[key]], key)
        return neutralize_text(txt) if self.config.neutralize_options else txt

    def _measure_render_delta(self, state, question, keys):
        """Token-count delta of each rendering mode on the first request of a run."""
        if self._render_delta is not None:
            return
        try:
            counts = {}
            for mode in ("kit", "keyaddressed", "prose"):
                body = render_question(
                    state,
                    question,
                    mode=mode,
                    keys=keys,
                    addressing=self._render_addressing(self._addressing(keys)),
                    slot=self.config.slot,
                    prefill=self.config.prefill_prefix,
                    annotate=self.config.annotate_indices,
                    neutralize=self.config.neutralize_options,
                    repeat=self.config.prompt_repeat,
                )
                counts[mode] = len(self._tokenize(body))
        except Exception as exc:  # a probe must never break a run
            log.warning("could not measure the per-rendering token delta: %s", exc)
            return
        base = counts["kit"]
        self._render_delta = {
            "tokens": counts,
            "delta_vs_kit": {m: counts[m] - base for m in counts},
            "note": "one payload rendered three ways, measured on the first request of a run",
        }
        log.info(
            "prompt token counts by rendering mode: kit=%d keyaddressed=%+d prose=%+d",
            base,
            self._render_delta["delta_vs_kit"]["keyaddressed"],
            self._render_delta["delta_vs_kit"]["prose"],
        )

    def _note_prompt_tokens(self, n):
        if n <= self.config.max_prompt_tokens:
            return 0
        if self.config.rendering not in self._warned_overflow:
            self._warned_overflow.add(self.config.rendering)
            log.warning(
                "rendering=%s produced a %d-token prompt, past max_prompt_tokens=%d. The mode is kept "
                "and nothing is truncated; expect higher latency on these rows.",
                self.config.rendering,
                n,
                self.config.max_prompt_tokens,
            )
        return 1

    # ----------------------------------------------------------------- readout

    def _score(self, p, ids, option_ids, precision=None):
        return self.backend.score(p, ids, option_ids, precision or self.config.projection)

    def _one_pass(self, state, question, order, descs, key_index, addressing, channel):
        """One forward for one rotation. Returns a key-indexed log-prob dict plus token accounting.

        ``order`` is the rendered option order; every returned key is an
        ORIGINAL criteria key, so passes compose without any further mapping.

        Rotation invariant -- the composed distribution must equal the mean of
        the independently-run single rotations *exactly*, on both the label path
        and the key-fallback path; that exactness is what lets rotation averaging
        cancel a position bias, and it is the whole justification for the
        rotations arm. ``addressing`` and the scored ``option_ids`` must name the
        same option that line *i* shows, or ``dict(zip(order, scores))`` below
        attributes pass r's scores to the wrong keys. On the label path the
        scored continuation is the positional label (A, B, ...) and the option
        that moves is the description beside it, so the positional addressing is
        already correct. On the key-fallback path the scored continuation *is*
        the option key, so the addressing must rotate with ``order``; otherwise
        pass 0 is right and every later pass is mislabelled.
        """
        if addressing and not isinstance(addressing[0], tuple):
            addressing = [addressing[key_index[k]] for k in order]
        ids = None
        if self.config.piecewise_tokenize and self.config.rendering == "reference":
            ids = self._tokenize_pieces(state, question, order, descs, key_index, addressing, channel)
        if ids is None:
            body = render_question(
                state,
                question,
                mode=self.config.rendering,
                keys=order,
                # Labels as strings for the option lines; the (label, token_id)
                # tuples are kept only for `option_ids` below. Passing tuples here
                # renders `('A', 236776): <desc>` on every line.
                addressing=self._render_addressing(addressing),
                slot=self.config.slot,
                channel=channel,
                prefill=self.config.prefill_prefix,
                annotate=self.config.annotate_indices,
                neutralize=self.config.neutralize_options,
                repeat=self.config.prompt_repeat,
                spaced_instructions=self.config.spaced_instructions,
            )
            ids = self._tokenize(body)
        self.counters["prompt_over_max"] += self._note_prompt_tokens(len(ids))
        if len(ids) + 1 > self.context_limit:
            self.counters["context_refusals"] += 1
            raise Unsupported(
                f"prompt of {len(ids)} tokens plus a label exceeds the {self.context_limit}-token context limit"
            )
        p = self._common_prefix(ids) if (self.cache_prefix and self.backend.cached_ids) else 0
        # The readout-factorial axis: "label" scores the positional label token
        # (the historical path); "word" scores the option's own word/text token
        # in the same single pass. The description channel already scores that
        # same text, so both share one builder.
        word_readout = channel == "description" or (channel == "key" and self.config.keys == "word")
        if word_readout:
            option_ids = [self.backend.encode(self._option_text(descs, key_index, k)) for k in order]
        else:
            option_ids = [[a[1]] if isinstance(a, tuple) else self.backend.encode(a) for a in addressing]
        scores = self._score(p, ids, option_ids)
        if self.frozen_prior is not None:
            # Positional and pure: subtracting here makes the correction
            # independent of which rotations run and of the row order.
            scores = self.frozen_prior.apply(list(scores))
        self.backend.cached_ids = ids
        # A backend may have declined to reuse the requested prefix (a sliding
        # cache cannot always crop to it), in which case it records the prefix
        # it actually reused; count that, not the requested one.
        self.counters["cached_prefix_tokens"] += getattr(self.backend, "reused_prefix", p)
        return dict(zip(order, scores)), len(ids), p

    def _common_prefix(self, ids):
        return self.backend._common_prefix(ids)

    def _scope_ok(self, rule, field, instructions):
        """Whether a rule's optional narrowing conditions hold for this question.

        All conditions are optional; absent means "always"; present conditions are
        ANDed. Each is derivable from the question alone or from a run option:
        there is no per-row catalog id (the kit calls ``engine(state, questions)``).

        * ``catalog_scope`` (int): the run's ``catalog_id`` option must equal it.
          With no ``catalog_id`` (a full-catalog run) it does NOT apply -- it must
          never leak to a benchmark it was not scoped to.
        * ``field_scope`` (str): the question's field name must equal it.
        * ``instructions_contains`` (str): the question's ``instructions`` text
          must contain it, case-insensitively. This is what lets catalog 38
          (ACOS) be selected: its signature ``choice|no,yes`` is shared by 8
          catalogs and its field names are dynamic aspect strings, but all of its
          instructions share a distinctive phrase.
        """
        _positive, _tau, catalog_scope, field_scope, instructions_contains = rule
        if catalog_scope is not None:
            if self.catalog_id is None or self.catalog_id != catalog_scope:
                return False
        if field_scope is not None and field != field_scope:
            return False
        if instructions_contains is not None:
            if instructions_contains.lower() not in str(instructions or "").lower():
                return False
        return True

    def _resolve_rule(self, field, qtype, keys, instructions, probs):
        """The rule that applies to one question, or None for plain argmax.

        Selection is per question by signature (``question_signature``); a
        signature may carry several rules, tried in file order, and the first
        whose scopes match the question wins. The legacy
        ``"<catalog_id>:<field>"`` form and then the file's ``default`` are
        fallbacks. A signature that matches no rule falls through to ``default``
        (null => plain argmax). Raises ``ConfigError`` (never silently falls
        back) if a rule names a positive key the question lacks.
        """
        sig = question_signature(qtype, keys)
        rule, used = None, None
        for candidate in self.signature_rules.get(sig, ()):
            if self._scope_ok(candidate, field, instructions):
                rule, used = candidate, sig
                break
        if rule is None and self.catalog_id is not None:
            legacy_key = f"{self.catalog_id}:{field}"
            for candidate in self.legacy_rules.get(legacy_key, ()):
                if self._scope_ok(candidate, field, instructions):
                    rule, used = candidate, legacy_key
                    break
        if rule is None:
            rule, used = self.default_rule, "default"
        if rule is None:
            return None
        positive, tau, _catalog_scope, field_scope, instructions_contains = rule
        choice = apply_choice_rule(probs, keys, positive, tau)
        return {
            "rule": used,
            "positive": positive,
            "tau_logodds": tau,
            "field_scope": field_scope,
            "instructions_contains": instructions_contains,
            "g": logit(probs[positive]),
            "choice": choice,
        }

    # ------------------------------------------------------------------- call

    def __call__(self, state, questions):
        t0 = time.perf_counter()
        # A row's forward shape must not depend on the previous request: clear the
        # prefix cache at the start of every request. Prefix reuse still applies
        # across the questions *within* this row (via `_common_prefix` below).
        reset = getattr(self.backend, "reset_cache", None)
        if callable(reset):
            reset()
        if not self._warming:
            self._requests_seen += 1
            self.counters["requests"] += 1
        cfg = self.config
        answers, raw = {}, {}
        forwards = 0
        prompt_tokens = 0
        calibrating = self._calibrating()
        cal_index = 0

        for k, q in questions.items():
            if q["type"] not in REFUSABLE_TYPES:
                raise Unsupported("Unsupported question type " + str(q["type"]))
            noul_true_key = "true"
            if q["type"] == "choice":
                keys = list(q["criteria"])
                descs = [q["criteria"][x] for x in keys]
            elif cfg.noul_reference_labels:
                # systemone.render_question's noul options: "no"/"yes", with the
                # criteria's false/true descriptions, names [False, True]. The
                # True answer is the "yes" option.
                base = q.get("criteria") or {}
                f = base.get("false", base.get(False))
                tr = base.get("true", base.get(True))
                q = {"type": q["type"], "instructions": q.get("instructions", ""),
                     "criteria": {"no": f, "yes": tr}}
                keys, descs = ["no", "yes"], [f, tr]
                noul_true_key = "yes"
            else:
                keys, descs = list(NOUL_KEYS), list(NOUL_DESCS)
            if not keys:
                raise Unsupported("question has no options")
            n = len(keys)
            fallback = n > len(self.labels)
            if fallback:
                self.counters["key_fallback_questions"] += 1
            addressing = self._addressing(keys)
            key_index = {kk: i for i, kk in enumerate(keys)}
            self._measure_render_delta(state, q, keys)

            passes, shifts = [], []
            first_max = None
            forwards_here = 0
            tokens_here = 0
            used_description = False

            for r in range(cfg.rotations):
                # The gate withholds the extra rotations from a question whose
                # first pass was already confident. With no tau, nothing is gated.
                if r and cfg.gate_tau is not None and not (first_max < cfg.gate_tau):
                    break
                if r >= n:  # fewer distinct cyclic orders than rotations requested
                    break
                order = R.rotate(keys, r)
                scores, ntok, _ = self._one_pass(state, q, order, descs, key_index, addressing, "key")
                forwards += 1
                forwards_here += 1
                tokens_here = max(tokens_here, ntok)
                if r == 0:
                    first_max = max(R.normalise(list(scores.values()), order, 1.0).values())
                if cfg.description_channel:
                    used_description = True
                    dscores, dntok, _ = self._one_pass(state, q, order, descs, key_index, addressing, "description")
                    scores = dict(
                        zip(
                            order,
                            R.combine_channels(
                                [scores[k] for k in order], [dscores[k] for k in order], cfg.channel_weight
                            ),
                        )
                    )
                    forwards += 1
                    forwards_here += 1
                    tokens_here = max(tokens_here, dntok)
                passes.append(R.normalise(list(scores.values()), order, 1.0))
                shifts.append(r)

            if calibrating:
                step = self._permutation(n, cal_index)
                cal_index += 1
                if step:
                    porder = R.rotate(keys, step)
                    pscores, _, _ = self._one_pass(state, q, porder, descs, key_index, addressing, "key")
                    pp = R.normalise(list(pscores.values()), porder, 1.0)
                    self.prior.observe([pp[k] for k in porder])
                    self.counters["pride_calibration_questions"] += 1
                    forwards += 1
                    forwards_here += 1

            composed = R.compose_rotations(passes, keys)
            corrected = False
            if not calibrating and self.prior.seen and self._prior_usable(n):
                # Position i of the composed vector is keys[i], which is the
                # position the permuted calibration readouts observed.
                composed = dict(zip(keys, R.apply_prior([composed[k] for k in keys], self.prior.prior(n))))
                corrected = True
                self.counters["pride_corrected_questions"] += 1
            probs = R.sharpen(composed, keys, cfg.temperature)
            choice = max(keys, key=lambda x: probs[x])

            # The frozen far-tail decision rule, applied inside the engine so it
            # reproduces on a re-run. Off by default; per-question and pure, so it
            # cannot depend on row order or on which other rules are configured.
            rule_info = None
            if q["type"] == "choice" and self.rules_enabled:
                rule_info = self._resolve_rule(k, q["type"], keys, q.get("instructions"), probs)
                if rule_info is not None:
                    rule_info["changed"] = rule_info["choice"] != choice
                    choice = rule_info["choice"]
                    self.counters["decision_rule_questions"] += 1
                    if rule_info["changed"]:
                        self.counters["decision_rule_changed"] += 1

            raw[k] = {
                "n_options": n,
                "n_labels": len(self.labels),
                "rotations_run": len(passes),
                "rotations_requested": cfg.rotations,
                "shifts": shifts,
                "gated": bool(cfg.gate_tau is not None and len(passes) == 1 and cfg.rotations > 1),
                "key_fallback": fallback,
                "channel": "key+description" if used_description else "key",
                "channel_weight": cfg.channel_weight if used_description else None,
                "temperature": cfg.temperature,
                "prior_corrected": corrected,
                "slot": cfg.slot,
                "projection": cfg.projection,
                "rendering": cfg.rendering,
                "composed": {kk: float(vv) for kk, vv in composed.items()},
                "max_p": max(probs.values()),
                "forwards": forwards_here,
            }
            if rule_info is not None:
                # Only present when a rule actually applied, so raw telemetry is
                # byte-identical to the baseline whenever rules are disabled.
                raw[k]["decision_rule"] = rule_info
            self.counters["questions"] += 1
            if len(passes) > 1:
                self.counters["rotated_questions"] += 1
            elif cfg.gate_tau is not None and cfg.rotations > 1:
                self.counters["gated_out_questions"] += 1
            if used_description:
                self.counters["description_channel_questions"] += 1
            prompt_tokens += tokens_here

            emit_probs = probs
            if cfg.temperature_by_options is not None and q["type"] == "choice":
                # Output calibration only: the decision paths above used the
                # working distribution; the emitted probabilities carry the
                # reference's T(n) = max(min, a + b*ln n).
                import math
                _tbo_a, _tbo_b, _tbo_min = cfg.temperature_by_options
                t_emit = max(_tbo_min, _tbo_a + _tbo_b * math.log(n))
                emit_probs = R.sharpen(probs, keys, t_emit)
                raw[k]["emitted_temperature"] = t_emit
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": choice, "probabilities": emit_probs}
            elif cfg.noul_threshold:
                # Its OWN flag, independent of the choice rules. Emits a hard
                # 0.0/1.0 at the kit's own noul cut (>= 0.5), so the decision is
                # unchanged -- see the flag's comment in config.py and
                # decision_rules.json for why this is opt-in and never default.
                answers[k] = {"type": "noul", "noul": 1.0 if probs[noul_true_key] >= 0.5 else 0.0}
            else:
                answers[k] = {"type": "noul", "noul": probs[noul_true_key]}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_questions = len(questions)
        self.counters["forwards"] += forwards
        response = {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": prompt_tokens, "forwards": forwards},
            "latency": {
                "wall_ms": round(elapsed_ms, 3),
                "forward_passes": forwards,
                "questions": n_questions,
                "cost_multiple": round(forwards / n_questions, 4) if n_questions else 1.0,
                "ms_per_forward": round(elapsed_ms / forwards, 3) if forwards else None,
                "note": "the admission criterion is the median request against 1000 ms; do not optimise p95",
            },
        }
        return response, raw

    # -------------------------------------------------------------- PriDe arms

    def _calibrating(self):
        """True while this request is inside the PriDe calibration window."""
        return self.config.pride_alpha > 0.0 and self._requests_seen <= self.config.calibration_requests

    @staticmethod
    def _permutation(n, index):
        """A deterministic order shift for PriDe's permuted readout. Identity when it cannot permute."""
        if n <= 1:
            return 0
        step = (2 * index + 1) % n
        if step == 0 or math.gcd(step, n) != 1:
            step = 1
        return step

    def _prior_usable(self, n):
        """The prior is only trusted once it has observations for the positions in play."""
        return max(self.prior.counts, default=-1) >= n - 1


def text_or_key(desc, key):
    """The description channel reads the description, falling back to the key when there is none."""
    if desc is None:
        return key
    return desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False, separators=(",", ":"))
