#!/usr/bin/env python3
"""Apply the frozen decision rules (engine/decision_rules.json) to kit records.

The production reference (``runs/full-rules``) was produced by the engine with
``rules_enabled=true``: per question, a signature-selected rule maps the
probabilities to a choice (ACOS, iSarcasm, FinEntity). The vLLM pipeline's
readout omits that step, so the fold applies the identical function here --
imported from the engine itself via ``vllm_fastpass._engine_modules()`` (never
re-implemented) -- before the full-cal transform.

Deterministic: a pure function of (question, probabilities); the rule file
travels with the engine source, so a maintainer re-run reproduces it.

Fold order: ``fold_records -> apply_rules_to_records -> temperature -> summarize``.
"""
from __future__ import annotations

import vllm_fastpass as vf  # noqa: E402  (path setup happens at its import)


def _engine():
    return vf._engine_modules()["engine"]


def _resolve_factory():
    eng = _engine()
    # NOTE: the shipped file has rules_enabled=false (opt-in); the production
    # reference ran with rules_enabled=true, so the fold applies the file's
    # rules unconditionally -- mirroring the reference configuration.
    _enabled, _default_rule, signature_rules, _legacy = eng.load_decision_rules()

    def resolve(question, field):
        """(keys, rule) for a question, or (None, None). First match wins."""
        keys = list((question.get("criteria") or {}).keys())
        if not keys:
            return None, None
        sig = eng.question_signature(question.get("type"), keys)
        instructions = str(question.get("instructions") or "")
        for rule in signature_rules.get(sig, ()):
            positive, tau, catalog_scope, field_scope, instructions_contains = rule
            # Mixed-catalog run: the engine has no per-row catalog id, so
            # catalog_scope rules cannot match here either.
            if catalog_scope is not None:
                continue
            if field_scope is not None and field != field_scope:
                continue
            if instructions_contains is not None and \
                    instructions_contains.lower() not in instructions.lower():
                continue
            return keys, rule
        return None, None

    return resolve


def load_rule_matcher():
    """Predicate: would a decision rule fire for this gate candidate?"""
    resolve = _resolve_factory()

    def matcher(candidate):
        question = candidate.get("question") or {}
        keys, rule = resolve(question, candidate.get("question_key"))
        return rule is not None

    return matcher


def apply_rules_to_records(rows, records, stats=None):
    """Override the choice of every rule-matched answer (in place).

    Returns per-count statistics: matched, changed, missing_probs, errors.
    """
    eng = _engine()
    resolve = _resolve_factory()
    stats = stats if stats is not None else {
        "matched": 0, "changed": 0, "missing_probs": 0, "errors": 0}
    index = {record.get("run_id"): record for record in records}
    for row in rows:
        run_id = (row.get("_evaluation") or {}).get("run_id") or row.get("id")
        record = index.get(run_id)
        if record is None:
            continue
        answers = (record.get("response") or {}).get("answers") or {}
        for field, question in (row.get("questions") or {}).items():
            answer = answers.get(field)
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                continue
            keys, rule = resolve(question, field)
            if rule is None:
                continue
            stats["matched"] += 1
            positive, tau = rule[0], rule[1]
            probs = answer.get("probabilities") or {}
            if positive not in probs:
                stats["missing_probs"] += 1
                continue
            try:
                new_choice = eng.apply_choice_rule(probs, keys, positive, tau)
            except Exception:  # noqa: BLE001 - count and keep the old choice
                stats["errors"] += 1
                continue
            if new_choice != answer.get("choice"):
                stats["changed"] += 1
            answer["choice"] = new_choice
    return stats
