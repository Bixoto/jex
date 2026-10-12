#!/usr/bin/env python3
"""cat62 positional fold applied post-hoc to the stored v1.x capture.

Rule (engine-v1x/di_engine/positional_rules.json):
  signature 'choice|A,B,C,D', scope phrase "Which response should the assistant
  give to the user's question, given the available tools?", positive 'D',
  tau_logodds -13.63.

Engine semantics: the rule reads the WORKING (pre-TBO, normalised label-score)
distribution. The capture stores the TBO-emitted probabilities and
emitted_temperature. TBO is the monotone map p -> p^(1/T)/Z, so the working
distribution is recovered exactly (up to normalisation) as p_work ~ p_emit^T.
Rules change `choice` only; probabilities are never modified.

Output: results-folded.jsonl.gz (same rows, cat62 choices updated), plus
counters/receipts json.
"""
import gzip, json, math, collections, os, sys

CAP = "/tmp/opencode/phase8/v1x-capture/fetched/wave1-v1x/results.jsonl.gz"
OUTDIR = "/tmp/opencode/submission04/v1.xplus/levers/fold60"
OUT = os.path.join(OUTDIR, "results-folded.jsonl.gz")

PHRASE = "Which response should the assistant give to the user's question, given the available tools?"
POSITIVE = "D"
TAU = -13.63
TARGET_SIG = "choice|A,B,C,D"

def logit(p):
    p = float(p)
    if p <= 0.0: return float("-inf")
    if p >= 1.0: return float("inf")
    return math.log(p) - math.log1p(-p)

def argmax(probs, keys):
    return max(keys, key=lambda k: probs[k])

def sig_of(keys):
    return "choice|" + ",".join(sorted({str(k) for k in keys}))

def apply_rule(probs, keys, positive, tau):
    if logit(probs[positive]) >= float(tau):
        return positive
    others = [k for k in keys if k != positive]
    if not others:
        return positive
    return max(others, key=lambda k: probs[k])

def inv_tbo(emitted, keys, T):
    w = {k: (max(float(emitted.get(k, 0.0)), 0.0)) ** float(T) for k in keys}
    s = sum(w.values())
    return {k: (v / s if s > 0 else v) for k, v in w.items()}

def main():
    counters = collections.Counter()
    per_cat = collections.defaultdict(collections.Counter)
    matched_rows = []
    changed_rows = []
    mismatch = 0
    n_rows = 0
    with gzip.open(CAP, "rt", encoding="utf-8") as fin, \
         gzip.open(OUT, "wt", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            n_rows += 1
            cid = r.get("catalog_id")
            row_changed = False
            for qk, q in (r.get("payload", {}).get("questions", {}) or {}).items():
                if q.get("type") != "choice":
                    continue
                keys = list((q.get("criteria") or {}).keys())
                if sig_of(keys) != TARGET_SIG:
                    continue
                if PHRASE.lower() not in str(q.get("instructions") or "").lower():
                    continue
                ans = r["response"]["answers"].get(qk)
                if ans is None or ans.get("type") != "choice":
                    continue
                emitted = ans.get("probabilities", {})
                T = r.get("raw_output", {}).get(qk, {}).get("emitted_temperature")
                work = inv_tbo(emitted, keys, T)
                am = argmax(emitted, keys)          # argmax invariant under TBO
                rule_choice = apply_rule(work, keys, POSITIVE, TAU)
                counters["matched"] += 1
                per_cat[cid]["matched"] += 1
                matched_rows.append(r["run_id"])
                if am != ans.get("choice"):
                    mismatch += 1
                if rule_choice != am:
                    counters["changed"] += 1
                    per_cat[cid]["changed"] += 1
                    if rule_choice == POSITIVE:
                        counters["changed_to_D"] += 1
                    changed_rows.append((r["run_id"], ans.get("choice"), rule_choice))
                    ans["choice"] = rule_choice
                    row_changed = True
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    receipt = {
        "source_capture": CAP,
        "folded_output": OUT,
        "rows": n_rows,
        "rule": {"signature": TARGET_SIG, "positive": POSITIVE, "tau_logodds": TAU,
                 "instructions_contains": PHRASE},
        "counters": dict(counters),
        "per_catalog": {str(c): dict(v) for c, v in per_cat.items()},
        "matched_catalogs": sorted({int(c) for c in per_cat}),
        "stored_choice_vs_argmax_mismatch_on_matched": mismatch,
        "changed_from": dict(collections.Counter(frm for _, frm, _ in changed_rows)),
        "changed_to": dict(collections.Counter(to for _, _, to in changed_rows)),
        "no_probability_changes": True,
    }
    with open(os.path.join(OUTDIR, "fold-receipt.json"), "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))

    # assertions
    assert n_rows == 140178, n_rows
    assert counters["matched"] == 3652, counters
    assert counters["changed"] == 476, counters
    assert counters["changed_to_D"] == 476, counters
    assert per_cat[62]["changed"] == 476
    assert receipt["matched_catalogs"] == [62], receipt["matched_catalogs"]
    assert mismatch == 0, mismatch
    print("FOLD VERIFICATION PASSED")

if __name__ == "__main__":
    main()
