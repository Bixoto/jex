#!/usr/bin/env python3
"""vllm_reason_resume.py — reasoning-only resume over an existing fast pass.

Gates the fast-pass results at ``--coverage`` and reasons the selected items in
margin order (lowest first = nested coverages), appending each finished chunk to
``--out`` as plain JSONL (truncation-safe, append-only), with progress / rate /
ETA logs. Stops cleanly after ``--max-seconds``; the partial output folds
normally with the fast file. Does NOT fold and does NOT upload — run it detached
with stdout redirected to a file (a dead SSH pipe must never kill it), then fold
locally.

Usage (on the VM, inside the staged repo):
  setsid nohup env HF_TOKEN=... python3 harness/vllm_reason_resume.py \
    --rows /tmp/vllm-production/panel.jsonl.gz \
    --fast-results /tmp/vllm-production/vllm-production-fast-results.jsonl.gz \
    --out /tmp/vllm-production/reasoned-prod.jsonl \
    --coverage 0.30 --cap 512 --reason-batch 32 --chunk 512 \
    --max-seconds 93600 > /tmp/reason-resume.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_engine(args):
    """Set BI env, construct the engine, resolve the tokenizer."""
    import vllm_fastpass as vf

    env_used = vf.apply_env(args)
    import vllm

    vf._engine_modules()  # ensure the engine package imports (side effect only)
    api_probe = vf.introspect_vllm(vllm)
    requested = vf._requested_kwargs(args, api_probe)
    llm, accepted, dropped, notes, err = vf._construct_llm(vllm.LLM, requested)
    if llm is None:
        raise SystemExit(f"LLM construction failed: {err}")
    tokenizer = vf._resolve_tokenizer(llm, args.model, args.trust_remote_code)
    return llm, tokenizer, env_used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--fast-results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coverage", type=float, default=0.30)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--reason-batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--skip-rule-matched", action="store_true",
                    help="drop gated items a decision rule will override at fold")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    ap.add_argument("--rendering", default="reference")
    ap.add_argument("--slot", default="paren")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    def emit(event, **kw):
        payload = {"event": event,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        payload.update(kw)
        print(json.dumps(payload), flush=True)

    import vllm_pipeline as pl

    selected, gate, join_stats = pl.select_gate(
        args.fast_results, args.rows, args.coverage)
    emit("gated", selected=len(selected), gate=gate, join_stats=join_stats)

    if args.skip_rule_matched:
        import decision_rules_apply
        matcher = decision_rules_apply.load_rule_matcher()
        before = len(selected)
        selected = [c for c in selected if not matcher(c)]
        emit("rule_skip", before=before, kept=len(selected),
             skipped=before - len(selected))

    llm, tokenizer, env_used = build_engine(args)
    emit("engine_ready", env=env_used)

    n = len(selected)
    done = 0
    errors_in_a_row = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, n, args.chunk):
            chunk = selected[start:start + args.chunk]
            try:
                records = pl.run_reasoning(llm, tokenizer, chunk, args,
                                           int(args.cap), int(args.reason_batch))
                errors_in_a_row = 0
            except Exception as exc:  # noqa: BLE001 - keep the partial run alive
                errors_in_a_row += 1
                emit("chunk_error", start=start, size=len(chunk),
                     error=f"{type(exc).__name__}: {exc}",
                     consecutive=errors_in_a_row)
                if errors_in_a_row >= 5:
                    emit("aborting_after_consecutive_errors", start=start)
                    break
                continue
            for record in records:
                out.write(json.dumps(record) + "\n")
            out.flush()
            os.fsync(out.fileno())
            done += len(chunk)
            el = time.time() - t0
            rate = (done / el) if el > 0 else 0.0
            emit("progress", done=done, total=n, elapsed_s=round(el, 1),
                 rate_items_per_s=round(rate, 4),
                 eta_s=(round((n - done) / rate) if rate > 0 else None))
            if args.max_seconds and el >= float(args.max_seconds):
                emit("max_seconds_reached", done=done)
                break
    emit("done", done=done, total=n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
