# LATENCY-CERTIFICATE.md — S1 GPU latency-v1 (v1.x+ entry)

Identical protocol and certified figures as the v1.0 entry (same frozen engine,
same BI/serial/no-thinking path); the levers are deterministic post-processing and
do not add inference latency.

| statistic | value | bar | pass |
|---|---:|---:|---|
| median | **116.4 ms** | ≤1000 | ✅ |
| mean | **369.6 ms** | ≤1000 | ✅ |
| p80 | **319.9 ms** | ≤1000 | ✅ |
| n | 749 timed + 1 (750-row sample) | — | — |

Sources and protocol: see `v1.0/latency/LATENCY-CERTIFICATE.md`. Cited:
`/tmp/opencode/COMPACT-STATE.md` (S1 gate line), `latency/latency_cpu.json`,
`latency/latency_rows.meta.json`.

The maintainers re-measure privately; this is the disclosure of what we measured.
