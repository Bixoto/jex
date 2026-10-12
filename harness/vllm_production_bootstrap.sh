#!/usr/bin/env bash
#
# vllm_production_bootstrap.sh -- run INSIDE a rented GPU VM to execute the
# integrated Route-B pipeline over the FULL 0.2.1 edition on one RTX PRO 6000
# (sm_120) and print the completion marker the wrapper waits for.
#
# It is a faithful clone of vllm_pipeline_bootstrap.sh (same provider, silicon
# gate, setup/install blocks and pins). The ONLY differences:
#
#   * the default bundle is the full-edition production bundle
#     (subsets/production_bundle.tar.gz: all 150,759 in-edition rows + the
#     matching full-rules reference), not the six-catalog pilot bundle;
#   * the default upload prefix is runs/route-b-production;
#   * the default coverage is 0.15 and the default reason batch is 32
#     (production calibration point), both still overridable via the forwarded
#     VLLM_PIPELINE_* envs;
#   * it passes --panel-catalogs (every catalog in the panel) when the staged
#     pipeline supports it, so the summary's full_index_delta covers the whole
#     edition rather than the pilot's six catalogs. If the staged pipeline does
#     not carry that flag, it logs a warning and falls back to the pipeline
#     default (the pilot six) -- the run still completes.
#
# pipeline:
#   [setup]   python+pip, C compiler + python3-dev
#   [install] vllm==0.30.0 (fallback: newest 0.30.x), transformers==5.14.1;
#             torch is left to vLLM's resolver. Resolved versions are logged.
#   [preflight] CUDA + RTX PRO 6000 gate
#   [unpack]  the production bundle (panel + reference fast pass + manifest)
#   [run]     harness/vllm_pipeline.py over the full edition
#   [upload]  merged results + fast-pass results + summary to
#             J-I-B/di-smoke:runs/route-b-production/
#
# Inputs (from the wrapper's environment):
#   DI_REPO_SRC   rebuilt repo dir on the VM (required)
#   DI_REPO_COMMIT tag of the staged tree (required, for the marker only)
#   HF_TOKEN      required for the gated model download and the upload
#   DI_MODEL      default google/gemma-4-31B-it
#   DI_ROWS       the production BUNDLE (default subsets/production_bundle.tar.gz)
#   VLLM_PIN                    default 0.30.0
#   VLLM_TRANSFORMERS_PIN       default 5.14.1 (research pin; fallback logged)
#   VLLM_PIPELINE_LIMIT         default empty (full edition); set N for a smoke
#   VLLM_PIPELINE_COVERAGE      default 0.15
#   VLLM_PIPELINE_CAP           default 512
#   VLLM_PIPELINE_FAST_BATCH    default 8
#   VLLM_PIPELINE_REASON_BATCH  default 32
#   VLLM_PIPELINE_TEMPERATURE_BY_OPTIONS  optional full-cal transform ("a b min")
#   VLLM_ATTENTION_BACKEND      default TRITON_ATTN
#   VLLM_CUDAGRAPH_MODE         default FULL_DECODE_ONLY
#   VLLM_MAX_MODEL_LEN          default 8192
#   VLLM_GPU_MEMORY_UTILIZATION default 0.9
#   DI_UPLOAD_PATH              default runs/route-b-production
#
# Usage via the wrapper:
#   DI_BOOTSTRAP_LOCAL=harness/vllm_production_bootstrap.sh \
#     ./harness/nebius_arm.sh --arm vllm-pipeline \
#       --rows subsets/production_bundle.tar.gz
set -uo pipefail

export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive

log() { printf 'vllm-production: %s\n' "$*"; }
STAMP="/tmp/runs/VLLM_PIPELINE_COMPLETE"
mkdir -p /tmp/runs
rm -f "$STAMP"
die() {
  printf 'FATAL: %s\n' "$*" >&2
  echo "VLLM_PIPELINE_COMPLETE status=fail reason=$*" > "$STAMP"
  printf 'VLLM_PIPELINE_COMPLETE status=fail reason=%s\n' "$*"
  exit "${2:-2}"
}

: "${DI_REPO_SRC:?vllm_production_bootstrap.sh: DI_REPO_SRC is required}"
: "${DI_REPO_COMMIT:?vllm_production_bootstrap.sh: DI_REPO_COMMIT is required}"
: "${HF_TOKEN:?vllm_production_bootstrap.sh: HF_TOKEN is required}"

MODEL="${DI_MODEL:-google/gemma-4-31B-it}"
BUNDLE_REL="${DI_ROWS:-subsets/production_bundle.tar.gz}"
RESULTS_REPO="${DI_RESULTS_REPO:-J-I-B/di-smoke}"
UPLOAD_PATH="${DI_UPLOAD_PATH:-runs/route-b-production}"
VLLM_PIN="${VLLM_PIN:-0.30.0}"
TRANSFORMERS_PIN="${VLLM_TRANSFORMERS_PIN:-5.14.1}"
LIMIT="${VLLM_PIPELINE_LIMIT:-}"
COVERAGE="${VLLM_PIPELINE_COVERAGE:-0.15}"
CAP="${VLLM_PIPELINE_CAP:-512}"
FAST_BATCH="${VLLM_PIPELINE_FAST_BATCH:-8}"
REASON_BATCH="${VLLM_PIPELINE_REASON_BATCH:-32}"
TBO="${VLLM_PIPELINE_TEMPERATURE_BY_OPTIONS:-}"
ATTN="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
CUDAGRAPH="${VLLM_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
WORK="/tmp/vllm-production"
REPO="${DI_REPO_SRC}"

PY="${DI_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="python3"
command -v "$PY" >/dev/null 2>&1 || die "no python on PATH" 2
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

log "arm=${DI_ARM:-vllm-production} commit=${DI_REPO_COMMIT} started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "repo=${REPO} model=${MODEL} bundle=${BUNDLE_REL}"
log "coverage=${COVERAGE} cap=${CAP} fast_batch=${FAST_BATCH} reason_batch=${REASON_BATCH} limit='${LIMIT}'"
log "temperature_by_options='${TBO}'"
log "attention=${ATTN} cudagraph=${CUDAGRAPH} max_model_len=${MAX_MODEL_LEN} gpu_util=${GPU_UTIL}"
log "pins: vllm==${VLLM_PIN} transformers==${TRANSFORMERS_PIN} (torch resolved by vLLM)"
log "upload=${RESULTS_REPO}:${UPLOAD_PATH}/"

# ---------------------------------------------------------------------------
# [fix 0] python + pip (bare CUDA image may ship neither pip nor PEP 668).
# ---------------------------------------------------------------------------
echo "############ [fix 0] python + pip ############"
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends python3-pip || die "could not install python3-pip" 2
fi
"$PY" -m pip --version || die "pip is still unavailable" 2
export PIP_BREAK_SYSTEM_PACKAGES=1
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# [fix 1] C compiler + Python headers BEFORE any model/Triton work.
# ---------------------------------------------------------------------------
echo "############ [fix 1] C compiler + python headers ############"
if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends gcc g++ make || true
fi
if ! ls /usr/include/python3*/Python.h >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends python3-dev || true
fi
command -v gcc >/dev/null 2>&1 || die "no gcc after install; Triton JIT will fail late and billed" 90
ls /usr/include/python3*/Python.h >/dev/null 2>&1 && echo "Python.h: present" || echo "Python.h: MISSING (Triton/JIT compile may fail)"
gcc --version | head -n 1

# ---------------------------------------------------------------------------
# [install] vLLM pinned, fallback to the newest 0.30.x (same block as the
# bake-off bootstrap); transformers pinned to the research version.
# ---------------------------------------------------------------------------
echo "############ [install] vLLM + transformers ############"
log "pip install vllm==${VLLM_PIN} (torch resolved by vLLM)"
if ! "$PY" -m pip install -q "vllm==${VLLM_PIN}"; then
  log "vllm==${VLLM_PIN} unavailable; falling back to newest 0.30.x"
  "$PY" -m pip install -q "vllm>=0.30.0,<0.31" || die "could not install any vLLM 0.30.x" 91
fi

log "pip install transformers==${TRANSFORMERS_PIN} (research pin)"
"$PY" -m pip install -q "transformers==${TRANSFORMERS_PIN}" || \
  log "WARNING: transformers==${TRANSFORMERS_PIN} install failed; leaving vLLM's resolved version"

TRANSFORMERS_ACTUAL="$("$PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'unimportable')"
if [ "$TRANSFORMERS_ACTUAL" != "$TRANSFORMERS_PIN" ]; then
  log "WARNING: transformers resolved to ${TRANSFORMERS_ACTUAL}, not the ${TRANSFORMERS_PIN} pin"
fi

if ! "$PY" -c 'import vllm' >/tmp/vllm-production-import.log 2>&1; then
  log "vLLM import failed after the transformers pin; reinstalling vLLM's own transformers"
  tail -n 20 /tmp/vllm-production-import.log
  "$PY" -m pip install -q --upgrade transformers || true
  "$PY" -c 'import vllm' >/tmp/vllm-production-import2.log 2>&1 || {
    log "vLLM still unimportable; install may need a different transformers pin"
    tail -n 20 /tmp/vllm-production-import2.log
  }
fi

# ---------------------------------------------------------------------------
# [versions] log the resolved stack.
# ---------------------------------------------------------------------------
echo "############ [versions] resolved stack ############"
"$PY" - <<'PYEOF' || true
for name in ("vllm", "torch", "transformers", "triton", "numpy"):
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: UNAVAILABLE ({type(exc).__name__}: {exc})")
try:
    import torch
    print("torch cuda:", getattr(torch.version, "cuda", None),
          "cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch device probe failed:", exc)
PYEOF

# ---------------------------------------------------------------------------
# device preflight: the same silicon gate as the bake-off.
# ---------------------------------------------------------------------------
echo "############ device preflight: CUDA + RTX PRO 6000 ############"
"$PY" - <<'PYEOF' || die "device preflight failed" 92
import shutil, subprocess, sys
if not shutil.which("gcc"):
    sys.exit("FATAL: gcc not on PATH")
print("gcc:", subprocess.run(["gcc","--version"], capture_output=True, text=True).stdout.splitlines()[0])
import torch
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), "cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible")
name = torch.cuda.get_device_name(0)
print("device:", name)
if "rtx pro 6000" not in name.lower():
    sys.exit("FATAL: device %r is not an RTX PRO 6000" % name)
print("vram_GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import transformers
print("transformers:", transformers.__version__)
try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as exc:
    sys.exit("FATAL: vllm not importable: %s" % exc)
print("DEVICE PREFLIGHT OK")
PYEOF

# ---------------------------------------------------------------------------
# verify the staged pipeline exists before running it.
# ---------------------------------------------------------------------------
echo "############ verify staged pipeline ############"
for f in harness/vllm_pipeline.py harness/vllm_fastpass.py harness/vllm_reason.py \
         harness/reason_pilot.py harness/_batching.py harness/pilot_analyze.py; do
  [ -f "$REPO/$f" ] || die "repo at ${DI_REPO_COMMIT} is missing ${f}" 3
done

# ---------------------------------------------------------------------------
# unpack the production bundle (full-edition panel + stored reference).
# ---------------------------------------------------------------------------
echo "############ unpack production bundle ############"
BUNDLE="${REPO}/${BUNDLE_REL}"
[ -f "$BUNDLE" ] || die "production bundle not found: $BUNDLE" 3
rm -rf "$WORK"; mkdir -p "$WORK"
tar xzf "$BUNDLE" -C "$WORK"
PANEL="$WORK/panel.jsonl.gz"
REFERENCE="$WORK/reference_panel_results.jsonl.gz"
[ -f "$PANEL" ] || die "bundle has no panel.jsonl.gz" 3
[ -f "$REFERENCE" ] || die "bundle has no reference_panel_results.jsonl.gz" 3
ls -la "$WORK"

# ---------------------------------------------------------------------------
# panel catalog list: the manifest is authoritative; fall back to streaming
# the panel if the bundle predates it. Passed only if the staged pipeline
# supports --panel-catalogs (otherwise the summary stays on the pilot six).
# ---------------------------------------------------------------------------
PANEL_CATALOGS=""
if [ -f "$WORK/production_manifest.json" ]; then
  PANEL_CATALOGS="$("$PY" - "$WORK/production_manifest.json" <<'PYEOF' 2>/dev/null || true
import json, sys
doc = json.load(open(sys.argv[1]))
print(" ".join(str(int(c)) for c in (doc.get("panel_catalogs") or [])))
PYEOF
)"
fi
if [ -z "$PANEL_CATALOGS" ]; then
  log "no usable manifest panel_catalogs; deriving from panel.jsonl.gz"
  PANEL_CATALOGS="$("$PY" - "$PANEL" <<'PYEOF' || true
import gzip, json, sys
ids = set()
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ev = row.get("_evaluation") or {}
        cid = ev.get("catalog_id", row.get("catalog_id"))
        if cid is not None:
            ids.add(int(cid))
print(" ".join(str(c) for c in sorted(ids)))
PYEOF
)"
fi
PANEL_FLAG=0
if "$PY" "$REPO/harness/vllm_pipeline.py" --help 2>&1 | grep -q -- '--panel-catalogs'; then
  PANEL_FLAG=1
fi
if [ -n "$PANEL_CATALOGS" ] && [ "$PANEL_FLAG" -eq 1 ]; then
  log "panel_catalogs($(echo "$PANEL_CATALOGS" | wc -w))=${PANEL_CATALOGS}"
else
  log "WARNING: staged pipeline lacks --panel-catalogs (or no catalogs found); summary will cover the pilot six only"
  PANEL_CATALOGS=""
fi

# ---------------------------------------------------------------------------
# assemble the pipeline arguments.
# ---------------------------------------------------------------------------
cd "$REPO"
OUT="$WORK/vllm-production-results.jsonl.gz"
FAST_OUT="$WORK/vllm-production-fast-results.jsonl.gz"
SUMMARY="$WORK/vllm-production-summary.json"

ARGS=(
  harness/vllm_pipeline.py
  --rows "$PANEL"
  --reference "$REFERENCE"
  --model "$MODEL"
  --coverage "$COVERAGE"
  --cap "$CAP"
  --fast-batch "$FAST_BATCH"
  --reason-batch "$REASON_BATCH"
  --attention-backend "$ATTN"
  --cudagraph-mode "$CUDAGRAPH"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --kit-root "$REPO/kit"
  --out "$OUT"
  --fast-out "$FAST_OUT"
  --summary "$SUMMARY"
)
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi
if [ -n "$TBO" ]; then
  ARGS+=(--temperature-by-options "$TBO")
fi
if [ -n "$PANEL_CATALOGS" ]; then
  # Intentional word-splitting: the catalogs are separate CLI values.
  ARGS+=(--panel-catalogs ${PANEL_CATALOGS})
fi
if [ -f "$REPO/runs/full-rules/scores.json" ]; then
  ARGS+=(--scores "$REPO/runs/full-rules/scores.json")
else
  log "no runs/full-rules/scores.json; the kit spec supplies weights/chance"
fi

# ---------------------------------------------------------------------------
# run the integrated pipeline on the BI engine.
# ---------------------------------------------------------------------------
echo "############ run vllm production pipeline ############"
echo "\$ $PY ${ARGS[*]}"
START_EPOCH="$(date +%s)"
"$PY" "${ARGS[@]}" || log "pipeline exited non-zero; summary may still exist"
WALL=$(( $(date +%s) - START_EPOCH ))
[ -f "$SUMMARY" ] || die "pipeline produced no summary; vLLM likely failed to install or start" 4

# ---------------------------------------------------------------------------
# summary.
# ---------------------------------------------------------------------------
echo "############ VLLM_PIPELINE_SUMMARY ############"
"$PY" - "$SUMMARY" "$WALL" <<'PYEOF' || true
import json, sys

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

doc = load(sys.argv[1])
wall = sys.argv[2] if len(sys.argv) > 2 else "?"
if doc is None:
    print("no summary")
else:
    print(f"kind={doc.get('kind')} fatal={doc.get('fatal_error')} wall_seconds={wall}")
    print(f"rows={doc.get('rows')} questions={doc.get('questions')} "
          f"gated={doc.get('gated')} reasoned={doc.get('reasoned')} "
          f"folded={doc.get('folded')} merged={doc.get('merged_questions')}")
    print(f"panel_catalogs={doc.get('panel_catalogs')} "
          f"panel_share={doc.get('panel_share')} "
          f"full_index_delta={doc.get('full_index_delta')} "
          f"panel_scale_delta={doc.get('panel_scale_delta')}")
    counts = doc.get("counts") or {}
    print(f"counts: n={counts.get('n')} gated={counts.get('gated')} "
          f"fixed={counts.get('fixed')} flipped={counts.get('flipped')} "
          f"fix_rate={counts.get('fix_rate')}")
    for err in doc.get("errors") or []:
        print(f"  error: {err}")
    gate = doc.get("gate") or {}
    print(f"gate: eligible={gate.get('eligible')} gated={gate.get('gated')} "
          f"cutoff_margin={gate.get('cutoff_margin')}")
PYEOF

# ---------------------------------------------------------------------------
# upload outputs + summary to J-I-B/di-smoke:runs/route-b-production/.
# ---------------------------------------------------------------------------
echo "############ upload ############"
rm -rf "$WORK/upload"; mkdir -p "$WORK/upload"
cp -f "$OUT" "$FAST_OUT" "$SUMMARY" "$WORK/upload/" 2>/dev/null || true
cp -f /tmp/vllm-production-import.log /tmp/vllm-production-import2.log "$WORK/upload/" 2>/dev/null || true
"$PY" - "$RESULTS_REPO" "$UPLOAD_PATH" <<'PYEOF' || die "upload failed" 5
import os, sys
from huggingface_hub import HfApi
repo, path = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True)
api.upload_folder(
    repo_id=repo, repo_type="dataset",
    folder_path="/tmp/vllm-production/upload",
    path_in_repo=path,
    commit_message="vllm-production: integrated Route-B full-edition run",
)
print(f"uploaded /tmp/vllm-production/upload -> {repo}:{path}")
PYEOF

echo "VLLM_PIPELINE_COMPLETE status=ok coverage=${COVERAGE} cap=${CAP} fast_batch=${FAST_BATCH} reason_batch=${REASON_BATCH} limit='${LIMIT}' upload=${RESULTS_REPO}:${UPLOAD_PATH}/"
echo "VLLM_PIPELINE_COMPLETE status=ok" > "$STAMP"
echo "PORTABILITY_ARM_COMPLETE vllm_production run=${DI_ARM:-vllm-production} commit=${DI_REPO_COMMIT} upload=${RESULTS_REPO}:${UPLOAD_PATH}/"
exit 0
