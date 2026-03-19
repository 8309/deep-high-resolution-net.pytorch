#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-preflight_$(date -u +%Y%m%d_%H%M%S)}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/hrnet}"
CONDA_ENV="${CONDA_ENV:-ai6103_hrnet}"
DATASET_ROOT="${DATASET_ROOT:-/home/msai/xjiang026/datasets/coco}"
COCO_BBOX_FILE="${COCO_BBOX_FILE:-$DATASET_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$PROJECT_DIR/models/pytorch/imagenet/hrnet_w32-36af842e.pth}"
CFG_FILE="${CFG_FILE:-$PROJECT_DIR/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_ID}"

required_dirs=(
  "$PROJECT_DIR"
  "$DATASET_ROOT"
  "$DATASET_ROOT/annotations"
  "$DATASET_ROOT/images/train2017"
  "$DATASET_ROOT/images/val2017"
)
required_files=(
  "$CFG_FILE"
  "$COCO_BBOX_FILE"
  "$PRETRAINED_MODEL"
  "$PROJECT_DIR/tools/train.py"
  "$PROJECT_DIR/tools/test.py"
)

errors=0
warns=0

info() { echo "[INFO] $*"; }
pass() { echo "[PASS] $*"; }
warn() { echo "[WARN] $*"; warns=$((warns + 1)); }
fail() { echo "[FAIL] $*"; errors=$((errors + 1)); }

activate_env() {
  module load anaconda >/dev/null 2>&1 || true
  export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$CONDA_ENV"
  elif command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
  else
    fail "No conda/micromamba found to activate env=$CONDA_ENV"
  fi
}

info "Run preflight with run_id=$RUN_ID"
info "project_dir=$PROJECT_DIR"
info "dataset_root=$DATASET_ROOT"
info "run_root=$RUN_ROOT"

for d in "${required_dirs[@]}"; do
  if [[ -d "$d" ]]; then
    pass "dir exists: $d"
  else
    fail "dir missing: $d"
  fi
done

for f in "${required_files[@]}"; do
  if [[ -f "$f" ]]; then
    pass "file exists: $f"
  else
    fail "file missing: $f"
  fi
done

mkdir -p "$RUN_ROOT" 2>/dev/null || fail "cannot create run_root: $RUN_ROOT"
if [[ -d "$RUN_ROOT" ]]; then
  touch "$RUN_ROOT/.preflight_write_test" 2>/dev/null || fail "run_root not writable: $RUN_ROOT"
  rm -f "$RUN_ROOT/.preflight_write_test" 2>/dev/null || true
  pass "run_root writable: $RUN_ROOT"
fi

for cmd in python sbatch squeue sacct rsync; do
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "command available: $cmd"
  else
    fail "command missing: $cmd"
  fi
done

activate_env

if [[ $errors -eq 0 ]]; then
  pass "environment activation succeeded: $CONDA_ENV"
fi

set +e
python - <<'PY'
import importlib
import json
import os
import sys

required = [
    "torch", "torchvision", "numpy", "cv2", "pycocotools",
    "tensorboard", "tensorboardX", "yacs", "json_tricks", "scipy", "pandas",
    "skimage", "yaml", "Cython"
]

report = {}
missing = []
for mod in required:
    try:
        m = importlib.import_module(mod)
        report[mod] = getattr(m, "__version__", "installed")
    except Exception:
        missing.append(mod)

try:
    import torch
    cuda_build = torch.version.cuda
    cuda_available = torch.cuda.is_available()
except Exception:
    cuda_build = None
    cuda_available = False

out = {
    "modules": report,
    "missing": missing,
    "torch_cuda_build": cuda_build,
    "torch_cuda_available": cuda_available,
}

print(json.dumps(out, indent=2, sort_keys=True))

if missing:
    sys.exit(3)
if cuda_build is None:
    sys.exit(4)
# cuda_available may be false on login nodes; do not fail hard
sys.exit(0)
PY
py_rc=$?
set -e

if [[ $py_rc -eq 0 ]]; then
  pass "python dependency check passed"
elif [[ $py_rc -eq 3 ]]; then
  fail "python dependency check failed: missing modules"
elif [[ $py_rc -eq 4 ]]; then
  fail "torch installed without CUDA build"
else
  fail "python dependency check failed with rc=$py_rc"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi accessible"
  else
    warn "nvidia-smi exists but not accessible on current node (often normal on login node)"
  fi
else
  warn "nvidia-smi not found on this node"
fi

if [[ $errors -gt 0 ]]; then
  echo "[RESULT] PRECHECK FAILED (errors=$errors warns=$warns)"
  exit 1
fi

echo "[RESULT] PRECHECK PASSED (errors=$errors warns=$warns)"
