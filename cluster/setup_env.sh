#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-ai6103_hrnet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda || true
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

activate_conda() {
  if command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    set -u
    return
  fi

  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    set -u
    return
  fi

  echo "[ERROR] conda not found"
  exit 1
}

activate_conda

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[INFO] conda env exists: $CONDA_ENV"
else
  echo "[INFO] creating conda env: $CONDA_ENV (python=$PYTHON_VERSION)"
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi

set +u
conda activate "$CONDA_ENV"
set -u

python -m pip install --upgrade pip setuptools wheel

# PyTorch GPU wheel (cluster CUDA 12.x compatible via cu121 wheel set)
python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision

# Project runtime dependencies
python -m pip install \
  "numpy<2" \
  "cython<3" \
  opencv-python==4.10.0.84 \
  easydict \
  shapely \
  scipy \
  pandas \
  pyyaml \
  json_tricks \
  scikit-image \
  yacs \
  tensorboard \
  tensorboardX \
  pycocotools

python - <<'PY'
import torch
import torchvision
import cv2
import tensorboard
print("torch", torch.__version__, "cuda_build", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("opencv", cv2.__version__)
print("tensorboard", tensorboard.__version__)
PY

pip freeze > "${PWD}/cluster_env_${CONDA_ENV}_$(date -u +%Y%m%d_%H%M%S).txt"

echo "[INFO] setup complete for conda env: $CONDA_ENV"
