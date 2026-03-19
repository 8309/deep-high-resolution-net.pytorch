#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <run_id> <chunk_epochs> <target_epoch> <chain_len>"
  exit 1
fi

RUN_ID="$1"
CHUNK_EPOCHS="$2"
TARGET_EPOCH="$3"
CHAIN_LEN="$4"

for n in "$CHUNK_EPOCHS" "$TARGET_EPOCH" "$CHAIN_LEN"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] numeric args required: chunk_epochs target_epoch chain_len"
    exit 1
  fi
  if (( n <= 0 )); then
    echo "[ERROR] numeric args must be > 0"
    exit 1
  fi
done

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="$PROJECT_DIR/cluster/slurm_hrnet_train.sbatch"

CONDA_ENV="${CONDA_ENV:-ai6103_hrnet}"
DATASET_ROOT="${DATASET_ROOT:-/home/msai/xjiang026/datasets/coco}"
COCO_BBOX_FILE="${COCO_BBOX_FILE:-$DATASET_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$PROJECT_DIR/models/pytorch/imagenet/hrnet_w32-36af842e.pth}"
CFG_FILE="${CFG_FILE:-$PROJECT_DIR/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_ID}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_FREQ="${PRINT_FREQ:-50}"
GPU_SAMPLE_SEC="${GPU_SAMPLE_SEC:-30}"
DEBUG_DUMP="${DEBUG_DUMP:-0}"
RUN_POST_EVAL="${RUN_POST_EVAL:-0}"
RUN_POST_DEMO="${RUN_POST_DEMO:-0}"
RUN_MODE="${RUN_MODE:-train}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "[ERROR] job script not found: $JOB_SCRIPT"
  exit 1
fi

mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/summary"

export RUN_ID PROJECT_DIR CONDA_ENV DATASET_ROOT COCO_BBOX_FILE PRETRAINED_MODEL CFG_FILE RUN_ROOT
export BATCH_SIZE_PER_GPU NUM_WORKERS PRINT_FREQ GPU_SAMPLE_SEC DEBUG_DUMP RUN_POST_EVAL RUN_POST_DEMO RUN_MODE
"$PROJECT_DIR/cluster/preflight_check.sh"

cfg_basename="$(basename "$CFG_FILE")"
cfg_name="${cfg_basename%.*}"
checkpoint_file="$RUN_ROOT/output/coco/pose_hrnet/$cfg_name/checkpoint.pth"

current_epoch="${BEGIN_EPOCH:-1}"
if [[ -f "$checkpoint_file" ]]; then
  detected_epoch=$(python - "$checkpoint_file" <<'PY'
import sys

try:
    import torch
except Exception:
    print("")
    raise SystemExit(0)

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location='cpu')
    epoch = ckpt.get('epoch', '')
    print(epoch)
except Exception:
    print("")
PY
)
  if [[ "$detected_epoch" =~ ^[0-9]+$ ]]; then
    current_epoch="$detected_epoch"
    echo "[INFO] detected checkpoint epoch=$current_epoch from $checkpoint_file"
  fi
fi

if (( current_epoch >= TARGET_EPOCH )); then
  echo "[INFO] checkpoint epoch ($current_epoch) already >= target ($TARGET_EPOCH); nothing to submit"
  exit 0
fi

schedule_file="$RUN_ROOT/summary/chain_submission_$(date -u +%Y%m%d_%H%M%S).csv"
echo "order,job_id,begin_epoch,end_epoch,dependency,submitted_utc" > "$schedule_file"

prev_job_id=""
submitted=0
for ((i=1; i<=CHAIN_LEN; i++)); do
  if (( current_epoch >= TARGET_EPOCH )); then
    break
  fi

  next_epoch=$((current_epoch + CHUNK_EPOCHS))
  if (( next_epoch > TARGET_EPOCH )); then
    next_epoch=$TARGET_EPOCH
  fi

  dep_args=()
  dep_note="none"
  if [[ -n "$prev_job_id" ]]; then
    dep_args=(--dependency "afterok:$prev_job_id")
    dep_note="afterok:$prev_job_id"
  fi

  job_id_raw=$(sbatch --parsable \
    --job-name "hrnet-c${i}" \
    --output "$RUN_ROOT/slurm/slurm-chain${i}-%j.out" \
    --error "$RUN_ROOT/slurm/slurm-chain${i}-%j.err" \
    "${dep_args[@]}" \
    --export=ALL,RUN_ID="$RUN_ID",PROJECT_DIR="$PROJECT_DIR",CONDA_ENV="$CONDA_ENV",DATASET_ROOT="$DATASET_ROOT",COCO_BBOX_FILE="$COCO_BBOX_FILE",PRETRAINED_MODEL="$PRETRAINED_MODEL",CFG_FILE="$CFG_FILE",RUN_ROOT="$RUN_ROOT",BEGIN_EPOCH="$current_epoch",END_EPOCH="$next_epoch",BATCH_SIZE_PER_GPU="$BATCH_SIZE_PER_GPU",NUM_WORKERS="$NUM_WORKERS",PRINT_FREQ="$PRINT_FREQ",GPU_SAMPLE_SEC="$GPU_SAMPLE_SEC",DEBUG_DUMP="$DEBUG_DUMP",RUN_POST_EVAL="$RUN_POST_EVAL",RUN_POST_DEMO="$RUN_POST_DEMO",RUN_MODE="$RUN_MODE" \
    "$JOB_SCRIPT")

  job_id="${job_id_raw%%;*}"
  echo "$i,$job_id,$current_epoch,$next_epoch,$dep_note,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$schedule_file"
  echo "[INFO] submitted job_id=$job_id begin=$current_epoch end=$next_epoch dependency=$dep_note"

  prev_job_id="$job_id"
  current_epoch="$next_epoch"
  submitted=$((submitted + 1))
done

if (( submitted == 0 )); then
  echo "[WARN] no jobs submitted"
  exit 1
fi

echo "[INFO] submitted $submitted chained jobs"
echo "[INFO] schedule: $schedule_file"
