#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_chain.sh <run_id> <chunk_epochs> <target_epoch> [chain_len]

  run_id         : identifier for the training run
  chunk_epochs   : epochs per Slurm job
  target_epoch   : final target epoch (e.g. 210)
  chain_len      : (optional) number of jobs to pre-submit with afterok deps.
                   If omitted or set to "auto", only the first job is submitted
                   and each job will self-submit the next chunk upon success
                   (auto-chain mode, avoids MaxJobsPU limits).

Environment variables (all have sensible defaults):
  PROJECT_DIR, CONDA_ENV, DATASET_ROOT, COCO_BBOX_FILE, PRETRAINED_MODEL,
  CFG_FILE, RUN_ROOT, BATCH_SIZE_PER_GPU, NUM_WORKERS, PRINT_FREQ,
  GPU_SAMPLE_SEC, DEBUG_DUMP, RUN_POST_EVAL, RUN_POST_DEMO, RUN_MODE
EOF
  exit 1
}

if [[ $# -lt 3 ]]; then
  usage
fi

RUN_ID="$1"
CHUNK_EPOCHS="$2"
TARGET_EPOCH="$3"
CHAIN_LEN="${4:-auto}"

for n in "$CHUNK_EPOCHS" "$TARGET_EPOCH"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n <= 0 )); then
    echo "[ERROR] chunk_epochs and target_epoch must be positive integers"
    exit 1
  fi
done

AUTO_CHAIN=0
if [[ "$CHAIN_LEN" == "auto" ]]; then
  AUTO_CHAIN=1
else
  if ! [[ "$CHAIN_LEN" =~ ^[0-9]+$ ]] || (( CHAIN_LEN <= 0 )); then
    echo "[ERROR] chain_len must be a positive integer or 'auto'"
    exit 1
  fi
fi

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

# ── Common export string for sbatch ──
common_exports="RUN_ID=$RUN_ID,PROJECT_DIR=$PROJECT_DIR,CONDA_ENV=$CONDA_ENV,DATASET_ROOT=$DATASET_ROOT,COCO_BBOX_FILE=$COCO_BBOX_FILE,PRETRAINED_MODEL=$PRETRAINED_MODEL,CFG_FILE=$CFG_FILE,RUN_ROOT=$RUN_ROOT,BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU,NUM_WORKERS=$NUM_WORKERS,PRINT_FREQ=$PRINT_FREQ,GPU_SAMPLE_SEC=$GPU_SAMPLE_SEC,DEBUG_DUMP=$DEBUG_DUMP,RUN_POST_EVAL=$RUN_POST_EVAL,RUN_POST_DEMO=$RUN_POST_DEMO,RUN_MODE=$RUN_MODE"

schedule_file="$RUN_ROOT/summary/chain_submission_$(date -u +%Y%m%d_%H%M%S).csv"
echo "order,job_id,begin_epoch,end_epoch,dependency,mode,submitted_utc" > "$schedule_file"

# ── Auto-chain mode: submit only the first chunk; the job self-submits the rest ──
if (( AUTO_CHAIN == 1 )); then
  first_end=$(( current_epoch + CHUNK_EPOCHS ))
  if (( first_end > TARGET_EPOCH )); then
    first_end=$TARGET_EPOCH
  fi

  echo "[INFO] auto-chain mode: submitting first chunk begin=$current_epoch end=$first_end target=$TARGET_EPOCH"

  job_id_raw=$(sbatch --parsable \
    --job-name "hrnet-c1" \
    --output "$RUN_ROOT/slurm/slurm-chain1-%j.out" \
    --error "$RUN_ROOT/slurm/slurm-chain1-%j.err" \
    --export=ALL,${common_exports},BEGIN_EPOCH="$current_epoch",END_EPOCH="$first_end",TARGET_EPOCH="$TARGET_EPOCH",CHUNK_EPOCHS="$CHUNK_EPOCHS",CHAIN_SEQ=1 \
    "$JOB_SCRIPT")

  job_id="${job_id_raw%%;*}"
  echo "1,$job_id,$current_epoch,$first_end,none,auto-chain,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$schedule_file"
  echo "[INFO] submitted job_id=$job_id begin=$current_epoch end=$first_end (auto-chain enabled, target=$TARGET_EPOCH)"
  echo "[INFO] schedule: $schedule_file"
  exit 0
fi

# ── Legacy pre-submit mode: submit CHAIN_LEN jobs with afterok dependencies ──
echo "[INFO] legacy pre-submit mode: chain_len=$CHAIN_LEN"

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
    --export=ALL,${common_exports},BEGIN_EPOCH="$current_epoch",END_EPOCH="$next_epoch" \
    "$JOB_SCRIPT")

  job_id="${job_id_raw%%;*}"
  echo "$i,$job_id,$current_epoch,$next_epoch,$dep_note,pre-submit,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$schedule_file"
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
