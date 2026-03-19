#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>"
  exit 1
fi

RUN_ID="$1"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="$PROJECT_DIR/cluster/slurm_hrnet_train.sbatch"

CONDA_ENV="${CONDA_ENV:-ai6103_hrnet}"
DATASET_ROOT="${DATASET_ROOT:-/home/msai/xjiang026/datasets/coco}"
COCO_BBOX_FILE="${COCO_BBOX_FILE:-$DATASET_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$PROJECT_DIR/models/pytorch/imagenet/hrnet_w32-36af842e.pth}"
CFG_FILE="${CFG_FILE:-$PROJECT_DIR/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_ID}"
BEGIN_EPOCH="${BEGIN_EPOCH:-0}"
END_EPOCH="${END_EPOCH:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRINT_FREQ="${PRINT_FREQ:-20}"
GPU_SAMPLE_SEC="${GPU_SAMPLE_SEC:-30}"
DEBUG_DUMP="${DEBUG_DUMP:-1}"
RUN_POST_EVAL="${RUN_POST_EVAL:-0}"
RUN_POST_DEMO="${RUN_POST_DEMO:-0}"
RUN_MODE="${RUN_MODE:-train}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "[ERROR] job script not found: $JOB_SCRIPT"
  exit 1
fi

mkdir -p "$RUN_ROOT/slurm" "$RUN_ROOT/summary"

export RUN_ID PROJECT_DIR CONDA_ENV DATASET_ROOT COCO_BBOX_FILE PRETRAINED_MODEL CFG_FILE RUN_ROOT
export BEGIN_EPOCH END_EPOCH BATCH_SIZE_PER_GPU NUM_WORKERS PRINT_FREQ GPU_SAMPLE_SEC DEBUG_DUMP
export RUN_POST_EVAL RUN_POST_DEMO RUN_MODE

"$PROJECT_DIR/cluster/preflight_check.sh"

job_id=$(sbatch --parsable \
  --job-name "hrnet-smoke" \
  --output "$RUN_ROOT/slurm/slurm-smoke-%j.out" \
  --error "$RUN_ROOT/slurm/slurm-smoke-%j.err" \
  --export=ALL,RUN_ID="$RUN_ID",PROJECT_DIR="$PROJECT_DIR",CONDA_ENV="$CONDA_ENV",DATASET_ROOT="$DATASET_ROOT",COCO_BBOX_FILE="$COCO_BBOX_FILE",PRETRAINED_MODEL="$PRETRAINED_MODEL",CFG_FILE="$CFG_FILE",RUN_ROOT="$RUN_ROOT",BEGIN_EPOCH="$BEGIN_EPOCH",END_EPOCH="$END_EPOCH",BATCH_SIZE_PER_GPU="$BATCH_SIZE_PER_GPU",NUM_WORKERS="$NUM_WORKERS",PRINT_FREQ="$PRINT_FREQ",GPU_SAMPLE_SEC="$GPU_SAMPLE_SEC",DEBUG_DUMP="$DEBUG_DUMP",RUN_POST_EVAL="$RUN_POST_EVAL",RUN_POST_DEMO="$RUN_POST_DEMO",RUN_MODE="$RUN_MODE" \
  "$JOB_SCRIPT")

echo "[INFO] submitted smoke job: $job_id"
echo "[INFO] run_root: $RUN_ROOT"

echo "job_id=$job_id" > "$RUN_ROOT/summary/smoke_submission.txt"
echo "submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/summary/smoke_submission.txt"
echo "begin_epoch=$BEGIN_EPOCH" >> "$RUN_ROOT/summary/smoke_submission.txt"
echo "end_epoch=$END_EPOCH" >> "$RUN_ROOT/summary/smoke_submission.txt"
