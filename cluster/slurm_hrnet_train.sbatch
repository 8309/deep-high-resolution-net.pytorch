#!/bin/bash
#SBATCH --job-name=hrnet-train
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Required interface variables (kept explicit by design)
RUN_ID="${RUN_ID:-hrnet_coco_w32_256x192_$(date -u +%Y%m%d_%H%M%S)}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/hrnet}"
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
DEBUG_DUMP="${DEBUG_DUMP:-0}"
RUN_MODE="${RUN_MODE:-train}"

# Auto-chain: if set, the job will submit the next chunk upon successful completion.
# TARGET_EPOCH=0 means auto-chain is disabled.
TARGET_EPOCH="${TARGET_EPOCH:-0}"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-0}"
CHAIN_SEQ="${CHAIN_SEQ:-1}"

# Optional switches for end-to-end pipeline
RUN_POST_EVAL="${RUN_POST_EVAL:-0}"
RUN_POST_DEMO="${RUN_POST_DEMO:-0}"
DEMO_IMAGE="${DEMO_IMAGE:-$PROJECT_DIR/demo/inference_1.jpg}"

required_vars=(
  RUN_ID PROJECT_DIR CONDA_ENV DATASET_ROOT COCO_BBOX_FILE PRETRAINED_MODEL
  CFG_FILE RUN_ROOT BEGIN_EPOCH END_EPOCH BATCH_SIZE_PER_GPU NUM_WORKERS
  PRINT_FREQ GPU_SAMPLE_SEC DEBUG_DUMP
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "[ERROR] Required variable $var_name is empty"
    exit 2
  fi
done

if [[ "$RUN_MODE" == "train" ]] && (( END_EPOCH <= BEGIN_EPOCH )); then
  echo "[ERROR] END_EPOCH ($END_EPOCH) must be > BEGIN_EPOCH ($BEGIN_EPOCH)"
  exit 2
fi

mkdir -p "$RUN_ROOT"/{slurm,system,telemetry,summary,demo,output,log}
cd "$PROJECT_DIR"

module load anaconda || true
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

activate_env() {
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
    echo "[ERROR] micromamba/conda not found; cannot activate Python environment"
    exit 1
  fi
}
activate_env

cfg_basename="$(basename "$CFG_FILE")"
cfg_name="${cfg_basename%.*}"
final_output_dir="$RUN_ROOT/output/coco/pose_hrnet/$cfg_name"
train_stdout="$RUN_ROOT/summary/train_stdout_${SLURM_JOB_ID:-manual}.log"
test_stdout="$RUN_ROOT/summary/test_stdout_${SLURM_JOB_ID:-manual}.log"

{
  echo "run_id=$RUN_ID"
  echo "project_dir=$PROJECT_DIR"
  echo "conda_env=$CONDA_ENV"
  echo "dataset_root=$DATASET_ROOT"
  echo "bbox_file=$COCO_BBOX_FILE"
  echo "pretrained_model=$PRETRAINED_MODEL"
  echo "cfg_file=$CFG_FILE"
  echo "run_root=$RUN_ROOT"
  echo "begin_epoch=$BEGIN_EPOCH"
  echo "end_epoch=$END_EPOCH"
  echo "batch_size_per_gpu=$BATCH_SIZE_PER_GPU"
  echo "num_workers=$NUM_WORKERS"
  echo "print_freq=$PRINT_FREQ"
  echo "gpu_sample_sec=$GPU_SAMPLE_SEC"
  echo "debug_dump=$DEBUG_DUMP"
  echo "run_mode=$RUN_MODE"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_job_name=${SLURM_JOB_NAME:-}"
  echo "hostname=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_ROOT/summary/train_env.txt"

printenv | sort > "$RUN_ROOT/system/env_full.txt" || true
python --version > "$RUN_ROOT/system/python_version.txt"
pip freeze > "$RUN_ROOT/system/pip_freeze.txt"
conda list > "$RUN_ROOT/system/conda_list.txt" 2>/dev/null || true
nvidia-smi > "$RUN_ROOT/system/nvidia_smi.txt" 2>/dev/null || true
nvidia-smi -L > "$RUN_ROOT/system/nvidia_smi_L.txt" 2>/dev/null || true

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$RUN_ROOT/summary/git_commit.txt" || true
  git status --porcelain > "$RUN_ROOT/summary/git_status.txt" || true
  git diff > "$RUN_ROOT/summary/git_diff.patch" || true
fi
cp "$CFG_FILE" "$RUN_ROOT/summary/cfg_source.yaml"

sampler_pid=""
gpu_metrics_csv="$RUN_ROOT/telemetry/gpu_metrics_${SLURM_JOB_ID:-manual}.csv"
echo "timestamp_utc,gpu_index,gpu_name,gpu_uuid,util_gpu,util_mem,mem_used_mb,mem_total_mb,temp_c,power_w" > "$gpu_metrics_csv"

start_gpu_sampler() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found; GPU telemetry disabled"
    return
  fi

  (
    while true; do
      ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      nvidia-smi \
        --query-gpu=index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | while IFS= read -r line; do
          echo "$ts,$line"
        done >> "$gpu_metrics_csv"
      sleep "$GPU_SAMPLE_SEC"
    done
  ) &
  sampler_pid=$!
}

stop_gpu_sampler() {
  if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" >/dev/null 2>&1; then
    kill "$sampler_pid" >/dev/null 2>&1 || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap stop_gpu_sampler EXIT INT TERM

start_gpu_sampler

if [[ "$DEBUG_DUMP" == "1" ]]; then
  debug_flag="True"
else
  debug_flag="False"
fi

if [[ "$RUN_MODE" == "train" ]]; then
  cat > "$RUN_ROOT/summary/train_command.sh" <<CMD
python -u tools/train.py \
  --cfg "$CFG_FILE" \
  --modelDir "$RUN_ROOT/output" \
  --logDir "$RUN_ROOT/log" \
  DATASET.ROOT "$DATASET_ROOT" \
  TEST.COCO_BBOX_FILE "$COCO_BBOX_FILE" \
  MODEL.PRETRAINED "$PRETRAINED_MODEL" \
  GPUS "(0,)" \
  WORKERS $NUM_WORKERS \
  PRINT_FREQ $PRINT_FREQ \
  TRAIN.BEGIN_EPOCH $BEGIN_EPOCH \
  TRAIN.END_EPOCH $END_EPOCH \
  TRAIN.BATCH_SIZE_PER_GPU $BATCH_SIZE_PER_GPU \
  TEST.BATCH_SIZE_PER_GPU $BATCH_SIZE_PER_GPU \
  DEBUG.DEBUG $debug_flag \
  DEBUG.SAVE_BATCH_IMAGES_GT $debug_flag \
  DEBUG.SAVE_BATCH_IMAGES_PRED $debug_flag \
  DEBUG.SAVE_HEATMAPS_GT $debug_flag \
  DEBUG.SAVE_HEATMAPS_PRED $debug_flag
CMD
  chmod +x "$RUN_ROOT/summary/train_command.sh"

  set +e
  python -u tools/train.py \
    --cfg "$CFG_FILE" \
    --modelDir "$RUN_ROOT/output" \
    --logDir "$RUN_ROOT/log" \
    DATASET.ROOT "$DATASET_ROOT" \
    TEST.COCO_BBOX_FILE "$COCO_BBOX_FILE" \
    MODEL.PRETRAINED "$PRETRAINED_MODEL" \
    GPUS "(0,)" \
    WORKERS "$NUM_WORKERS" \
    PRINT_FREQ "$PRINT_FREQ" \
    TRAIN.BEGIN_EPOCH "$BEGIN_EPOCH" \
    TRAIN.END_EPOCH "$END_EPOCH" \
    TRAIN.BATCH_SIZE_PER_GPU "$BATCH_SIZE_PER_GPU" \
    TEST.BATCH_SIZE_PER_GPU "$BATCH_SIZE_PER_GPU" \
    DEBUG.DEBUG "$debug_flag" \
    DEBUG.SAVE_BATCH_IMAGES_GT "$debug_flag" \
    DEBUG.SAVE_BATCH_IMAGES_PRED "$debug_flag" \
    DEBUG.SAVE_HEATMAPS_GT "$debug_flag" \
    DEBUG.SAVE_HEATMAPS_PRED "$debug_flag" \
    2>&1 | tee "$train_stdout"
  train_rc=${PIPESTATUS[0]}
  set -e

  if (( train_rc != 0 )); then
    echo "[ERROR] training command failed with rc=$train_rc"
    exit "$train_rc"
  fi

  checkpoint_file="$final_output_dir/checkpoint.pth"
  if [[ -f "$checkpoint_file" ]]; then
    python - "$checkpoint_file" "$RUN_ROOT/summary/checkpoint_epoch.txt" <<'PY'
import sys
import torch

checkpoint_path = sys.argv[1]
out_path = sys.argv[2]
ckpt = torch.load(checkpoint_path, map_location='cpu')
epoch = ckpt.get('epoch', '')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(str(epoch))
print(epoch)
PY
  fi
else
  echo "[INFO] RUN_MODE=$RUN_MODE -> skip training phase"
fi

# ── Auto-chain: submit next chunk if training succeeded and more epochs remain ──
if [[ "$RUN_MODE" == "train" ]] && (( TARGET_EPOCH > 0 && CHUNK_EPOCHS > 0 && END_EPOCH < TARGET_EPOCH )); then
  next_begin="$END_EPOCH"
  next_end=$(( END_EPOCH + CHUNK_EPOCHS ))
  if (( next_end > TARGET_EPOCH )); then
    next_end="$TARGET_EPOCH"
  fi
  next_seq=$(( CHAIN_SEQ + 1 ))

  echo "[AUTO-CHAIN] submitting next chunk: begin=$next_begin end=$next_end seq=$next_seq target=$TARGET_EPOCH"

  next_job_id_raw=$(sbatch --parsable \
    --job-name "hrnet-c${next_seq}" \
    --output "$RUN_ROOT/slurm/slurm-chain${next_seq}-%j.out" \
    --error "$RUN_ROOT/slurm/slurm-chain${next_seq}-%j.err" \
    --export=ALL,RUN_ID="$RUN_ID",PROJECT_DIR="$PROJECT_DIR",CONDA_ENV="$CONDA_ENV",DATASET_ROOT="$DATASET_ROOT",COCO_BBOX_FILE="$COCO_BBOX_FILE",PRETRAINED_MODEL="$PRETRAINED_MODEL",CFG_FILE="$CFG_FILE",RUN_ROOT="$RUN_ROOT",BEGIN_EPOCH="$next_begin",END_EPOCH="$next_end",BATCH_SIZE_PER_GPU="$BATCH_SIZE_PER_GPU",NUM_WORKERS="$NUM_WORKERS",PRINT_FREQ="$PRINT_FREQ",GPU_SAMPLE_SEC="$GPU_SAMPLE_SEC",DEBUG_DUMP="$DEBUG_DUMP",RUN_POST_EVAL="$RUN_POST_EVAL",RUN_POST_DEMO="$RUN_POST_DEMO",RUN_MODE="$RUN_MODE",TARGET_EPOCH="$TARGET_EPOCH",CHUNK_EPOCHS="$CHUNK_EPOCHS",CHAIN_SEQ="$next_seq" \
    "$PROJECT_DIR/cluster/slurm_hrnet_train.sbatch")

  next_job_id="${next_job_id_raw%%;*}"
  echo "[AUTO-CHAIN] submitted job_id=$next_job_id begin=$next_begin end=$next_end"

  # Log to chain schedule
  chain_log="$RUN_ROOT/summary/auto_chain_log.csv"
  if [[ ! -f "$chain_log" ]]; then
    echo "seq,job_id,begin_epoch,end_epoch,parent_job_id,submitted_utc" > "$chain_log"
  fi
  echo "$next_seq,$next_job_id,$next_begin,$next_end,${SLURM_JOB_ID:-manual},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$chain_log"
else
  if [[ "$RUN_MODE" == "train" ]] && (( TARGET_EPOCH > 0 && END_EPOCH >= TARGET_EPOCH )); then
    echo "[AUTO-CHAIN] target epoch $TARGET_EPOCH reached — no more chunks to submit"
  fi
fi

model_for_eval="$final_output_dir/final_state.pth"
if [[ ! -f "$model_for_eval" && -f "$final_output_dir/model_best.pth" ]]; then
  model_for_eval="$final_output_dir/model_best.pth"
fi

if [[ ("$RUN_POST_EVAL" == "1" || "$RUN_POST_DEMO" == "1") && ! -f "$model_for_eval" ]]; then
  echo "[ERROR] requested post-eval/demo but model file not found under $final_output_dir"
  exit 3
fi

if [[ "$RUN_POST_EVAL" == "1" ]]; then
  set +e
  python -u tools/test.py \
    --cfg "$CFG_FILE" \
    --modelDir "$RUN_ROOT/output" \
    --logDir "$RUN_ROOT/log" \
    DATASET.ROOT "$DATASET_ROOT" \
    TEST.COCO_BBOX_FILE "$COCO_BBOX_FILE" \
    TEST.MODEL_FILE "$model_for_eval" \
    GPUS "(0,)" \
    WORKERS "$NUM_WORKERS" \
    PRINT_FREQ "$PRINT_FREQ" \
    TEST.BATCH_SIZE_PER_GPU "$BATCH_SIZE_PER_GPU" \
    2>&1 | tee "$test_stdout"
  test_rc=${PIPESTATUS[0]}
  set -e
  if (( test_rc != 0 )); then
    echo "[WARN] post-eval failed rc=$test_rc"
  fi
fi

if [[ "$RUN_POST_DEMO" == "1" ]]; then
  demo_out="$RUN_ROOT/demo/demo_output_${SLURM_JOB_ID:-manual}.jpg"
  set +e
  python -u scripts/run_demo_snapshot.py \
    --cfg "$CFG_FILE" \
    --image "$DEMO_IMAGE" \
    --out "$demo_out" \
    --model-file "$model_for_eval" \
    DATASET.ROOT "$DATASET_ROOT" \
    GPUS "(0,)" \
    > "$RUN_ROOT/summary/demo_stdout_${SLURM_JOB_ID:-manual}.log" 2>&1
  demo_rc=$?
  set -e
  if (( demo_rc != 0 )); then
    echo "[WARN] post-demo failed rc=$demo_rc"
  fi
fi

if command -v sacct >/dev/null 2>&1; then
  sacct -j "${SLURM_JOB_ID:-}" \
    --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES \
    > "$RUN_ROOT/summary/sacct_${SLURM_JOB_ID:-manual}.txt" 2>/dev/null || true
fi

if [[ -f "$PROJECT_DIR/scripts/export_tb_scalars.py" ]]; then
  python -u scripts/export_tb_scalars.py \
    --tb_dir "$RUN_ROOT/log" \
    --out_csv "$RUN_ROOT/summary/tb_scalars.csv" || true
fi

if [[ -f "$PROJECT_DIR/scripts/build_run_manifest.py" ]]; then
  python -u scripts/build_run_manifest.py \
    --run_root "$RUN_ROOT" \
    --run_id "$RUN_ID" \
    --project_dir "$PROJECT_DIR" \
    --cfg_file "$CFG_FILE" \
    --begin_epoch "$BEGIN_EPOCH" \
    --end_epoch "$END_EPOCH" \
    --output "$RUN_ROOT/summary/run_manifest.json" \
    --artifact_index "$RUN_ROOT/summary/artifact_index.csv" \
    --run_summary "$RUN_ROOT/summary/run_summary.md" || true
fi

echo "[INFO] completed run_id=$RUN_ID begin_epoch=$BEGIN_EPOCH end_epoch=$END_EPOCH mode=$RUN_MODE"
