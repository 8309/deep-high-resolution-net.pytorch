# HRNet TC2 Cluster Workflow (COCO w32-256x192)

This folder provides an end-to-end Slurm workflow for:

- smoke run (1 epoch)
- chained training to target epochs with checkpoint resume
- full telemetry/data capture
- artifact pullback to local machine

## Scripts

- `slurm_hrnet_train.sbatch`: main training job script (env-driven)
- `setup_env.sh`: create/activate conda env and install required packages
- `preflight_check.sh`: checks dataset/model/env before submit
- `submit_smoke.sh <run_id>`: submit 1-epoch smoke job
- `recommend_chunk_epochs.sh <run_id> [epochs_ran] [budget_seconds]`: estimate chunk size from smoke logs
- `submit_chain.sh <run_id> <chunk_epochs> <target_epoch> <chain_len>`: submit chained jobs
- `submit_finalize.sh <run_id> [epoch_marker]`: run final eval/demo and refresh manifests without extra training
- `pull_artifacts.sh <run_id> <local_dest>`: rsync run artifacts to local

## Required Environment Variables

These variables are consumed by `slurm_hrnet_train.sbatch`:

- `RUN_ID`
- `PROJECT_DIR`
- `CONDA_ENV`
- `DATASET_ROOT`
- `COCO_BBOX_FILE`
- `PRETRAINED_MODEL`
- `CFG_FILE`
- `RUN_ROOT`
- `BEGIN_EPOCH`
- `END_EPOCH`
- `BATCH_SIZE_PER_GPU`
- `NUM_WORKERS`
- `PRINT_FREQ`
- `GPU_SAMPLE_SEC`
- `DEBUG_DUMP`

## Default Assumptions

- host: `xjiang026@10.96.189.12`
- QoS: `normal`
- resources: `gpu:1 cpu:10 mem:30G time=06:00:00`
- training config: `experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml`
- env includes `tensorboard` package for scalar export (`scripts/export_tb_scalars.py`)

## Quick Start

### 0) Push local project directly to TC2

Run this on your local machine if you want to skip GitHub fork/pull and sync the current repo snapshot straight to TC2:

```bash
cd /Users/albert/coursework/AI6103-group/src/hrnet
bash cluster/push_project.sh
```

### 1) On cluster login node

```bash
cd ~/hrnet
bash cluster/setup_env.sh
export RUN_ID=hrnet_coco_w32_256x192_$(date +%Y%m%d_%H%M%S)
export DATASET_ROOT=/home/msai/xjiang026/datasets/coco
export COCO_BBOX_FILE=$DATASET_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json
export PRETRAINED_MODEL=$PWD/models/pytorch/imagenet/hrnet_w32-36af842e.pth
```

### 2) Smoke run (1 epoch)

```bash
bash cluster/submit_smoke.sh "$RUN_ID"
```

### 3) Chain run

Example: submit 8 jobs, each 10 epochs, up to 210 epochs.

```bash
# Optional: estimate chunk size from smoke logs (default budget 5.5h)
bash cluster/recommend_chunk_epochs.sh "$RUN_ID" 1 19800

bash cluster/submit_chain.sh "$RUN_ID" 10 210 8
```

### 4) Monitor

```bash
squeue -u $USER
sacct -u $USER --starttime today --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES
```

### 5) Finalize (eval + demo + manifest)

```bash
bash cluster/submit_finalize.sh "$RUN_ID" 210
```

### 6) Pull artifacts back to local

Run this on your local machine:

```bash
cd /Users/albert/coursework/AI6103-group/src/hrnet
bash cluster/pull_artifacts.sh "$RUN_ID" /Users/albert/coursework/AI6103-group/documents/runs
```

## Output Layout

`$RUN_ROOT` contains:

- `output/` training checkpoints and model files
- `log/` TensorBoard event logs
- `telemetry/` GPU time-series (`gpu_metrics_<jobid>.csv`)
- `summary/` command snapshots, `run_manifest.json`, `artifact_index.csv`, `run_summary.md`
- `slurm/` per-job out/err logs
- `demo/` optional demo outputs (headless snapshot from `scripts/run_demo_snapshot.py`)

## Notes

- `submit_chain.sh` will detect existing checkpoint epoch and continue from there.
- set `DEBUG_DUMP=1` only for smoke phase; keep `DEBUG_DUMP=0` for long runs.
- set `RUN_POST_EVAL=1` and `RUN_POST_DEMO=1` in environment if you want the job to run post-train validation/demo.
