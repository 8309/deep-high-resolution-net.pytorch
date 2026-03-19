#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [epochs_ran] [budget_seconds]"
  exit 1
fi

RUN_ID="$1"
EPOCHS_RAN="${2:-1}"
BUDGET_SECONDS="${3:-19800}" # 6h - 30m safety = 5.5h

if ! [[ "$EPOCHS_RAN" =~ ^[0-9]+$ ]] || (( EPOCHS_RAN <= 0 )); then
  echo "[ERROR] epochs_ran must be a positive integer"
  exit 1
fi
if ! [[ "$BUDGET_SECONDS" =~ ^[0-9]+$ ]] || (( BUDGET_SECONDS <= 0 )); then
  echo "[ERROR] budget_seconds must be a positive integer"
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG_FILE="${CFG_FILE:-$PROJECT_DIR/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_ID}"

cfg_basename="$(basename "$CFG_FILE")"
cfg_name="${cfg_basename%.*}"
log_dir="$RUN_ROOT/output/coco/pose_hrnet/$cfg_name"

if [[ ! -d "$log_dir" ]]; then
  echo "[ERROR] log dir not found: $log_dir"
  exit 1
fi

train_log="$(ls -1t "$log_dir"/*_train.log 2>/dev/null | head -n 1 || true)"
if [[ -z "$train_log" ]]; then
  echo "[ERROR] train log not found under $log_dir"
  exit 1
fi

python - "$train_log" "$EPOCHS_RAN" "$BUDGET_SECONDS" <<'PY'
import math
import re
import sys
from datetime import datetime

log_path = sys.argv[1]
epochs_ran = int(sys.argv[2])
budget_seconds = int(sys.argv[3])

pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

stamps = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pattern.match(line)
        if not m:
            continue
        stamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"))

if len(stamps) < 2:
    print(f"[ERROR] not enough timestamps in {log_path}")
    raise SystemExit(2)

elapsed = (max(stamps) - min(stamps)).total_seconds()
if elapsed <= 0:
    print(f"[ERROR] non-positive elapsed seconds from {log_path}")
    raise SystemExit(2)

epoch_seconds = elapsed / epochs_ran
chunk_epochs = max(1, int(math.floor(budget_seconds / epoch_seconds)))

print(f"[INFO] train_log={log_path}")
print(f"[INFO] elapsed_seconds={elapsed:.2f}")
print(f"[INFO] epochs_ran={epochs_ran}")
print(f"[INFO] epoch_seconds={epoch_seconds:.2f}")
print(f"[INFO] budget_seconds={budget_seconds}")
print(f"[RESULT] recommended_chunk_epochs={chunk_epochs}")
PY
