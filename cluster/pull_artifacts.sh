#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_id> <local_dest>"
  exit 1
fi

RUN_ID="$1"
LOCAL_DEST="$2"

SSH_USER_HOST="${SSH_USER_HOST:-xjiang026@10.96.189.12}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/msai/xjiang026/hrnet}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-$REMOTE_PROJECT_DIR/runs/$RUN_ID}"

LOCAL_RUN_DIR="$LOCAL_DEST/$RUN_ID"
mkdir -p "$LOCAL_RUN_DIR"

echo "[INFO] pulling from $SSH_USER_HOST:$REMOTE_RUN_ROOT"
echo "[INFO] local target: $LOCAL_RUN_DIR"

rsync -avh --progress \
  "$SSH_USER_HOST:$REMOTE_RUN_ROOT/" \
  "$LOCAL_RUN_DIR/"

file_count=$(find "$LOCAL_RUN_DIR" -type f | wc -l | tr -d ' ')
dir_count=$(find "$LOCAL_RUN_DIR" -type d | wc -l | tr -d ' ')
bytes=$(du -sb "$LOCAL_RUN_DIR" | awk '{print $1}')

summary_file="$LOCAL_RUN_DIR/pull_summary.txt"
{
  echo "run_id=$RUN_ID"
  echo "remote=$SSH_USER_HOST:$REMOTE_RUN_ROOT"
  echo "local=$LOCAL_RUN_DIR"
  echo "pulled_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "files=$file_count"
  echo "dirs=$dir_count"
  echo "bytes=$bytes"
} > "$summary_file"

if command -v shasum >/dev/null 2>&1; then
  index_file="$LOCAL_RUN_DIR/pull_sha256_index.txt"
  (
    cd "$LOCAL_RUN_DIR"
    find . -type f ! -name "pull_sha256_index.txt" -print0 | \
      xargs -0 shasum -a 256
  ) > "$index_file"
fi

echo "[INFO] pull completed: files=$file_count bytes=$bytes"
echo "[INFO] summary: $summary_file"
