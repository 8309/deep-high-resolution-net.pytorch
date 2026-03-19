#!/usr/bin/env bash
set -euo pipefail

SSH_USER_HOST="${SSH_USER_HOST:-xjiang026@10.96.189.12}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/msai/xjiang026/hrnet}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found"
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "[ERROR] ssh not found"
  exit 1
fi

echo "[INFO] local project:  $LOCAL_PROJECT_DIR"
echo "[INFO] remote target: $SSH_USER_HOST:$REMOTE_PROJECT_DIR"

ssh "$SSH_USER_HOST" "mkdir -p '$REMOTE_PROJECT_DIR'"

rsync -avh --progress \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'data/' \
  --exclude 'output/' \
  --exclude 'log/' \
  --exclude 'runs/' \
  --exclude '*.zip' \
  "$LOCAL_PROJECT_DIR/" \
  "$SSH_USER_HOST:$REMOTE_PROJECT_DIR/"

echo "[INFO] project sync complete"
echo "[INFO] next: ssh $SSH_USER_HOST"
echo "[INFO] then: cd $REMOTE_PROJECT_DIR"
