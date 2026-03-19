#!/usr/bin/env python3
"""Build run metadata files for reproducible training artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class FileRecord:
    rel_path: str
    size_bytes: int
    mtime_utc: str
    sha256: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run manifest and artifact indexes")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--project_dir", default=os.getcwd())
    parser.add_argument("--cfg_file", default="")
    parser.add_argument("--begin_epoch", default="")
    parser.add_argument("--end_epoch", default="")
    parser.add_argument("--output", required=True, help="run_manifest.json output path")
    parser.add_argument("--artifact_index", default="", help="artifact_index.csv output path")
    parser.add_argument("--run_summary", default="", help="run_summary.md output path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_cmd(cmd: list[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def build_file_records(run_root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in iter_files(run_root):
        stat = path.stat()
        records.append(
            FileRecord(
                rel_path=str(path.relative_to(run_root)),
                size_bytes=stat.st_size,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                sha256=sha256_file(path),
            )
        )
    records.sort(key=lambda r: r.rel_path)
    return records


def first_existing_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def relpaths(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in paths if p.exists())


def main() -> int:
    args = parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    artifact_index = (
        Path(args.artifact_index).expanduser().resolve()
        if args.artifact_index
        else run_root / "summary" / "artifact_index.csv"
    )
    run_summary = (
        Path(args.run_summary).expanduser().resolve()
        if args.run_summary
        else run_root / "summary" / "run_summary.md"
    )

    if not run_root.exists():
        print(f"[ERROR] run_root not found: {run_root}")
        return 2

    records = build_file_records(run_root)
    total_bytes = sum(r.size_bytes for r in records)

    artifact_index.parent.mkdir(parents=True, exist_ok=True)
    with artifact_index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "size_bytes", "mtime_utc", "sha256"])
        for r in records:
            writer.writerow([r.rel_path, r.size_bytes, r.mtime_utc, r.sha256])

    checkpoint_files = list(run_root.rglob("checkpoint.pth"))
    best_model_files = list(run_root.rglob("model_best.pth"))
    final_state_files = list(run_root.rglob("final_state.pth"))
    tb_event_files = list(run_root.rglob("events.out.tfevents*"))
    results_json_files = list(run_root.rglob("keypoints_*_results_*.json"))
    gpu_csv_files = list(run_root.rglob("gpu_metrics_*.csv"))
    slurm_out_files = list((run_root / "slurm").glob("*.out")) if (run_root / "slurm").exists() else []
    slurm_err_files = list((run_root / "slurm").glob("*.err")) if (run_root / "slurm").exists() else []

    git_commit = run_cmd(["git", "rev-parse", "HEAD"], project_dir)
    git_remote_origin = run_cmd(["git", "remote", "get-url", "origin"], project_dir)
    git_status = run_cmd(["git", "status", "--short"], project_dir)

    checkpoint_epoch = first_existing_text(run_root / "summary" / "checkpoint_epoch.txt")

    manifest = {
        "run_id": args.run_id,
        "generated_at_utc": utc_now_iso(),
        "paths": {
            "run_root": str(run_root),
            "project_dir": str(project_dir),
            "cfg_file": args.cfg_file,
            "artifact_index": str(artifact_index),
            "run_summary": str(run_summary),
        },
        "epochs": {
            "begin_epoch": args.begin_epoch,
            "end_epoch": args.end_epoch,
            "checkpoint_epoch": checkpoint_epoch,
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "job_name": os.environ.get("SLURM_JOB_NAME", ""),
            "node": os.environ.get("SLURMD_NODENAME", ""),
        },
        "git": {
            "commit": git_commit,
            "origin": git_remote_origin,
            "has_uncommitted_changes": bool(git_status),
        },
        "inventory": {
            "file_count": len(records),
            "total_bytes": total_bytes,
        },
        "artifacts": {
            "checkpoints": relpaths(checkpoint_files, run_root),
            "model_best": relpaths(best_model_files, run_root),
            "final_state": relpaths(final_state_files, run_root),
            "tensorboard_events": relpaths(tb_event_files, run_root),
            "results_json": relpaths(results_json_files, run_root),
            "gpu_metrics_csv": relpaths(gpu_csv_files, run_root),
            "slurm_out": relpaths(slurm_out_files, run_root),
            "slurm_err": relpaths(slurm_err_files, run_root),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    run_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"# Run Summary: {args.run_id}",
        "",
        f"- Generated (UTC): {manifest['generated_at_utc']}",
        f"- Run root: `{run_root}`",
        f"- Project dir: `{project_dir}`",
        f"- Config: `{args.cfg_file}`",
        f"- Epoch window: `{args.begin_epoch}` -> `{args.end_epoch}`",
        f"- Checkpoint epoch: `{checkpoint_epoch}`",
        f"- File count: `{len(records)}`",
        f"- Total bytes: `{total_bytes}`",
        f"- Git commit: `{git_commit}`",
        f"- Git dirty: `{bool(git_status)}`",
        "",
        "## Key Artifact Counts",
        f"- checkpoints: {len(checkpoint_files)}",
        f"- model_best: {len(best_model_files)}",
        f"- final_state: {len(final_state_files)}",
        f"- tensorboard_events: {len(tb_event_files)}",
        f"- results_json: {len(results_json_files)}",
        f"- gpu_metrics_csv: {len(gpu_csv_files)}",
        f"- slurm_out: {len(slurm_out_files)}",
        f"- slurm_err: {len(slurm_err_files)}",
        "",
        "## Outputs",
        f"- Manifest: `{output}`",
        f"- Artifact index: `{artifact_index}`",
    ]
    run_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[INFO] wrote manifest: {output}")
    print(f"[INFO] wrote artifact index: {artifact_index}")
    print(f"[INFO] wrote run summary: {run_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
