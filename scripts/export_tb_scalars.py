#!/usr/bin/env python3
"""Export TensorBoard scalar events to a flat CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar events to CSV")
    parser.add_argument("--tb_dir", required=True, help="TensorBoard log root directory")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tb_dir = Path(args.tb_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()

    if not tb_dir.exists():
      print(f"[ERROR] tb_dir not found: {tb_dir}", file=sys.stderr)
      return 2

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        print(
            "[ERROR] tensorboard package is required for scalar export. "
            "Install with: pip install tensorboard",
            file=sys.stderr,
        )
        print(f"[ERROR] import detail: {exc}", file=sys.stderr)
        return 3

    event_files = sorted(tb_dir.rglob("events.out.tfevents*"))
    if not event_files:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_file", "tag", "step", "wall_time", "value"])
        print(f"[WARN] no TensorBoard event files found under {tb_dir}")
        print(f"[INFO] wrote empty CSV: {out_csv}")
        return 0

    rows = []
    for event_file in event_files:
        relative = event_file.relative_to(tb_dir)
        try:
            accumulator = EventAccumulator(
                str(event_file),
                size_guidance={
                    "scalars": 0,
                    "images": 0,
                    "histograms": 0,
                    "tensors": 0,
                },
            )
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
        except Exception as exc:  # noqa: PERF203
            print(f"[WARN] skip unreadable event file {event_file}: {exc}", file=sys.stderr)
            continue

        for tag in scalar_tags:
            try:
                scalar_events = accumulator.Scalars(tag)
            except Exception as exc:  # noqa: PERF203
                print(f"[WARN] skip tag {tag} in {event_file}: {exc}", file=sys.stderr)
                continue
            for scalar in scalar_events:
                rows.append(
                    {
                        "source_file": str(relative),
                        "tag": tag,
                        "step": int(scalar.step),
                        "wall_time": float(scalar.wall_time),
                        "value": float(scalar.value),
                    }
                )

    rows.sort(key=lambda x: (x["tag"], x["step"], x["source_file"], x["wall_time"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_file", "tag", "step", "wall_time", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] processed event files: {len(event_files)}")
    print(f"[INFO] exported scalar rows: {len(rows)}")
    print(f"[INFO] output: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
