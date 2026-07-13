"""
定量評価の一括実行：ラベルCSVの全ファイルに対して
  1. ラベルの区間定義（--shots-csv）で Phase 1 検出を実行
  2. その結果で Phase 2 動き分類を実行
  3. evaluate_hold_frames / evaluate_motion で正答率を算出
を通しで行う。

使い方：
  python run_full_evaluation.py \
      --labels ../test-assets/labels/shot_labels_v0.3.csv \
      --raw-dir ../test-assets/raw \
      --out-dir ../test-assets/labels/evaluation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list) -> int:
    print(f"\n=== {' '.join(str(c) for c in cmd[1:])}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dust-robust", action="store_true",
                        help="Phase 1 にダスト耐性再判定を適用する")
    args = parser.parse_args()

    py = sys.executable
    here = Path(__file__).parent
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.labels)
    files = sorted(set(Path(str(f)).name for f in df["source_file"].dropna()))
    print(f"{len(files)} ファイルを評価します")

    merged_shots = []
    failed = []
    for name in files:
        video = Path(args.raw_dir) / name
        stem = Path(name).stem
        hold_json = out / f"{stem}_hold_groups.json"
        motion_json = out / f"{stem}_motion.json"

        cmd = [py, here / "run_detection.py", "--input", video,
               "--shots-csv", args.labels, "--output", hold_json]
        if args.dust_robust:
            cmd.append("--dust-robust")
        if run(cmd) != 0:
            failed.append((name, "detection"))
            continue

        if run([py, here / "run_motion_classification.py", "--input", video,
                "--hold-groups", hold_json, "--output", motion_json]) != 0:
            failed.append((name, "motion"))
            continue

        with open(hold_json, encoding="utf-8") as f:
            merged_shots.extend(json.load(f)["shots"])

    merged_path = out / "all_hold_groups.json"
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({"source_file": "(merged)", "shots": merged_shots}, f,
                  ensure_ascii=False, indent=1)

    print("\n" + "=" * 60)
    print("Phase 1（コマ打ちパターン）評価")
    print("=" * 60, flush=True)
    run([py, here / "evaluate_hold_frames.py",
         "--predictions", merged_path, "--labels", args.labels])

    print("\n" + "=" * 60)
    print("Phase 2（動き分類）評価")
    print("=" * 60, flush=True)
    run([py, here / "evaluate_motion.py",
         "--predictions", out, "--labels", args.labels])

    if failed:
        print("\n失敗したステップ:")
        for name, step in failed:
            print(f"  {name}: {step}")


if __name__ == "__main__":
    main()
