"""
パイプライン一括実行：Phase 1（保持フレーム検出）→ 欠陥スキャン →
Phase 2（動き分類）→ Phase 3（デノイズ）を1コマンドで通す。

各ステップの中間成果物（JSON）は --work-dir に残るので、
個別に再実行・検証したい場合はそれぞれの run_*.py を直接使えばよい。

使い方：
  python run_pipeline.py --input ../test-assets/raw/05_3coma_01.mov \
      --work-dir ../test-assets/pipeline/05_3coma_01 \
      --mode texture_preserving --extend 2 --dust-robust
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list) -> None:
    print(f"\n=== {' '.join(str(c) for c in cmd[1:])}")
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--work-dir", required=True,
                        help="中間JSON・出力動画の置き場所")
    parser.add_argument("--mode", default="texture_preserving",
                        choices=["texture_preserving", "full_temporal_integration"])
    parser.add_argument("--extend", type=int, default=2)
    parser.add_argument("--grain-reduction", type=float, default=0.0)
    parser.add_argument("--dust-robust", action="store_true")
    parser.add_argument("--flicker-correction", action="store_true")
    parser.add_argument("--remove-line-noise", action="store_true")
    parser.add_argument("--remove-scan-noise", action="store_true")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--skip-motion", action="store_true",
                        help="Phase 2（動き分類）をスキップする")
    args = parser.parse_args()

    py = sys.executable
    here = Path(__file__).parent
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    name = Path(args.input).stem

    hold_groups = work / f"{name}_hold_groups.json"
    defects = work / f"{name}_defects.json"
    motion = work / f"{name}_motion.json"
    output = work / f"{name}_{args.mode}.mov"

    # Phase 1：保持フレーム検出
    cmd = [py, here / "run_detection.py", "--input", args.input,
           "--output", hold_groups]
    if args.dust_robust:
        cmd.append("--dust-robust")
    run(cmd)

    # 欠陥スキャン（傷・ラインノイズのレポート）
    run([py, here / "run_defect_scan.py", "--input", args.input,
         "--hold-groups", hold_groups, "--output", defects])

    # Phase 2：動き分類
    if not args.skip_motion:
        run([py, here / "run_motion_classification.py", "--input", args.input,
             "--hold-groups", hold_groups, "--output", motion])

    # Phase 3：デノイズ
    cmd = [py, here / "run_denoise.py", "--input", args.input,
           "--hold-groups", hold_groups, "--output", output,
           "--mode", args.mode, "--extend", args.extend,
           "--grain-reduction", args.grain_reduction]
    if args.flicker_correction:
        cmd.append("--flicker-correction")
    if args.remove_line_noise:
        cmd.append("--remove-line-noise")
    if args.remove_scan_noise:
        cmd.append("--remove-scan-noise")
    if args.side_by_side:
        cmd.append("--side-by-side")
    run(cmd)

    print(f"\nパイプライン完了。成果物: {work}")


if __name__ == "__main__":
    main()
