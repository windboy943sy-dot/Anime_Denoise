"""
欠陥スキャン実行スクリプト：Phase 1 の保持グループ検出結果（JSON）を入力に、
傷（縦スクラッチ）とラインノイズを検出してレポートJSONを出力する。

- 傷：グループ間の持続性で検出（静止カットでは絵柄の縦線と区別できないため
  候補の提示まで。除去は run_denoise 側での適用を予定＝ユーザー確認前提）
- ラインノイズ：グループごとの参照像に対する行・列の外れ値検定

使い方：
  python run_defect_scan.py --input ../test-assets/raw/sample.mov \
      --hold-groups ../test-assets/labels/detections/sample_hold_groups.json \
      --output ../test-assets/labels/detections/sample_defects.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from denoise import DenoiseParams, analyze_hold_group
from denoise.linenoise import detect_line_noise
from denoise.scratch import detect_scratch_columns


def read_frames(cap: cv2.VideoCapture, start: int, end: int) -> list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end + 1):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hold-groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch-threshold", type=float, default=6.0)
    parser.add_argument("--max-groups", type=int, default=12,
                        help="傷の持続性判定に使う最大グループ数（等間隔サンプリング）")
    args = parser.parse_args()

    with open(args.hold_groups, encoding="utf-8") as f:
        detection = json.load(f)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {args.input}")

    params = DenoiseParams(dust_detection=False)
    report = {"source_file": args.input, "shots": []}

    for shot in detection["shots"]:
        groups = shot["hold_groups"]
        # 等間隔にサンプリングしたグループの参照像を作る
        step = max(1, len(groups) // args.max_groups)
        sampled = groups[::step][:args.max_groups]

        refs = []
        line_noise_by_group = []
        for g in sampled:
            frames = read_frames(cap, g["start"], g["end"])
            if not frames:
                continue
            analysis = analyze_hold_group(frames, params)
            refs.append(analysis["reference"])
            rows = detect_line_noise(analysis["reference"], axis=0)
            cols = detect_line_noise(analysis["reference"], axis=1)
            if rows or cols:
                line_noise_by_group.append({
                    "group": [g["start"], g["end"]],
                    "rows": rows, "columns": cols,
                })

        scratch = {"columns": []}
        if len(refs) >= 2:
            scratch = detect_scratch_columns(
                refs, response_threshold=args.scratch_threshold)

        shot_report = {
            "shot_id": shot["shot_id"],
            "frame_range": shot["frame_range"],
            "sampled_groups": len(refs),
            "scratch_columns": scratch["columns"],
            "line_noise": line_noise_by_group,
        }
        report["shots"].append(shot_report)
        print(f"shot {shot['shot_id']}: 傷候補 {len(scratch['columns'])} 列, "
              f"ラインノイズあり {len(line_noise_by_group)} グループ")

    cap.release()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"結果を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
