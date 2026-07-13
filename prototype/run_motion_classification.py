"""
Phase 2 実行スクリプト：Phase 1 の保持グループ検出結果（JSON）を入力に、
保持グループ間の動きを分類し、共通データフォーマットの
global_motion / local_motion セクションを出力する。

使い方：

  # まず Phase 1 を実行して hold_groups JSON を作っておく
  python run_detection.py --input ../test-assets/raw/sample01.mov \
      --output ../test-assets/labels/sample01_hold_groups.json

  # その JSON を入力に動き分類を実行
  python run_motion_classification.py \
      --input ../test-assets/raw/sample01.mov \
      --hold-groups ../test-assets/labels/sample01_hold_groups.json \
      --output ../test-assets/labels/sample01_motion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from motion_classification import (
    MotionThresholds,
    classify_local_motion,
    dominant_global_motion,
    estimate_global_motion,
    estimate_noise_floor,
)
from motion_classification.core import analyze_camera_path

MAX_NOISE_CALIBRATION_PAIRS = 3


def read_frame(cap: cv2.VideoCapture, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        raise IOError(f"フレーム {index} を読み込めませんでした")
    return frame


def collect_noise_pairs(cap: cv2.VideoCapture, groups: list[dict]) -> list:
    """保持グループ「内」のフレームペア（動きゼロの教師データ）を集める。"""
    pairs = []
    # 長いグループほどウィーブ・グレインのばらつきを広く含むので優先する
    for g in sorted(groups, key=lambda g: g["end"] - g["start"], reverse=True):
        if g["end"] - g["start"] < 1:
            continue
        pairs.append((read_frame(cap, g["start"]), read_frame(cap, g["end"])))
        if len(pairs) >= MAX_NOISE_CALIBRATION_PAIRS:
            break
    return pairs


def process_shot(cap: cv2.VideoCapture, shot: dict, thresholds: MotionThresholds) -> dict:
    groups = shot["hold_groups"]
    # 各保持グループの代表フレーム（中央）を使う。グレインはフレームごとに
    # 揺らぐが内容は同一なので、どのフレームを選んでも動き推定には影響しない想定
    rep_indices = [(g["start"] + g["end"]) // 2 for g in groups]
    rep_frames = [read_frame(cap, i) for i in rep_indices]

    # カット固有のノイズ床を保持グループ内ペアから較正する
    noise_pairs = collect_noise_pairs(cap, groups)
    noise_floor = estimate_noise_floor(noise_pairs, thresholds)

    global_segments = []
    local_segments = []
    global_motions = []

    for i in range(1, len(rep_frames)):
        g = estimate_global_motion(rep_frames[i - 1], rep_frames[i], thresholds)
        l = classify_local_motion(rep_frames[i - 1], rep_frames[i], g,
                                  noise_floor, thresholds)
        global_motions.append(g)

        # 区間は「前グループの先頭〜後グループの末尾」（この遷移が影響する範囲）
        seg_start, seg_end = groups[i - 1]["start"], groups[i]["end"]

        params = {"tx": round(g.tx, 2), "ty": round(g.ty, 2)}
        if g.type == "zoom":
            params["scale"] = round(g.scale, 4)
        if g.type == "rotation":
            params["rotation_deg"] = round(g.rotation_deg, 2)

        global_segments.append({
            "start": seg_start, "end": seg_end,
            "type": g.type, "params": params,
            "method": g.method, "confidence": round(g.confidence, 3),
        })
        local_segments.append({
            "start": seg_start, "end": seg_end,
            "type": l.type,
            "moving_ratio": round(l.moving_ratio, 4),
            "largest_component_ratio": round(l.largest_component_ratio, 3),
            "bbox": list(l.bbox) if l.bbox else None,
        })

    return {
        "shot_id": shot["shot_id"],
        "frame_range": shot["frame_range"],
        "noise_floor": round(noise_floor, 2),
        "dominant_global_motion": dominant_global_motion(global_motions),
        "camera_path": analyze_camera_path(global_motions),
        "global_motion": global_segments,
        "local_motion": local_segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="入力動画ファイル")
    parser.add_argument("--hold-groups", required=True,
                        help="Phase 1 (run_detection.py) が出力した hold_groups JSON")
    parser.add_argument("--output", required=True, help="出力JSONのパス")
    parser.add_argument("--work-width", type=int, default=640)
    parser.add_argument("--moving-ratio-full", type=float, default=0.35)
    parser.add_argument("--no-ecc", action="store_true",
                        help="ECCによるサブピクセル精密位置合わせを無効化（高速・低精度）")
    args = parser.parse_args()

    thresholds = MotionThresholds(
        work_width=args.work_width,
        moving_ratio_full=args.moving_ratio_full,
        use_ecc_refinement=not args.no_ecc,
    )

    with open(args.hold_groups, encoding="utf-8") as f:
        detection = json.load(f)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {args.input}")

    results = []
    for shot in detection["shots"]:
        if len(shot["hold_groups"]) < 2:
            print(f"shot {shot['shot_id']}: 保持グループが1つのためスキップ（動きなし）")
            continue
        result = process_shot(cap, shot, thresholds)
        results.append(result)
        local_types = [seg["type"] for seg in result["local_motion"]]
        from collections import Counter
        print(f"shot {result['shot_id']}: global={result['dominant_global_motion']}, "
              f"noise_floor={result['noise_floor']}, "
              f"local={dict(Counter(local_types))}, "
              f"{len(result['global_motion'])} transitions")

    cap.release()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"source_file": args.input, "shots": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"結果を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
