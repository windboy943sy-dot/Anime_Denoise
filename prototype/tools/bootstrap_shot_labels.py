"""
Phase 0 補助ツール：カット候補を自動抽出し、ラベリングCSVの下書きを生成する。

目的：
  test-assets/raw/ 以下の動画から、ヒストグラム差分ベースの簡易カット検出で
  カット境界の候補を切り出し、shot_labels_template.csv と同じ列を持つ
  下書きCSVを出力する。koma_pattern等の判定列は空欄のまま出力するので、
  人手で目視確認しながら埋めていく前提（実装ロードマップ docs/phase0_checklist.md 参照）。

  ここで使うカット検出はあくまで「候補の下書き」を作るための簡易版であり、
  設計提案書1章・9章で述べた本実装の保持フレーム解析ロジックとは別物。

使い方：
  python bootstrap_shot_labels.py --input ../../test-assets/raw/sample01_scan.mov \
      --output ../../test-assets/labels/sample01_draft.csv
"""

import argparse
import csv
import os

import cv2
import numpy as np


def compute_hist_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """2フレーム間のヒストグラム差分（Bhattacharyya距離）を計算する。"""
    hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)


def detect_shot_boundaries(video_path: str, threshold: float = 0.4) -> list[int]:
    """ヒストグラム差分がしきい値を超えるフレームをカット境界候補として返す。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {video_path}")

    boundaries = [0]
    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            diff = compute_hist_diff(prev_frame, frame)
            if diff > threshold:
                boundaries.append(frame_idx)

        prev_frame = frame
        frame_idx += 1

    total_frames = frame_idx
    if boundaries[-1] != total_frames - 1:
        boundaries.append(total_frames - 1)

    cap.release()
    return boundaries


def write_draft_csv(source_file: str, boundaries: list[int], output_path: str) -> None:
    fieldnames = [
        "shot_id", "source_file", "frame_start", "frame_end",
        "koma_pattern", "global_motion", "local_motion",
        "defects_present", "intentional_effects", "source_type", "notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(boundaries) - 1):
            writer.writerow({
                "shot_id": f"{i + 1:04d}",
                "source_file": source_file,
                "frame_start": boundaries[i],
                "frame_end": boundaries[i + 1] - 1,
                "koma_pattern": "",
                "global_motion": "",
                "local_motion": "",
                "defects_present": "",
                "intentional_effects": "",
                "source_type": "",
                "notes": "auto-detected shot boundary (要目視確認)",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="入力動画ファイル")
    parser.add_argument("--output", required=True, help="出力する下書きCSVのパス")
    parser.add_argument("--threshold", type=float, default=0.4,
                         help="カット境界とみなすヒストグラム差分のしきい値（デフォルト0.4）")
    args = parser.parse_args()

    boundaries = detect_shot_boundaries(args.input, args.threshold)
    write_draft_csv(os.path.relpath(args.input), boundaries, args.output)

    print(f"{len(boundaries) - 1} 個のカット候補を検出しました。")
    print(f"下書きCSVを書き出しました: {args.output}")
    print("koma_pattern 等の空欄列は目視確認のうえ記入してください。")


if __name__ == "__main__":
    main()
