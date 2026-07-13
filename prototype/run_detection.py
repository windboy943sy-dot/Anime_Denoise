"""
Phase 1 実行スクリプト：動画から保持フレームグループを検出し、
実装ロードマップ「解析結果の共通データフォーマット」に沿ったJSONを出力する。

使い方：

  # 動画全体を1カットとして解析する場合
  python run_detection.py --input ../test-assets/raw/sample01.mov \
      --output ../test-assets/labels/sample01_hold_groups.json

  # ラベリング済みのカット単位CSV（shot_labels_template.csv形式）を使い、
  # カットごとに解析する場合（推奨：設計提案書1章で「区間ごとに独立判定」とされているため）
  python run_detection.py --input ../test-assets/raw/sample01.mov \
      --shots-csv ../test-assets/labels/sample01_labels.csv \
      --output ../test-assets/labels/sample01_hold_groups.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from hold_frame_detection import (
    DetectionThresholds,
    detect_hold_groups,
    dominant_pattern_for_shot,
    estimate_koma_pattern,
    refine_hold_groups,
    split_drifting_groups,
)


def read_frames(video_path: str, frame_start: int = 0, frame_end: int | None = None) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {video_path}")

    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    frames = []
    idx = frame_start
    while True:
        if frame_end is not None and idx > frame_end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1

    cap.release()
    return frames


def process_shot(video_path: str, shot_id: str, frame_start: int, frame_end: int,
                  thresholds: DetectionThresholds, dust_robust: bool = False,
                  drift_check: bool = True) -> dict:
    frames = read_frames(video_path, frame_start, frame_end)
    groups = detect_hold_groups(frames, thresholds)
    if dust_robust:
        # ダストの点滅で分裂したグループを統合（実装ロードマップの反復設計）
        groups = refine_hold_groups(frames, groups, thresholds)
    if drift_check:
        # 累積ドリフト検査：超低速ズーム等で融合した巨大グループを分割
        # （時間統合時のゴースト防止。既定で有効）
        groups = split_drifting_groups(frames, groups, thresholds)
    groups = estimate_koma_pattern(groups)

    return {
        "shot_id": shot_id,
        "frame_range": [frame_start, frame_end if frame_end is not None else frame_start + len(frames) - 1],
        "dominant_koma_pattern": dominant_pattern_for_shot(groups),
        "hold_groups": [
            {
                # 出力はカット先頭からの相対番号ではなく、動画全体でのフレーム番号に変換する
                "start": frame_start + g.start,
                "end": frame_start + g.end,
                "pattern": g.pattern,
                "confidence": round(g.confidence, 3),
            }
            for g in groups
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="入力動画ファイル")
    parser.add_argument("--output", required=True, help="出力JSONのパス")
    parser.add_argument("--shots-csv", default=None,
                         help="カット単位のラベルCSV（shot_labels_template.csv形式）。"
                              "指定しない場合は動画全体を1カットとして解析する")
    parser.add_argument("--coarse-phash-threshold", type=int, default=16)
    parser.add_argument("--diff-threshold", type=float, default=3.0)
    parser.add_argument("--ssim-threshold", type=float, default=0.92)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--dust-robust", action="store_true",
                        help="ダストの点滅で分裂したグループを再判定で統合する"
                             "（ダスト・白ゴミの多い素材向け）")
    parser.add_argument("--no-drift-check", action="store_true",
                        help="累積ドリフト検査（超低速動きによる巨大グループの分割）を無効化")
    args = parser.parse_args()

    thresholds = DetectionThresholds(
        coarse_phash_threshold=args.coarse_phash_threshold,
        diff_threshold=args.diff_threshold,
        ssim_threshold=args.ssim_threshold,
        block_size=args.block_size,
    )

    results = []

    if args.shots_csv:
        df = pd.read_csv(args.shots_csv)
        # 複数ファイル分をまとめたラベルCSVに対応：source_file 列があれば
        # 入力動画のファイル名と一致する行だけを処理する
        if "source_file" in df.columns:
            input_name = Path(args.input).name
            df = df[df["source_file"].astype(str).str.endswith(input_name)]
        for _, row in df.iterrows():
            if pd.isna(row.get("frame_start")) or pd.isna(row.get("frame_end")):
                continue  # 未記入行はスキップ
            result = process_shot(
                args.input,
                shot_id=str(row["shot_id"]),
                frame_start=int(row["frame_start"]),
                frame_end=int(row["frame_end"]),
                thresholds=thresholds,
                dust_robust=args.dust_robust,
                drift_check=not args.no_drift_check,
            )
            results.append(result)
            print(f"shot {result['shot_id']}: "
                  f"{len(result['hold_groups'])} groups, "
                  f"dominant={result['dominant_koma_pattern']}")
    else:
        result = process_shot(args.input, shot_id="0001", frame_start=0, frame_end=None,
                               thresholds=thresholds, dust_robust=args.dust_robust,
                               drift_check=not args.no_drift_check)
        results.append(result)
        print(f"shot {result['shot_id']}: "
              f"{len(result['hold_groups'])} groups, "
              f"dominant={result['dominant_koma_pattern']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"source_file": args.input, "shots": results}, f, ensure_ascii=False, indent=2)

    print(f"結果を書き出しました: {args.output}")


if __name__ == "__main__":
    main()
