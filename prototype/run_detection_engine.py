#!/usr/bin/env python3
"""Phase 1 検知・解析エンジン CLI。

入力映像(または合成デモ)を解析し、DefectMap レポート(JSON)と、検知のみ
オーバーレイ(PNG)を出力する。除去は一切行わない(Phase 1 の責務)。

使用例:
  # 合成デモ(外部素材・cv2 不要。エンジンの動作確認)
  python run_detection_engine.py --demo --output ../test-assets/detect_demo

  # 実素材(cv2 が必要)
  python run_detection_engine.py --input clip.mov --output out_dir --max-frames 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from detection_engine import DefectAnalyzer, AnalyzerConfig
from detection_engine.io_png import write_png
from detection_engine import visualize


def load_video(path: str, max_frames: int):
    import cv2  # 実素材読み込みは cv2 に委譲(存在しなければ ImportError)
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(rgb)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description="Phase 1 検知・解析エンジン")
    ap.add_argument("--input", help="入力動画(cv2 が必要)")
    ap.add_argument("--demo", action="store_true", help="合成デモを解析")
    ap.add_argument("--output", required=True, help="出力ディレクトリ")
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--no-overlay", action="store_true", help="PNG 出力を省略")
    ap.add_argument("--dust-k", type=float, default=3.0)
    ap.add_argument("--scratch-k", type=float, default=4.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.demo:
        from detection_engine.synth import make_clip
        frames, _ = make_clip(n_frames=min(args.max_frames, 16), h=180, w=240,
                              sigma=0.025, dust_per_frame=8, n_scratches=2, seed=7)
        color_space = "synthetic_linear"
    elif args.input:
        frames = load_video(args.input, args.max_frames)
        color_space = "host_working_space"
    else:
        ap.error("--input か --demo のいずれかを指定してください")

    cfg = AnalyzerConfig(dust_k=args.dust_k, scratch_k=args.scratch_k)
    analyzer = DefectAnalyzer(cfg)
    analysis = analyzer.analyze_clip(frames, color_space=color_space)

    # クリップ診断
    diag = dict(analysis.diagnostics)
    diag["scratch_ledger"] = [
        {"track_id": t.track_id, "x": round(t.x, 2), "width": round(t.width, 2),
         "polarity": t.polarity, "persistence": t.persistence,
         "confidence": round(t.confidence, 3)}
        for t in analysis.scratch_ledger
    ]
    prof = analysis.noise_profile
    diag["noise_profile"] = {
        "dominant_model": prof.dominant_model,
        "global_sigma": round(prof.global_sigma, 5),
        "is_white": prof.is_white,
        "spatial_correlation_length": round(prof.spatial_correlation_length, 3),
        "temporal_sigma": (round(prof.temporal_sigma, 5)
                           if prof.temporal_sigma is not None else None),
        "chroma_dominant": prof.chroma_dominant,
        "color_space": prof.color_space,
    }
    with open(os.path.join(args.output, "clip_diagnostics.json"), "w") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    # フレーム毎レポート + オーバーレイ
    per_frame = []
    for i, fmap in enumerate(analysis.defect_maps):
        per_frame.append({
            "frame": i,
            "stats": fmap.frame_stats,
            "instances": [ins.to_dict() for ins in fmap.instances],
        })
        if not args.no_overlay:
            ov = visualize.overlay(frames[i], fmap, alpha=0.6)
            write_png(os.path.join(args.output, f"overlay_{i:04d}.png"), ov)
    with open(os.path.join(args.output, "defect_report.json"), "w") as f:
        json.dump(per_frame, f, ensure_ascii=False, indent=2)

    print(f"解析完了: {len(frames)} フレーム")
    print(f"  ノイズモデル: {prof.dominant_model} (sigma={prof.global_sigma:.4f})")
    print(f"  支配欠陥    : {diag['dominant_defect']}")
    print(f"  確定スクラッチ: {len(analysis.scratch_ledger)} 本")
    print(f"  平均欠陥密度: {diag['mean_defect_density']:.4f}")
    print(f"  出力先      : {args.output}/")


if __name__ == "__main__":
    main()
