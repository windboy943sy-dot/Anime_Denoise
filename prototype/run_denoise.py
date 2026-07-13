"""
Phase 3 実行スクリプト：Phase 1 の保持グループ検出結果（JSON）を入力に、
保持グループ単位でデノイズした動画を書き出す。

使い方：

  python run_denoise.py --input ../test-assets/raw/05_3coma_01.mov \
      --hold-groups ../test-assets/labels/detections/05_3coma_01_hold_groups.json \
      --output ../test-assets/denoised/05_3coma_01_texpres.mov \
      --mode texture_preserving

  # 完全時間統合モード＋第2層（前後2グループのカット間拡張統合）＋左右比較動画
  python run_denoise.py --input ../test-assets/raw/01_Pan_01.mov \
      --hold-groups ../test-assets/labels/detections/01_Pan_01_hold_groups.json \
      --output ../test-assets/denoised/01_Pan_01_full.mov \
      --mode full_temporal_integration --extend 2 --side-by-side
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from denoise import DenoiseParams, analyze_hold_group, render_hold_group
from denoise.extend import ExtendParams, blend_spatial_fallback, extend_reference
from denoise.linenoise import correct_line_noise, detect_line_noise
from denoise.scannoise import correct_scan_noise, detect_scan_noise
from denoise.scratch import build_scratch_mask, repair_scratches


def read_frames(cap: cv2.VideoCapture, start: int, end: int) -> list:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end + 1):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    return frames


def render_with_extension(window: deque, center_pos: int,
                          params: DenoiseParams,
                          extend_params: ExtendParams) -> tuple:
    """ウィンドウ内の中心グループを、両隣を使って拡張統合してから出力する。"""
    center = window[center_pos][1]
    neighbors = [window[i][1] for i in range(len(window))
                 if i != center_pos and abs(i - center_pos) <= extend_params.radius]
    ext = extend_reference(center, neighbors, extend_params, params)
    outputs = render_hold_group(center, params, reference_out=ext["reference"])

    # 第3層：時間統合が効かなかった画素（実効Nが低い）にだけ空間NRを混ぜる
    # 連続ブレンド（完全時間統合モードのみ。texture_preservingは
    # render_hold_group 内の grain_reduction が同じ空間NRを担当する）
    if params.mode == "full_temporal_integration" and params.grain_reduction > 0:
        outputs = [blend_spatial_fallback(o, ext["effective_n"],
                                          center["grain_sigma"],
                                          strength=params.grain_reduction)
                   for o in outputs]
    stats = {
        "used_neighbors": ext["used_neighbors"],
        "accept_ratios": ext["accept_ratios"],
        "mean_effective_n": round(float(ext["effective_n"].mean()), 2),
    }
    return outputs, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--hold-groups", required=True,
                        help="Phase 1 (run_detection.py) の出力JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="texture_preserving",
                        choices=["texture_preserving", "full_temporal_integration"])
    parser.add_argument("--reference-method", default="trimmed_mean",
                        choices=["median", "trimmed_mean", "mean"])
    parser.add_argument("--grain-reduction", type=float, default=0.0,
                        help="texture_preservingでの単独フレーム空間NR強度(0-1)")
    parser.add_argument("--dust-sigma", type=float, default=5.0)
    parser.add_argument("--extend", type=int, default=0,
                        help="第2層：前後このグループ数まで拡張統合する（0=無効）")
    parser.add_argument("--flicker-correction", action="store_true")
    parser.add_argument("--remove-line-noise", action="store_true",
                        help="行/列ラインノイズの検出・補正を有効化")
    parser.add_argument("--remove-scan-noise", action="store_true",
                        help="FFT周期スキャンノイズの検出・補正を有効化")
    parser.add_argument("--scratch-defects", default=None,
                        help="run_defect_scan.py の出力JSON。指定すると傷候補列を"
                             "inpaintで補修する（候補は目視確認済みであること）")
    parser.add_argument("--no-align", action="store_true",
                        help="グループ内サブピクセル位置合わせを無効化（比較検証用）")
    parser.add_argument("--no-dust", action="store_true")
    parser.add_argument("--side-by-side", action="store_true",
                        help="左：オリジナル／右：処理後 の比較動画も出力する")
    args = parser.parse_args()

    params = DenoiseParams(
        mode=args.mode,
        reference_method=args.reference_method,
        grain_reduction=args.grain_reduction,
        dust_sigma=args.dust_sigma,
        align=not args.no_align,
        dust_detection=not args.no_dust,
        flicker_correction=args.flicker_correction,
    )
    extend_params = ExtendParams(radius=args.extend)

    with open(args.hold_groups, encoding="utf-8") as f:
        detection = json.load(f)

    # 傷候補（run_defect_scan.py の出力）をショットIDごとに読み込む
    scratch_by_shot = {}
    if args.scratch_defects:
        with open(args.scratch_defects, encoding="utf-8") as f:
            for s in json.load(f)["shots"]:
                if s.get("scratch_columns"):
                    scratch_by_shot[s["shot_id"]] = s["scratch_columns"]

    # 対象別ON/OFFの一覧表示（設計提案書3.2節：何がオンかを一目でわかるように）
    print(f"Mode: {args.mode}")
    print(f"  Dust/Dirt Removal        [{'ON' if not args.no_dust else 'OFF'}]"
          f" sigma={args.dust_sigma}")
    print(f"  Scratch Removal          [{'ON' if scratch_by_shot else 'OFF'}]")
    print(f"  Grain Reduction          [{'ON' if args.grain_reduction > 0 else 'OFF'}]"
          f" strength={args.grain_reduction}")
    print(f"  Flicker Correction       [{'ON' if args.flicker_correction else 'OFF'}]")
    print(f"  Line Noise Removal       [{'ON' if args.remove_line_noise else 'OFF'}]")
    print(f"  Scan Noise Removal       [{'ON' if args.remove_scan_noise else 'OFF'}]")
    print(f"  Layer2 Extend            [{'ON' if args.extend > 0 else 'OFF'}]"
          f" radius={args.extend}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # VideoWriter は最初の書き込み直前に遅延オープンする。
    # 単一の長い保持グループでは開いてから最初の write まで数分空くことがあり、
    # ネットワークボリューム上でファイルハンドルが失効して全書き込みが
    # 失敗する事故が実際に起きたため（36バイトの壊れた出力になる）
    writers = {}

    def get_writer(key: str, path: str) -> cv2.VideoWriter:
        if key not in writers:
            writers[key] = cv2.VideoWriter(path, fourcc, fps, (w, h))
        return writers[key]

    sbs_path = None
    if args.side_by_side:
        sbs_path = str(Path(args.output).with_suffix("")) + "_sbs" + Path(args.output).suffix

    def post_process(outputs, analysis, scratch_mask):
        """対象別ON/OFF機構（3.2節）：ラインノイズ・スキャンノイズ・傷の補正。"""
        ref = analysis["reference"]
        if args.remove_line_noise:
            rows = detect_line_noise(ref, axis=0)
            cols = detect_line_noise(ref, axis=1)
            if rows or cols:
                outputs = [correct_line_noise(correct_line_noise(o, rows, axis=0),
                                              cols, axis=1) for o in outputs]
        if args.remove_scan_noise:
            det_h = detect_scan_noise(ref, axis=0)
            det_v = detect_scan_noise(ref, axis=1)
            if det_h or det_v:
                outputs = [correct_scan_noise(correct_scan_noise(o, ref, det_h, axis=0),
                                              ref, det_v, axis=1) for o in outputs]
        if scratch_mask is not None:
            outputs = [repair_scratches(o, scratch_mask) for o in outputs]
        return outputs

    def emit(group_range, orig_frames, outputs, note=""):
        for orig, out in zip(orig_frames, outputs):
            get_writer("main", args.output).write(out)
            if sbs_path is not None:
                half = w // 2
                sbs = np.hstack([orig[:, :half], out[:, half:]])
                cv2.line(sbs, (half, 0), (half, h), (0, 0, 255), 2)
                get_writer("sbs", sbs_path).write(sbs)
        print(f"group [{group_range[0]}-{group_range[1]}]{note}", flush=True)

    for shot in detection["shots"]:
        groups = shot["hold_groups"]
        scratch_mask = None
        if shot["shot_id"] in scratch_by_shot:
            scratch_mask = build_scratch_mask((h, w), scratch_by_shot[shot["shot_id"]])

        if args.extend <= 0:
            for g in groups:
                frames = read_frames(cap, g["start"], g["end"])
                if not frames:
                    continue
                analysis = analyze_hold_group(frames, params)
                outputs = render_hold_group(analysis, params)
                outputs = post_process(outputs, analysis, scratch_mask)
                dust = sum(int((m > 0).sum()) for m in analysis["dust_masks"]
                           if m is not None)
                emit((g["start"], g["end"]), frames, outputs,
                     f" sigma={analysis['grain_sigma']:.2f} dust_px={dust}")
            continue

        # 第2層：スライディングウィンドウ（中心±radius）で拡張統合しながら出力。
        # ウィンドウには (グループ情報, 解析結果, 元フレーム) を保持する
        radius = args.extend
        window: deque = deque()

        def flush(center_pos: int):
            g, analysis, orig = window[center_pos]
            outputs, stats = render_with_extension(window, center_pos, params, extend_params)
            outputs = post_process(outputs, analysis, scratch_mask)
            emit((g["start"], g["end"]), orig, outputs,
                 f" ext: neighbors={stats['used_neighbors']} "
                 f"effN={stats['mean_effective_n']} accept={stats['accept_ratios']}")

        pending_left = 0  # まだ出力していない左端グループ数（ウィンドウ充填前）
        for g in groups:
            frames = read_frames(cap, g["start"], g["end"])
            if not frames:
                continue
            analysis = analyze_hold_group(frames, params)
            window.append((g, analysis, frames))
            pending_left += 1

            if len(window) == 2 * radius + 1:
                # ウィンドウが埋まった：中心を出力。ただし最初の充填時は
                # 左端〜中心-1（片側ウィンドウしか無かったグループ）も先に出力する
                while pending_left > radius + 1:
                    flush(len(window) - pending_left)
                    pending_left -= 1
                flush(radius)
                pending_left -= 1
                window.popleft()

        # ショット末尾：残りを truncated window で出力
        while pending_left > 0:
            flush(len(window) - pending_left)
            pending_left -= 1

    cap.release()
    for wr in writers.values():
        wr.release()
    print(f"完了: {args.output}")


if __name__ == "__main__":
    main()
