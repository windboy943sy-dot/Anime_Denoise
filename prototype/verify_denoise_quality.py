"""
デノイズ品質の定量検証：処理前後の動画を比較して
  - 平坦部の高周波ノイズ量（グレイン残量の指標。低いほど良い）
  - エッジ鮮鋭度（Cannyエッジ上の平均Sobel勾配。下がっていなければエッジ非劣化）
  - 平坦部の時間方向ゆらぎ（フレーム間のちらつき。full統合ではほぼ0になるはず）
を算出する。

使い方：
  python verify_denoise_quality.py \
      --original ../test-assets/raw/05_3coma_01.mov \
      --processed ../test-assets/denoised/05_3coma_01_full.mov \
      [--sample-frames 12] [--json 出力パス]
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _flat_mask(gray: np.ndarray, crop: float = 0.12) -> np.ndarray:
    """フィルム枠を除いた中央領域のうち、勾配の小さい（平坦な）画素。"""
    g = cv2.GaussianBlur(gray, (7, 7), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    flat = np.sqrt(gx * gx + gy * gy) < 15
    h, w = gray.shape
    my, mx = int(h * crop), int(w * crop)
    border = np.zeros_like(flat)
    border[my:h - my, mx:w - mx] = True
    return flat & border


def flat_hf_noise(frame: np.ndarray) -> float:
    """平坦部の高周波成分std（グレイン残量の指標）。"""
    gray = _gray(frame)
    mask = _flat_mask(gray)
    hf = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(hf[mask].std()) if mask.any() else 0.0


def edge_sharpness(frame: np.ndarray) -> float:
    """Cannyエッジ上の平均Sobel勾配強度（高いほどシャープ）。"""
    gray = _gray(frame)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150) > 0
    return float(mag[edges].mean()) if edges.any() else 0.0


def temporal_flicker(pairs: list) -> float:
    """平坦部の「真の隣接フレーム」差の平均（時間方向のちらつき指標）。

    飛び飛びサンプル同士を比較すると絵柄の変化（コマ打ちの動き）が混入するため、
    必ず (i, i+1) の連続ペアで計測する。
    """
    if not pairs:
        return 0.0
    diffs = []
    for a, b in pairs:
        ga, gb = _gray(a), _gray(b)
        mask = _flat_mask(ga)
        diffs.append(float(np.abs(ga - gb)[mask].mean()))
    return float(np.mean(diffs))


def sample_pairs(path: str, n: int) -> list:
    """等間隔に選んだ位置から連続フレームペア (i, i+1) を読む。"""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return []
    idxs = np.linspace(0, total - 2, min(n, total - 1)).astype(int)
    pairs = []
    for i in sorted(set(int(x) for x in idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ra, fa = cap.read()
        rb, fb = cap.read()
        if ra and rb:
            pairs.append((fa, fb))
    cap.release()
    return pairs


def sample_video(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 2, 0), min(n, total)).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--json", default=None, help="結果をJSONでも書き出す")
    args = parser.parse_args()

    orig = sample_video(args.original, args.sample_frames)
    proc = sample_video(args.processed, args.sample_frames)
    if not proc:
        raise IOError(f"処理後動画を読めませんでした（壊れている可能性）: {args.processed}")
    orig_pairs = sample_pairs(args.original, args.sample_frames)
    proc_pairs = sample_pairs(args.processed, args.sample_frames)

    result = {
        "original": args.original,
        "processed": args.processed,
        "flat_hf_noise": {
            "original": round(float(np.mean([flat_hf_noise(f) for f in orig])), 2),
            "processed": round(float(np.mean([flat_hf_noise(f) for f in proc])), 2),
        },
        "edge_sharpness": {
            "original": round(float(np.mean([edge_sharpness(f) for f in orig])), 1),
            "processed": round(float(np.mean([edge_sharpness(f) for f in proc])), 1),
        },
        "temporal_flicker": {
            "original": round(temporal_flicker(orig_pairs), 3),
            "processed": round(temporal_flicker(proc_pairs), 3),
        },
    }

    n = result["flat_hf_noise"]
    e = result["edge_sharpness"]
    t = result["temporal_flicker"]
    noise_drop = (1 - n["processed"] / max(n["original"], 1e-6)) * 100
    sharp_delta = (e["processed"] / max(e["original"], 1e-6) - 1) * 100
    print(f"平坦部HFノイズ : {n['original']:6.2f} → {n['processed']:6.2f}  ({noise_drop:+.1f}% 削減)")
    print(f"エッジ鮮鋭度   : {e['original']:6.1f} → {e['processed']:6.1f}  ({sharp_delta:+.1f}%)")
    print(f"時間ちらつき   : {t['original']:6.3f} → {t['processed']:6.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
