"""
ラインノイズ検出・補正プロトタイプ（設計提案書 4章 対応）

スキャナ・テレシネ由来のラインノイズは「行（または列）全体の輝度が
周囲の行から一様にずれる」形で現れる。検出は設計どおり
「行・列単位の平均値が周囲と統計的に有意にずれているか」の外れ値検定：

  1. 行ごとの輝度中央値プロファイルを取り、メディアン平滑との差（dev）を計算
  2. devのロバストσ（MAD）に対して外れ値の行を候補にする
  3. 一様性チェック：本物のラインノイズは行全体が同じ量だけずれる。
     絵柄の水平線・エッジは位置によってずれ量がばらつくので除外できる
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .core import _luma


def detect_line_noise(reference: np.ndarray,
                      axis: int = 0,
                      sigma_factor: float = 5.0,
                      min_offset: float = 1.5,
                      uniformity_ratio: float = 0.6,
                      active_area_crop: float = 0.10) -> list[dict]:
    """行（axis=0）または列（axis=1）単位のラインノイズを検出する。

    返り値：[{"index": 行/列番号, "offset": ずれ量, "uniformity": 一様性}]
    """
    y = _luma(reference)
    h, w = y.shape
    my, mx = int(h * active_area_crop), int(w * active_area_crop)
    inner = y[my:h - my, mx:w - mx]
    if axis == 1:
        inner = inner.T  # 以降は「行」として扱う

    prof = np.median(inner, axis=1)
    smooth = median_filter(prof, size=9)
    dev = prof - smooth

    mad = np.median(np.abs(dev - np.median(dev)))
    threshold = max(min_offset, sigma_factor * mad * 1.4826)

    results = []
    for i in np.where(np.abs(dev) > threshold)[0]:
        # 一様性チェック：行全体が同じ量ずれているか。
        # 上下の行の平均を「本来の値」とみなし、ずれ量の位置ごとのばらつきを見る
        lo, hi = max(0, i - 2), min(inner.shape[0], i + 3)
        neighbor_rows = [r for r in range(lo, hi) if r != i]
        baseline = inner[neighbor_rows].mean(axis=0)
        line_dev = inner[i] - baseline
        med_dev = float(np.median(line_dev))
        if abs(med_dev) < min_offset:
            continue
        # ずれの符号が行の大半で一致していること（絵柄エッジならばらける）
        agree = float((np.sign(line_dev) == np.sign(med_dev)).mean())
        if agree < uniformity_ratio:
            continue
        results.append({
            "index": int(i + (my if axis == 0 else mx)),
            "offset": round(med_dev, 2),
            "uniformity": round(agree, 3),
        })
    return results


def correct_line_noise(frame: np.ndarray, detections: list[dict],
                       axis: int = 0, strength: float = 1.0) -> np.ndarray:
    """検出済みラインノイズのオフセットを差し引いて補正する。"""
    if not detections or strength <= 0:
        return frame
    out = frame.astype(np.float32)
    for d in detections:
        i = d["index"]
        corr = d["offset"] * strength
        if axis == 0:
            out[i, :, :] -= corr
        else:
            out[:, i, :] -= corr
    return np.clip(out, 0, 255).astype(np.uint8)
