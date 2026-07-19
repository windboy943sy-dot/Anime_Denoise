"""検知結果の可視化(検知のみモードのオーバーレイ)。§9.1 B7

除去はしない。欠陥を色分けオーバーレイし、確率ヒートマップも出せる。
依存は numpy のみ(PNG 書き出しは呼び出し側で cv2/PIL を使う)。
"""
from __future__ import annotations

import numpy as np

from .contracts import DefectMap, DefectType

# 種別ごとの色(RGB, 0..1)
_TYPE_COLORS = {
    DefectType.DUST_WHITE: (1.0, 0.2, 0.2),
    DefectType.DUST_BLACK: (0.2, 0.5, 1.0),
    DefectType.PARTICLE: (1.0, 0.6, 0.0),
    DefectType.MOLD: (0.6, 0.0, 0.8),
    DefectType.SCRATCH_VERTICAL: (0.0, 1.0, 0.4),
    DefectType.SCRATCH_HORIZONTAL: (0.0, 0.8, 0.8),
    DefectType.SCRATCH_CURVED: (0.8, 0.8, 0.0),
    DefectType.DROPOUT: (1.0, 0.0, 1.0),
    DefectType.UNKNOWN: (0.7, 0.7, 0.7),
}


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1).astype(np.float32)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2).astype(np.float32)
    return img[..., :3].astype(np.float32)


def overlay(frame: np.ndarray, dmap: DefectMap, alpha: float = 0.5) -> np.ndarray:
    """フレームに欠陥インスタンスを種別色でオーバーレイ(0..1 RGB)。"""
    out = _ensure_rgb(frame).copy()
    for ins in dmap.instances:
        color = np.array(_TYPE_COLORS.get(ins.type, _TYPE_COLORS[DefectType.UNKNOWN]),
                         np.float32)
        m = dmap.labels == ins.id
        if not m.any():
            continue
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0.0, 1.0)


def probability_heatmap(dmap: DefectMap) -> np.ndarray:
    """確率マップを赤系ヒートマップに(0..1 RGB)。"""
    p = dmap.prob
    heat = np.stack([p, p * 0.3, np.zeros_like(p)], axis=-1)
    return np.clip(heat, 0.0, 1.0)
