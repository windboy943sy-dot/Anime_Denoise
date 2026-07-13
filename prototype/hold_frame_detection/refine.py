"""
Phase 1 の反復改善：ダスト耐性のある保持グループ再判定
（実装ロードマップ Phase 1「既知のリスク」の反復設計に対応）

ダスト・白ゴミが多い素材では、欠陥の点滅がフレーム間差分を押し上げ、
本当は同一絵柄のフレームが別グループに分裂する（実測：ダスト素材が
ほぼ全フレーム 1koma 判定になる）。

対策：隣接グループの境界フレーム同士を「ダストを除外して」再比較し、
  1. 差分の大きい画素がすべて「ダスト状」（コンパクトな小塊）で、
  2. その面積が画面のごく一部で、
  3. ダスト画素を除いた残りの差分が通常の保持判定を満たす
場合に限りグループを統合する。動きのある境界ではセルのエッジに沿った
細長い・大面積の差分が出るため統合されない（安全側）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .core import DetectionThresholds, HoldGroup


def _same_except_dust(frame_a: np.ndarray, frame_b: np.ndarray,
                      thresholds: DetectionThresholds,
                      max_dust_fraction: float = 0.10,
                      dust_max_area_ratio: float = 0.0008) -> bool:
    """2フレームが「ダスト等の単発欠陥を除けば同一」かを判定する。"""
    k = thresholds.blur_ksize
    gray_a = cv2.GaussianBlur(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), (k, k), 0)
    gray_b = cv2.GaussianBlur(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), (k, k), 0)
    d = np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))

    if float(d.mean()) < thresholds.diff_threshold:
        return True  # ダスト抜きでもそもそも同一判定

    # 差分の大きい画素（＝ダスト候補 or 動き）
    high = (d > max(6.0, thresholds.diff_threshold * 2.0)).astype(np.uint8)
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    frac = float(high.mean())
    if frac > max_dust_fraction:
        return False  # 画面の広範囲が変わっている＝動き

    h, w = d.shape
    max_area = int(h * w * dust_max_area_ratio)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
    for j in range(1, n_labels):
        area = stats[j, cv2.CC_STAT_AREA]
        bw = stats[j, cv2.CC_STAT_WIDTH]
        bh = stats[j, cv2.CC_STAT_HEIGHT]
        if area > max_area:
            return False  # 大きな塊＝絵の変化
        elongation = max(bw, bh) / max(1, min(bw, bh))
        fill_ratio = area / max(1, bw * bh)
        if elongation > 6.0 and fill_ratio < 0.3:
            return False  # 細長い成分＝セルエッジの動き

    # ダスト候補（少し膨らませる）を除いた残りで通常の保持判定
    dust_region = cv2.dilate(high, np.ones((5, 5), np.uint8)) > 0
    rest = d[~dust_region]
    if rest.size == 0:
        return False
    return float(rest.mean()) < thresholds.diff_threshold


def refine_hold_groups(frames: list[np.ndarray],
                       groups: list[HoldGroup],
                       thresholds: DetectionThresholds | None = None) -> list[HoldGroup]:
    """ダストで分裂した隣接グループを統合して返す。

    frames はカット内フレーム列（groups の start/end と同じ座標系）。
    統合は「境界フレーム同士がダスト抜きで同一」の場合のみ。連鎖統合に対応
    （A-B が統合されたら、次は (A+B)-C の境界を判定する）。
    """
    thresholds = thresholds or DetectionThresholds()
    if len(groups) < 2:
        return groups

    merged: list[HoldGroup] = [groups[0]]
    for g in groups[1:]:
        prev = merged[-1]
        if _same_except_dust(frames[prev.end], frames[g.start], thresholds):
            merged[-1] = HoldGroup(
                start=prev.start,
                end=g.end,
                confidence=float((prev.confidence * prev.length +
                                  g.confidence * g.length) /
                                 (prev.length + g.length)),
            )
        else:
            merged.append(g)
    return merged
