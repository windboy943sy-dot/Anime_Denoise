"""
線画マップ抽出（XDoG）— docs/reference/ 特化設計書 2.2節の採用実装。

用途：ダスト検出のエッジ保護・第2層のエッジ除外・統合ガードなど、
「線画・輝度勾配のある場所」を保護対象として参照する全モジュールの共有データ。

Canny(50,150) との違い：
  - Canny はヒステリシス閾値未満の低コントラスト境界（グラデーション帯・
    彩度差のみの境界）を落とす。実測では 05_3coma_01 の塔の縞がこれで漏れ、
    フィルム局所歪みの弧状誤検出（f81）の遠因になった
  - XDoG は DoG の連続応答を tanh で軟判定するため、低コントラスト線も
    「弱い線」として拾える。グレイン耐性は事前ぼかし（σ）で確保する

パラメータは 2560x1920 のフィルムスキャン実測で調整（下記デフォルト）。
"""

from __future__ import annotations

import cv2
import numpy as np


def xdog_response(gray: np.ndarray,
                  sigma: float = 1.4,
                  k: float = 1.6,
                  tau: float = 0.98) -> np.ndarray:
    """XDoG の連続応答（負側が「線」）を返す。gray は float32 想定。"""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    return g1 - tau * g2


def xdog_line_mask(image: np.ndarray,
                   sigma: float = 1.4,
                   k: float = 1.6,
                   tau: float = 0.98,
                   epsilon: float = -0.3,
                   dilate_px: int = 3) -> np.ndarray:
    """線画・輝度勾配マスク（255=線・保護対象）を返す。

    暗い線（アニメの輪郭線）と低コントラスト境界の両方を拾う。
    ノイズ斑点は面積フィルタで除去する。
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.float32)

    u = xdog_response(gray, sigma, k, tau)
    mask = (u < epsilon).astype(np.uint8)

    # グレイン由来の孤立斑点を除去（線は連結して長い）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= 12:
            out[labels == j] = 255

    if dilate_px > 0:
        out = cv2.dilate(out, np.ones((dilate_px, dilate_px), np.uint8))
    return out
