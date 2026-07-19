"""空間解析による候補生成(ダスト/スクラッチサーベイ 第1章)。

  §1.1 DoG/LoG(点状欠陥の一次スクリーニング)   [通説]
  §1.2 Top-Hat / Bottom-Hat(サイズ・極性選択)  [事実/通説]
  §1.4 Hessian / vesselness(線状構造)          [事実]

役割: いずれも「高感度候補生成」段。単独では画像本来の小構造/線と区別
できない(サーベイ §1.6)。確証は時間軸(temporal.py)・追跡に委ねる。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dog_response(luma: np.ndarray, sigmas=(1.0, 2.0, 4.0), k: float = 1.6) -> np.ndarray:
    """複数スケール DoG のうち絶対値最大の応答(符号つき)を返す。§1.1

    正の応答 = 暗い塊(黒ダスト)、負 = 明るい塊(白ダスト)。
    LoG の高速近似。極性が白/黒分類の第一特徴になる。
    """
    luma = luma.astype(np.float32)
    best = np.zeros_like(luma)
    for s in sigmas:
        g1 = ndimage.gaussian_filter(luma, s)
        g2 = ndimage.gaussian_filter(luma, s * k)
        resp = g1 - g2
        replace = np.abs(resp) > np.abs(best)
        best = np.where(replace, resp, best)
    return best


def white_tophat(luma: np.ndarray, size: int = 7) -> np.ndarray:
    """White Top-Hat: size より小さい明構造(白ダスト)。§1.2"""
    return ndimage.white_tophat(luma.astype(np.float32), size=size)


def black_tophat(luma: np.ndarray, size: int = 7) -> np.ndarray:
    """Black Top-Hat(Bottom-Hat): size より小さい暗構造(黒ダスト)。§1.2"""
    return ndimage.black_tophat(luma.astype(np.float32), size=size)


def tophat_candidates(luma: np.ndarray, sigma_map: np.ndarray, k: float = 4.0,
                      size: int = 7):
    """Top-Hat による白/黒ダスト候補の二値マスク。適応しきい値 T=k*sigma。§1.2/§5.1

    戻り値 (white_mask, black_mask, response) — response は符号つき(+白/-黒)。
    """
    wth = white_tophat(luma, size)
    bth = black_tophat(luma, size)
    thr = k * np.maximum(sigma_map, 1e-4)
    white = wth > thr
    black = bth > thr
    response = np.where(wth >= bth, wth, -bth).astype(np.float32)
    return white, black, response


def hessian_vesselness(luma: np.ndarray, sigmas=(1.0, 2.0, 3.0),
                       beta: float = 0.5, c_factor: float = 0.5):
    """Frangi 型 vesselness(§1.4)。複数スケールの最大応答と線方向を返す。

    戻り値 (vesselness[0..], orientation_deg, polarity)。
      vesselness: 線らしさ(>0)
      orientation_deg: 線の方向(0=水平, 90=垂直)
      polarity: +1=明線(白/ベース傷), -1=暗線
    Frangi et al., MICCAI 1998 [事実]。2x2 固有値は閉形式。
    """
    luma = luma.astype(np.float32)
    H, W = luma.shape
    best_v = np.zeros((H, W), np.float32)
    best_ori = np.zeros((H, W), np.float32)
    best_pol = np.zeros((H, W), np.float32)

    for s in sigmas:
        g = ndimage.gaussian_filter(luma, s)
        # ガウシアン微分(スケール正規化 s^2)
        Ixx = ndimage.gaussian_filter(g, s, order=(0, 2)) * (s ** 2)
        Iyy = ndimage.gaussian_filter(g, s, order=(2, 0)) * (s ** 2)
        Ixy = ndimage.gaussian_filter(g, s, order=(1, 1)) * (s ** 2)
        # 2x2 対称行列の固有値(閉形式)
        tr = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.maximum((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy, 0.0)) * 0.5
        half_tr = 0.5 * tr
        l1 = half_tr - disc   # |l1| <= |l2| になるよう後で並べ替え
        l2 = half_tr + disc
        # |lambda1| <= |lambda2| を保証
        swap = np.abs(l1) > np.abs(l2)
        lam1 = np.where(swap, l2, l1)
        lam2 = np.where(swap, l1, l2)
        # vesselness
        Rb = np.abs(lam1) / (np.abs(lam2) + 1e-12)   # ブロブ抑制
        S = np.sqrt(lam1 ** 2 + lam2 ** 2)           # 構造強度
        c = c_factor * float(S.max() + 1e-12)
        v = np.exp(-(Rb ** 2) / (2 * beta ** 2)) * (1.0 - np.exp(-(S ** 2) / (2 * c ** 2)))
        v = np.where(np.abs(lam2) > 1e-8, v, 0.0).astype(np.float32)
        # 線方向: lambda2 に対応する固有ベクトルに直交する向きが線方向。
        # 主曲率方向の角度は 0.5*atan2(2Ixy, Ixx-Iyy)
        theta = 0.5 * np.arctan2(2 * Ixy, (Ixx - Iyy))    # ラジアン
        line_dir = np.degrees(theta) + 90.0               # 線の伸びる方向
        line_dir = np.mod(line_dir, 180.0)
        polarity = np.where(lam2 < 0, 1.0, -1.0).astype(np.float32)  # lam2<0=明線

        take = v > best_v
        best_v = np.where(take, v, best_v)
        best_ori = np.where(take, line_dir.astype(np.float32), best_ori)
        best_pol = np.where(take, polarity, best_pol)

    return best_v, best_ori, best_pol


def edge_mask(luma: np.ndarray, sigma: float, k: float = 3.0) -> np.ndarray:
    """勾配強度によるエッジマスク(平坦領域選別・ノイズ推定の間引き用)。"""
    gx = ndimage.sobel(luma.astype(np.float32), axis=1)
    gy = ndimage.sobel(luma.astype(np.float32), axis=0)
    grad = np.hypot(gx, gy)
    return grad > (k * max(sigma, 1e-4) * 4.0)
