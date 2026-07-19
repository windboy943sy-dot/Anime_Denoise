"""時間解析によるダスト検知(ダスト/スクラッチサーベイ 第2章)。

  §2.1 SDI(Kokaram): 前後フレームと同符号かつ大 = 時間的インパルス  [事実]
  §2.2 ROD(Nadenau-Mitra): 順位統計による頑健なスパイク検出         [事実]
  §2.3 時間メディアン乖離                                            [事実/通説]

中核テーゼ(§0.2): ダスト = 時間的外れ値。これが最も特異度の高い検知軸。
持続する欠陥(スクラッチ・カビ)は原理的に検知されない(それは §2.5 で扱う)。

動き補償(MC)について [考察/要実測]:
本リファレンスは MC を motion.compensate に委譲する。既定は恒等(静止前提)。
実素材では粗いブロックマッチングを与える。MC 誤差が最大の誤検知源であるため
(§2.1 短所)、シーンチェンジ無効化と近傍多数決を必ず併用する。
"""
from __future__ import annotations

import numpy as np

from .noise_profile import to_luma


def sdi_detector(luma_t: np.ndarray, luma_prev: np.ndarray, luma_next: np.ndarray,
                 sigma_map: np.ndarray, k: float = 3.0):
    """Kokaram SDI(Spike Detection Index)。§2.1

    両方向差分が同符号かつ min(|e_b|,|e_f|) > k*sigma の画素を blotch とする。
    片側だけ大 = オクルージョン/動き誤差の可能性が高く棄却。

    戻り値 (mask[bool], polarity[+1明/-1暗], strength[float]).
    """
    eb = luma_t - luma_prev   # 後方差分
    ef = luma_t - luma_next   # 前方差分
    thr = k * np.maximum(sigma_map, 1e-4)
    same_sign = (np.sign(eb) == np.sign(ef)) & (eb != 0)
    magnitude = np.minimum(np.abs(eb), np.abs(ef))
    mask = same_sign & (magnitude > thr)
    polarity = np.sign(eb).astype(np.int8)
    strength = magnitude.astype(np.float32)
    return mask, polarity, strength


def rod_detector(luma_t: np.ndarray, luma_prev: np.ndarray, luma_next: np.ndarray,
                 sigma_map: np.ndarray, k: float = 3.0):
    """ROD(Rank-Order Difference)。§2.2

    補償済み前後フレームの参照画素集合(ここでは各方向の 3x3 近傍から代表値)
    をソートし、現画素が順位統計(min/max)からどれだけ逸脱するかで判定。
    ガウス仮定不要でグレインに強い。SDI と AND を取ると特異度が上がる(§2.2)。
    """
    from scipy import ndimage

    # 前後フレームの局所 min/max(参照画素群の順位統計を近傍で近似)
    ref_lo = np.minimum(ndimage.minimum_filter(luma_prev, size=3),
                        ndimage.minimum_filter(luma_next, size=3))
    ref_hi = np.maximum(ndimage.maximum_filter(luma_prev, size=3),
                        ndimage.maximum_filter(luma_next, size=3))
    thr = k * np.maximum(sigma_map, 1e-4)
    over = luma_t - ref_hi     # 参照最大を上回る量(明インパルス)
    under = ref_lo - luma_t    # 参照最小を下回る量(暗インパルス)
    mask_hi = over > thr
    mask_lo = under > thr
    mask = mask_hi | mask_lo
    polarity = np.where(mask_hi, 1, np.where(mask_lo, -1, 0)).astype(np.int8)
    strength = np.maximum(over, under).astype(np.float32)
    return mask, polarity, strength


def temporal_median_deviation(luma_t: np.ndarray, luma_prev: np.ndarray,
                              luma_next: np.ndarray, sigma_map: np.ndarray,
                              k: float = 3.0):
    """時間3点メディアンとの乖離で検知。§2.3

    med(prev,t,next) は1フレーム欠陥を必ず中央値から外す。無差別置換(除去)は
    しないが、|t - med| を検知量として使う。
    """
    stack = np.stack([luma_prev, luma_t, luma_next], axis=0)
    med = np.median(stack, axis=0)
    dev = luma_t - med
    thr = k * np.maximum(sigma_map, 1e-4)
    mask = np.abs(dev) > thr
    polarity = np.sign(dev).astype(np.int8)
    return mask, polarity, np.abs(dev).astype(np.float32)


def neighbor_majority(mask: np.ndarray, min_count: int = 5) -> np.ndarray:
    """3x3 近傍の多数決でノイズ起因の点誤検知を抑制(§2.1 ベストプラクティス)。"""
    from scipy import ndimage
    count = ndimage.uniform_filter(mask.astype(np.float32), size=3) * 9.0
    return (mask & (count >= min_count))


def scene_change(luma_t: np.ndarray, luma_prev: np.ndarray, thresh: float = 0.15) -> bool:
    """大域差分エネルギーの急増でシーンチェンジ判定。§2.1 (フラッシュ/カット)。

    シーンチェンジ跨ぎでは全画素が「両側不一致」になり SDI が破綻するため、
    当該フレームのダスト検知を無効化する(必須)。
    """
    return float(np.mean(np.abs(luma_t - luma_prev))) > thresh


def detect_dust_temporal(frame_t, frame_prev, frame_next, sigma_map,
                         k: float = 3.0, use_rod: bool = True,
                         majority: bool = True):
    """時間系ダスト検知の統合。SDI ∧ (ROD or median) の投票で特異度を上げる。

    戻り値 (mask, polarity, strength)。前後フレームが無い端では空マスク。
    """
    lt = to_luma(frame_t)
    lp = to_luma(frame_prev)
    ln = to_luma(frame_next)

    if scene_change(lt, lp) or scene_change(lt, ln):
        z = np.zeros(lt.shape, bool)
        return z, np.zeros(lt.shape, np.int8), np.zeros(lt.shape, np.float32)

    sdi_m, sdi_p, sdi_s = sdi_detector(lt, lp, ln, sigma_map, k)
    if use_rod:
        rod_m, rod_p, rod_s = rod_detector(lt, lp, ln, sigma_map, k)
    else:
        rod_m, rod_p, rod_s = temporal_median_deviation(lt, lp, ln, sigma_map, k)

    # 投票: 両検知器が反応した画素を採用(高特異度)。極性は SDI 優先。
    mask = sdi_m & rod_m
    if majority:
        mask = neighbor_majority(mask, min_count=4)
    polarity = np.where(sdi_p != 0, sdi_p, rod_p).astype(np.int8)
    strength = np.maximum(sdi_s, rod_s).astype(np.float32)
    return mask, polarity, strength
