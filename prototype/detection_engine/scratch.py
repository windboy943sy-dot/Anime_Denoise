"""スクラッチ検知(ダスト/スクラッチサーベイ 第4章 + §2.5)。

  §4.2 垂直射影ヒストグラム(Joyeux 系): 縦傷検知の実用定番       [事実]
  §2.5 時間持続性追跡: 複数フレーム同位置に出続ける = 傷確定     [事実/考察]

中核テーゼ(§0.2): スクラッチは時間差分では消える。空間・射影・形状が主軸で、
時間軸は「持続による確証」に使う。単一フレームの垂直構造(柱・線画)は
「持続かつシーン動きに非追従」で棄却する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .noise_profile import to_luma


@dataclass
class ScratchColumn:
    x: float
    width: float
    polarity: int          # +1=明傷, -1=暗傷
    strength: float
    band: int = 0          # 帯番号(縦帯別射影)


@dataclass
class ScratchTrack:
    """フレーム間追跡された傷。持続数が閾値未満なら棄却(誤検知抑制の最重要ルール)。"""
    track_id: int
    x: float
    width: float
    polarity: int
    persistence: int = 1
    velocity: float = 0.0
    last_frame: int = 0
    confidence: float = 0.0
    history: list = field(default_factory=list)


def vertical_projection_scratches(frame: np.ndarray, sigma: float,
                                  n_bands: int = 4, k: float = 4.0,
                                  min_width: float = 1.0, max_width: float = 6.0,
                                  active_crop: float = 0.05,
                                  min_strength: float = 0.02) -> list[ScratchColumn]:
    """帯別垂直射影から縦傷候補列を抽出。§4.2

    1. 各帯で垂直射影 p(x) = mean_y I(x,y)
    2. 局所背景(中域通過)との差分から幅の鋭いピーク/ディップを抽出
    3. 射影で SNR が sqrt(H) 倍に増幅(傷だけ残る)

    min_strength: ノイズ床すれすれの微小残差を棄却する絶対下限(0..1 輝度)。
    本物の傷は可視コントラストを持つ(§4.3 断面モデルの含意)。
    """
    luma = to_luma(frame)
    H, W = luma.shape
    cols: list[ScratchColumn] = []
    band_h = H // n_bands
    for b in range(n_bands):
        y0 = b * band_h
        y1 = H if b == n_bands - 1 else (b + 1) * band_h
        crop = int(active_crop * (y1 - y0))
        strip = luma[y0 + crop:y1 - crop, :]
        if strip.shape[0] < 4:
            continue
        p = strip.mean(axis=0)                      # 1D プロファイル(長さ W)
        # 局所背景 = 幅の広いメディアン。residual = p - bg
        bg = ndimage.median_filter(p, size=15, mode="reflect")
        resid = p - bg
        # 射影後のノイズは sigma/sqrt(H) に減る。閾値もそれに合わせる。
        proj_sigma = sigma / np.sqrt(max(strip.shape[0], 1))
        thr = max(k * max(proj_sigma, 1e-5), min_strength)
        for pol, signed in ((1, resid), (-1, -resid)):
            # pol に合わせた向きの射影(明傷は p、暗傷は -p)で ridge/edge を判定
            oriented_p = p if pol == 1 else -p
            peaks = _find_peaks(signed, oriented_p, thr, min_width, max_width,
                                active_crop, W)
            for x, width, strength in peaks:
                cols.append(ScratchColumn(x=x, width=width, polarity=pol,
                                          strength=strength, band=b))
    return cols


def _is_ridge_not_edge(oriented_p: np.ndarray, cx: int, width: float,
                       max_fwhm: float = 9.0) -> bool:
    """線(ridge)か段(edge)かの弁別。§4.3 断面モデル / §1.4 R_B の1D版。

    細い傷は元射影 p でも狭い山(半値全幅 FWHM が小)。内容の帯は広く持ち上がり、
    段差は片側が落ちない(プラトー)。山頂から左右へ半値まで落ちる距離を測り、
    片側が上限に達する(=プラトー)か合計幅が max_fwhm を超えれば内容として棄却。
    """
    n = oriented_p.size
    peak = float(oriented_p[cx])
    cap = int(max_fwhm) + 3
    # 左右の局所背景(cap 先の中央値)
    left_bg_seg = oriented_p[max(0, cx - cap - 3):max(1, cx - cap)]
    right_bg_seg = oriented_p[min(n - 1, cx + cap):min(n, cx + cap + 3)]
    left_bg = float(np.median(left_bg_seg)) if left_bg_seg.size else peak
    right_bg = float(np.median(right_bg_seg)) if right_bg_seg.size else peak
    baseline = min(left_bg, right_bg)
    half = baseline + 0.5 * (peak - baseline)
    if peak - baseline <= 1e-6:
        return False
    # 左へ半値まで
    li = 0
    i = cx
    while i > 0 and oriented_p[i] > half and li <= cap:
        i -= 1; li += 1
    ri = 0
    i = cx
    while i < n - 1 and oriented_p[i] > half and ri <= cap:
        i += 1; ri += 1
    if li > cap or ri > cap:          # 片側が落ちない = プラトー(段差/帯)
        return False
    return (li + ri) <= max_fwhm


def _find_peaks(signed: np.ndarray, oriented_p: np.ndarray, thr: float,
                min_w: float, max_w: float, active_crop: float, W: int):
    """1D プロファイルの正のピーク(x, width, strength)を返す。ridge/edge 弁別込み。"""
    above = signed > thr
    peaks = []
    xmin = int(active_crop * W)
    xmax = int((1 - active_crop) * W)
    idx = 0
    n = signed.size
    while idx < n:
        if above[idx]:
            j = idx
            while j < n and above[j]:
                j += 1
            width = j - idx
            if min_w <= width <= max_w:
                seg = signed[idx:j]
                cxi = idx + int(np.argmax(seg))
                cx = float(cxi)
                if xmin <= cx <= xmax and _is_ridge_not_edge(oriented_p, cxi, width):
                    peaks.append((cx, float(width), float(seg.max())))
            idx = j
        else:
            idx += 1
    return peaks


class ScratchTracker:
    """1次元 Kalman/αβ 相当の簡易追跡。§4.2/§2.5

    x 位置(位置+速度)で追跡し、数フレーム未満しか持続しない検知は棄却する。
    """

    def __init__(self, match_dist: float = 4.0, min_persistence: int = 3,
                 max_gap: int = 2):
        self.tracks: dict[int, ScratchTrack] = {}
        self.match_dist = match_dist
        self.min_persistence = min_persistence
        self.max_gap = max_gap
        self._next_id = 1

    def update(self, columns: list[ScratchColumn], frame_index: int):
        """当該フレームの検知列でトラックを更新。確定済みトラックを返す。"""
        # 帯をまたいで同じ x のものは統合(平均)
        merged = _merge_columns_by_x(columns, self.match_dist)
        used = set()
        for col in merged:
            best_id, best_d = None, self.match_dist + 1
            for tid, tr in self.tracks.items():
                if tid in used or tr.polarity != col.polarity:
                    continue
                pred = tr.x + tr.velocity
                d = abs(pred - col.x)
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is not None:
                tr = self.tracks[best_id]
                tr.velocity = 0.5 * tr.velocity + 0.5 * (col.x - tr.x)
                tr.x = col.x
                tr.width = 0.5 * tr.width + 0.5 * col.width
                tr.persistence += 1
                tr.last_frame = frame_index
                tr.confidence = min(1.0, tr.confidence + 0.15)
                tr.history.append((frame_index, col.x))
                used.add(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = ScratchTrack(
                    track_id=tid, x=col.x, width=col.width, polarity=col.polarity,
                    persistence=1, last_frame=frame_index, confidence=0.2,
                    history=[(frame_index, col.x)])
                used.add(tid)

        # 古いトラックを掃除
        stale = [tid for tid, tr in self.tracks.items()
                 if frame_index - tr.last_frame > self.max_gap]
        for tid in stale:
            del self.tracks[tid]

        return [tr for tr in self.tracks.values()
                if tr.persistence >= self.min_persistence]

    def confirmed_tracks(self):
        return [tr for tr in self.tracks.values() if tr.persistence >= self.min_persistence]


def _merge_columns_by_x(columns: list[ScratchColumn], dist: float):
    if not columns:
        return []
    cols = sorted(columns, key=lambda c: c.x)
    out = [cols[0]]
    for c in cols[1:]:
        last = out[-1]
        if abs(c.x - last.x) <= dist and c.polarity == last.polarity:
            last.x = 0.5 * (last.x + c.x)
            last.strength = max(last.strength, c.strength)
            last.width = 0.5 * (last.width + c.width)
        else:
            out.append(c)
    return out
