"""
スキャンノイズ検出・補正プロトタイプ（設計提案書 4章 対応）

スキャナ・テレシネ由来の周期ノイズ（走査方向の規則的な縞）を
FFT周波数解析で検出する：

  1. 参照像Rの輝度を走査方向に平均してプロファイル化
     （平均によりグレインは相殺され、行/列全体に一様な周期成分だけが残る）
  2. プロファイルのFFTスペクトルからスパイク（局所メディアンの何倍も
     強い周波数成分）を検出
  3. 補正はスパイク成分だけを逆FFTで再構成してプロファイルから差し引く
     （ノッチフィルタ相当。絵柄の大域的な明暗はスパイクにならないので保たれる）

ラインノイズ（単発の行/列ずれ、linenoise.py）とは検出手法が異なるが対象が近い。
設計提案書3.2節の注意どおり、両方を有効にした際のクロストークは実データで要検証。
"""

from __future__ import annotations

import numpy as np

from .core import _luma


def _profile_spectrum(reference: np.ndarray, axis: int,
                      active_area_crop: float) -> tuple[np.ndarray, np.ndarray]:
    """走査方向平均プロファイルとその実FFTスペクトルを返す。

    axis=0：水平縞（行方向の周期）を対象に、各行を横方向に平均する。
    """
    y = _luma(reference)
    h, w = y.shape
    my, mx = int(h * active_area_crop), int(w * active_area_crop)
    inner = y[my:h - my, mx:w - mx]
    if axis == 1:
        inner = inner.T
    profile = inner.mean(axis=1)
    profile = profile - profile.mean()
    return profile, np.fft.rfft(profile)


def detect_scan_noise(reference: np.ndarray,
                      axis: int = 0,
                      spike_factor: float = 8.0,
                      min_period_px: float = 2.0,
                      max_period_px: float = 64.0,
                      min_amplitude: float = 0.3,
                      active_area_crop: float = 0.10) -> list[dict]:
    """周期スキャンノイズのスペクトルスパイクを検出する。

    返り値：[{"period_px": 周期, "amplitude": 振幅, "snr": スパイク強度比}]
    """
    profile, spec = _profile_spectrum(reference, axis, active_area_crop)
    n = len(profile)
    mag = np.abs(spec)

    # 局所メディアン（周辺15ビン）に対するスパイク検出。
    # 絵柄由来の低周波は近傍もまとめて大きいためスパイクにならない
    results = []
    for k in range(2, len(mag) - 1):
        period = n / k
        if not (min_period_px <= period <= max_period_px):
            continue
        lo, hi = max(1, k - 8), min(len(mag), k + 9)
        neighborhood = np.delete(mag[lo:hi], k - lo)
        local_med = float(np.median(neighborhood)) + 1e-6
        snr = float(mag[k]) / local_med
        amplitude = 2.0 * float(mag[k]) / n  # 正弦成分の振幅（輝度単位）
        if snr >= spike_factor and amplitude >= min_amplitude:
            results.append({
                "period_px": round(period, 2),
                "amplitude": round(amplitude, 3),
                "snr": round(snr, 1),
                "bin": k,
            })
    # 近接ビン（同一ノイズの裾）をまとめて最強のものだけ残す
    results.sort(key=lambda r: -r["snr"])
    kept = []
    for r in results:
        if all(abs(r["bin"] - o["bin"]) > 2 for o in kept):
            kept.append(r)
    return kept


def correct_scan_noise(frame: np.ndarray, reference: np.ndarray,
                       detections: list[dict], axis: int = 0,
                       strength: float = 1.0,
                       active_area_crop: float = 0.10) -> np.ndarray:
    """検出済みの周期成分をプロファイルから再構成して差し引く（ノッチ補正）。"""
    if not detections or strength <= 0:
        return frame
    profile, spec = _profile_spectrum(reference, axis, active_area_crop)
    n = len(profile)

    notch = np.zeros_like(spec)
    for d in detections:
        k = d["bin"]
        notch[max(0, k - 1):k + 2] = spec[max(0, k - 1):k + 2]
    periodic = np.fft.irfft(notch, n=n)  # 周期ノイズ成分の1次元波形

    out = frame.astype(np.float32)
    h, w = out.shape[:2]
    m = int((h if axis == 0 else w) * active_area_crop)
    corr = periodic * strength
    if axis == 0:
        for i, y in enumerate(range(m, h - m)):
            out[y, :, :] -= corr[i]
    else:
        for i, x in enumerate(range(m, w - m)):
            out[:, x, :] -= corr[i]
    return np.clip(out, 0, 255).astype(np.uint8)
