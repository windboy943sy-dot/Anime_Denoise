"""ノイズプロファイル推定(4軸)。

ノイズサーベイ v1.0 の以下を実装:
  §1.2 Immerkaer 法(高速分散推定)         [事実]
  §1.1 局所分散 + 輝度ビン別集計(強度依存)  [事実/通説]
  §3.1 時間差分 sigma(Var(d)/2)            [通説]
  §4.1 空間相関長(高域残差の自己相関)       [通説]
  §0.1 支配モデル分類                        [考察]

依存: numpy, scipy のみ(OFX/OpenCV 非依存コア)。C++/OpenCV への移植時は
scipy.ndimage.* を対応する cv2/自作カーネルに置換する。移植対応表は
docs/detection_engine_architecture.md の付録を参照。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import NoiseProfile

# Immerkaer の 3x3 オペレータ(定数・平面成分を打ち消す)。§1.2 [事実]
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_SQRT_PI_2 = float(np.sqrt(np.pi / 2.0))


def to_luma(img: np.ndarray) -> np.ndarray:
    """RGB(HxWxC, 0..1) を輝度に。単チャンネルはそのまま返す。Rec.709 係数。"""
    if img.ndim == 2:
        return img.astype(np.float32)
    c = img.shape[2]
    if c == 1:
        return img[..., 0].astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


def immerkaer_sigma(luma: np.ndarray, edge_mask: np.ndarray | None = None) -> float:
    """Immerkaer 高速分散推定。edge_mask(True=エッジ)を渡すと masked 版。§1.2"""
    resp = ndimage.convolve(luma.astype(np.float32), _IMMERKAER, mode="reflect")
    a = np.abs(resp)
    if edge_mask is not None:
        a = a[~edge_mask]
        if a.size == 0:
            a = np.abs(resp).ravel()
    # sigma = sqrt(pi/2) * mean(|N*I|) / 6   (6 = sqrt(sum(N^2)) = sqrt(36))
    return _SQRT_PI_2 * float(np.mean(a)) / 6.0


def _block_view(img: np.ndarray, bs: int) -> np.ndarray:
    """画像を bs x bs のブロックに切り出し (nby, nbx, bs, bs) を返す。端は切り捨て。"""
    h, w = img.shape
    nby, nbx = h // bs, w // bs
    img = img[: nby * bs, : nbx * bs]
    return img.reshape(nby, bs, nbx, bs).transpose(0, 2, 1, 3)


def structure_oriented_flat_blocks(luma: np.ndarray, bs: int = 16):
    """Amer-Dubois 系(§1.5)の簡易版: 各ブロックの方向別高域応答が小さい
    =均質ブロックを選別し、(ブロック平均輝度, ブロック標準偏差) を返す。"""
    blocks = _block_view(luma, bs)                      # (nby, nbx, bs, bs)
    nby, nbx = blocks.shape[:2]
    flat = blocks.reshape(nby * nbx, bs, bs)
    # 方向別高域(水平・垂直・2対角の1次差分の分散)
    gh = np.diff(flat, axis=2)
    gv = np.diff(flat, axis=1)
    dir_energy = np.maximum(gh.var(axis=(1, 2)), gv.var(axis=(1, 2)))
    means = flat.mean(axis=(1, 2))
    stds = flat.std(axis=(1, 2))
    # 均質判定: 方向エネルギーが下位40%のブロックのみ採用
    if dir_energy.size == 0:
        return means, stds, np.zeros(0, bool)
    thr = np.percentile(dir_energy, 40.0)
    homog = dir_energy <= thr
    return means, stds, homog


def estimate_intensity_binned_sigma(luma: np.ndarray, n_bins: int = 16, bs: int = 16):
    """輝度ビン別 sigma テーブル(§1.1 実用形 + §1.7 の簡易一般化)。

    均質ブロックのみを使い、ブロック平均輝度でビン分けし、各ビン内の
    ブロック標準偏差の下位パーセンタイル(平坦ブロックの分散≈sigma²)を取る。
    """
    means, stds, homog = structure_oriented_flat_blocks(luma, bs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sigma = np.zeros(n_bins, np.float32)
    m, s = means[homog], stds[homog]
    for i in range(n_bins):
        sel = (m >= bins[i]) & (m < bins[i + 1] if i < n_bins - 1 else m <= bins[i + 1])
        vals = s[sel]
        if vals.size >= 3:
            sigma[i] = float(np.percentile(vals, 25.0))  # 下位25%(平坦ブロック)
        else:
            sigma[i] = np.nan
    # 空きビンを単調性を壊さない範囲で内挿(近傍の有効値で補間)
    valid = ~np.isnan(sigma)
    if valid.any():
        sigma = np.interp(centers, centers[valid], sigma[valid]).astype(np.float32)
    else:
        sigma[:] = immerkaer_sigma(luma)
    return centers.astype(np.float32), sigma


def estimate_spatial_correlation_length(luma: np.ndarray) -> float:
    """高域残差の水平自己相関から相関長を推定。白色ノイズは~1画素。§4.1

    NR済み・圧縮素材では有色(相関長 >1.5)になり、単純分散法が過小推定する。
    """
    hp = ndimage.convolve(luma.astype(np.float32), _IMMERKAER, mode="reflect")
    # 平坦部だけ見るため大きい応答(構造)を除去
    thr = np.percentile(np.abs(hp), 80.0)
    hp = np.where(np.abs(hp) <= thr, hp, 0.0)
    hp = hp - hp.mean()
    # 水平方向ラグ0,1,2 の正規化自己相関
    denom = float((hp * hp).sum()) + 1e-12
    r1 = float((hp[:, :-1] * hp[:, 1:]).sum()) / denom
    r2 = float((hp[:, :-2] * hp[:, 2:]).sum()) / denom
    # 指数減衰 r(k)=exp(-k/L) を仮定して L を粗く推定
    r1 = max(r1, 1e-3)
    corr_len = 1.0 / max(-np.log(r1), 1e-3) if r1 < 0.9 else 3.0
    if r2 > 0.3:  # ラグ2でも高い相関 → 明確に有色
        corr_len = max(corr_len, 2.5)
    return float(np.clip(corr_len, 0.5, 8.0))


def estimate_temporal_sigma(frame_t: np.ndarray, frame_prev: np.ndarray,
                            still_thresh: float = 0.05):
    """時間差分 sigma(§3.1)。静止ブロックのみで Var(d)/2 を推定。

    戻り値 (temporal_sigma, has_fixed_pattern)。動き補償なし版なので静止領域
    前提。動きの大きい素材では sigma_map と併用して静止判定を厳しくする。
    """
    lt, lp = to_luma(frame_t), to_luma(frame_prev)
    d = lt - lp
    bs = 16
    db = _block_view(d, bs).reshape(-1, bs, bs)
    # 静止ブロック: 平均絶対差分が小さい
    still = np.abs(db).mean(axis=(1, 2)) < still_thresh
    if still.sum() < 4:
        still = np.abs(db).mean(axis=(1, 2)) <= np.percentile(
            np.abs(db).mean(axis=(1, 2)), 30.0)
    var_d = db[still].var(axis=(1, 2))
    if var_d.size == 0:
        return None, False
    temporal_sigma = float(np.sqrt(max(np.percentile(var_d, 25.0), 0.0) / 2.0))
    # FPN の疑い: 時間差分がほぼ0なのに空間高域が残る(=時間不変の縞)
    spatial_sigma = immerkaer_sigma(lt)
    has_fpn = temporal_sigma < 0.3 * spatial_sigma and spatial_sigma > 1e-3
    return temporal_sigma, bool(has_fpn)


def estimate_noise_profile(frame: np.ndarray, frame_prev: np.ndarray | None = None,
                           color_space: str = "unknown", n_bins: int = 16) -> NoiseProfile:
    """1フレーム(任意で前フレーム)からノイズプロファイル4軸を推定。"""
    prof = NoiseProfile(color_space=color_space)
    luma = to_luma(frame)

    # 軸1: 強度依存
    centers, sigma = estimate_intensity_binned_sigma(luma, n_bins=n_bins)
    prof.intensity_bins = centers
    prof.sigma_by_bin = sigma
    prof.global_sigma = float(np.median(sigma[sigma > 0])) if (sigma > 0).any() else immerkaer_sigma(luma)
    # ポアソン-ガウス Var = a*mu + b の線形当てはめ(参考値)
    var = sigma.astype(np.float64) ** 2
    if (var > 0).sum() >= 3:
        A = np.vstack([centers, np.ones_like(centers)]).T
        coef, *_ = np.linalg.lstsq(A, var, rcond=None)
        prof.poisson_a, prof.gauss_b = float(coef[0]), float(coef[1])

    # 軸2: 空間相関
    prof.spatial_correlation_length = estimate_spatial_correlation_length(luma)
    prof.is_white = prof.spatial_correlation_length < 1.5

    # 軸3: 時間特性
    if frame_prev is not None:
        ts, fpn = estimate_temporal_sigma(frame, frame_prev)
        prof.temporal_sigma = ts
        prof.has_fixed_pattern = fpn

    # 軸4: 色チャンネル依存
    if frame.ndim == 3 and frame.shape[2] >= 3:
        sig_ch = np.array([immerkaer_sigma(frame[..., c].astype(np.float32))
                           for c in range(min(3, frame.shape[2]))], np.float32)
        prof.sigma_per_channel = sig_ch
        luma_sigma = max(prof.global_sigma, 1e-6)
        # クロマ優勢: 色差方向のノイズが輝度より顕著
        prof.chroma_dominant = bool(sig_ch.max() > 1.6 * luma_sigma)

    prof.dominant_model = _classify_model(prof)
    return prof


def _classify_model(prof: NoiseProfile) -> str:
    """支配ノイズモデルの粗い分類(§0.1)。[考察]"""
    if prof.has_fixed_pattern:
        return "fpn"
    if not prof.is_white:
        return "correlated"       # NR済み・圧縮・グレイン系(有色)
    if prof.poisson_a is not None and prof.poisson_a > 2.0 * abs(prof.gauss_b or 0.0) + 1e-6:
        return "poisson_gauss"    # 強度依存が顕著
    if prof.spatial_correlation_length > 1.2:
        return "grain"
    return "awgn"
