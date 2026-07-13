"""
第2層：大域動きのみのカット間 拡張統合（docs/denoise_method_survey.md 2章）

パン・ズーム等の大域動きしかない区間では、隣接する保持グループの参照像を
大域ワープで現グループ座標に変換し、「一致する画素だけ」統合に参加させる。
これにより：

  - 1コマ撮り（毎フレーム動く）パン素材でも実効フレーム数Nを稼げる
    （第1層はグループ長1で無力なため、こうした素材の唯一の時間統合手段）
  - 2〜4コマ打ちの短いグループでも品質が第1層単独より上がる

安全設計（エッジを甘くしないための要点）：
  - 統合参加は画素単位の受け入れマスク制。ワープ後のトレランス付き残差が
    ノイズ床以下の画素だけが混ざる。セルが動いた領域・視差のある
    マルチプレーンレイヤー・ワープ誤差の大きいエッジは自動的に除外され、
    その画素は第1層の結果のまま残る（劣化しない）
  - 受け入れ率が低すぎる隣接グループ（＝実は絵が変わっている）は丸ごと棄却
  - ワープの信頼度が低い場合・移動量が大きすぎる場合も棄却
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from motion_classification.core import MotionThresholds, estimate_global_motion

from .core import DenoiseParams, _line_art_edge_mask, _luma


@dataclass
class ExtendParams:
    radius: int = 2                    # 前後それぞれ何グループまで統合に使うか
    accept_sigma_factor: float = 3.0   # 受け入れ閾値 = max(floor, factor×σ_pair)
    accept_floor: float = 3.0
    min_accept_ratio: float = 0.05     # 受け入れ画素率がこれ未満の隣接グループは棄却
    max_translation_ratio: float = 0.25  # 画面幅に対する並進がこれ超のワープは棄却
    min_warp_confidence: float = 0.2
    active_area_crop: float = 0.10     # 受け入れ率の算定・統計はフィルム枠を除いた中央で


def _estimate_warp_full(ref_src: np.ndarray, ref_dst: np.ndarray,
                        motion_thresholds: MotionThresholds) -> tuple[np.ndarray, float] | None:
    """ref_src → ref_dst 座標系への元解像度2x3ワープと信頼度を返す（失敗時 None）。"""
    src_u8 = np.clip(ref_src, 0, 255).astype(np.uint8)
    dst_u8 = np.clip(ref_dst, 0, 255).astype(np.uint8)
    g = estimate_global_motion(src_u8, dst_u8, motion_thresholds)
    if g.warp_work is None:
        return None
    warp = g.warp_work.copy().astype(np.float32)
    # workスケール→元解像度：線形部はそのまま、並進のみスケール
    to_orig = ref_src.shape[1] / motion_thresholds.work_width
    if to_orig > 1.0:
        warp[0, 2] *= to_orig
        warp[1, 2] *= to_orig
    return warp, g.confidence


def _tolerant_luma_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """±1pxの位置ずれを許容した輝度残差（どちらも同座標系の画像）。"""
    ya = np.clip(_luma(a), 0, 255).astype(np.uint8)
    yb = np.clip(_luma(b), 0, 255).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    over = cv2.subtract(yb, cv2.dilate(ya, k3)).astype(np.float32)
    under = cv2.subtract(cv2.erode(ya, k3), yb).astype(np.float32)
    return cv2.GaussianBlur(np.maximum(over, under), (5, 5), 0)


def extend_reference(center: dict, neighbors: list[dict],
                     params: ExtendParams | None = None,
                     denoise_params: DenoiseParams | None = None) -> dict:
    """中心グループの参照像を、隣接グループの参照像で拡張統合する。

    center / neighbors は analyze_hold_group の返り値
    （"reference"(float32) と "n" と "grain_sigma" を使う）。

    返り値：{
      "reference": 拡張後の参照像 (float32),
      "effective_n": 画素ごとの実効フレーム数マップ,
      "used_neighbors": 実際に統合に使えた隣接グループ数,
      "accept_ratios": 各隣接グループの受け入れ画素率,
    }
    """
    params = params or ExtendParams()
    denoise_params = denoise_params or DenoiseParams()

    ref_c = center["reference"].astype(np.float32)
    n_c = float(center["n"])
    sigma_c = max(center["grain_sigma"], 0.5)
    h, w = ref_c.shape[:2]

    motion_thresholds = MotionThresholds()

    acc = ref_c * n_c
    weight = np.full((h, w), n_c, np.float32)

    # エッジ画素は拡張統合から除外する。ワープには±1px程度の残留誤差があり、
    # トレランス受け入れをすり抜けた微小ずれエッジが混ざると鮮鋭度が落ちる
    # （実測：-4%）。グレイン除去の利益はほぼ平坦部にあるので、エッジは
    # 第1層（グループ内統合）の結果をそのまま保つのが最も安全
    not_edge = _line_art_edge_mask(np.clip(ref_c, 0, 255), dilate_px=5) == 0

    # 受け入れ率の算定はフィルム枠を除いた中央領域で行う
    my, mx = int(h * params.active_area_crop), int(w * params.active_area_crop)
    inner = np.zeros((h, w), bool)
    inner[my:h - my, mx:w - mx] = True

    used = 0
    accept_ratios = []

    for nb in neighbors:
        ref_n = nb["reference"].astype(np.float32)
        n_n = float(nb["n"])

        est = _estimate_warp_full(ref_n, ref_c, motion_thresholds)
        if est is None:
            accept_ratios.append(0.0)
            continue
        warp, confidence = est
        translation = float(np.hypot(warp[0, 2], warp[1, 2]))
        if confidence < params.min_warp_confidence or \
                translation > params.max_translation_ratio * w:
            accept_ratios.append(0.0)
            continue

        warped = cv2.warpAffine(ref_n, warp, (w, h),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(-1000.0, -1000.0, -1000.0, 0.0))
        valid = (warped[..., 0] > -500.0)  # ワープで画面外から来た画素を除外

        resid = _tolerant_luma_residual(np.clip(warped, 0, 255), np.clip(ref_c, 0, 255))
        # 両参照像とも時間平均済みなのでノイズは σ/√N に減っている
        sigma_pair = sigma_c * float(np.sqrt(1.0 / max(n_c, 1) + 1.0 / max(n_n, 1)))
        threshold = max(params.accept_floor,
                        params.accept_sigma_factor * sigma_pair)

        accept = (resid <= threshold) & valid & not_edge
        # エッジ際のにじみ込みを防ぐため受け入れマスクを少し痩せさせる
        accept = cv2.erode(accept.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

        ratio = float(accept[inner].mean())
        accept_ratios.append(round(ratio, 4))
        if ratio < params.min_accept_ratio:
            continue

        a = accept.astype(np.float32) * n_n
        acc += warped * a[..., None]
        weight += a
        used += 1

    reference_ext = acc / weight[..., None]

    return {
        "reference": reference_ext.astype(np.float32),
        "effective_n": weight,
        "used_neighbors": used,
        "accept_ratios": accept_ratios,
    }


def blend_spatial_fallback(image: np.ndarray, effective_n: np.ndarray,
                           grain_sigma: float, strength: float = 1.0) -> np.ndarray:
    """実効Nの低い画素にだけ空間NR（第3層）を混ぜる連続ブレンド。

    設計提案書3章「時間方向重みと空間方向重みを連続的に変化させるブレンド方式」の実装：
      - 実効N=1（時間統合が効かなかった画素）→ 空間NRを重み1.0で適用
      - 実効Nが増えるほど空間NRの重みを 1/N で減衰（時間統合で既にσ/√Nまで
        減っているため空間処理は不要になっていく）
    重みマップは空間的にぼかして分類境界のフリッカーを防ぐ（同章の注意点）。
    空間NR自体は線画エッジ保護つき（spatial_denoise_edge_preserving）なので、
    どの重みでもエッジには触れない。
    """
    if strength <= 0:
        return image
    from .core import spatial_denoise_edge_preserving

    u8 = np.clip(image, 0, 255).astype(np.uint8)
    denoised = spatial_denoise_edge_preserving(u8, grain_sigma, strength=strength)

    w = 1.0 / np.maximum(effective_n.astype(np.float32), 1.0)
    w = cv2.GaussianBlur(w, (31, 31), 0)[..., None]
    out = denoised.astype(np.float32) * w + u8.astype(np.float32) * (1.0 - w)
    return np.clip(out, 0, 255).astype(np.uint8)
