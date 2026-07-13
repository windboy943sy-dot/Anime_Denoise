"""
Phase 3：参照像R生成・欠陥検出・デノイズ出力プロトタイプ（設計提案書 3章・4章 対応）

エッジを甘くしないための設計原則（docs/denoise_method_survey.md に詳細）：

  1. 保持グループ内の時間統合が主役。空間フィルタを使わないため、
     原理的にエッジ劣化ゼロでグレインを除去できる。
  2. ただしフィルムスキャンにはゲートウィーブ（フレームごとの±0.5px程度の揺れ）が
     あるため、統合前にECCでサブピクセル位置合わせする。位置合わせなしで
     median/平均を取ると、それ自体がエッジを甘くする（実測済みの前提）。
  3. 空間デノイズは「動いている領域」への最後の手段。使う場合も
     線画エッジのマスクで保護し、ベタ塗り領域だけに強く効かせる
     （アニメは「線画＋平坦な色面」なので、平坦部の空間平滑はエッジと無関係にできる）。
  4. 欠陥（ダスト等）は振幅・形状・出現パターンでグレインと分離し、
     怪しい場合は欠陥扱いしない（保守的判定）。

出力モード（設計提案書3.1節）：
  - full_temporal_integration：参照像Rをそのまま出力（除去品質最大・質感は静止）
  - texture_preserving：出力は各フレーム自身。欠陥画素のみRで補正し、
    グレインの揺らぎは残す（grain_reduction>0 なら単独フレーム内の空間NRを併用）
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DenoiseParams:
    # 位置合わせ
    align: bool = True                 # グループ内サブピクセル位置合わせ（ウィーブ補正）
    align_work_width: int = 640        # ECCを回す解像度（ワープは元解像度に換算して適用）
    ecc_iterations: int = 30

    # 参照像R
    reference_method: str = "trimmed_mean"  # "median" / "trimmed_mean" / "mean"
    trim_ratio: float = 0.25           # trimmed_mean で上下から落とす割合（各側）

    # 欠陥（ダスト・ゴミ）検出
    dust_detection: bool = True
    dust_sigma: float = 5.0            # ノイズ標準偏差の何倍を「欠陥候補」とするか
    dust_min_area: int = 4             # 元解像度での最小面積(px)。これ未満はグレイン扱い
    dust_max_area_ratio: float = 0.0008  # 画面に対する最大面積比。これ超は「絵の変化」扱い。
                                         # 2560x1920で約4000px（直径70px相当）まで。実際の
                                         # ダストは数px〜数十px径なので十分な上限
    dust_protect_edges: bool = True    # 線画エッジ近傍では検出閾値を上げる（誤検出防止）
    dust_active_area_crop: float = 0.10  # フィルム枠・パーフォレーション近傍を検出対象外に
                                         # する外周割合。枠線はフレームごとに揺れるため
                                         # 枠沿いは恒常的に偽残差が出る（実測済み）

    # 出力モード
    mode: str = "texture_preserving"   # "texture_preserving" / "full_temporal_integration"
    grain_reduction: float = 0.0       # 0-1。texture_preservingでの単独フレーム空間NR強度
    feather_boundary_frames: bool = True  # グループ境界フレームは統合の重みを下げる

    # 位置合わせ品質チェック：残差中央値がグループ標準のこの倍数を超えるフレームは
    # 「位置合わせ不良」とみなし、ダスト検出閾値を2倍・欠陥補正を抑制する
    # （グループ末尾フレームで低コントラストエッジ沿いの弧状誤検出が出る実測対策）
    misalign_factor: float = 2.5

    # フリッカー補正（設計提案書4章：保持区間内の輝度時間変動の正規化）。
    # 「味」として残す選択肢があるためデフォルトOFF（3.2節のオン/オフ機構）
    flicker_correction: bool = False


# ---------------------------------------------------------------------------
# サブピクセル位置合わせ（ゲートウィーブ補正）
# ---------------------------------------------------------------------------

def _ecc_align_pair(gray_ref_small: np.ndarray, gray_mov_small: np.ndarray,
                    params: DenoiseParams) -> np.ndarray | None:
    """縮小画像同士でECCを回し、workスケールの2x3行列を返す（失敗時 None）。

    MOTION_EUCLIDEAN を使う（ウィーブ＝並進＋微小回転の補正が目的）。
    ※MOTION_AFFINE は自由度が多すぎてグレインに適合し、真の止めでも
    Rを劣化させることを実測で確認済み（02_Zoom_02 調査時）。
    低速ズーム等の「グループ内で絵が動く」ケースは位置合わせでは救えないため、
    analyze_hold_group の統合安全ガード（端点エッジ残差）で統合自体を止める。
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                params.ecc_iterations, 1e-5)
    try:
        # warpAffine(mov, warp) ≒ ref になる向き
        _, warp = cv2.findTransformECC(gray_ref_small, gray_mov_small, warp,
                                       cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        return warp
    except cv2.error:
        return None


def align_group_frames(frames: list[np.ndarray],
                       params: DenoiseParams | None = None,
                       ref_index: int | None = None) -> list[np.ndarray]:
    """保持グループ内の全フレームを基準フレームへサブピクセル位置合わせする。

    ウィーブは並進＋微小回転なので MOTION_EUCLIDEAN で十分（スケールは固定）。
    ECC は縮小画像で回し、得た並進成分を元解像度に換算して適用する。
    """
    params = params or DenoiseParams()
    if len(frames) < 2 or not params.align:
        return list(frames)

    if ref_index is None:
        ref_index = len(frames) // 2

    h, w = frames[0].shape[:2]
    scale = params.align_work_width / w
    small_size = (params.align_work_width, int(round(h * scale)))

    def small_gray(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, small_size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g, (5, 5), 0)

    ref_small = small_gray(frames[ref_index])
    aligned = []
    for i, f in enumerate(frames):
        if i == ref_index:
            aligned.append(f)
            continue
        warp = _ecc_align_pair(ref_small, small_gray(f), params)
        if warp is None:
            aligned.append(f)  # 収束失敗時は位置合わせなしで採用（安全側）
            continue
        # 並進成分を元解像度へ換算（回転中心のずれは並進に吸収される）
        warp_full = warp.copy()
        warp_full[0, 2] /= scale
        warp_full[1, 2] /= scale
        aligned.append(cv2.warpAffine(
            f, warp_full, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE))
    return aligned


# ---------------------------------------------------------------------------
# 参照像R
# ---------------------------------------------------------------------------

def compute_reference(frames_aligned: list[np.ndarray],
                      params: DenoiseParams | None = None) -> np.ndarray:
    """位置合わせ済みフレーム群からロバストな参照像Rを生成する。

    - median：ダスト等の単発外れ値に最も強い（フレーム数が少ないと量子化が残る）
    - trimmed_mean：外れ値を落としつつ残りを平均するので、グレイン抑制と
      外れ値耐性のバランスが良い（既定）
    - mean：抑制力最大だがダストがそのまま平均に混入するので単体では非推奨
    """
    params = params or DenoiseParams()
    stack = np.stack(frames_aligned).astype(np.float32)  # (N, H, W, C)
    n = stack.shape[0]

    if params.reference_method == "median" or n <= 2:
        return np.median(stack, axis=0)

    if params.reference_method == "trimmed_mean":
        k = int(n * params.trim_ratio)
        if k == 0 or n - 2 * k < 1:
            return np.median(stack, axis=0)
        sorted_stack = np.sort(stack, axis=0)
        return sorted_stack[k:n - k].mean(axis=0)

    return stack.mean(axis=0)  # "mean"


def _luma(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    return cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)


def _luma_residual_stack(frames_aligned: list[np.ndarray],
                         reference: np.ndarray) -> np.ndarray:
    """各フレームの輝度残差 |Y(frame) − Y(R)| を (N, H, W) で返す。"""
    ref_y = _luma(reference)
    return np.stack([np.abs(_luma(f) - ref_y) for f in frames_aligned])


def estimate_grain_sigma(frames_aligned: list[np.ndarray],
                         reference: np.ndarray,
                         residual_stack: np.ndarray | None = None) -> float:
    """グループ内の輝度残差からグレインの標準偏差をロバスト推定する（MAD×1.4826）。

    ダスト等の外れ値の影響を受けないよう中央値ベースで推定する。
    ここで得たσは欠陥判定閾値・空間NR強度の自動設定の両方に使う
    （カットごとのグレイン量への自動適応）。
    """
    if residual_stack is None:
        residual_stack = _luma_residual_stack(frames_aligned, reference)
    # 残差は絶対値なので中央値≒MADに相当する（半正規分布のスケール推定）
    mad = np.median(residual_stack)
    return float(max(mad * 1.4826, 0.5))


# ---------------------------------------------------------------------------
# 欠陥（ダスト・ゴミ）検出
# ---------------------------------------------------------------------------

def _line_art_edge_mask(reference: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    """線画・セル境界のエッジマスク（255=エッジ近傍）。

    参照像R（グレイン抑制済み）から取るのが重要：生フレームから取ると
    グレインまでエッジ扱いになり保護マスクが画面全体に広がる。

    実装は XDoG（lineart.py）：Canny(50,150) は低コントラストの縞・
    グラデーション境界を落とし（実測：塔の縞領域で被覆0%）、それが
    弧状誤検出の遠因になった。XDoG は完全な線画＋低コントラスト境界を
    拾える（同領域4.3%、目視でも線画を完全捕捉）。
    """
    from .lineart import xdog_line_mask
    return xdog_line_mask(np.clip(reference, 0, 255), dilate_px=dilate_px)


def detect_dust_masks_group(residual_stack: np.ndarray,
                            reference: np.ndarray,
                            grain_sigma: float,
                            params: DenoiseParams | None = None,
                            frame_threshold_scales: np.ndarray | None = None
                            ) -> list[np.ndarray]:
    """保持グループ全フレーム分のダスト・ゴミ（単発欠陥）マスクを返す（255=欠陥）。

    設計提案書4章の「時間方向中央値との差分＋形状フィルタ」を、保持グループ構造を
    活かした3条件に具体化している：

      1. 振幅：残差 > dust_sigma × グレインσ
      2. 時間的単発性：**同グループの他フレームでは同位置の残差が小さい**こと。
         グレインは（振幅は小さいが）全フレームでランダムに揺らぎ続けるのに対し、
         ダストはそのフレームにしか存在しない。フィルムグレインは裾が重く
         空間相関もあるため振幅閾値だけでは大量に貫通する（実測済み）。
         この条件が誤検出を桁で減らす。傷（複数フレーム持続）はここで意図的に
         除外される＝別ロジックで扱う（4章の表どおり）
      3. 形状：孤立した小面積の塊（大面積は「絵の変化」、微小はグレイン扱い）

    保守的判定（5章）：線画エッジ近傍は振幅閾値を2倍にする。
    """
    params = params or DenoiseParams()
    n = residual_stack.shape[0]
    if n < 2:
        return [np.zeros(residual_stack.shape[1:], np.uint8)] * max(n, 1)

    threshold = max(8.0, params.dust_sigma * grain_sigma)

    edge_mask = None
    grad_map = None
    if params.dust_protect_edges:
        edge_mask = _line_art_edge_mask(reference) > 0
        # Cannyが拾えない低コントラストの輝度勾配（グラデーション帯など）用の
        # 連続勾配マップ。フィルムの局所歪み（大域ワープで補正できないうねり）は
        # こうした勾配に沿った弧状の偽残差を作るため、勾配の高い場所の候補は
        # 棄却する（ダストは平坦部に出るものだけ拾う保守的判定）
        ref_y_blur = cv2.GaussianBlur(_luma(reference), (5, 5), 0)
        gx = cv2.Sobel(ref_y_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ref_y_blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_map = np.sqrt(gx * gx + gy * gy)

    # フィルム枠・パーフォレーション領域は検出対象外（枠線の揺れが偽残差になる）
    hh, ww = residual_stack.shape[1:]
    border = np.ones((hh, ww), bool)
    my, mx = int(hh * params.dust_active_area_crop), int(ww * params.dust_active_area_crop)
    if my > 0 and mx > 0:
        border[:my, :] = False
        border[-my:, :] = False
        border[:, :mx] = False
        border[:, -mx:] = False

    masks = []
    h, w = residual_stack.shape[1:]
    max_area = int(h * w * params.dust_max_area_ratio)
    all_idx = np.arange(n)

    for i in range(n):
        resid_i = residual_stack[i]
        thr_i = threshold
        if frame_threshold_scales is not None:
            thr_i = threshold * float(frame_threshold_scales[i])
        # 他フレームの残差の中央値：グレインなら同程度、ダストならほぼゼロになる
        others = residual_stack[all_idx != i]
        others_med = np.median(others, axis=0)

        candidates = (resid_i > thr_i) & (others_med < threshold * 0.4) & border

        if edge_mask is not None:
            strong = (resid_i > thr_i * 2.0) & (others_med < threshold * 0.4) & border
            candidates = np.where(edge_mask, strong, candidates)

        candidates = candidates.astype(np.uint8)
        # 微小成分（グレイン）の除去と、塊の穴埋め
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
        mask = np.zeros((h, w), np.uint8)
        for j in range(1, n_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if not (params.dust_min_area <= area <= max_area):
                continue
            # ダストはコンパクトな塊。細長い成分（サブピクセル位置ずれによる
            # エッジ沿いの残差線や傷）は除外する。傷は持続性ベースの別ロジックで
            # 扱う（設計提案書4章）
            bw = stats[j, cv2.CC_STAT_WIDTH]
            bh = stats[j, cv2.CC_STAT_HEIGHT]
            elongation = max(bw, bh) / max(1, min(bw, bh))
            fill_ratio = area / max(1, bw * bh)
            if elongation > 4.0 and fill_ratio < 0.4:
                continue
            comp = labels == j
            if grad_map is not None and float(np.median(grad_map[comp])) > 10.0:
                continue  # 輝度勾配上の候補＝局所歪みの偽残差の可能性が高い
            mask[comp] = 255

        # 補正時のにじみ防止に1pxだけ広げる（ダストの縁の薄い部分を拾う）
        masks.append(cv2.dilate(mask, np.ones((3, 3), np.uint8)))

    return masks


# ---------------------------------------------------------------------------
# 空間デノイズ（動領域用・エッジ保護つき）
# ---------------------------------------------------------------------------

def spatial_denoise_edge_preserving(frame: np.ndarray, grain_sigma: float,
                                    strength: float = 1.0,
                                    protect_edges: bool = True) -> np.ndarray:
    """単独フレーム内で完結する、線画保護つき空間デノイズ。

    アニメは「線画＋平坦な色面」で構成されるため：
      - ベースはNLM（fastNlMeansDenoisingColored）。hはグレインσから自動設定
      - 線画エッジ近傍は元画素へフォールバックし、輪郭のシャープさを完全保持
    時間方向の情報は使わないため、texture_preserving モードの
    grain_reduction としても安全に使える（フレーム間の揺らぎは保たれる）。
    """
    if strength <= 0:
        return frame
    h_luma = float(np.clip(grain_sigma * 1.2 * strength, 1.0, 15.0))
    denoised = cv2.fastNlMeansDenoisingColored(
        frame, None, h_luma, h_luma, 7, 21)

    if protect_edges:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        # エッジ近傍をぼかした重みでブレンド（境界の不連続を避ける）
        w_edge = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
        w_edge = w_edge[..., None]
        out = denoised.astype(np.float32) * (1 - w_edge) + frame.astype(np.float32) * w_edge
        return np.clip(out, 0, 255).astype(np.uint8)
    return denoised


# ---------------------------------------------------------------------------
# 保持グループ単位の処理（パイプライン本体）
# ---------------------------------------------------------------------------

def analyze_hold_group(frames: list[np.ndarray],
                       params: DenoiseParams | None = None) -> dict:
    """保持グループ1つ分の解析（位置合わせ・R生成・σ推定・欠陥検出）を行う。

    出力生成は render_hold_group が担当。第2層（カット間拡張統合）では
    解析後に参照像を差し替えてから出力を作るため、解析と出力を分離している。
    """
    params = params or DenoiseParams()

    aligned = align_group_frames(frames, params)
    reference = compute_reference(aligned, params)

    # フリッカー（保持区間内の輝度時間変動）の推定。補正ONなら各フレームの
    # 輝度オフセットをRに合わせて正規化してからRを作り直す
    flicker_offsets = []
    ref_y = _luma(reference)
    for f in aligned:
        flicker_offsets.append(float(np.median(_luma(f) - ref_y)))
    if params.flicker_correction and any(abs(o) > 0.25 for o in flicker_offsets):
        aligned = [np.clip(f.astype(np.float32) - o, 0, 255).astype(np.uint8)
                   for f, o in zip(aligned, flicker_offsets)]
        reference = compute_reference(aligned, params)

    residual_stack = _luma_residual_stack(aligned, reference)
    grain_sigma = estimate_grain_sigma(aligned, reference, residual_stack)
    n = len(frames)

    # 統合安全ガード（グループ全体）：端点フレーム同士の「エッジ画素上の」残差が
    # 大きいグループは、検出閾値以下の連続的な動き（超低速ズーム等）が累積して
    # おり、統合するとエッジが二重化する（実測＝02_Zoom_02）。
    # Phase 1 のドリフト検査は全画面平均のため平坦部に薄められてこれを見逃す。
    # 閾値は実測8ケースで較正：統合可は ≤4.1σ、不可は ≥6.2σ に分離
    integration_unsafe = False
    if n >= 2:
        guard_edges = cv2.Canny(
            np.clip(_luma(aligned[n // 2]), 0, 255).astype(np.uint8), 50, 150) > 0
        if guard_edges.any():
            # 端±1を除いた内側端点で判定（フィルムの局所歪みは端フレームに
            # 集中するため。モーションガードと同じ対処）
            ia, ib = (1, n - 2) if n >= 4 else (0, n - 1)
            end_a = cv2.GaussianBlur(_luma(aligned[ia]), (3, 3), 0)
            end_b = cv2.GaussianBlur(_luma(aligned[ib]), (3, 3), 0)
            endpoint_edge_resid = float(
                np.median(np.abs(end_a - end_b)[guard_edges]))
            integration_unsafe = endpoint_edge_resid > max(6.0, 5.0 * grain_sigma)

    # 画素単位モーションガード：グループ内で「一定割合以上のフレーム」が
    # グレインを超えて変動する画素（微細な口パク・まばたき等）を検出する。
    # こうした動きは Phase 1 の閾値以下でグループが統合されることがあり、
    # そのまま完全時間統合すると芝居が平均化されて消える（実測：口パク消失）。
    # ダストは単発（1フレームのみ）なので割合条件に該当せず、共存できる。
    guard_threshold = max(6.0, 4.0 * grain_sigma)
    # 端のフレームは判定から除外する：フィルムの局所歪み（大域ワープで補正
    # できないうねり）は基準フレームから時間的に遠い端フレームに集中するため。
    # 本物の芝居（口パク等）は内部フレームにも分布するので検出できる
    if n >= 5:
        interior = residual_stack[1:-1]
        deviating_ratio = (interior > guard_threshold).mean(axis=0)
        motion_guard = (deviating_ratio >= 0.25).astype(np.uint8)
        # フィルム枠・パーフォレーションは検出対象外（枠はフレームごとに揺れるため
        # 「持続的に変動する画素」に該当してしまう。実測で誤検出の主成分だった）
        gh, gw = motion_guard.shape
        gmy = int(gh * params.dust_active_area_crop)
        gmx = int(gw * params.dust_active_area_crop)
        border_mask = np.zeros_like(motion_guard)
        border_mask[gmy:gh - gmy, gmx:gw - gmx] = 1
        motion_guard &= border_mask
    else:
        # 短いグループは統合枚数が少なく芝居消失の被害も小さいため対象外
        motion_guard = np.zeros(residual_stack.shape[1:], np.uint8)
    if motion_guard.any():
        k3 = np.ones((3, 3), np.uint8)
        motion_guard = cv2.morphologyEx(motion_guard, cv2.MORPH_OPEN, k3)
        # 本物の芝居（口パク等）は大きな連結成分になる。グレインの重い裾が
        # 偶然そろった画素は小斑点なので、面積フィルタで先に落とす
        # （先に CLOSE で膨らませると斑点同士が併合して除去できなくなる）
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(motion_guard, connectivity=8)
        filtered = np.zeros_like(motion_guard)
        for j in range(1, n_lbl):
            if stats[j, cv2.CC_STAT_AREA] >= 300:
                filtered[labels == j] = 1
        motion_guard = filtered
        if motion_guard.any():
            motion_guard = cv2.dilate(
                cv2.morphologyEx(motion_guard, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8)),
                np.ones((7, 7), np.uint8))

    # 位置合わせ品質：残差中央値がグループ標準より大きいフレームは
    # 「位置合わせ不良」とみなす（ダスト検出閾値2倍・欠陥補正の抑制）
    frame_resid_med = np.array([float(np.median(residual_stack[i])) for i in range(n)])
    group_med = float(np.median(frame_resid_med)) if n else 0.0
    misaligned = frame_resid_med > max(2.0, params.misalign_factor * max(group_med, 0.1))
    threshold_scales = np.where(misaligned, 2.0, 1.0)

    if params.dust_detection and len(aligned) >= 2:
        dust_masks = detect_dust_masks_group(residual_stack, reference,
                                             grain_sigma, params,
                                             frame_threshold_scales=threshold_scales)
    else:
        dust_masks = [None] * len(aligned)

    return {
        "aligned": aligned,
        "reference": reference,
        "n": n,
        "grain_sigma": grain_sigma,
        "dust_masks": dust_masks,
        "flicker_offsets": flicker_offsets,
        "misaligned": misaligned,
        "motion_guard": motion_guard,  # 255/1=グループ内で動いている画素
        "integration_unsafe": integration_unsafe,  # True=統合禁止（低速動きの累積）
    }


def render_hold_group(analysis: dict,
                      params: DenoiseParams | None = None,
                      reference_out: np.ndarray | None = None) -> list[np.ndarray]:
    """解析結果から出力フレーム列を生成する。

    reference_out に第2層で拡張した参照像を渡すと、出力・欠陥補正に
    そちらを使う（欠陥検出自体はグループ内のRで済んでいる）。
    """
    params = params or DenoiseParams()
    reference = reference_out if reference_out is not None else analysis["reference"]
    aligned = analysis["aligned"]
    misaligned = analysis["misaligned"]
    grain_sigma = analysis["grain_sigma"]
    n = analysis["n"]

    # モーションガードの重みマップ（フェザリングして境界の不連続を防ぐ）
    guard = analysis.get("motion_guard")
    guard_w = None
    if guard is not None and guard.any():
        guard_w = cv2.GaussianBlur(
            (guard > 0).astype(np.float32), (15, 15), 0)[..., None]

    outputs = []
    for i, f_aligned in enumerate(aligned):
        mask = analysis["dust_masks"][i]

        if analysis.get("integration_unsafe"):
            # 統合禁止グループ：フレーム自身を出力（グレイン低減のみ任意適用）。
            # Rは信頼できないため欠陥補正も行わない（保守的判定）
            out = f_aligned
            if params.grain_reduction > 0:
                out = spatial_denoise_edge_preserving(
                    f_aligned, grain_sigma, strength=params.grain_reduction)
            outputs.append(out)
            continue

        if params.mode == "full_temporal_integration":
            out = reference
            feather = (params.feather_boundary_frames and n >= 3 and i in (0, n - 1))
            if feather or misaligned[i]:
                # 境界フレーム・位置合わせ不良フレームは誤判定時のにじみ対策として
                # 元フレーム50%ブレンド（設計提案書3章のフェザリング）
                out = reference * 0.5 + f_aligned.astype(np.float32) * 0.5
            if guard_w is not None:
                # グループ内で動いている画素（口パク等）は統合せず
                # フレーム自身の値を保持する（grain_reduction>0なら空間NRのみ適用）
                moving_src = f_aligned
                if params.grain_reduction > 0:
                    moving_src = spatial_denoise_edge_preserving(
                        f_aligned, grain_sigma, strength=params.grain_reduction)
                out = np.asarray(out, np.float32) * (1 - guard_w) \
                    + moving_src.astype(np.float32) * guard_w
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:  # texture_preserving
            base = f_aligned
            if params.grain_reduction > 0:
                base = spatial_denoise_edge_preserving(
                    f_aligned, grain_sigma, strength=params.grain_reduction)
            out = base
            if mask is not None and mask.any():
                m = (mask[..., None] > 0)
                out = np.where(m, np.clip(reference, 0, 255).astype(np.uint8), base)
        outputs.append(out)

    return outputs


def process_hold_group(frames: list[np.ndarray],
                       params: DenoiseParams | None = None) -> dict:
    """保持グループ1つ分をデノイズし、出力フレーム列と解析情報を返す。

    analyze_hold_group ＋ render_hold_group の一括実行（第1層のみの標準経路）。
    """
    params = params or DenoiseParams()
    analysis = analyze_hold_group(frames, params)
    outputs = render_hold_group(analysis, params)

    return {
        "output": outputs,
        "reference": analysis["reference"],
        "grain_sigma": analysis["grain_sigma"],
        "dust_masks": analysis["dust_masks"],
        "dust_pixel_counts": [int((m > 0).sum()) if m is not None else 0
                              for m in analysis["dust_masks"]],
        "flicker_offsets": analysis["flicker_offsets"],
        "misaligned_frames": [int(i) for i in np.where(analysis["misaligned"])[0]],
    }
