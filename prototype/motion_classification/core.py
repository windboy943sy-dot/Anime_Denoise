"""
Phase 2：動き分類プロトタイプ（設計提案書 2章 対応）

処理の流れ：
  1. 大域動き推定：ORB特徴点マッチング＋RANSACで部分アフィン変換を初期推定し、
     ECC（cv2.findTransformECC）でサブピクセル精度に追い込む。
     並進量・スケール比・回転角に分解して「静止／パン／ズーム／回転」に分類。
     特徴点不足時（ベタ塗り主体のカット）は位相相関法で並進のみ推定にフォールバック。
  2. ノイズ床の較正：Phase 1 の保持グループ「内」のフレームペアは
     動きゼロ・グレイン／ウィーブのみの教師データとみなせるため、
     そこから「動きなしでもこれだけの残差が出る」というカット固有のノイズ床を実測する
     （設計提案書2章「閾値設計はカット単位で正規化するのが望ましい」に対応）。
  3. 局所動き分類：大域動き補正後のトレランス付き残差がノイズ床を超えた画素を
     「動き画素」とし、その割合と空間分布で「なし／局所／全体」に分類。

コマ打ちアニメでは動きは保持グループ（hold-group）間でしか起きないため、
本モジュールは「連続フレーム」ではなく「保持グループの代表フレーム同士」の
比較を前提とする（Phase 1の検出結果を入力にする）。

フィルムスキャン素材固有の対策：
  - スキャンにはフィルム枠・パーフォレーションが写り込んでいる。これらは
    スキャナ座標系で固定なのに絵はカメラワークで動くため、大域補正すると
    枠側が巨大な偽残差になる。よって残差評価は中央のアクティブピクチャ領域
    （active_area_crop で指定）に限定する。
  - サブピクセルの位置ずれはシャープなエッジ上で大きな差分を生むため、
    残差は dilate/erode によるトレランス付き（±1px の位置ずれを許容）で取る。

備考：ロードマップPhase 2ではOptical Flow（Farneback）の使用が挙げられているが、
グレインの強い素材では平坦領域でフローが大きなベクトルを幻覚する問題を実測で確認
したため、なし／局所／全体の3クラス分類には残差ベースを採用する。動きベクトルの
方向・大きさが必要になる後段（補間・位置合わせ）で高精度フロー（RAFT等）を検討する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionThresholds:
    # 前処理
    work_width: int = 640          # 解析用ダウンスケール後の幅（速度とグレイン耐性のため）
    blur_ksize: int = 5

    # 大域動き分類（work_width スケールでの値）
    static_translation_px: float = 0.5   # これ未満の並進は「静止」
    zoom_scale_delta: float = 0.002      # |scale-1| がこれ以上なら「ズーム」成分あり
    rotation_deg: float = 0.5            # |回転角| がこれ以上なら「回転」成分あり。
                                         # フィルムのゲートウィーブで±0.3°程度揺らぐため
                                         # それより大きい値にする

    # ORB / RANSAC / ECC
    orb_features: int = 2000
    min_matches: int = 20                # これ未満なら位相相関フォールバック
    ransac_reproj_threshold: float = 3.0
    use_ecc_refinement: bool = True
    ecc_iterations: int = 50

    # 残差・局所動き分類
    active_area_crop: float = 0.12       # フィルム枠除外のため上下左右をこの割合だけ切り落とす
    residual_floor: float = 3.0          # ノイズ床の下限（0-255スケール）
    noise_floor_percentile: float = 99.5 # 保持グループ内残差のこのパーセンタイルをノイズ床にする
    moving_ratio_none: float = 0.01      # 動き画素割合がこれ未満なら「動きなし」
    moving_ratio_full: float = 0.35      # これ以上なら「全体が動く」


@dataclass
class GlobalMotion:
    type: str          # "static" / "pan" / "zoom" / "rotation"
    tx: float          # 元解像度スケールでの並進(px)
    ty: float
    scale: float       # スケール比（1.0=変化なし）
    rotation_deg: float
    method: str        # "orb_ransac" / "orb_ransac+ecc" / "phase_correlation"
    confidence: float  # インライア率／ECC相関係数など推定の確からしさ（0-1）
    warp_work: np.ndarray | None = None  # workスケールでの2x3アフィン行列（残差計算で再利用）
    scale_dev_ratio: float | None = None  # フルアフィンのスケール偏差比 min/max。
                                          # 等方（真のズーム）≈0.9+、視差パンは小。
                                          # ショット単位判定（dominant_global_motion）で使用


@dataclass
class LocalMotion:
    type: str            # "none" / "local" / "full"
    moving_ratio: float  # アクティブ領域内の動き画素割合（0-1）
    largest_component_ratio: float  # 動き画素中、最大連結成分が占める割合（0-1）
    bbox: tuple[int, int, int, int] | None  # 元解像度での動き領域外接矩形 (x, y, w, h)
    noise_floor: float   # 使用したノイズ床（デバッグ・チューニング用）


def _prep(frame_bgr: np.ndarray, thresholds: MotionThresholds) -> tuple[np.ndarray, float]:
    """解析用のグレースケール縮小画像と、work座標→元解像度の倍率を返す。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = thresholds.work_width / w
    if scale < 1.0:
        gray = cv2.resize(gray, (thresholds.work_width, int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    k = thresholds.blur_ksize
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray, 1.0 / scale


def _decompose_partial_affine(m: np.ndarray) -> tuple[float, float, float, float]:
    """2x3 のアフィン行列を (tx, ty, scale, rot_deg) に分解する。

    スケールは行列式の平方根（面積比の平方根）で取る。ECC等がシアーを含む
    フルアフィンを返した場合でも、hypot による行ノルムより安定するため。
    """
    tx, ty = m[0, 2], m[1, 2]
    det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
    scale = math.sqrt(max(det, 1e-12))
    rot_deg = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    return float(tx), float(ty), float(scale), float(rot_deg)


def estimate_global_motion(frame_a: np.ndarray, frame_b: np.ndarray,
                           thresholds: MotionThresholds | None = None) -> GlobalMotion:
    """frame_a → frame_b の大域動きを推定し分類する。"""
    thresholds = thresholds or MotionThresholds()
    gray_a, to_orig = _prep(frame_a, thresholds)
    gray_b, _ = _prep(frame_b, thresholds)

    warp = None
    method = "orb_ransac"
    confidence = 0.0

    orb = cv2.ORB_create(nfeatures=thresholds.orb_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    matches = []
    if des_a is not None and des_b is not None and len(kp_a) > 0 and len(kp_b) > 0:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)

    matched_pts = None
    if len(matches) >= thresholds.min_matches:
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
        m_affine, inliers = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC,
            ransacReprojThreshold=thresholds.ransac_reproj_threshold)
        if m_affine is not None and inliers is not None and inliers.sum() >= thresholds.min_matches // 2:
            warp = m_affine.astype(np.float32)
            confidence = float(inliers.sum() / len(matches))
            matched_pts = (pts_a, pts_b)

    if warp is None:
        # ベタ塗り等で特徴点が足りない場合：位相相関で並進のみ推定
        # （設計提案書2章「位相相関法で並進成分を高速推定」に対応）
        method = "phase_correlation"
        (dx, dy), response = cv2.phaseCorrelate(gray_a.astype(np.float32),
                                                gray_b.astype(np.float32))
        warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        confidence = float(np.clip(response, 0.0, 1.0))

    # 分類にはORB／位相相関の推定値を使う（下記ECC前に分解しておく）。
    # マルチプレーン撮影のカットでは、特徴点ベースのORBは前景レイヤー、
    # 輝度ベースのECCは面積の大きい背景レイヤーに合わせる傾向があり、
    # カメラワークの「種類」は前景基準（ORB）の方が人間のラベルと一致しやすい
    tx, ty, scale, rot = _decompose_partial_affine(warp)
    warp_pre_ecc = warp.copy()

    def _tolerant_map_of(warp_matrix: np.ndarray | None) -> np.ndarray:
        """指定ワープ適用後（Noneならidentity）のトレランス付き残差マップ。"""
        img_a = gray_a
        if warp_matrix is not None:
            img_a = cv2.warpAffine(gray_a, warp_matrix,
                                   (gray_a.shape[1], gray_a.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        k3 = np.ones((3, 3), np.uint8)
        over = cv2.subtract(gray_b, cv2.dilate(img_a, k3)).astype(np.float32)
        under = cv2.subtract(cv2.erode(img_a, k3), gray_b).astype(np.float32)
        resid = np.maximum(over, under)
        h, w = resid.shape
        my = int(h * thresholds.active_area_crop)
        mx = int(w * thresholds.active_area_crop)
        return resid[my:h - my, mx:w - mx]

    # ECC でサブピクセル精度に追い込む（残差計算用のワープのみ。分類には使わない）
    if thresholds.use_ecc_refinement:
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        thresholds.ecc_iterations, 1e-5)
            # findTransformECC は warpAffine(gray_a, warp) ≒ gray_b になる向き
            cc, warp_ecc = cv2.findTransformECC(gray_b, gray_a, warp.copy(),
                                                cv2.MOTION_AFFINE, criteria, None, 5)
            # ECCはグレインの強い素材で発散してワープを壊すことがあるため、
            # 実際に残差が改善した場合のみ採用する
            if float(_tolerant_map_of(warp_ecc).mean()) <= \
                    float(_tolerant_map_of(warp_pre_ecc).mean()):
                warp = warp_ecc
                method += "+ecc"
                confidence = max(confidence, float(np.clip(cc, 0.0, 1.0)))
        except cv2.error:
            pass  # 収束失敗時はORB/位相相関の結果をそのまま使う

    # 分類（優先順位：回転 > ズーム > パン > 静止。複合動きは支配的な成分で代表する）
    translation = math.hypot(tx, ty)
    if abs(rot) >= thresholds.rotation_deg:
        motion_type = "rotation"
    elif abs(scale - 1.0) >= thresholds.zoom_scale_delta:
        motion_type = "zoom"
    elif translation >= thresholds.static_translation_px:
        motion_type = "pan"
    else:
        motion_type = "static"

    # ワープ改善チェック：推定したワープが identity（静止）に対して
    # 「改善した画素」と「新たに壊した画素」のどちらが多いかで検証する。
    # 静止カメラ＋大面積のキャラ芝居では、ORB特徴点がキャラに乗って
    # キャラの変形をカメラワークと誤推定するが、そのワープを適用すると
    # 静止していた背景が新たにずれる（壊れる画素が発生する）。
    # 真のカメラワークならワープは改善一方で、壊す画素はほぼ出ない。
    # 平均残差の比較ではグレインが支配して差が薄まるため、画素数の対決にする。
    # チェックは分類の根拠である ECC 前のワープで行う（ECCとは独立に判定）。
    if motion_type != "static":
        resid_id = _tolerant_map_of(None)
        resid_warp = _tolerant_map_of(warp_pre_ecc)
        t = 4.0  # ぼかし・トレランス後のグレイン残差(1〜2)より十分上の閾値
        improved = float(((resid_id > t) & (resid_warp <= t)).mean())
        broken = float(((resid_id <= t) & (resid_warp > t)).mean())
        if broken >= improved:
            motion_type = "static"

    # マルチプレーン判定用の非等方性（ショット単位で集計して使う）：
    # 真のズームは等方スケール（|sx-1|≈|sy-1|、ショット中央値0.94〜0.99）、
    # 多層視差はパン方向の軸だけスケールが立つ（同0.60/0.28）。
    # 遷移単位ではノイズで不安定なため（実測）、ここでは比率の記録のみ行い、
    # 判定は dominant_global_motion がショット中央値で行う
    scale_dev_ratio = None
    if motion_type in ("zoom", "rotation") and matched_pts is not None:
        a_full, _ = cv2.estimateAffine2D(
            matched_pts[0], matched_pts[1], method=cv2.RANSAC,
            ransacReprojThreshold=thresholds.ransac_reproj_threshold)
        if a_full is not None:
            sx = math.hypot(a_full[0, 0], a_full[1, 0])
            sy = math.hypot(a_full[0, 1], a_full[1, 1])
            dev_lo, dev_hi = sorted([abs(sx - 1.0), abs(sy - 1.0)])
            if dev_hi > 0.005:
                scale_dev_ratio = dev_lo / dev_hi

    return GlobalMotion(
        type=motion_type,
        tx=tx * to_orig, ty=ty * to_orig,
        scale=scale, rotation_deg=rot,
        method=method, confidence=confidence,
        warp_work=warp,
        scale_dev_ratio=scale_dev_ratio,
    )


def _tolerant_residual(frame_a: np.ndarray, frame_b: np.ndarray,
                       global_motion: GlobalMotion,
                       thresholds: MotionThresholds) -> np.ndarray:
    """大域動き補正後の、±1px位置ずれを許容した残差マップ（workスケール）を返す。"""
    gray_a, _ = _prep(frame_a, thresholds)
    gray_b, _ = _prep(frame_b, thresholds)
    warp = global_motion.warp_work
    if warp is None:
        warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    warped_a = cv2.warpAffine(gray_a, warp, (gray_a.shape[1], gray_a.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    k3 = np.ones((3, 3), np.uint8)
    over = cv2.subtract(gray_b, cv2.dilate(warped_a, k3)).astype(np.float32)
    under = cv2.subtract(cv2.erode(warped_a, k3), gray_b).astype(np.float32)
    return cv2.GaussianBlur(np.maximum(over, under), (5, 5), 0)


def _active_area(residual: np.ndarray, thresholds: MotionThresholds) -> np.ndarray:
    """フィルム枠・パーフォレーションを除いた中央領域を切り出す。"""
    h, w = residual.shape
    my = int(h * thresholds.active_area_crop)
    mx = int(w * thresholds.active_area_crop)
    return residual[my:h - my, mx:w - mx]


def estimate_noise_floor(intra_group_pairs: list[tuple[np.ndarray, np.ndarray]],
                         thresholds: MotionThresholds | None = None) -> float:
    """保持グループ内（＝動きゼロ）のフレームペアから、カット固有のノイズ床を実測する。

    ペアごとにウィーブ補正（大域推定）→残差を取り、その高パーセンタイル値を
    「動きが無くても出る残差の上限」とみなす。
    """
    thresholds = thresholds or MotionThresholds()
    samples = []
    for fa, fb in intra_group_pairs:
        g = estimate_global_motion(fa, fb, thresholds)
        resid = _active_area(_tolerant_residual(fa, fb, g, thresholds), thresholds)
        samples.append(np.percentile(resid, thresholds.noise_floor_percentile))
    if not samples:
        return thresholds.residual_floor
    # ペア間の中央値を採用（ダスト等の突発欠陥を含むペアに引きずられないように）
    return max(thresholds.residual_floor, float(np.median(samples)))


def classify_local_motion(frame_a: np.ndarray, frame_b: np.ndarray,
                          global_motion: GlobalMotion,
                          noise_floor: float,
                          thresholds: MotionThresholds | None = None) -> LocalMotion:
    """大域動き補正後の残差から局所動きを分類する。

    noise_floor には estimate_noise_floor で較正したカット固有の値を渡す。
    """
    thresholds = thresholds or MotionThresholds()
    resid = _tolerant_residual(frame_a, frame_b, global_motion, thresholds)
    inner = _active_area(resid, thresholds)

    moving = (inner > noise_floor).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, k3)  # 孤立ノイズ画素の除去

    moving_ratio = float(moving.mean())

    largest_ratio = 0.0
    bbox = None
    if moving.any():
        # 近接した動き領域（キャラの輪郭など）をまとめてから連結成分を取る
        merged = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        if n_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = int(np.argmax(areas)) + 1
            largest_ratio = float(areas.max() / max(1, merged.sum()))
            # bbox は元解像度に換算（クロップ分のオフセットも戻す）
            gray_h, gray_w = resid.shape
            src_w = frame_a.shape[1]
            to_orig = src_w / gray_w if gray_w else 1.0
            off_x = int(gray_w * thresholds.active_area_crop)
            off_y = int(gray_h * thresholds.active_area_crop)
            x, y, w, h = (stats[largest_idx, cv2.CC_STAT_LEFT],
                          stats[largest_idx, cv2.CC_STAT_TOP],
                          stats[largest_idx, cv2.CC_STAT_WIDTH],
                          stats[largest_idx, cv2.CC_STAT_HEIGHT])
            bbox = (int((x + off_x) * to_orig), int((y + off_y) * to_orig),
                    int(w * to_orig), int(h * to_orig))

    # 分類（ロードマップPhase 2完了基準の方針に従い、まずは
    # なし／局所／全体 の3クラス。キャラ／エフェクトの区別は後段で追加する）
    if moving_ratio < thresholds.moving_ratio_none:
        motion_type = "none"
    elif moving_ratio >= thresholds.moving_ratio_full:
        motion_type = "full"
    else:
        motion_type = "local"

    return LocalMotion(
        type=motion_type,
        moving_ratio=moving_ratio,
        largest_component_ratio=largest_ratio,
        bbox=bbox,
        noise_floor=noise_floor,
    )


def analyze_camera_path(motions: list[GlobalMotion],
                        min_translation_px: float = 2.0) -> dict:
    """遷移列の並進ベクトルから「パン」と「カメラシェイク」を区別する統計を返す。

    パン：変位の方向が一貫している（単位ベクトルの合成長 R が 1 に近い）
    シェイク：変位はあるが方向がばらばらで、累積してもほぼ元の位置に留まる（R が小さい）

    R（方向一貫性）は circular statistics の mean resultant length。
    外れ値（モーションブラーでORB/ECCが壊れた遷移など）が1つあっても、
    単位ベクトル平均なので大きさに引きずられない。
    """
    vecs = [(m.tx, m.ty) for m in motions]
    mags = [math.hypot(x, y) for x, y in vecs]
    moving = [(x / m, y / m) for (x, y), m in zip(vecs, mags)
              if m >= min_translation_px]

    if len(moving) < 3:
        consistency = 1.0  # サンプル不足時はシェイク判定しない（安全側）
    else:
        mean_x = sum(v[0] for v in moving) / len(moving)
        mean_y = sum(v[1] for v in moving) / len(moving)
        consistency = math.hypot(mean_x, mean_y)

    return {
        "direction_consistency": round(consistency, 3),
        "moving_transitions": len(moving),
        "median_translation_px": round(
            float(np.median(mags)) if mags else 0.0, 2),
        "camera_shake": len(moving) >= 3 and consistency < 0.5,
    }


def dominant_global_motion(motions: list[GlobalMotion]) -> str:
    """カット全体の代表大域動き（ラベルCSVのglobal_motion列と比較する用）。

    並進主体だが方向が一貫しない場合はカメラシェイク（フィックス＋揺れ演出）と
    みなし static を返す。シェイクか否かの詳細は analyze_camera_path で取れる。
    """
    if not motions:
        return "unknown"
    from collections import Counter
    counts = Counter(m.type for m in motions)
    # 「静止」以外の動きが一定数あればそれを代表とする（止め主体のカットでも
    # パン区間があれば pan と答えたいため）
    non_static = {k: v for k, v in counts.items() if k != "static"}
    # 閾値25%：低速パン（遷移の多くが静止閾値未満に落ちる）でも、
    # 非静止遷移が1/4以上あればカメラワークありとみなす。
    # シェイク素材の誤昇格は analyze_camera_path の方向一貫性チェックが防ぐ
    if non_static and sum(non_static.values()) >= max(2, len(motions) * 0.25):
        dominant = max(non_static, key=lambda k: non_static[k])
        if dominant == "pan" and analyze_camera_path(motions)["camera_shake"]:
            return "static"  # 方向が一貫しない並進＝シェイク（フィックス扱い）
        if dominant in ("zoom", "rotation"):
            # ショット単位の視差判定：zoom/rotation 遷移の非等方比の中央値が
            # 低ければ多層視差のパンとみなす（実測較正：真のズーム0.94〜0.99、
            # 視差0.60/0.28。遷移単位はノイズ支配のためショット集計が正解）
            ratios = [m.scale_dev_ratio for m in motions
                      if m.type == dominant and m.scale_dev_ratio is not None]
            if len(ratios) >= 3 and float(np.median(ratios)) < 0.75:
                return "pan"
        return dominant
    return counts.most_common(1)[0][0]
