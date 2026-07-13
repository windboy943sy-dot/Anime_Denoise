"""
Phase 1：保持フレーム解析プロトタイプ（設計提案書 1章 対応）

処理の流れ（設計提案書1章の疑似コードに対応）：
  1. 粗選別：pHashで直前フレームとの類似度を高速判定
  2. 精密判定：ぼかし後の差分 ＋ ブロック単位SSIMで保持フレーム境界を確定
  3. 保持グループ（hold-group）の確定
  4. カット内の保持グループ長の最頻値からコマ打ちパターンを推定

このモジュール単体では動画ファイルの入出力までは行わない（run_detection.py が担当）。
テストしやすいよう、フレーム列（numpy配列のリスト）を受け取る関数として実装する。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


@dataclass
class HoldGroup:
    start: int  # カット内での相対フレーム番号（0始まり）
    end: int
    confidence: float
    pattern: str = ""  # 後段で estimate_koma_pattern により付与

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class DetectionThresholds:
    coarse_phash_threshold: int = 16      # これを超えたら即座に「新規内容」と判定（粗選別）。
                                          # グレインでpHashは±10程度揺らぐため、粗選別は
                                          # 「明らかに別内容」だけを弾く緩めの値にする
    diff_threshold: float = 3.0           # ぼかし後の平均絶対差分のしきい値（0-255スケール）。
                                          # 実フィルムスキャン素材の実測で、保持フレーム間は約1.5、
                                          # 微妙な動きの境界でも約4.8以上だったため中間の3.0とする
    ssim_threshold: float = 0.92          # ブロック単位SSIMの平均値のしきい値
    block_size: int = 32                  # ブロックSSIMのブロックサイズ(px)
    blur_ksize: int = 5                   # ぼかし後差分に使うガウシアンぼかしのカーネルサイズ


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def compute_phash(frame_bgr: np.ndarray) -> imagehash.ImageHash:
    """pHash（知覚ハッシュ）を計算する。粗選別に使用。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def phash_distance(hash_a: imagehash.ImageHash, hash_b: imagehash.ImageHash) -> int:
    return hash_a - hash_b


def blurred_mean_abs_diff(frame_a: np.ndarray, frame_b: np.ndarray, ksize: int) -> float:
    """ぼかし後フレーム差分（グレイン耐性のある軽量差分）。"""
    gray_a = cv2.GaussianBlur(_to_gray(frame_a), (ksize, ksize), 0)
    gray_b = cv2.GaussianBlur(_to_gray(frame_b), (ksize, ksize), 0)
    return float(np.mean(np.abs(gray_a.astype(np.float32) - gray_b.astype(np.float32))))


def block_ssim(frame_a: np.ndarray, frame_b: np.ndarray, block_size: int,
               blur_ksize: int = 5, flat_std_threshold: float = 2.0) -> float:
    """ブロック単位SSIMの平均値。局所的な差分をならしすぎず検出できる。

    グレイン（フィルムノイズ）はフレーム間で無相関なため、ベタ塗り領域で
    生画素のままSSIMを取ると 0 付近に崩壊する。そのため差分判定と同様に
    ぼかしを掛けてからSSIMを計算し、それでも内容がほぼ平坦なブロックは
    差分ベースの簡易スコアで代替する。
    """
    gray_a = cv2.GaussianBlur(_to_gray(frame_a), (blur_ksize, blur_ksize), 0)
    gray_b = cv2.GaussianBlur(_to_gray(frame_b), (blur_ksize, blur_ksize), 0)
    h, w = gray_a.shape

    scores = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_a = gray_a[y:y + block_size, x:x + block_size]
            block_b = gray_b[y:y + block_size, x:x + block_size]
            # ぼかし後もほぼ平坦なブロック（空・ベタ塗り等）は残留グレインが
            # SSIMの「構造」として支配的になり不安定なため、差分ベースで代替する
            if block_a.std() < flat_std_threshold and block_b.std() < flat_std_threshold:
                diff = np.mean(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32)))
                scores.append(max(0.0, 1.0 - diff / 255.0))
                continue
            score = ssim(block_a, block_b, data_range=255)
            scores.append(score)

    return float(np.mean(scores)) if scores else 1.0


def _confidence(diff_score: float, ssim_score: float, thresholds: DetectionThresholds) -> float:
    """diffとssimがしきい値からどれだけ余裕を持って「同一」判定されたかをスコア化する。
    0〜1に正規化し、1に近いほど「確実に同一フレーム」と判断できる。
    """
    diff_margin = max(0.0, (thresholds.diff_threshold - diff_score) / thresholds.diff_threshold)
    ssim_margin = max(0.0, (ssim_score - thresholds.ssim_threshold) / (1.0 - thresholds.ssim_threshold + 1e-6))
    return float(np.clip((diff_margin + ssim_margin) / 2.0, 0.0, 1.0))


def detect_hold_groups(
    frames: list[np.ndarray],
    thresholds: DetectionThresholds | None = None,
) -> list[HoldGroup]:
    """フレーム列（1カット分を想定）から保持グループ（hold-group）を検出する。

    設計提案書1章の疑似コードに対応：
        for each frame i:
            quick_score = phash_distance(...)
            if quick_score > coarse_threshold: 新規内容
            else:
                diff_score, ssim_score を計算
                しきい値を満たせば現グループを延長、満たさなければ新規グループ
    """
    thresholds = thresholds or DetectionThresholds()
    if not frames:
        return []

    groups: list[HoldGroup] = []
    group_start = 0
    prev_hash = compute_phash(frames[0])
    confidences: list[float] = [1.0]  # 先頭フレームは常にグループの開始点

    for i in range(1, len(frames)):
        cur_hash = compute_phash(frames[i])
        quick_score = phash_distance(prev_hash, cur_hash)

        is_same = False
        conf = 0.0

        if quick_score <= thresholds.coarse_phash_threshold:
            diff_score = blurred_mean_abs_diff(frames[i - 1], frames[i], thresholds.blur_ksize)
            ssim_score = block_ssim(frames[i - 1], frames[i], thresholds.block_size,
                                    blur_ksize=thresholds.blur_ksize)

            if diff_score < thresholds.diff_threshold and ssim_score > thresholds.ssim_threshold:
                is_same = True
                conf = _confidence(diff_score, ssim_score, thresholds)

        if is_same:
            confidences.append(conf)
        else:
            # 現在のグループを確定
            groups.append(HoldGroup(
                start=group_start,
                end=i - 1,
                confidence=float(np.mean(confidences)),
            ))
            group_start = i
            confidences = [1.0]

        prev_hash = cur_hash

    # 最後のグループを確定
    groups.append(HoldGroup(
        start=group_start,
        end=len(frames) - 1,
        confidence=float(np.mean(confidences)),
    ))

    return groups


def _frames_same(frames: list[np.ndarray], a: int, b: int,
                 thresholds: DetectionThresholds) -> bool:
    """フレーム a, b が「同一絵柄」かを精密判定と同じ基準で判定する。"""
    diff = blurred_mean_abs_diff(frames[a], frames[b], thresholds.blur_ksize)
    if diff >= thresholds.diff_threshold:
        return False
    ssim_score = block_ssim(frames[a], frames[b], thresholds.block_size,
                            blur_ksize=thresholds.blur_ksize)
    return ssim_score > thresholds.ssim_threshold


def split_drifting_groups(frames: list[np.ndarray],
                          groups: list[HoldGroup],
                          thresholds: DetectionThresholds | None = None,
                          min_span: int = 3) -> list[HoldGroup]:
    """累積ドリフト検査：グループの端点フレーム同士を直接比較し、
    累積で動いていたら再帰的に二分割する。

    隣接フレーム間の差分が閾値を下回り続ける超低速ズーム・パンや微細な口パクは、
    通常の検出では1つの巨大グループに融合する（実測：263フレームの低速ズームが
    1グループ化）。そのまま時間統合するとゴーストが出るため、
    「グループの端と端が同一絵柄であること」を保証するのがこの検査の目的。
    真の静止グループは端点比較（グレインのみの差）を通過するので影響を受けない。
    """
    thresholds = thresholds or DetectionThresholds()
    result: list[HoldGroup] = []

    def rec(start: int, end: int, confidence: float):
        if end - start + 1 <= min_span or _frames_same(frames, start, end, thresholds):
            result.append(HoldGroup(start=start, end=end, confidence=confidence))
            return
        mid = (start + end) // 2
        # 分割されたグループの信頼度は下げる（端点不一致＝内容が動いている兆候）
        rec(start, mid, confidence * 0.8)
        rec(mid + 1, end, confidence * 0.8)

    for g in groups:
        if g.length <= min_span:
            result.append(g)
            continue
        rec(g.start, g.end, g.confidence)

    return result


def estimate_koma_pattern(groups: list[HoldGroup]) -> list[HoldGroup]:
    """カット内の保持グループ長の最頻値から、各グループにコマ打ちパターンのラベルを付与する。

    設計提案書1章：「同じカット内で打ち方が変わることもあるため、区間ごとに独立判定」の通り、
    最頻パターンと異なる長さのグループは "irregular" として区別する。
    """
    if not groups:
        return groups

    length_counts = Counter(g.length for g in groups)
    dominant_length, _ = length_counts.most_common(1)[0]

    pattern_map = {1: "1koma", 2: "2koma", 3: "3koma", 4: "4koma"}
    dominant_pattern = pattern_map.get(dominant_length, f"{dominant_length}koma")

    for g in groups:
        if g.length == dominant_length:
            g.pattern = dominant_pattern
        elif g.length in pattern_map:
            g.pattern = f"{pattern_map[g.length]}_irregular"
        else:
            g.pattern = f"{g.length}koma_irregular"

    return groups


def dominant_pattern_for_shot(groups: list[HoldGroup]) -> str:
    """カット全体の代表コマ打ちパターン（ラベルCSVのkoma_pattern列と比較する用）。

    グループ長の種類が複数かつ十分な数存在する場合は "mixed" とする。
    """
    if not groups:
        return "unknown"

    length_counts = Counter(g.length for g in groups)
    if len(length_counts) == 1:
        length = next(iter(length_counts))
        return {1: "1koma", 2: "2koma", 3: "3koma", 4: "4koma"}.get(length, f"{length}koma")

    total = sum(length_counts.values())
    dominant_length, dominant_count = length_counts.most_common(1)[0]
    if dominant_count / total >= 0.7:
        return {1: "1koma", 2: "2koma", 3: "3koma", 4: "4koma"}.get(dominant_length, f"{dominant_length}koma")

    return "mixed"
