"""検知・解析エンジン 共通データ契約 (Data Contracts)

このモジュールは検知エンジンの「中核」となる出力契約を定義する。除去段
(Phase 2-4) や UI・品質評価モジュールは、すべてこの契約だけに依存する。
検知アルゴリズムを差し替えても契約が変わらなければ後段は影響を受けない、
という拡張性(最重要要件)の担保がこのファイルの役割である。

根拠となる設計文書:
  - ダスト/スクラッチ検知サーベイ v1.0 §0.3「検知器の出力契約 DefectMap」
    (画素マスク層 + インスタンス層の二層構造)
  - ノイズ検知サーベイ v1.0 §0.1「ノイズプロファイル4軸」

記述区分(サーベイの規約を踏襲):
  [事実]   査読論文・公式仕様に基づく
  [考察]   本実装独自の設計判断
  [要実測] 実データで検証すべき事項
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class DefectType(enum.Enum):
    """欠陥種別。サーベイ第0章の分類に対応。

    UI の個別 on/off・除去手法の選択・誤検知分析はすべてこの型で分岐する。
    値を追加する際は末尾に足すこと(シリアライズ互換のため)。
    """

    UNKNOWN = "unknown"
    DUST_WHITE = "dust_white"       # 白ダスト(明インパルス) [事実 §0.1]
    DUST_BLACK = "dust_black"       # 黒ダスト(暗インパルス)
    PARTICLE = "particle"           # 大型ゴミ・繊維(細長・不定形)
    MOLD = "mold"                   # カビ(樹枝状・多フレーム持続)
    SCRATCH_VERTICAL = "scratch_v"  # 縦傷(持続) [事実 §0.1]
    SCRATCH_HORIZONTAL = "scratch_h"
    SCRATCH_CURVED = "scratch_curved"
    DROPOUT = "dropout"             # VHS等の水平ドロップアウト線


# 時間挙動による2系統分類(サーベイ §0.2 の中核テーゼ)
# インパルス系統(単一フレーム外れ値)と持続系統(複数フレーム持続)は
# 原理的に別の検知器で扱う。
IMPULSE_TYPES = frozenset(
    {DefectType.DUST_WHITE, DefectType.DUST_BLACK, DefectType.PARTICLE, DefectType.DROPOUT}
)
PERSISTENT_TYPES = frozenset(
    {DefectType.MOLD, DefectType.SCRATCH_VERTICAL, DefectType.SCRATCH_HORIZONTAL,
     DefectType.SCRATCH_CURVED}
)


class DetectorSource(enum.Flag):
    """どの検知器が寄与したか(誤検知分析・重み付け投票用)。Flag なので OR 合成可。"""

    NONE = 0
    DOG = enum.auto()          # §1.1 DoG/LoG
    TOPHAT = enum.auto()       # §1.2 Top-Hat
    HESSIAN = enum.auto()      # §1.4 Hessian/vesselness
    SDI = enum.auto()          # §2.1 Kokaram SDI
    ROD = enum.auto()          # §2.2 Nadenau-Mitra ROD
    TEMPORAL_MEDIAN = enum.auto()  # §2.3
    PROJECTION = enum.auto()   # §4.2 垂直射影(Joyeux)
    PERSISTENCE = enum.auto()  # §2.5 持続性追跡
    CNN = enum.auto()          # §6 AI検証(将来)


@dataclass
class NoiseProfile:
    """ノイズプロファイル4軸(ノイズサーベイ §0.1)。

    ダスト検知の適応しきい値 T = k * sigma はこのプロファイルを共通基盤にする
    (ダストサーベイ §5.1「ダスト検知精度はノイズ推定精度に律速される」)。

    [考察] どの色空間で推定したかを必ず記録する(img-processing 規律)。
    """

    # 軸1: 強度依存性。輝度ビン(0..1 正規化)別の sigma テーブル。
    intensity_bins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_by_bin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # ポアソン-ガウス Var = a*E[y] + b の当てはめ結果(任意)
    poisson_a: Optional[float] = None
    gauss_b: Optional[float] = None

    # 軸2: 空間相関。白色(相関長~1)か有色(塊状)か。
    spatial_correlation_length: float = 1.0   # 画素単位。>1.5 で有色とみなす [考察]
    is_white: bool = True

    # 軸3: 時間特性。
    temporal_sigma: Optional[float] = None     # フレーム間差分から推定した sigma
    has_fixed_pattern: bool = False            # FPN の疑い(時間平均に縞が残る)
    flicker_detected: bool = False

    # 軸4: 色チャンネル依存性。
    sigma_per_channel: Optional[np.ndarray] = None  # 例: [R,G,B] or [Y,Cb,Cr]
    chroma_dominant: bool = False              # クロマノイズ優勢(VHS・高ISO)

    # メタ
    color_space: str = "unknown"               # "rec709_gamma" 等
    dominant_model: str = "unknown"            # "awgn"|"poisson_gauss"|"correlated"|"grain"|"fpn"
    global_sigma: float = 0.0                  # 代表 sigma(0..1 スケール)

    def sigma_at(self, luma: float) -> float:
        """輝度 luma(0..1)におけるノイズ sigma を返す。強度依存の適応しきい値用。"""
        if self.sigma_by_bin.size == 0:
            return self.global_sigma
        idx = int(np.clip(luma, 0.0, 1.0) * (self.sigma_by_bin.size - 1) + 0.5)
        idx = int(np.clip(idx, 0, self.sigma_by_bin.size - 1))
        return float(self.sigma_by_bin[idx])

    def sigma_map(self, luma_img: np.ndarray) -> np.ndarray:
        """輝度画像から画素毎の sigma マップを生成(適応しきい値の空間展開)。"""
        if self.sigma_by_bin.size == 0:
            return np.full(luma_img.shape, self.global_sigma, dtype=np.float32)
        idx = np.clip(luma_img, 0.0, 1.0) * (self.sigma_by_bin.size - 1)
        idx = np.clip(np.round(idx).astype(np.int32), 0, self.sigma_by_bin.size - 1)
        return self.sigma_by_bin[idx].astype(np.float32)


@dataclass
class DefectInstance:
    """欠陥インスタンス(連結成分)。誤検知抑制・分類・追跡・UI はここで動く。

    サーベイ §0.3 のインスタンス層。画素マスクと二層で保持する。
    """

    id: int
    type: DefectType = DefectType.UNKNOWN
    # 幾何
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    centroid: tuple[float, float] = (0.0, 0.0)
    area: int = 0
    elongation: float = 1.0        # 慣性モーメント比 λ1/λ2(細長度)
    circularity: float = 0.0       # 4πA/P²
    orientation_deg: float = 0.0   # 主軸方向(度)。0=水平, 90=垂直
    # 外観
    polarity: int = 0              # +1=明(白), -1=暗(黒), 0=不明
    contrast: float = 0.0          # 背景との輝度差
    translucency: float = 0.0      # 半透明度 α(0=不透明, 1=完全透明) §0.3
    # 時間
    first_frame: int = -1
    track_id: int = -1
    persistence: int = 1           # 連続して検知されたフレーム数
    # 由来
    confidence: float = 0.0        # 0..1
    sources: DetectorSource = DetectorSource.NONE

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["type"] = self.type.value
        d["sources"] = int(self.sources.value)
        return d


@dataclass
class DefectMap:
    """検知エンジンの最終出力(サーベイ §0.3)。

    - prob:   画素別欠陥確率(0..1)。除去(インペイント)はこの層で動く。
    - alpha:  半透明度マップ。減算/ブレンド修復に使う(§7.6)。
    - labels: 画素→インスタンスID。
    - instances: インスタンス層。
    - frame_stats: 欠陥密度・種別ヒストグラム(劣化診断・UI用)。
    """

    width: int
    height: int
    frame_index: int = 0
    prob: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    instances: list[DefectInstance] = field(default_factory=list)
    frame_stats: dict = field(default_factory=dict)
    noise_profile: Optional[NoiseProfile] = None

    def binary_mask(self, threshold: float = 0.5) -> np.ndarray:
        """確率マスクを二値化。除去段が要求する dilation はここでは行わない。"""
        return (self.prob >= threshold).astype(np.uint8)

    def instances_of(self, *types: DefectType) -> list[DefectInstance]:
        s = set(types)
        return [ins for ins in self.instances if ins.type in s]

    def type_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for ins in self.instances:
            hist[ins.type.value] = hist.get(ins.type.value, 0) + 1
        return hist

    def compute_frame_stats(self) -> None:
        total = self.width * self.height
        defect_px = int((self.prob >= 0.5).sum()) if self.prob.size else 0
        self.frame_stats = {
            "defect_pixel_ratio": (defect_px / total) if total else 0.0,
            "instance_count": len(self.instances),
            "type_histogram": self.type_histogram(),
            "mean_confidence": (
                float(np.mean([i.confidence for i in self.instances])) if self.instances else 0.0
            ),
        }
