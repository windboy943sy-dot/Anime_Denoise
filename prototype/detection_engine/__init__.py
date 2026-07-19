"""検知・解析エンジン(Phase 1)。

映像中のノイズ・ダスト・スクラッチを高精度に「検知・分類・解析」する中核
モジュール。除去(Phase 2-4)はこのモジュールの出力 DefectMap / NoiseProfile
のみに依存する。設計は docs/detection_engine_architecture.md を参照。

依存: numpy, scipy のみ(OFX/OpenCV 非依存)。C++/OpenFX への移植のため、
画像演算は scipy.ndimage の分離可能フィルタ・モルフォロジーに限定している。
"""
from .analyzer import AnalyzerConfig, ClipAnalysis, DefectAnalyzer
from .contracts import (DefectInstance, DefectMap, DefectType, DetectorSource,
                        IMPULSE_TYPES, PERSISTENT_TYPES, NoiseProfile)
from .noise_profile import estimate_noise_profile

__all__ = [
    "DefectAnalyzer", "AnalyzerConfig", "ClipAnalysis",
    "DefectMap", "DefectInstance", "DefectType", "DetectorSource",
    "NoiseProfile", "IMPULSE_TYPES", "PERSISTENT_TYPES",
    "estimate_noise_profile",
]
