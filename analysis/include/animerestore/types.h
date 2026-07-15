// 共通型定義（docs/cpp_port_design.md 2章）
// Python プロトタイプ prototype/hold_frame_detection/core.py の
// dataclass 群と1対1対応させる。閾値のデフォルト値も一致させること。
#pragma once

#include <string>
#include <vector>

namespace animerestore {

struct HoldGroup {
    int start = 0;              // カット内相対フレーム番号（0始まり）
    int end = 0;
    double confidence = 0.0;
    std::string pattern;        // "3koma" 等（estimateKomaPattern が付与）

    int length() const { return end - start + 1; }
};

struct DetectionThresholds {
    // Python 側 DetectionThresholds と同値（変更時は両方を更新すること）
    int coarsePhashThreshold = 16;  // pHash粗選別。グレインで±10揺らぐため緩め
    double diffThreshold = 3.0;     // ぼかし後平均絶対差分（実測: 保持間1.5/境界4.8）
    double ssimThreshold = 0.92;    // ブロックSSIM平均
    int blockSize = 32;
    int blurKsize = 5;
    bool useRegionSegment = false;  // 領域別（マルチプレーン）保持判定を有効化するフラグ
};

}  // namespace animerestore
