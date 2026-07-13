// Phase 1：保持フレーム検出（prototype/hold_frame_detection/ の C++ 移植）
#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "types.h"

namespace animerestore {

// pHash（imagehash.phash 互換：32x32グレー化 → DCT-II(非正規化) → 左上8x8 の
// 中央値二値化）。PILとのリサイズ差により1〜2bit揺れる可能性がある（golden
// テストではハミング距離≤4を許容）。粗選別用途なので分類には影響しない。
uint64_t computePHash(const cv::Mat& frameBgr);
int phashDistance(uint64_t a, uint64_t b);

// ぼかし後の平均絶対差分（グレイン耐性のある軽量差分）
double blurredMeanAbsDiff(const cv::Mat& frameA, const cv::Mat& frameB, int ksize);

// ブロック単位SSIMの平均（ぼかし後に計算。skimage の
// structural_similarity(gaussian_weights=False, win_size=7) と同式）
double blockSsim(const cv::Mat& frameA, const cv::Mat& frameB,
                 int blockSize, int blurKsize = 5,
                 double flatStdThreshold = 2.0);

// 保持グループ検出（Python detect_hold_groups と同一ロジック）
std::vector<HoldGroup> detectHoldGroups(const std::vector<cv::Mat>& frames,
                                        const DetectionThresholds& th = {});

// 累積ドリフト検査：端点比較で再帰二分割（split_drifting_groups 相当）
std::vector<HoldGroup> splitDriftingGroups(const std::vector<cv::Mat>& frames,
                                           const std::vector<HoldGroup>& groups,
                                           const DetectionThresholds& th = {},
                                           int minSpan = 3);

// ダスト耐性再判定：ダストの点滅で分裂した隣接グループを統合
// （hold_frame_detection/refine.py 相当。差分の大きい画素がすべて
// コンパクトなダスト状で、除外後の残差が保持判定を満たす場合のみ統合）
std::vector<HoldGroup> refineHoldGroups(const std::vector<cv::Mat>& frames,
                                        const std::vector<HoldGroup>& groups,
                                        const DetectionThresholds& th = {});

// コマ打ちパターン付与・カット代表パターン（estimate_koma_pattern 相当）
void estimateKomaPattern(std::vector<HoldGroup>& groups);
std::string dominantPatternForShot(const std::vector<HoldGroup>& groups);

}  // namespace animerestore
