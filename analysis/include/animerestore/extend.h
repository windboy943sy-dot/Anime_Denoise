// 第2層：カット間拡張統合＋第3層ブレンド（prototype/denoise/extend.py の移植）
// 設計と実測根拠は docs/denoise_method_survey.md 2章・第2層の実測結果を参照。
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "denoise.h"

namespace animerestore {

struct ExtendParams {
    int radius = 2;
    double acceptSigmaFactor = 3.0;
    double acceptFloor = 3.0;
    double minAcceptRatio = 0.05;
    double maxTranslationRatio = 0.25;
    double minWarpConfidence = 0.2;
    double activeAreaCrop = 0.10;
};

struct ExtendResult {
    cv::Mat reference;    // 拡張後のR（32FC3）
    cv::Mat effectiveN;   // 画素ごとの実効フレーム数（32F）
    int usedNeighbors = 0;
    std::vector<double> acceptRatios;
};

// 中心グループのRを隣接グループのRで拡張統合する。
// エッジ画素は統合から除外（ワープ残留誤差±1pxによる鮮鋭度低下の実測対策）
ExtendResult extendReference(const GroupAnalysis& center,
                             const std::vector<const GroupAnalysis*>& neighbors,
                             const ExtendParams& p = {});

// 第3層：実効Nの低い画素にだけ線画保護つき空間NRを混ぜる連続ブレンド
cv::Mat blendSpatialFallback(const cv::Mat& image, const cv::Mat& effectiveN,
                             double grainSigma, double strength = 1.0);

}  // namespace animerestore
