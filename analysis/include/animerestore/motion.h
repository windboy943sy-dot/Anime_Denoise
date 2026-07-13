// Phase 2：動き分類（prototype/motion_classification/core.py の C++ 移植）
// 実測に基づく設計判断（ワープ改善チェック・ECC採用ゲート・シェイク判定等）は
// Python 側コメントと docs/phase1_phase2_status.md を参照。
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace animerestore {

struct MotionThresholds {
    int workWidth = 640;
    int blurKsize = 5;

    double staticTranslationPx = 0.5;   // workスケール
    double zoomScaleDelta = 0.002;
    double rotationDeg = 0.5;           // ゲートウィーブ±0.3°より大きく

    int orbFeatures = 2000;
    int minMatches = 20;
    double ransacReprojThreshold = 3.0;
    bool useEccRefinement = true;
    int eccIterations = 50;

    double activeAreaCrop = 0.12;       // フィルム枠除外
    double residualFloor = 3.0;
    double noiseFloorPercentile = 99.5;
    double movingRatioNone = 0.01;
    double movingRatioFull = 0.35;
};

struct GlobalMotion {
    std::string type;        // "static" / "pan" / "zoom" / "rotation"
    double tx = 0, ty = 0;   // 元解像度スケール(px)
    double scale = 1.0;
    double rotationDeg = 0;
    std::string method;      // "orb_ransac" / "orb_ransac+ecc" / "phase_correlation"
    double confidence = 0;
    cv::Mat warpWork;        // workスケール2x3（残差計算で再利用）
    double scaleDevRatio = -1;  // フルアフィンのスケール偏差比 min/max（-1=無効）。
                                // 等方（真のズーム）≈0.9+、視差パンは小。ショット単位判定で使用
};

struct LocalMotion {
    std::string type;        // "none" / "local" / "full"
    double movingRatio = 0;
    double largestComponentRatio = 0;
    cv::Rect bbox;           // 元解像度。無効なら width==0
    double noiseFloor = 0;
};

struct CameraPath {
    double directionConsistency = 1.0;  // 単位ベクトル合成長R（circular statistics）
    int movingTransitions = 0;
    double medianTranslationPx = 0;
    bool cameraShake = false;
};

GlobalMotion estimateGlobalMotion(const cv::Mat& frameA, const cv::Mat& frameB,
                                  const MotionThresholds& th = {});

// 保持グループ内ペア（動きゼロの教師データ）からカット固有のノイズ床を較正
double estimateNoiseFloor(const std::vector<std::pair<cv::Mat, cv::Mat>>& intraGroupPairs,
                          const MotionThresholds& th = {});

LocalMotion classifyLocalMotion(const cv::Mat& frameA, const cv::Mat& frameB,
                                const GlobalMotion& globalMotion,
                                double noiseFloor,
                                const MotionThresholds& th = {});

CameraPath analyzeCameraPath(const std::vector<GlobalMotion>& motions,
                             double minTranslationPx = 2.0);

std::string dominantGlobalMotion(const std::vector<GlobalMotion>& motions);

}  // namespace animerestore
