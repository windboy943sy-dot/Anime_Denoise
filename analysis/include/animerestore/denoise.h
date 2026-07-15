// Phase 3：参照像R生成・欠陥検出・デノイズ出力（prototype/denoise/core.py の
// C++ 移植）。設計原則（4層構成・エッジ非劣化・モーションガード）は
// docs/denoise_method_survey.md を参照。
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace animerestore {

enum class DenoiseMode { TexturePreserving, FullTemporalIntegration };
enum class ReferenceMethod { Median, TrimmedMean, Mean };

// ソース世代（★3、docs/design_integration_review.md）。
// 現行の処理内容は3世代とも共通で、これは「どの欠陥除去を既定で有効に
// するか」の推奨値を切り替えるだけ。ビデオ収録特有の処理
// （デインターレース・クロマにじみ）とデジタル特有の処理（バンディング）
// は将来ここへフックする前提のパラメータ設計
enum class SourceMedia { FilmScan, VideoTape, DigitalNative };

struct SourceMediaDefaults {
    bool dust;       // フィルムのダスト・ダート（ビデオ/デジタルでは誤検出源）
    bool lineNoise;  // テープドロップアウト由来の行/列ノイズ
    bool scanNoise;  // スキャナ/伝送系の周期ノイズ
};

// 世代別の推奨既定値。UI/CLI で共有し、ユーザーの個別指定は常にこれを上書き
SourceMediaDefaults sourceMediaDefaults(SourceMedia m);

struct DenoiseParams {
    // 位置合わせ
    bool align = true;
    int alignWorkWidth = 640;
    int eccIterations = 30;

    // 参照像R
    ReferenceMethod referenceMethod = ReferenceMethod::TrimmedMean;
    double trimRatio = 0.25;

    // ダスト検出（4条件：振幅・時間的単発性・形状・場所）
    bool dustDetection = true;
    double dustSigma = 5.0;
    int dustMinArea = 4;
    double dustMaxAreaRatio = 0.0008;
    bool dustProtectEdges = true;
    double dustActiveAreaCrop = 0.10;

    // 出力
    DenoiseMode mode = DenoiseMode::TexturePreserving;
    double grainReduction = 0.0;
    bool featherBoundaryFrames = true;
    double misalignFactor = 2.5;
    bool flickerCorrection = false;
};

struct GroupAnalysis {
    std::vector<cv::Mat> aligned;      // 位置合わせ済みフレーム（8UC3）
    cv::Mat reference;                 // 参照像R（32FC3）
    double grainSigma = 0;
    std::vector<cv::Mat> dustMasks;    // 8U（255=欠陥）。無効時は empty
    std::vector<double> flickerOffsets;
    std::vector<bool> misaligned;
    cv::Mat motionGuard;               // 8U。グループ内で動いている画素（口パク等）
    bool integrationUnsafe = false;    // true=統合禁止（検出閾値以下の動きの累積）
};

// グループ内サブピクセル位置合わせ（ゲートウィーブ補正、MOTION_EUCLIDEAN）
std::vector<cv::Mat> alignGroupFrames(const std::vector<cv::Mat>& frames,
                                      const DenoiseParams& p = {});

cv::Mat computeReference(const std::vector<cv::Mat>& framesAligned,
                         const DenoiseParams& p = {});

// 解析（位置合わせ・R・σ・ダスト・フリッカー・モーションガード）
GroupAnalysis analyzeHoldGroup(const std::vector<cv::Mat>& frames,
                               const DenoiseParams& p = {});

// 出力生成。referenceOut に第2層で拡張したRを渡せる（empty可）
std::vector<cv::Mat> renderHoldGroup(const GroupAnalysis& analysis,
                                     const DenoiseParams& p = {},
                                     const cv::Mat& referenceOut = {});

// 線画・セル境界のエッジマスク（XDoG、255=線近傍。ダスト保護・第2層で共有）
cv::Mat lineArtEdgeMask(const cv::Mat& reference, int dilatePx = 3);

// 線画保護つき空間NR（フレーム単独で完結。Guided Filter / NLM＋Cannyエッジフォールバック）
cv::Mat spatialDenoiseEdgePreserving(const cv::Mat& frame, double grainSigma,
                                     double strength = 1.0,
                                     bool protectEdges = true,
                                     bool useNlm = false);

}  // namespace animerestore
