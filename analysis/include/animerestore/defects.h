// 欠陥検出：傷（縦スクラッチ）・ラインノイズ・スキャンノイズ
// （prototype/denoise/{scratch,linenoise,scannoise}.py の C++ 移植）
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace animerestore {

struct ScratchColumn {
    int x = 0;
    double coverage = 0;   // 縦方向被覆率
    double strength = 0;
};

struct LineNoise {
    int index = 0;         // 行/列番号（元画像座標）
    double offset = 0;     // ずれ量
    double uniformity = 0;
};

struct ScanNoise {
    double periodPx = 0;
    double amplitude = 0;
    double snr = 0;
    int bin = 0;
};

// 傷：複数グループの参照像（32FC3 または 8UC3）にわたり固定x位置に持続する
// 細い縦線を検出（グループ間の応答最小値）。静止カットでは絵柄と区別できない
// ため候補提示まで（除去はユーザー確認前提）
std::vector<ScratchColumn> detectScratchColumns(
    const std::vector<cv::Mat>& references,
    double responseThreshold = 6.0,
    double minColumnCoverage = 0.25,
    double activeAreaCrop = 0.10);

// ラインノイズ：行(axis=0)/列(axis=1)の輝度中央値プロファイルの外れ値検定
// ＋一様性チェック（本物は行全体が同符号で一様にずれる）
std::vector<LineNoise> detectLineNoise(const cv::Mat& reference, int axis = 0,
                                       double sigmaFactor = 5.0,
                                       double minOffset = 1.5,
                                       double uniformityRatio = 0.6,
                                       double activeAreaCrop = 0.10);
cv::Mat correctLineNoise(const cv::Mat& frame,
                         const std::vector<LineNoise>& detections,
                         int axis = 0, double strength = 1.0);

// スキャンノイズ：走査方向平均プロファイルのFFTスペクトルからスパイク検出。
// 補正はスパイク成分のみ逆FFTで差し引くノッチ方式
std::vector<ScanNoise> detectScanNoise(const cv::Mat& reference, int axis = 0,
                                       double spikeFactor = 8.0,
                                       double minPeriodPx = 2.0,
                                       double maxPeriodPx = 64.0,
                                       double minAmplitude = 0.3,
                                       double activeAreaCrop = 0.10);
cv::Mat correctScanNoise(const cv::Mat& frame, const cv::Mat& reference,
                         const std::vector<ScanNoise>& detections, int axis = 0,
                         double strength = 1.0, double activeAreaCrop = 0.10);

}  // namespace animerestore
