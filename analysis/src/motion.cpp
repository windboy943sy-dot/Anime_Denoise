// Phase 2：動き分類の C++ 実装。
// prototype/motion_classification/core.py を仕様として忠実に移植する。

#include "animerestore/motion.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace animerestore {

namespace {

// 解析用グレースケール縮小画像と、work座標→元解像度の倍率
std::pair<cv::Mat, double> prep(const cv::Mat& frameBgr, const MotionThresholds& th) {
    cv::Mat gray;
    cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);
    double scale = static_cast<double>(th.workWidth) / gray.cols;
    if (scale < 1.0) {
        cv::resize(gray, gray,
                   cv::Size(th.workWidth,
                            static_cast<int>(std::lround(gray.rows * scale))),
                   0, 0, cv::INTER_AREA);
    } else {
        scale = 1.0;
    }
    cv::GaussianBlur(gray, gray, cv::Size(th.blurKsize, th.blurKsize), 0);
    return {gray, 1.0 / scale};
}

// 2x3アフィンを (tx, ty, scale, rot_deg) に分解。スケールは行列式の平方根
// （ECCがシアーを含むフルアフィンを返しても安定）
void decompose(const cv::Mat& m, double& tx, double& ty, double& scale, double& rotDeg) {
    tx = m.at<float>(0, 2);
    ty = m.at<float>(1, 2);
    double det = m.at<float>(0, 0) * m.at<float>(1, 1) -
                 m.at<float>(0, 1) * m.at<float>(1, 0);
    scale = std::sqrt(std::max(det, 1e-12));
    rotDeg = std::atan2(m.at<float>(1, 0), m.at<float>(0, 0)) * 180.0 / M_PI;
}

// ±1pxの位置ずれを許容した残差マップ（アクティブ領域のみ）
cv::Mat tolerantResidualCropped(const cv::Mat& imgA, const cv::Mat& grayB,
                                const MotionThresholds& th) {
    cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
    cv::Mat dil, ero, over, under;
    cv::dilate(imgA, dil, k3);
    cv::erode(imgA, ero, k3);
    cv::subtract(grayB, dil, over, cv::noArray(), CV_32F);
    cv::subtract(ero, grayB, under, cv::noArray(), CV_32F);
    cv::Mat resid = cv::max(over, under);
    resid = cv::max(resid, 0.0f);
    int my = static_cast<int>(resid.rows * th.activeAreaCrop);
    int mx = static_cast<int>(resid.cols * th.activeAreaCrop);
    return resid(cv::Rect(mx, my, resid.cols - 2 * mx, resid.rows - 2 * my));
}

cv::Mat warpToB(const cv::Mat& grayA, const cv::Mat& warp) {
    cv::Mat out;
    cv::warpAffine(grayA, out, warp, grayA.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    return out;
}

double percentileOf(const cv::Mat& m, double p) {
    cv::Mat flat = m.reshape(1, 1).clone();
    flat.convertTo(flat, CV_32F);
    std::vector<float> v(flat.begin<float>(), flat.end<float>());
    size_t k = static_cast<size_t>(std::min<double>(
        v.size() - 1, std::max(0.0, p / 100.0 * (v.size() - 1))));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

}  // namespace

GlobalMotion estimateGlobalMotion(const cv::Mat& frameA, const cv::Mat& frameB,
                                  const MotionThresholds& th) {
    auto [grayA, toOrig] = prep(frameA, th);
    auto [grayB, toOrigB] = prep(frameB, th);
    (void)toOrigB;

    cv::Mat warp;
    std::string method = "orb_ransac";
    double confidence = 0.0;

    auto orb = cv::ORB::create(th.orbFeatures);
    std::vector<cv::KeyPoint> kpA, kpB;
    cv::Mat desA, desB;
    orb->detectAndCompute(grayA, cv::noArray(), kpA, desA);
    orb->detectAndCompute(grayB, cv::noArray(), kpB, desB);

    std::vector<cv::DMatch> matches;
    if (!desA.empty() && !desB.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING, true);
        matcher.match(desA, desB, matches);
        std::sort(matches.begin(), matches.end(),
                  [](const cv::DMatch& a, const cv::DMatch& b) {
                      return a.distance < b.distance;
                  });
    }

    std::vector<cv::Point2f> matchedA, matchedB;
    if (static_cast<int>(matches.size()) >= th.minMatches) {
        std::vector<cv::Point2f> ptsA, ptsB;
        for (const auto& m : matches) {
            ptsA.push_back(kpA[m.queryIdx].pt);
            ptsB.push_back(kpB[m.trainIdx].pt);
        }
        cv::Mat inliers;
        cv::Mat affine = cv::estimateAffinePartial2D(
            ptsA, ptsB, inliers, cv::RANSAC, th.ransacReprojThreshold);
        int inlierCount = cv::countNonZero(inliers);
        if (!affine.empty() && inlierCount >= th.minMatches / 2) {
            affine.convertTo(warp, CV_32F);
            confidence = static_cast<double>(inlierCount) / matches.size();
            matchedA = ptsA;
            matchedB = ptsB;
        }
    }

    if (warp.empty()) {
        // ベタ塗り等で特徴点不足：位相相関で並進のみ推定
        method = "phase_correlation";
        cv::Mat fa, fb;
        grayA.convertTo(fa, CV_32F);
        grayB.convertTo(fb, CV_32F);
        double response = 0.0;
        cv::Point2d shift = cv::phaseCorrelate(fa, fb, cv::noArray(), &response);
        warp = (cv::Mat_<float>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
        confidence = std::clamp(response, 0.0, 1.0);
    }

    // 分類はECC前のワープで行う（マルチプレーンではORB=前景基準が
    // 人間のラベルと一致しやすい）
    double tx, ty, scale, rot;
    decompose(warp, tx, ty, scale, rot);
    cv::Mat warpPreEcc = warp.clone();

    // ECC前ワープの残差マップは ECCゲート・改善チェック・面積対決の3箇所で
    // 使うため一度だけ計算する（数値は従来と完全同一）
    cv::Mat residPreMap = tolerantResidualCropped(warpToB(grayA, warpPreEcc),
                                                  grayB, th);

    // ECC精密化：残差が実際に改善した場合のみ採用
    // （グレインの強いパン素材でECCが発散しワープを壊す実測事例への対策）
    if (th.useEccRefinement) {
        try {
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                      th.eccIterations, 1e-5);
            cv::Mat warpEcc = warp.clone();
            double cc = cv::findTransformECC(grayB, grayA, warpEcc,
                                             cv::MOTION_AFFINE, criteria,
                                             cv::noArray(), 5);
            double residEcc = cv::mean(
                tolerantResidualCropped(warpToB(grayA, warpEcc), grayB, th))[0];
            double residPre = cv::mean(residPreMap)[0];
            if (residEcc <= residPre) {
                warp = warpEcc;
                method += "+ecc";
                confidence = std::max(confidence, std::clamp(cc, 0.0, 1.0));
            }
        } catch (const cv::Exception&) {
            // 収束失敗時はORB/位相相関の結果をそのまま使う
        }
    }

    // 分類（優先順位：回転 > ズーム > パン > 静止）
    double translation = std::hypot(tx, ty);
    std::string type;
    if (std::abs(rot) >= th.rotationDeg) type = "rotation";
    else if (std::abs(scale - 1.0) >= th.zoomScaleDelta) type = "zoom";
    else if (translation >= th.staticTranslationPx) type = "pan";
    else type = "static";

    // ワープ改善チェック：「改善した画素数 vs 新たに壊した画素数」の対決。
    // 静止カメラ＋大面積キャラ芝居のORB誤フィット対策（実測で主因）。
    // 平均残差比較はグレインが支配して機能しない
    if (type != "static") {
        cv::Mat residId = tolerantResidualCropped(grayA, grayB, th);
        const float t = 4.0f;
        cv::Mat idHigh = residId > t, warpHigh = residPreMap > t;
        double improved = cv::countNonZero(idHigh & ~warpHigh);
        double broken = cv::countNonZero(~idHigh & warpHigh);
        if (broken >= improved) type = "static";
    }

    // マルチプレーン対策（v4確定構成 = 評価29/30の最良値。Python側と同値）：
    //  (1) 非等方性 ratio<0.75 → 視差、(2) 面積対決 → 並進が同等以上なら視差。
    // 既知の限界：真の超低速ズーム＋キャラ動き（02_Zoom_02）は pan と誤答
    // （分類ラベルのみの問題でデノイズ品質には影響しない）
    double scaleDevRatio = -1;
    if (type == "zoom" || type == "rotation") {
        bool isParallax = false;
        if (!matchedA.empty()) {
            cv::Mat inliersFull;
            cv::Mat aFull = cv::estimateAffine2D(matchedA, matchedB, inliersFull,
                                                 cv::RANSAC, th.ransacReprojThreshold);
            if (!aFull.empty()) {
                double sx = std::hypot(aFull.at<double>(0, 0), aFull.at<double>(1, 0));
                double sy = std::hypot(aFull.at<double>(0, 1), aFull.at<double>(1, 1));
                double devLo = std::min(std::abs(sx - 1.0), std::abs(sy - 1.0));
                double devHi = std::max(std::abs(sx - 1.0), std::abs(sy - 1.0));
                if (devHi > 0.005) {
                    scaleDevRatio = devLo / devHi;
                    if (scaleDevRatio < 0.75) isParallax = true;
                }
            }
        }
        cv::Mat fa, fb;
        grayA.convertTo(fa, CV_32F);
        grayB.convertTo(fb, CV_32F);
        cv::Point2d shift = cv::phaseCorrelate(fa, fb);
        if (!isParallax) {
            cv::Mat warpTrans =
                (cv::Mat_<float>(2, 3) << 1, 0, shift.x, 0, 1, shift.y);
            const float t = 4.0f;
            double areaModel = cv::countNonZero(residPreMap <= t);
            double areaTrans = cv::countNonZero(
                tolerantResidualCropped(warpToB(grayA, warpTrans), grayB, th) <= t);
            isParallax = areaTrans >= areaModel;
        }
        if (isParallax) {
            if (std::hypot(shift.x, shift.y) >= th.staticTranslationPx) {
                type = "pan";
                tx = shift.x;
                ty = shift.y;
                scale = 1.0;
                rot = 0.0;
            } else {
                type = "static";
            }
        }
    }

    GlobalMotion g;
    g.type = type;
    g.tx = tx * toOrig;
    g.ty = ty * toOrig;
    g.scale = scale;
    g.rotationDeg = rot;
    g.method = method;
    g.confidence = confidence;
    g.warpWork = warp;
    g.scaleDevRatio = scaleDevRatio;
    return g;
}

double estimateNoiseFloor(const std::vector<std::pair<cv::Mat, cv::Mat>>& pairs,
                          const MotionThresholds& th) {
    std::vector<double> samples;
    for (const auto& [fa, fb] : pairs) {
        GlobalMotion g = estimateGlobalMotion(fa, fb, th);
        auto [grayA, s] = prep(fa, th);
        auto [grayB, s2] = prep(fb, th);
        (void)s; (void)s2;
        cv::Mat imgA = g.warpWork.empty() ? grayA : warpToB(grayA, g.warpWork);
        cv::Mat resid = tolerantResidualCropped(imgA, grayB, th);
        cv::Mat blurred;
        cv::GaussianBlur(resid, blurred, cv::Size(5, 5), 0);
        samples.push_back(percentileOf(blurred, th.noiseFloorPercentile));
    }
    if (samples.empty()) return th.residualFloor;
    std::nth_element(samples.begin(), samples.begin() + samples.size() / 2,
                     samples.end());
    return std::max(th.residualFloor, samples[samples.size() / 2]);
}

LocalMotion classifyLocalMotion(const cv::Mat& frameA, const cv::Mat& frameB,
                                const GlobalMotion& globalMotion,
                                double noiseFloor, const MotionThresholds& th) {
    auto [grayA, toOrig] = prep(frameA, th);
    auto [grayB, s2] = prep(frameB, th);
    (void)s2;

    cv::Mat imgA = globalMotion.warpWork.empty()
                       ? grayA
                       : warpToB(grayA, globalMotion.warpWork);
    cv::Mat resid = tolerantResidualCropped(imgA, grayB, th);
    cv::Mat blurred;
    cv::GaussianBlur(resid, blurred, cv::Size(5, 5), 0);

    cv::Mat moving = blurred > noiseFloor;
    cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
    cv::morphologyEx(moving, moving, cv::MORPH_OPEN, k3);

    LocalMotion lm;
    lm.noiseFloor = noiseFloor;
    lm.movingRatio = static_cast<double>(cv::countNonZero(moving)) / moving.total();

    if (cv::countNonZero(moving) > 0) {
        cv::Mat merged;
        cv::morphologyEx(moving, merged, cv::MORPH_CLOSE,
                         cv::Mat::ones(9, 9, CV_8U));
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(merged, labels, stats,
                                                       centroids, 8);
        int bestIdx = -1, bestArea = 0, totalArea = 0;
        for (int j = 1; j < nLabels; ++j) {
            int area = stats.at<int>(j, cv::CC_STAT_AREA);
            totalArea += area;
            if (area > bestArea) { bestArea = area; bestIdx = j; }
        }
        if (bestIdx > 0) {
            lm.largestComponentRatio =
                static_cast<double>(bestArea) / std::max(1, totalArea);
            int offX = static_cast<int>(grayA.cols * th.activeAreaCrop);
            int offY = static_cast<int>(grayA.rows * th.activeAreaCrop);
            lm.bbox = cv::Rect(
                static_cast<int>((stats.at<int>(bestIdx, cv::CC_STAT_LEFT) + offX) * toOrig),
                static_cast<int>((stats.at<int>(bestIdx, cv::CC_STAT_TOP) + offY) * toOrig),
                static_cast<int>(stats.at<int>(bestIdx, cv::CC_STAT_WIDTH) * toOrig),
                static_cast<int>(stats.at<int>(bestIdx, cv::CC_STAT_HEIGHT) * toOrig));
        }
    }

    if (lm.movingRatio < th.movingRatioNone) lm.type = "none";
    else if (lm.movingRatio >= th.movingRatioFull) lm.type = "full";
    else lm.type = "local";
    return lm;
}

CameraPath analyzeCameraPath(const std::vector<GlobalMotion>& motions,
                             double minTranslationPx) {
    CameraPath cp;
    std::vector<double> mags;
    double sumX = 0, sumY = 0;
    for (const auto& m : motions) {
        double mag = std::hypot(m.tx, m.ty);
        mags.push_back(mag);
        if (mag >= minTranslationPx) {
            sumX += m.tx / mag;
            sumY += m.ty / mag;
            cp.movingTransitions++;
        }
    }
    if (cp.movingTransitions >= 3) {
        cp.directionConsistency =
            std::hypot(sumX, sumY) / cp.movingTransitions;
    }
    if (!mags.empty()) {
        std::nth_element(mags.begin(), mags.begin() + mags.size() / 2, mags.end());
        cp.medianTranslationPx = mags[mags.size() / 2];
    }
    cp.cameraShake = cp.movingTransitions >= 3 && cp.directionConsistency < 0.5;
    return cp;
}

std::string dominantGlobalMotion(const std::vector<GlobalMotion>& motions) {
    if (motions.empty()) return "unknown";
    std::map<std::string, int> counts;
    for (const auto& m : motions) counts[m.type]++;

    int nonStaticTotal = 0, best = 0;
    std::string dominant;
    for (const auto& [type, c] : counts) {
        if (type == "static") continue;
        nonStaticTotal += c;
        if (c > best) { best = c; dominant = type; }
    }
    if (nonStaticTotal >= std::max<int>(2, static_cast<int>(motions.size() * 0.25)) &&
        !dominant.empty()) {
        if (dominant == "pan" && analyzeCameraPath(motions).cameraShake) {
            return "static";  // 方向が一貫しない並進＝シェイク（フィックス扱い）
        }
        return dominant;
    }
    // 最頻値
    std::string top;
    int topCount = 0;
    for (const auto& [type, c] : counts)
        if (c > topCount) { topCount = c; top = type; }
    return top;
}

}  // namespace animerestore
