// 第2層：カット間拡張統合の C++ 実装。
// prototype/denoise/extend.py を仕様として忠実に移植する。

#include "animerestore/extend.h"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "animerestore/motion.h"

namespace animerestore {

namespace {

cv::Mat lumaU8(const cv::Mat& img32f3) {
    cv::Mat u8, gray;
    img32f3.convertTo(u8, CV_8UC3);
    cv::cvtColor(u8, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// ±1pxの位置ずれを許容した輝度残差（同座標系の2画像）
cv::Mat tolerantLumaResidual(const cv::Mat& a32f3, const cv::Mat& b32f3) {
    cv::Mat ya = lumaU8(a32f3), yb = lumaU8(b32f3);
    cv::Mat k3 = cv::Mat::ones(3, 3, CV_8U);
    cv::Mat dil, ero, over, under;
    cv::dilate(ya, dil, k3);
    cv::erode(ya, ero, k3);
    cv::subtract(yb, dil, over, cv::noArray(), CV_32F);
    cv::subtract(ero, yb, under, cv::noArray(), CV_32F);
    cv::Mat resid = cv::max(cv::max(over, under), 0.0f);
    cv::GaussianBlur(resid, resid, cv::Size(5, 5), 0);
    return resid;
}

}  // namespace

ExtendResult extendReference(const GroupAnalysis& center,
                             const std::vector<const GroupAnalysis*>& neighbors,
                             const ExtendParams& p) {
    const cv::Mat& refC = center.reference;
    const double nC = static_cast<double>(center.aligned.size());
    const double sigmaC = std::max(center.grainSigma, 0.5);
    const int h = refC.rows, w = refC.cols;

    ExtendResult r;
    cv::Mat acc = refC * nC;
    r.effectiveN = cv::Mat(h, w, CV_32F, cv::Scalar(nC));

    // エッジ除外マスクは denoise 側の XDoG 実装を共有（Python extend.py と同値。
    // 従来ここだけ Canny のままでパリティが逸脱していた＝監査で発見・修正）
    cv::Mat notEdge = ~lineArtEdgeMask(refC, 5);

    int my = static_cast<int>(h * p.activeAreaCrop);
    int mx = static_cast<int>(w * p.activeAreaCrop);
    cv::Rect inner(mx, my, w - 2 * mx, h - 2 * my);

    MotionThresholds mth;
    cv::Mat refCu8;
    refC.convertTo(refCu8, CV_8UC3);

    for (const GroupAnalysis* nb : neighbors) {
        const cv::Mat& refN = nb->reference;
        const double nN = static_cast<double>(nb->aligned.size());

        cv::Mat refNu8;
        refN.convertTo(refNu8, CV_8UC3);
        GlobalMotion g = estimateGlobalMotion(refNu8, refCu8, mth);
        if (g.warpWork.empty()) {
            r.acceptRatios.push_back(0.0);
            continue;
        }
        // workスケール→元解像度：並進のみスケール
        cv::Mat warp = g.warpWork.clone();
        double toOrig = static_cast<double>(w) / mth.workWidth;
        if (toOrig > 1.0) {
            warp.at<float>(0, 2) *= static_cast<float>(toOrig);
            warp.at<float>(1, 2) *= static_cast<float>(toOrig);
        }
        double translation = std::hypot(warp.at<float>(0, 2), warp.at<float>(1, 2));
        if (g.confidence < p.minWarpConfidence ||
            translation > p.maxTranslationRatio * w) {
            r.acceptRatios.push_back(0.0);
            continue;
        }

        cv::Mat warped;
        cv::warpAffine(refN, warped, warp, cv::Size(w, h), cv::INTER_LANCZOS4,
                       cv::BORDER_CONSTANT, cv::Scalar(-1000, -1000, -1000));
        cv::Mat channels[3];
        cv::split(warped, channels);
        cv::Mat valid = channels[0] > -500.0f;

        cv::Mat warpedClip;
        cv::max(warped, 0.0f, warpedClip);
        cv::min(warpedClip, 255.0f, warpedClip);
        cv::Mat resid = tolerantLumaResidual(warpedClip, refC);
        double sigmaPair = sigmaC * std::sqrt(1.0 / std::max(nC, 1.0) +
                                              1.0 / std::max(nN, 1.0));
        double threshold = std::max(p.acceptFloor, p.acceptSigmaFactor * sigmaPair);

        cv::Mat accept = (resid <= threshold) & valid & notEdge;
        cv::erode(accept, accept, cv::Mat::ones(3, 3, CV_8U));

        double ratio = static_cast<double>(cv::countNonZero(accept(inner))) /
                       inner.area();
        r.acceptRatios.push_back(ratio);
        if (ratio < p.minAcceptRatio) continue;

        cv::Mat aW;
        accept.convertTo(aW, CV_32F, nN / 255.0);
        cv::Mat aW3;
        cv::cvtColor(aW, aW3, cv::COLOR_GRAY2BGR);
        acc += warpedClip.mul(aW3);
        r.effectiveN += aW;
        r.usedNeighbors++;
    }

    cv::Mat n3;
    cv::cvtColor(r.effectiveN, n3, cv::COLOR_GRAY2BGR);
    r.reference = acc / n3;
    return r;
}

cv::Mat blendSpatialFallback(const cv::Mat& image, const cv::Mat& effectiveN,
                             double grainSigma, double strength) {
    if (strength <= 0) return image;
    cv::Mat u8;
    if (image.type() == CV_8UC3) u8 = image;
    else image.convertTo(u8, CV_8UC3);

    cv::Mat denoised = spatialDenoiseEdgePreserving(u8, grainSigma, strength);

    cv::Mat w;
    cv::max(effectiveN, 1.0f, w);
    cv::divide(1.0, w, w);
    cv::GaussianBlur(w, w, cv::Size(31, 31), 0);
    cv::Mat w3;
    cv::cvtColor(w, w3, cv::COLOR_GRAY2BGR);

    cv::Mat fD, fO;
    denoised.convertTo(fD, CV_32FC3);
    u8.convertTo(fO, CV_32FC3);
    cv::Mat out = fD.mul(w3) + fO.mul(cv::Scalar::all(1.0) - w3);
    out.convertTo(out, CV_8UC3);
    return out;
}

}  // namespace animerestore
