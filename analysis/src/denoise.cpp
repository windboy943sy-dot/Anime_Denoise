// Phase 3：デノイズコアの C++ 実装。
#include <cstdlib>
// prototype/denoise/core.py を仕様として忠実に移植する。

#include "animerestore/denoise.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video/tracking.hpp>

namespace animerestore {

namespace {

cv::Mat luma(const cv::Mat& frame) {
    cv::Mat gray;
    if (frame.channels() == 1) {
        frame.convertTo(gray, CV_32F);
    } else {
        cv::Mat f32;
        frame.convertTo(f32, CV_32FC3);
        cv::cvtColor(f32, gray, cv::COLOR_BGR2GRAY);
    }
    return gray;
}

double medianOf(cv::Mat m) {
    m = m.reshape(1, 1).clone();
    m.convertTo(m, CV_32F);
    std::vector<float> v(m.begin<float>(), m.end<float>());
    size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

}  // namespace

// 線画・セル境界のエッジマスク（Rから取る：生フレームだとグレインが
// エッジ扱いになり保護マスクが画面全体に広がる）。
// 実装は XDoG（Python: denoise/lineart.py と同値）。Canny は低コントラストの
// 縞・グラデーション境界を落とし弧状誤検出の遠因になった（実測）。
// extend（第2層のエッジ除外）とも共有するため公開シンボルにしている
cv::Mat lineArtEdgeMask(const cv::Mat& reference, int dilatePx) {
    const double sigma = 1.4, k = 1.6, tau = 0.98, epsilon = -0.3;
    cv::Mat gray = luma(reference);
    cv::Mat g1, g2;
    cv::GaussianBlur(gray, g1, cv::Size(0, 0), sigma);
    cv::GaussianBlur(gray, g2, cv::Size(0, 0), sigma * k);
    cv::Mat u = g1 - tau * g2;
    cv::Mat mask = (u < epsilon);

    // グレイン由来の孤立斑点を除去（線は連結して長い）
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(2, 2, CV_8U));
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    // 線画は数百〜数千成分になるため、成分ごとの setTo（成分数×全画素の走査）
    // ではなくラベルLUTの一括適用にする（結果はビット同一、O(HW)）
    std::vector<uint8_t> keep(n, 0);
    for (int j = 1; j < n; ++j)
        if (stats.at<int>(j, cv::CC_STAT_AREA) >= 12) keep[j] = 255;
    cv::Mat out(mask.size(), CV_8U);
    for (int y = 0; y < out.rows; ++y) {
        const int32_t* lb = labels.ptr<int32_t>(y);
        uint8_t* o = out.ptr<uint8_t>(y);
        for (int x = 0; x < out.cols; ++x) o[x] = keep[lb[x]];
    }

    if (dilatePx > 0)
        cv::dilate(out, out, cv::Mat::ones(dilatePx, dilatePx, CV_8U));
    return out;
}

std::vector<cv::Mat> alignGroupFrames(const std::vector<cv::Mat>& frames,
                                      const DenoiseParams& p) {
    if (frames.size() < 2 || !p.align) return frames;

    int refIndex = static_cast<int>(frames.size()) / 2;
    double scale = static_cast<double>(p.alignWorkWidth) / frames[0].cols;
    cv::Size smallSize(p.alignWorkWidth,
                       static_cast<int>(std::lround(frames[0].rows * scale)));

    auto smallGray = [&](const cv::Mat& f) {
        cv::Mat g;
        cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
        cv::resize(g, g, smallSize, 0, 0, cv::INTER_AREA);
        cv::GaussianBlur(g, g, cv::Size(5, 5), 0);
        return g;
    };

    cv::Mat refSmall = smallGray(frames[refIndex]);
    std::vector<cv::Mat> aligned;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              p.eccIterations, 1e-5);

    for (size_t i = 0; i < frames.size(); ++i) {
        if (static_cast<int>(i) == refIndex) {
            aligned.push_back(frames[i]);
            continue;
        }
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        try {
            cv::findTransformECC(refSmall, smallGray(frames[i]), warp,
                                 cv::MOTION_EUCLIDEAN, criteria, cv::noArray(), 5);
        } catch (const cv::Exception&) {
            aligned.push_back(frames[i]);  // 収束失敗：位置合わせなし（安全側）
            continue;
        }
        // 並進成分を元解像度へ換算
        warp.at<float>(0, 2) /= static_cast<float>(scale);
        warp.at<float>(1, 2) /= static_cast<float>(scale);
        cv::Mat out;
        cv::warpAffine(frames[i], out, warp, frames[i].size(),
                       cv::INTER_LANCZOS4, cv::BORDER_REPLICATE);
        aligned.push_back(out);
    }
    return aligned;
}

cv::Mat computeReference(const std::vector<cv::Mat>& framesAligned,
                         const DenoiseParams& p) {
    const int n = static_cast<int>(framesAligned.size());
    const int rows = framesAligned[0].rows, cols = framesAligned[0].cols;

    ReferenceMethod method = p.referenceMethod;
    int k = static_cast<int>(n * p.trimRatio);
    if (method == ReferenceMethod::TrimmedMean && (k == 0 || n - 2 * k < 1))
        method = ReferenceMethod::Median;
    if (n <= 2) method = ReferenceMethod::Median;

    if (method == ReferenceMethod::Mean) {
        cv::Mat acc = cv::Mat::zeros(rows, cols, CV_32FC3);
        for (const auto& f : framesAligned) {
            cv::Mat f32;
            f.convertTo(f32, CV_32FC3);
            acc += f32;
        }
        return acc / n;
    }

    // median / trimmed_mean：画素・チャンネルごとにソートして集計。
    // 行単位で処理してメモリを抑える（2560x1920でもワークは1行分×N）
    cv::Mat result(rows, cols, CV_32FC3);
    std::vector<float> vals(n);
    for (int y = 0; y < rows; ++y) {
        std::vector<const cv::Vec3b*> rowPtrs(n);
        for (int i = 0; i < n; ++i) rowPtrs[i] = framesAligned[i].ptr<cv::Vec3b>(y);
        cv::Vec3f* dst = result.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < n; ++i) vals[i] = rowPtrs[i][x][c];
                std::sort(vals.begin(), vals.end());
                if (method == ReferenceMethod::Median) {
                    dst[x][c] = (n % 2) ? vals[n / 2]
                                        : (vals[n / 2 - 1] + vals[n / 2]) * 0.5f;
                } else {  // trimmed mean
                    float s = std::accumulate(vals.begin() + k, vals.end() - k, 0.0f);
                    dst[x][c] = s / (n - 2 * k);
                }
            }
        }
    }
    return result;
}

cv::Mat spatialDenoiseEdgePreserving(const cv::Mat& frame, double grainSigma,
                                     double strength, bool protectEdges) {
    if (strength <= 0) return frame;
    float h = static_cast<float>(
        std::clamp(grainSigma * 1.2 * strength, 1.0, 15.0));
    cv::Mat denoised;
    cv::fastNlMeansDenoisingColored(frame, denoised, h, h, 7, 21);

    if (!protectEdges) return denoised;

    cv::Mat gray, edges;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Canny(gray, edges, 50, 150);
    cv::dilate(edges, edges, cv::Mat::ones(3, 3, CV_8U));
    cv::Mat w;
    edges.convertTo(w, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(w, w, cv::Size(5, 5), 0);

    cv::Mat fD, fO, w3, out;
    denoised.convertTo(fD, CV_32FC3);
    frame.convertTo(fO, CV_32FC3);
    cv::cvtColor(w, w3, cv::COLOR_GRAY2BGR);
    out = fD.mul(cv::Scalar::all(1.0) - w3) + fO.mul(w3);
    out.convertTo(out, CV_8UC3);
    return out;
}

GroupAnalysis analyzeHoldGroup(const std::vector<cv::Mat>& frames,
                               const DenoiseParams& p) {
    GroupAnalysis a;
    a.aligned = alignGroupFrames(frames, p);
    a.reference = computeReference(a.aligned, p);
    const int n = static_cast<int>(a.aligned.size());

    // フリッカー推定（補正ONなら正規化してRを作り直す）。
    // ここで計算する luma は補正が入らなければ後段でそのまま再利用する
    cv::Mat refY = luma(a.reference);
    std::vector<cv::Mat> lumas(n);
    for (int i = 0; i < n; ++i) {
        lumas[i] = luma(a.aligned[i]);
        a.flickerOffsets.push_back(medianOf(lumas[i] - refY));
    }
    bool anyFlicker = std::any_of(a.flickerOffsets.begin(), a.flickerOffsets.end(),
                                  [](double o) { return std::abs(o) > 0.25; });
    if (p.flickerCorrection && anyFlicker) {
        for (int i = 0; i < n; ++i) {
            cv::Mat f32;
            a.aligned[i].convertTo(f32, CV_32FC3);
            f32 -= cv::Scalar::all(a.flickerOffsets[i]);
            f32.convertTo(a.aligned[i], CV_8UC3);
        }
        a.reference = computeReference(a.aligned, p);
        refY = luma(a.reference);
        for (int i = 0; i < n; ++i) lumas[i] = luma(a.aligned[i]);  // 補正後に再計算
    }

    // 輝度残差スタックとグレインσ（MAD×1.4826、半正規分布のスケール推定）
    std::vector<cv::Mat> residual(n);
    for (int i = 0; i < n; ++i)
        residual[i] = cv::abs(lumas[i] - refY);
    {
        cv::Mat all;
        cv::vconcat(residual, all);
        a.grainSigma = std::max(medianOf(all) * 1.4826, 0.5);
    }

    // 統合安全ガード（グループ全体）：端点フレーム同士の「エッジ画素上の」残差が
    // 大きいグループは、検出閾値以下の連続的な動き（超低速ズーム等）が累積して
    // おり、統合するとエッジが二重化する（実測＝02_Zoom_02）。
    // 閾値は実測8ケースで較正：統合可 ≤4.1σ、不可 ≥6.2σ（Python側と同値）
    if (n >= 2) {
        cv::Mat midU8, guardEdges;
        lumas[n / 2].convertTo(midU8, CV_8U);
        cv::Canny(midU8, guardEdges, 50, 150);
        if (cv::countNonZero(guardEdges) > 0) {
            // 端±1を除いた内側端点で判定（フィルムの局所歪みは端フレームに
            // 集中するため。モーションガードと同じ対処）
            int ia = (n >= 4) ? 1 : 0;
            int ib = (n >= 4) ? n - 2 : n - 1;
            cv::Mat endA, endB;
            cv::GaussianBlur(lumas[ia], endA, cv::Size(3, 3), 0);
            cv::GaussianBlur(lumas[ib], endB, cv::Size(3, 3), 0);
            cv::Mat diff = cv::abs(endA - endB);
            std::vector<float> vals;
            for (int y = 0; y < diff.rows; ++y) {
                const uint8_t* e = guardEdges.ptr<uint8_t>(y);
                const float* d = diff.ptr<float>(y);
                for (int x = 0; x < diff.cols; ++x)
                    if (e[x]) vals.push_back(d[x]);
            }
            size_t k = vals.size() / 2;
            std::nth_element(vals.begin(), vals.begin() + k, vals.end());
            a.integrationUnsafe = vals[k] > std::max(6.0, 5.0 * a.grainSigma);
        }
    }

    // 画素単位モーションガード（微細な口パク等の芝居消失防止）：
    // 内部フレームの25%以上が変動 ＋ フィルム枠除外 ＋ 面積≥300
    const int rows = refY.rows, cols = refY.cols;
    a.motionGuard = cv::Mat::zeros(rows, cols, CV_8U);
    if (n >= 5) {
        double guardThreshold = std::max(6.0, 4.0 * a.grainSigma);
        cv::Mat count = cv::Mat::zeros(rows, cols, CV_32F);
        for (int i = 1; i < n - 1; ++i)
            cv::add(count, residual[i] > guardThreshold, count, cv::noArray(), CV_32F);
        count /= 255.0;
        cv::Mat guard = (count / (n - 2)) >= 0.25;

        int my = static_cast<int>(rows * p.dustActiveAreaCrop);
        int mx = static_cast<int>(cols * p.dustActiveAreaCrop);
        cv::Mat border = cv::Mat::zeros(rows, cols, CV_8U);
        border(cv::Rect(mx, my, cols - 2 * mx, rows - 2 * my)) = 255;
        guard &= border;

        cv::morphologyEx(guard, guard, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
        cv::Mat labels, stats, centroids;
        int nl = cv::connectedComponentsWithStats(guard, labels, stats, centroids, 8);
        cv::Mat filtered = cv::Mat::zeros(rows, cols, CV_8U);
        for (int j = 1; j < nl; ++j)
            if (stats.at<int>(j, cv::CC_STAT_AREA) >= 300)
                filtered.setTo(255, labels == j);
        if (cv::countNonZero(filtered) > 0) {
            cv::morphologyEx(filtered, filtered, cv::MORPH_CLOSE,
                             cv::Mat::ones(9, 9, CV_8U));
            cv::dilate(filtered, filtered, cv::Mat::ones(7, 7, CV_8U));
        }
        a.motionGuard = filtered;
    }

    // 位置合わせ品質（残差中央値がグループ標準の misalignFactor 倍超）
    std::vector<double> frameMed(n);
    for (int i = 0; i < n; ++i) frameMed[i] = medianOf(residual[i]);
    std::vector<double> sorted = frameMed;
    std::nth_element(sorted.begin(), sorted.begin() + n / 2, sorted.end());
    double groupMed = sorted[n / 2];
    for (int i = 0; i < n; ++i)
        a.misaligned.push_back(
            frameMed[i] > std::max(2.0, p.misalignFactor * std::max(groupMed, 0.1)));

    // ダスト検出（振幅・時間的単発性・形状・場所の4条件AND）
    a.dustMasks.assign(n, cv::Mat());
    if (p.dustDetection && n >= 2) {
        double threshold = std::max(8.0, p.dustSigma * a.grainSigma);
        cv::Mat edgeMask, gradMap;
        if (p.dustProtectEdges) {
            edgeMask = lineArtEdgeMask(a.reference);
            cv::Mat refBlur, gx, gy;
            cv::GaussianBlur(refY, refBlur, cv::Size(5, 5), 0);
            cv::Sobel(refBlur, gx, CV_32F, 1, 0, 3);
            cv::Sobel(refBlur, gy, CV_32F, 0, 1, 3);
            cv::magnitude(gx, gy, gradMap);
        }
        int my = static_cast<int>(rows * p.dustActiveAreaCrop);
        int mx = static_cast<int>(cols * p.dustActiveAreaCrop);
        cv::Mat border = cv::Mat::zeros(rows, cols, CV_8U);
        border(cv::Rect(mx, my, cols - 2 * mx, rows - 2 * my)) = 255;
        int maxArea = static_cast<int>(rows * cols * p.dustMaxAreaRatio);

        for (int i = 0; i < n; ++i) {
            double thrI = threshold * (a.misaligned[i] ? 2.0 : 1.0);
            // 他フレーム残差の中央値（時間的単発性の判定）
            cv::Mat othersMed = cv::Mat::zeros(rows, cols, CV_32F);
            {
                std::vector<cv::Mat> others;
                for (int j = 0; j < n; ++j)
                    if (j != i) others.push_back(residual[j]);
                // 中央値の近似としてソート済みマージは重いので、
                // ここは正確に：行単位で計算
                // 偶数個の中央値は np.median と同じく「中央2値の平均」にする
                // （nth_elementで上側を取ると条件が過度に厳しくなり、実測で
                // ダスト円盤の半分が欠けて形状フィルタを通らなくなった）
                std::vector<float> vals(others.size());
                const size_t m = others.size();
                for (int y = 0; y < rows; ++y) {
                    float* dst = othersMed.ptr<float>(y);
                    std::vector<const float*> ptrs(m);
                    for (size_t j = 0; j < m; ++j)
                        ptrs[j] = others[j].ptr<float>(y);
                    for (int x = 0; x < cols; ++x) {
                        for (size_t j = 0; j < m; ++j) vals[j] = ptrs[j][x];
                        std::sort(vals.begin(), vals.end());
                        dst[x] = (m % 2) ? vals[m / 2]
                                         : (vals[m / 2 - 1] + vals[m / 2]) * 0.5f;
                    }
                }
            }
            cv::Mat cand = (residual[i] > thrI) & (othersMed < threshold * 0.4) & border;
            // AR_DEBUG=<dir> で中間マスクをダンプ（パリティ調査用）
            if (const char* dbg = getenv("AR_DEBUG")) {
                std::fprintf(stderr, "[dust f%d] thr=%.1f amp=%d uniq=%d cand=%d\n", i,
                             thrI, cv::countNonZero(residual[i] > thrI),
                             cv::countNonZero(othersMed < threshold * 0.4),
                             cv::countNonZero(cand));
                cv::imwrite(std::string(dbg) + "/cand_" + std::to_string(i) + ".png",
                            cand);
            }
            if (!edgeMask.empty()) {
                cv::Mat strong =
                    (residual[i] > thrI * 2.0) & (othersMed < threshold * 0.4) & border;
                strong.copyTo(cand, edgeMask);
            }
            cv::morphologyEx(cand, cand, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));
            cv::morphologyEx(cand, cand, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8U));
            if (getenv("AR_DEBUG"))
                std::fprintf(stderr, "[dust f%d] after morph=%d\n", i,
                             cv::countNonZero(cand));

            cv::Mat labels, stats, centroids;
            int nl = cv::connectedComponentsWithStats(cand, labels, stats, centroids, 8);
            cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8U);
            for (int j = 1; j < nl; ++j) {
                int area = stats.at<int>(j, cv::CC_STAT_AREA);
                if (area < p.dustMinArea || area > maxArea) continue;
                int bw = stats.at<int>(j, cv::CC_STAT_WIDTH);
                int bh = stats.at<int>(j, cv::CC_STAT_HEIGHT);
                double elong = static_cast<double>(std::max(bw, bh)) /
                               std::max(1, std::min(bw, bh));
                double fill = static_cast<double>(area) / std::max(1, bw * bh);
                if (elong > 4.0 && fill < 0.4) continue;  // 細長い＝位置ずれ/傷
                if (!gradMap.empty()) {
                    cv::Mat comp = (labels == j);
                    cv::Mat gradVals;
                    gradMap.copyTo(gradVals, comp);
                    // 成分下の勾配中央値が高い候補は棄却（局所歪みの偽残差）
                    std::vector<float> gv;
                    for (int y = stats.at<int>(j, cv::CC_STAT_TOP);
                         y < stats.at<int>(j, cv::CC_STAT_TOP) + bh; ++y) {
                        const uint8_t* cp = comp.ptr<uint8_t>(y);
                        const float* gp = gradMap.ptr<float>(y);
                        for (int x = stats.at<int>(j, cv::CC_STAT_LEFT);
                             x < stats.at<int>(j, cv::CC_STAT_LEFT) + bw; ++x)
                            if (cp[x]) gv.push_back(gp[x]);
                    }
                    if (!gv.empty()) {
                        size_t k = gv.size() / 2;
                        std::nth_element(gv.begin(), gv.begin() + k, gv.end());
                        if (gv[k] > 10.0f) continue;
                    }
                }
                mask.setTo(255, labels == j);
            }
            cv::dilate(mask, mask, cv::Mat::ones(3, 3, CV_8U));
            a.dustMasks[i] = mask;
        }
    }
    return a;
}

std::vector<cv::Mat> renderHoldGroup(const GroupAnalysis& analysis,
                                     const DenoiseParams& p,
                                     const cv::Mat& referenceOut) {
    const cv::Mat& reference =
        referenceOut.empty() ? analysis.reference : referenceOut;
    const int n = static_cast<int>(analysis.aligned.size());

    cv::Mat guardW;
    if (cv::countNonZero(analysis.motionGuard) > 0) {
        analysis.motionGuard.convertTo(guardW, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(guardW, guardW, cv::Size(15, 15), 0);
        cv::cvtColor(guardW, guardW, cv::COLOR_GRAY2BGR);
    }

    std::vector<cv::Mat> outputs;
    for (int i = 0; i < n; ++i) {
        const cv::Mat& f = analysis.aligned[i];
        cv::Mat out;

        if (analysis.integrationUnsafe) {
            // 統合禁止グループ：フレーム自身を出力（グレイン低減のみ任意適用）。
            // Rは信頼できないため欠陥補正も行わない（保守的判定）
            out = f;
            if (p.grainReduction > 0)
                out = spatialDenoiseEdgePreserving(f, analysis.grainSigma,
                                                   p.grainReduction);
            outputs.push_back(out);
            continue;
        }

        if (p.mode == DenoiseMode::FullTemporalIntegration) {
            cv::Mat o = reference.clone();
            bool feather = p.featherBoundaryFrames && n >= 3 && (i == 0 || i == n - 1);
            if (feather || analysis.misaligned[i]) {
                cv::Mat f32;
                f.convertTo(f32, CV_32FC3);
                o = o * 0.5 + f32 * 0.5;
            }
            if (!guardW.empty()) {
                cv::Mat movingSrc = f;
                if (p.grainReduction > 0)
                    movingSrc = spatialDenoiseEdgePreserving(
                        f, analysis.grainSigma, p.grainReduction);
                cv::Mat m32;
                movingSrc.convertTo(m32, CV_32FC3);
                o = o.mul(cv::Scalar::all(1.0) - guardW) + m32.mul(guardW);
            }
            o.convertTo(out, CV_8UC3);
        } else {  // TexturePreserving
            cv::Mat base = f;
            if (p.grainReduction > 0)
                base = spatialDenoiseEdgePreserving(f, analysis.grainSigma,
                                                    p.grainReduction);
            out = base.clone();
            const cv::Mat& mask = analysis.dustMasks[i];
            if (!mask.empty() && cv::countNonZero(mask) > 0) {
                cv::Mat ref8;
                reference.convertTo(ref8, CV_8UC3);
                ref8.copyTo(out, mask);
            }
        }
        outputs.push_back(out);
    }
    return outputs;
}

}  // namespace animerestore
