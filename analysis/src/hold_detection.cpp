// Phase 1：保持フレーム検出の C++ 実装。
// prototype/hold_frame_detection/core.py を仕様として忠実に移植する。
// アルゴリズム上の判断・閾値の根拠は Python 側のコメントと
// docs/phase1_phase2_status.md を参照。

#include "animerestore/hold_detection.h"

#include <algorithm>
#include <cmath>
#include <map>

#include <opencv2/imgproc.hpp>

namespace animerestore {

namespace {

cv::Mat toGray(const cv::Mat& frameBgr) {
    cv::Mat gray;
    cv::cvtColor(frameBgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// PIL の 'L' 変換（ITU-R 601: L = (R*299 + G*587 + B*114) / 1000）。
// OpenCV の COLOR_BGR2GRAY と係数がわずかに異なるため pHash 用に別実装
cv::Mat toGrayPil(const cv::Mat& frameBgr) {
    cv::Mat gray(frameBgr.rows, frameBgr.cols, CV_32F);
    for (int y = 0; y < frameBgr.rows; ++y) {
        const cv::Vec3b* src = frameBgr.ptr<cv::Vec3b>(y);
        float* dst = gray.ptr<float>(y);
        for (int x = 0; x < frameBgr.cols; ++x) {
            // BGR 並び
            dst[x] = (src[x][2] * 299.0f + src[x][1] * 587.0f + src[x][0] * 114.0f) / 1000.0f;
        }
    }
    return gray;
}

// scipy.fftpack.dct(type=2, norm=None) の2次元適用（行方向→列方向）。
// 8x8 の低周波のみ必要なので 32x32 入力に対する部分計算で十分
void dct2dLowFreq(const cv::Mat& img32, double out[8][8]) {
    constexpr int N = 32;
    // 軸0（列ごとに縦方向DCT）→ 軸1（行ごとに横方向DCT）の順は可換なので
    // out[u][v] = Σy Σx img(y,x)·2cos(π(2y+1)u/2N)·2cos(π(2x+1)v/2N)
    static double cosTable[8][N];
    static bool init = false;
    if (!init) {
        for (int u = 0; u < 8; ++u)
            for (int i = 0; i < N; ++i)
                cosTable[u][i] = 2.0 * std::cos(M_PI * (2 * i + 1) * u / (2.0 * N));
        init = true;
    }
    double tmp[8][N];  // 縦方向のみDCT済み
    for (int u = 0; u < 8; ++u) {
        for (int x = 0; x < N; ++x) {
            double s = 0.0;
            for (int y = 0; y < N; ++y) s += img32.at<float>(y, x) * cosTable[u][y];
            tmp[u][x] = s;
        }
    }
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            double s = 0.0;
            for (int x = 0; x < N; ++x) s += tmp[u][x] * cosTable[v][x];
            out[u][v] = s;
        }
}

}  // namespace

uint64_t computePHash(const cv::Mat& frameBgr) {
    cv::Mat gray = toGrayPil(frameBgr);
    cv::Mat small;
    // PIL の ANTIALIAS(Lanczos) と完全一致はしないが、面積平均は近い挙動。
    // pHash は粗選別のみに使うため hamming 数bit の揺れは許容（ヘッダ参照）
    cv::resize(gray, small, cv::Size(32, 32), 0, 0, cv::INTER_AREA);

    double low[8][8];
    dct2dLowFreq(small, low);

    double flat[64];
    for (int i = 0; i < 64; ++i) flat[i] = low[i / 8][i % 8];
    double sorted[64];
    std::copy(flat, flat + 64, sorted);
    std::nth_element(sorted, sorted + 31, sorted + 64);
    double m1 = sorted[31];
    std::nth_element(sorted, sorted + 32, sorted + 64);
    double median = (m1 + sorted[32]) / 2.0;

    uint64_t hash = 0;
    for (int i = 0; i < 64; ++i)
        if (flat[i] > median) hash |= (1ULL << (63 - i));
    return hash;
}

int phashDistance(uint64_t a, uint64_t b) {
    return static_cast<int>(__builtin_popcountll(a ^ b));
}

double blurredMeanAbsDiff(const cv::Mat& frameA, const cv::Mat& frameB, int ksize) {
    cv::Mat ga, gb;
    cv::GaussianBlur(toGray(frameA), ga, cv::Size(ksize, ksize), 0);
    cv::GaussianBlur(toGray(frameB), gb, cv::Size(ksize, ksize), 0);
    cv::Mat fa, fb;
    ga.convertTo(fa, CV_32F);
    gb.convertTo(fb, CV_32F);
    return cv::mean(cv::abs(fa - fb))[0];
}

namespace {

// skimage.metrics.structural_similarity(gaussian_weights=False, win_size=7,
// data_range=255) と同じ式。box filter 平均 → 不偏共分散 → SSIMマップ →
// 端 (win-1)/2 を落として平均
double ssimUniform7(const cv::Mat& a8u, const cv::Mat& b8u) {
    constexpr int win = 7;
    constexpr double dataRange = 255.0;
    const double C1 = std::pow(0.01 * dataRange, 2);
    const double C2 = std::pow(0.03 * dataRange, 2);
    const double w = win * win;
    const double covNorm = w / (w - 1.0);

    cv::Mat a, b;
    a8u.convertTo(a, CV_64F);
    b8u.convertTo(b, CV_64F);

    auto boxf = [](const cv::Mat& src) {
        cv::Mat dst;
        cv::blur(src, dst, cv::Size(win, win), cv::Point(-1, -1), cv::BORDER_REFLECT);
        return dst;
    };
    cv::Mat ux = boxf(a), uy = boxf(b);
    cv::Mat uxx = boxf(a.mul(a)), uyy = boxf(b.mul(b)), uxy = boxf(a.mul(b));
    cv::Mat vx = covNorm * (uxx - ux.mul(ux));
    cv::Mat vy = covNorm * (uyy - uy.mul(uy));
    cv::Mat vxy = covNorm * (uxy - ux.mul(uy));

    cv::Mat num = (2 * ux.mul(uy) + C1).mul(2 * vxy + C2);
    cv::Mat den = (ux.mul(ux) + uy.mul(uy) + C1).mul(vx + vy + C2);
    cv::Mat s = num / den;

    const int pad = (win - 1) / 2;
    cv::Rect valid(pad, pad, s.cols - 2 * pad, s.rows - 2 * pad);
    if (valid.width <= 0 || valid.height <= 0) return 1.0;
    return cv::mean(s(valid))[0];
}

}  // namespace

double blockSsim(const cv::Mat& frameA, const cv::Mat& frameB,
                 int blockSize, int blurKsize, double flatStdThreshold) {
    cv::Mat ga, gb;
    cv::GaussianBlur(toGray(frameA), ga, cv::Size(blurKsize, blurKsize), 0);
    cv::GaussianBlur(toGray(frameB), gb, cv::Size(blurKsize, blurKsize), 0);

    double sum = 0.0;
    int count = 0;
    for (int y = 0; y + blockSize <= ga.rows; y += blockSize) {
        for (int x = 0; x + blockSize <= ga.cols; x += blockSize) {
            cv::Rect r(x, y, blockSize, blockSize);
            cv::Mat ba = ga(r), bb = gb(r);
            cv::Scalar meanA, stdA, meanB, stdB;
            cv::meanStdDev(ba, meanA, stdA);
            cv::meanStdDev(bb, meanB, stdB);
            // ぼかし後もほぼ平坦なブロックは残留グレインがSSIMの「構造」として
            // 支配的になり不安定なため、差分ベースの簡易スコアで代替（Python同様）
            if (stdA[0] < flatStdThreshold && stdB[0] < flatStdThreshold) {
                cv::Mat fa, fb;
                ba.convertTo(fa, CV_32F);
                bb.convertTo(fb, CV_32F);
                double diff = cv::mean(cv::abs(fa - fb))[0];
                sum += std::max(0.0, 1.0 - diff / 255.0);
            } else {
                sum += ssimUniform7(ba, bb);
            }
            ++count;
        }
    }
    return count ? sum / count : 1.0;
}

std::vector<HoldGroup> detectHoldGroups(const std::vector<cv::Mat>& frames,
                                        const DetectionThresholds& th) {
    std::vector<HoldGroup> groups;
    if (frames.empty()) return groups;

    int groupStart = 0;
    uint64_t prevHash = computePHash(frames[0]);
    std::vector<double> confidences{1.0};

    auto confidence = [&](double diffScore, double ssimScore) {
        double diffMargin = std::max(0.0, (th.diffThreshold - diffScore) / th.diffThreshold);
        double ssimMargin = std::max(
            0.0, (ssimScore - th.ssimThreshold) / (1.0 - th.ssimThreshold + 1e-6));
        return std::clamp((diffMargin + ssimMargin) / 2.0, 0.0, 1.0);
    };
    auto meanOf = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / v.size();
    };

    for (size_t i = 1; i < frames.size(); ++i) {
        uint64_t curHash = computePHash(frames[i]);
        int quickScore = phashDistance(prevHash, curHash);

        bool isSame = false;
        double conf = 0.0;
        if (quickScore <= th.coarsePhashThreshold) {
            double diffScore = blurredMeanAbsDiff(frames[i - 1], frames[i], th.blurKsize);
            double ssimScore = blockSsim(frames[i - 1], frames[i], th.blockSize, th.blurKsize);
            if (diffScore < th.diffThreshold && ssimScore > th.ssimThreshold) {
                isSame = true;
                conf = confidence(diffScore, ssimScore);
            }
        }

        if (isSame) {
            confidences.push_back(conf);
        } else {
            groups.push_back({groupStart, static_cast<int>(i) - 1, meanOf(confidences), ""});
            groupStart = static_cast<int>(i);
            confidences = {1.0};
        }
        prevHash = curHash;
    }
    groups.push_back({groupStart, static_cast<int>(frames.size()) - 1,
                      meanOf(confidences), ""});
    return groups;
}

namespace {

bool framesSame(const std::vector<cv::Mat>& frames, int a, int b,
                const DetectionThresholds& th) {
    double diff = blurredMeanAbsDiff(frames[a], frames[b], th.blurKsize);
    if (diff >= th.diffThreshold) return false;
    return blockSsim(frames[a], frames[b], th.blockSize, th.blurKsize) > th.ssimThreshold;
}

void splitRec(const std::vector<cv::Mat>& frames, int start, int end,
              double confidence, const DetectionThresholds& th, int minSpan,
              std::vector<HoldGroup>& out) {
    if (end - start + 1 <= minSpan || framesSame(frames, start, end, th)) {
        out.push_back({start, end, confidence, ""});
        return;
    }
    int mid = (start + end) / 2;
    splitRec(frames, start, mid, confidence * 0.8, th, minSpan, out);
    splitRec(frames, mid + 1, end, confidence * 0.8, th, minSpan, out);
}

}  // namespace

std::vector<HoldGroup> splitDriftingGroups(const std::vector<cv::Mat>& frames,
                                           const std::vector<HoldGroup>& groups,
                                           const DetectionThresholds& th,
                                           int minSpan) {
    std::vector<HoldGroup> result;
    for (const auto& g : groups) {
        if (g.length() <= minSpan) {
            result.push_back(g);
        } else {
            splitRec(frames, g.start, g.end, g.confidence, th, minSpan, result);
        }
    }
    return result;
}

namespace {

// 2フレームが「ダスト等の単発欠陥を除けば同一」か（refine.py _same_except_dust）
bool sameExceptDust(const cv::Mat& frameA, const cv::Mat& frameB,
                    const DetectionThresholds& th,
                    double maxDustFraction = 0.10,
                    double dustMaxAreaRatio = 0.0008) {
    cv::Mat ga, gb;
    cv::GaussianBlur(toGray(frameA), ga, cv::Size(th.blurKsize, th.blurKsize), 0);
    cv::GaussianBlur(toGray(frameB), gb, cv::Size(th.blurKsize, th.blurKsize), 0);
    cv::Mat fa, fb;
    ga.convertTo(fa, CV_32F);
    gb.convertTo(fb, CV_32F);
    cv::Mat d = cv::abs(fa - fb);

    if (cv::mean(d)[0] < th.diffThreshold) return true;

    cv::Mat high = d > std::max(6.0, th.diffThreshold * 2.0);
    cv::morphologyEx(high, high, cv::MORPH_OPEN, cv::Mat::ones(3, 3, CV_8U));

    double frac = static_cast<double>(cv::countNonZero(high)) / high.total();
    if (frac > maxDustFraction) return false;

    int maxArea = static_cast<int>(d.total() * dustMaxAreaRatio);
    cv::Mat labels, stats, centroids;
    int nl = cv::connectedComponentsWithStats(high, labels, stats, centroids, 8);
    for (int j = 1; j < nl; ++j) {
        int area = stats.at<int>(j, cv::CC_STAT_AREA);
        if (area > maxArea) return false;
        int bw = stats.at<int>(j, cv::CC_STAT_WIDTH);
        int bh = stats.at<int>(j, cv::CC_STAT_HEIGHT);
        double elong = static_cast<double>(std::max(bw, bh)) /
                       std::max(1, std::min(bw, bh));
        double fill = static_cast<double>(area) / std::max(1, bw * bh);
        if (elong > 6.0 && fill < 0.3) return false;  // セルエッジの動き
    }

    cv::Mat dustRegion;
    cv::dilate(high, dustRegion, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat rest;
    d.copyTo(rest, ~dustRegion);
    int restCount = static_cast<int>(d.total()) - cv::countNonZero(dustRegion);
    if (restCount <= 0) return false;
    return cv::sum(rest)[0] / restCount < th.diffThreshold;
}

}  // namespace

std::vector<HoldGroup> refineHoldGroups(const std::vector<cv::Mat>& frames,
                                        const std::vector<HoldGroup>& groups,
                                        const DetectionThresholds& th) {
    if (groups.size() < 2) return groups;
    std::vector<HoldGroup> merged{groups[0]};
    for (size_t i = 1; i < groups.size(); ++i) {
        const HoldGroup& g = groups[i];
        HoldGroup& prev = merged.back();
        if (sameExceptDust(frames[prev.end], frames[g.start], th)) {
            double conf = (prev.confidence * prev.length() +
                           g.confidence * g.length()) /
                          (prev.length() + g.length());
            prev = {prev.start, g.end, conf, ""};
        } else {
            merged.push_back(g);
        }
    }
    return merged;
}

namespace {

std::string patternName(int length) {
    switch (length) {
        case 1: return "1koma";
        case 2: return "2koma";
        case 3: return "3koma";
        case 4: return "4koma";
        default: return std::to_string(length) + "koma";
    }
}

}  // namespace

void estimateKomaPattern(std::vector<HoldGroup>& groups) {
    if (groups.empty()) return;
    std::map<int, int> counts;
    for (const auto& g : groups) counts[g.length()]++;
    int dominant = counts.begin()->first;
    int best = 0;
    for (const auto& [len, c] : counts)
        if (c > best) { best = c; dominant = len; }

    for (auto& g : groups) {
        if (g.length() == dominant) {
            g.pattern = patternName(dominant);
        } else {
            g.pattern = patternName(g.length()) + "_irregular";
        }
    }
}

std::string dominantPatternForShot(const std::vector<HoldGroup>& groups) {
    if (groups.empty()) return "unknown";
    std::map<int, int> counts;
    for (const auto& g : groups) counts[g.length()]++;
    if (counts.size() == 1) return patternName(counts.begin()->first);

    int total = 0, best = 0, dominant = 0;
    for (const auto& [len, c] : counts) {
        total += c;
        if (c > best) { best = c; dominant = len; }
    }
    if (static_cast<double>(best) / total >= 0.7) return patternName(dominant);
    return "mixed";
}

}  // namespace animerestore
