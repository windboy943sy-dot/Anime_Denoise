// 欠陥検出（傷・ラインノイズ・スキャンノイズ）の C++ 実装。
// prototype/denoise/{scratch,linenoise,scannoise}.py を仕様として移植する。

#include "animerestore/defects.h"

#include <algorithm>
#include <cmath>
#include <complex>

#include <opencv2/imgproc.hpp>

namespace animerestore {

namespace {

cv::Mat luma(const cv::Mat& frame) {
    cv::Mat f32, gray;
    if (frame.channels() == 1) {
        frame.convertTo(gray, CV_32F);
        return gray;
    }
    frame.convertTo(f32, CV_32FC3);
    cv::cvtColor(f32, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

int reflectIndex(int i, int n) {
    // scipy.ndimage の 'reflect'（(d c b a | a b c d | d c b a)）
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - i - 1;
    }
    return i;
}

// 水平方向メディアンフィルタ（size = 1 x win、reflect境界）
cv::Mat medianFilterHorizontal(const cv::Mat& src, int win) {
    cv::Mat dst(src.size(), CV_32F);
    const int half = win / 2;
    std::vector<float> vals(win);
    for (int y = 0; y < src.rows; ++y) {
        const float* s = src.ptr<float>(y);
        float* d = dst.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            for (int k = 0; k < win; ++k)
                vals[k] = s[reflectIndex(x - half + k, src.cols)];
            std::nth_element(vals.begin(), vals.begin() + half, vals.end());
            d[x] = vals[half];
        }
    }
    return dst;
}

std::vector<double> medianFilter1D(const std::vector<double>& v, int win) {
    const int n = static_cast<int>(v.size());
    const int half = win / 2;
    std::vector<double> out(n), vals(win);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < win; ++k)
            vals[k] = v[reflectIndex(i - half + k, n)];
        std::nth_element(vals.begin(), vals.begin() + half, vals.end());
        out[i] = vals[half];
    }
    return out;
}

double medianVec(std::vector<double> v) {
    if (v.empty()) return 0;
    size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    double m = v[k];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + k - 1, v.end());
        m = (m + v[k - 1]) / 2.0;
    }
    return m;
}

// アクティブ領域の切り出し（axis=1なら転置して「行」として扱う）
cv::Mat innerAsRows(const cv::Mat& y, int axis, double crop) {
    int my = static_cast<int>(y.rows * crop);
    int mx = static_cast<int>(y.cols * crop);
    cv::Mat inner = y(cv::Rect(mx, my, y.cols - 2 * mx, y.rows - 2 * my));
    if (axis == 1) {
        cv::Mat t;
        cv::transpose(inner, t);
        return t;
    }
    return inner.clone();
}

}  // namespace

// --- 傷 ---------------------------------------------------------------

namespace {

cv::Mat verticalLineResponse(const cv::Mat& reference,
                             int maxLineWidth = 5, int minVerticalExtent = 25) {
    cv::Mat y = luma(reference);
    cv::GaussianBlur(y, y, cv::Size(3, 3), 0);
    cv::Mat medH = medianFilterHorizontal(y, maxLineWidth * 2 + 1);
    cv::Mat resp = cv::abs(y - medH);
    cv::Mat out;
    cv::erode(resp, out, cv::Mat::ones(minVerticalExtent, 1, CV_8U));
    return out;
}

}  // namespace

std::vector<ScratchColumn> detectScratchColumns(
    const std::vector<cv::Mat>& references, double responseThreshold,
    double minColumnCoverage, double activeAreaCrop) {
    std::vector<ScratchColumn> result;
    if (references.empty()) return result;

    cv::Mat persist = verticalLineResponse(references[0]);
    for (size_t i = 1; i < references.size(); ++i)
        persist = cv::min(persist, verticalLineResponse(references[i]));

    int my = static_cast<int>(persist.rows * activeAreaCrop);
    int mx = static_cast<int>(persist.cols * activeAreaCrop);
    cv::Mat inner = persist(cv::Rect(mx, my, persist.cols - 2 * mx,
                                     persist.rows - 2 * my));

    for (int x = 0; x < inner.cols; ++x) {
        int hits = 0;
        double sum = 0;
        for (int y = 0; y < inner.rows; ++y) {
            float v = inner.at<float>(y, x);
            if (v > responseThreshold) {
                ++hits;
                sum += v;
            }
        }
        double coverage = static_cast<double>(hits) / inner.rows;
        if (coverage > minColumnCoverage) {
            result.push_back({x + mx, coverage, hits ? sum / hits : 0.0});
        }
    }
    return result;
}

// --- ラインノイズ -------------------------------------------------------

std::vector<LineNoise> detectLineNoise(const cv::Mat& reference, int axis,
                                       double sigmaFactor, double minOffset,
                                       double uniformityRatio,
                                       double activeAreaCrop) {
    cv::Mat y = luma(reference);
    cv::Mat inner = innerAsRows(y, axis, activeAreaCrop);
    const int rows = inner.rows, cols = inner.cols;

    std::vector<double> prof(rows);
    std::vector<double> rowVals(cols);
    for (int i = 0; i < rows; ++i) {
        const float* r = inner.ptr<float>(i);
        for (int x = 0; x < cols; ++x) rowVals[x] = r[x];
        prof[i] = medianVec(rowVals);
    }
    std::vector<double> smooth = medianFilter1D(prof, 9);
    std::vector<double> dev(rows), absdev(rows);
    for (int i = 0; i < rows; ++i) dev[i] = prof[i] - smooth[i];
    double devMed = medianVec(dev);
    for (int i = 0; i < rows; ++i) absdev[i] = std::abs(dev[i] - devMed);
    double mad = medianVec(absdev);
    double threshold = std::max(minOffset, sigmaFactor * mad * 1.4826);

    int offset = static_cast<int>((axis == 0 ? y.rows : y.cols) * activeAreaCrop);
    std::vector<LineNoise> result;
    for (int i = 0; i < rows; ++i) {
        if (std::abs(dev[i]) <= threshold) continue;
        int lo = std::max(0, i - 2), hi = std::min(rows, i + 3);
        // 近傍行の平均を「本来の値」とし、行全体のずれの一様性を見る
        std::vector<double> lineDev(cols);
        for (int x = 0; x < cols; ++x) {
            double baseline = 0;
            int cnt = 0;
            for (int r2 = lo; r2 < hi; ++r2) {
                if (r2 == i) continue;
                baseline += inner.at<float>(r2, x);
                ++cnt;
            }
            baseline /= std::max(1, cnt);
            lineDev[x] = inner.at<float>(i, x) - baseline;
        }
        double medDev = medianVec(lineDev);
        if (std::abs(medDev) < minOffset) continue;
        int agree = 0;
        for (double v : lineDev)
            if ((v >= 0) == (medDev >= 0)) ++agree;
        double agreeRatio = static_cast<double>(agree) / cols;
        if (agreeRatio < uniformityRatio) continue;
        result.push_back({i + offset, medDev, agreeRatio});
    }
    return result;
}

cv::Mat correctLineNoise(const cv::Mat& frame,
                         const std::vector<LineNoise>& detections, int axis,
                         double strength) {
    if (detections.empty() || strength <= 0) return frame;
    cv::Mat out;
    frame.convertTo(out, CV_32FC3);
    for (const auto& d : detections) {
        double corr = d.offset * strength;
        if (axis == 0) out.row(d.index) -= cv::Scalar::all(corr);
        else out.col(d.index) -= cv::Scalar::all(corr);
    }
    cv::Mat u8;
    out.convertTo(u8, CV_8UC3);
    return u8;
}

// --- スキャンノイズ -----------------------------------------------------

namespace {

// 走査方向平均プロファイル（平均0化）とその複素スペクトル
void profileSpectrum(const cv::Mat& reference, int axis, double crop,
                     std::vector<double>& profile,
                     std::vector<std::complex<double>>& spec) {
    cv::Mat inner = innerAsRows(luma(reference), axis, crop);
    const int n = inner.rows;
    profile.resize(n);
    double mean = 0;
    for (int i = 0; i < n; ++i) {
        profile[i] = cv::mean(inner.row(i))[0];
        mean += profile[i];
    }
    mean /= n;
    for (auto& v : profile) v -= mean;

    // 実DFT（プロファイル長は高々数千なので直接計算で十分）
    spec.assign(n / 2 + 1, {0, 0});
    for (int k = 0; k <= n / 2; ++k) {
        std::complex<double> s{0, 0};
        for (int i = 0; i < n; ++i) {
            double ang = -2.0 * M_PI * k * i / n;
            s += profile[i] * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        spec[k] = s;
    }
}

}  // namespace

std::vector<ScanNoise> detectScanNoise(const cv::Mat& reference, int axis,
                                       double spikeFactor, double minPeriodPx,
                                       double maxPeriodPx, double minAmplitude,
                                       double activeAreaCrop) {
    std::vector<double> profile;
    std::vector<std::complex<double>> spec;
    profileSpectrum(reference, axis, activeAreaCrop, profile, spec);
    const int n = static_cast<int>(profile.size());

    std::vector<double> mag(spec.size());
    for (size_t k = 0; k < spec.size(); ++k) mag[k] = std::abs(spec[k]);

    std::vector<ScanNoise> candidates;
    for (int k = 2; k < static_cast<int>(mag.size()) - 1; ++k) {
        double period = static_cast<double>(n) / k;
        if (period < minPeriodPx || period > maxPeriodPx) continue;
        std::vector<double> neigh;
        for (int j = std::max(1, k - 8);
             j < std::min<int>(mag.size(), k + 9); ++j)
            if (j != k) neigh.push_back(mag[j]);
        double localMed = medianVec(neigh) + 1e-6;
        double snr = mag[k] / localMed;
        double amplitude = 2.0 * mag[k] / n;
        if (snr >= spikeFactor && amplitude >= minAmplitude)
            candidates.push_back({period, amplitude, snr, k});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const ScanNoise& a, const ScanNoise& b) { return a.snr > b.snr; });
    std::vector<ScanNoise> kept;
    for (const auto& c : candidates) {
        bool near = false;
        for (const auto& o : kept)
            if (std::abs(c.bin - o.bin) <= 2) near = true;
        if (!near) kept.push_back(c);
    }
    return kept;
}

cv::Mat correctScanNoise(const cv::Mat& frame, const cv::Mat& reference,
                         const std::vector<ScanNoise>& detections, int axis,
                         double strength, double activeAreaCrop) {
    if (detections.empty() || strength <= 0) return frame;

    std::vector<double> profile;
    std::vector<std::complex<double>> spec;
    profileSpectrum(reference, axis, activeAreaCrop, profile, spec);
    const int n = static_cast<int>(profile.size());

    // スパイク近傍ビンのみを残した逆DFTで周期ノイズ成分を再構成
    std::vector<std::complex<double>> notch(spec.size(), {0, 0});
    for (const auto& d : detections)
        for (int k = std::max(0, d.bin - 1);
             k <= std::min<int>(spec.size() - 1, d.bin + 1); ++k)
            notch[k] = spec[k];

    std::vector<double> periodic(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (size_t k = 0; k < notch.size(); ++k) {
            if (notch[k] == std::complex<double>{0, 0}) continue;
            double ang = 2.0 * M_PI * k * i / n;
            double contrib = notch[k].real() * std::cos(ang) -
                             notch[k].imag() * std::sin(ang);
            // 実信号のrfft：k=0とk=n/2以外は共役対称分の2倍
            s += (k == 0 || static_cast<int>(k) == n / 2 ? contrib : 2.0 * contrib);
        }
        periodic[i] = s / n;
    }

    cv::Mat out;
    frame.convertTo(out, CV_32FC3);
    int m = static_cast<int>((axis == 0 ? frame.rows : frame.cols) * activeAreaCrop);
    for (int i = 0; i < n; ++i) {
        double corr = periodic[i] * strength;
        if (axis == 0) out.row(m + i) -= cv::Scalar::all(corr);
        else out.col(m + i) -= cv::Scalar::all(corr);
    }
    cv::Mat u8;
    out.convertTo(u8, CV_8UC3);
    return u8;
}

}  // namespace animerestore
