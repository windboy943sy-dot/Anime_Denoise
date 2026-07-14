// パリティ検証用 CLI（docs/cpp_port_design.md 3章）。
// Python の run_detection.py 相当の出力を生成し、golden データと比較する。
//
//   ar_cli detect  --frames-dir <dir> --output <json> [--no-drift-check]
//   ar_cli detect  --input <video>    --output <json> [--no-drift-check]
//   ar_cli metrics --frames-dir <dir> --output <json>   # pHash・ペア指標のダンプ
//   ar_cli motion  --frames-dir <dir> --output <json>   # 隣接ペアの大域動き分類
//   ar_cli denoise-group --frames-dir <dir> --start N --end M --out-dir <dir>
//   ar_cli denoise --input <video> --output <video>     # 一括デノイズ（run_denoise相当）
//     [--mode full|texture] [--extend N] [--grain-reduction X]
//     [--dust-robust] [--no-drift-check]
//     メモ：入力フレームは全読み（検出のランダムアクセスのため）。解析は
//     スライディングウィンドウ＋非同期先読みでメモリと時間を抑制

#include <cstdio>
#include <deque>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "animerestore/defects.h"
#include "animerestore/denoise.h"
#include "animerestore/extend.h"
#include "animerestore/hold_detection.h"
#include "animerestore/motion.h"

using namespace animerestore;

namespace {

std::vector<cv::Mat> loadFramesDir(const std::string& dir) {
    std::vector<cv::String> files;
    cv::glob(dir + "/frame_*.png", files, false);
    std::sort(files.begin(), files.end());
    std::vector<cv::Mat> frames;
    for (const auto& f : files) frames.push_back(cv::imread(f, cv::IMREAD_COLOR));
    return frames;
}

std::vector<cv::Mat> loadVideo(const std::string& path) {
    cv::VideoCapture cap(path);
    std::vector<cv::Mat> frames;
    cv::Mat f;
    while (cap.read(f)) frames.push_back(f.clone());
    return frames;
}

std::string getArg(int argc, char** argv, const std::string& name,
                   const std::string& def = "") {
    for (int i = 0; i < argc - 1; ++i)
        if (name == argv[i]) return argv[i + 1];
    return def;
}

bool hasFlag(int argc, char** argv, const std::string& name) {
    for (int i = 0; i < argc; ++i)
        if (name == argv[i]) return true;
    return false;
}

int cmdDetect(int argc, char** argv) {
    std::string framesDir = getArg(argc, argv, "--frames-dir");
    std::string input = getArg(argc, argv, "--input");
    std::string output = getArg(argc, argv, "--output", "hold_groups.json");

    std::vector<cv::Mat> frames =
        !framesDir.empty() ? loadFramesDir(framesDir) : loadVideo(input);
    if (frames.empty()) {
        std::fprintf(stderr, "フレームを読み込めませんでした\n");
        return 1;
    }

    DetectionThresholds th;
    auto groups = detectHoldGroups(frames, th);
    if (!hasFlag(argc, argv, "--no-drift-check")) {
        groups = splitDriftingGroups(frames, groups, th);
    }
    estimateKomaPattern(groups);

    FILE* fp = std::fopen(output.c_str(), "w");
    std::fprintf(fp, "{\n \"dominant_koma_pattern\": \"%s\",\n \"hold_groups\": [\n",
                 dominantPatternForShot(groups).c_str());
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        std::fprintf(fp,
                     "  {\"start\": %d, \"end\": %d, \"pattern\": \"%s\", "
                     "\"confidence\": %.3f}%s\n",
                     g.start, g.end, g.pattern.c_str(), g.confidence,
                     i + 1 < groups.size() ? "," : "");
    }
    std::fprintf(fp, " ]\n}\n");
    std::fclose(fp);
    std::printf("%zu groups, dominant=%s -> %s\n", groups.size(),
                dominantPatternForShot(groups).c_str(), output.c_str());
    return 0;
}

int cmdMetrics(int argc, char** argv) {
    std::string framesDir = getArg(argc, argv, "--frames-dir");
    std::string output = getArg(argc, argv, "--output", "metrics.json");
    auto frames = loadFramesDir(framesDir);
    if (frames.empty()) {
        std::fprintf(stderr, "フレームを読み込めませんでした\n");
        return 1;
    }

    DetectionThresholds th;
    FILE* fp = std::fopen(output.c_str(), "w");
    std::fprintf(fp, "{\n \"phash\": [");
    for (size_t i = 0; i < frames.size(); ++i) {
        std::fprintf(fp, "\"%016llx\"%s",
                     static_cast<unsigned long long>(computePHash(frames[i])),
                     i + 1 < frames.size() ? ", " : "");
    }
    std::fprintf(fp, "],\n \"pairs\": [\n");
    for (size_t i = 1; i < frames.size(); ++i) {
        double d = blurredMeanAbsDiff(frames[i - 1], frames[i], th.blurKsize);
        double s = blockSsim(frames[i - 1], frames[i], th.blockSize, th.blurKsize);
        std::fprintf(fp,
                     "  {\"pair\": [%zu, %zu], \"blurred_diff\": %.4f, "
                     "\"block_ssim\": %.5f}%s\n",
                     i - 1, i, d, s, i + 1 < frames.size() ? "," : "");
    }
    std::fprintf(fp, " ]\n}\n");
    std::fclose(fp);
    std::printf("metrics -> %s\n", output.c_str());
    return 0;
}

int cmdMotion(int argc, char** argv) {
    std::string framesDir = getArg(argc, argv, "--frames-dir");
    std::string output = getArg(argc, argv, "--output", "motion.json");
    auto frames = loadFramesDir(framesDir);
    if (frames.size() < 2) {
        std::fprintf(stderr, "フレームが2枚未満です\n");
        return 1;
    }

    MotionThresholds th;
    FILE* fp = std::fopen(output.c_str(), "w");
    std::fprintf(fp, "[\n");
    for (size_t i = 1; i < frames.size(); ++i) {
        GlobalMotion g = estimateGlobalMotion(frames[i - 1], frames[i], th);
        std::fprintf(fp,
                     " {\"type\": \"%s\", \"tx\": %.2f, \"ty\": %.2f, "
                     "\"scale\": %.4f, \"rotation_deg\": %.3f, \"method\": \"%s\"}%s\n",
                     g.type.c_str(), g.tx, g.ty, g.scale, g.rotationDeg,
                     g.method.c_str(), i + 1 < frames.size() ? "," : "");
    }
    std::fprintf(fp, "]\n");
    std::fclose(fp);
    std::printf("motion -> %s\n", output.c_str());
    return 0;
}

int cmdDenoiseGroup(int argc, char** argv) {
    std::string framesDir = getArg(argc, argv, "--frames-dir");
    std::string outDir = getArg(argc, argv, "--out-dir", ".");
    int start = std::stoi(getArg(argc, argv, "--start", "0"));
    int end = std::stoi(getArg(argc, argv, "--end", "-1"));

    auto all = loadFramesDir(framesDir);
    if (all.empty()) {
        std::fprintf(stderr, "フレームを読み込めませんでした\n");
        return 1;
    }
    if (end < 0) end = static_cast<int>(all.size()) - 1;
    std::vector<cv::Mat> frames(all.begin() + start, all.begin() + end + 1);

    DenoiseParams p;
    GroupAnalysis a = analyzeHoldGroup(frames, p);

    cv::Mat ref8;
    a.reference.convertTo(ref8, CV_8UC3);
    cv::imwrite(outDir + "/reference.png", ref8);
    for (size_t i = 0; i < a.dustMasks.size(); ++i) {
        if (!a.dustMasks[i].empty())
            cv::imwrite(outDir + "/dust_mask_" + std::to_string(i) + ".png",
                        a.dustMasks[i]);
    }
    FILE* fp = std::fopen((outDir + "/denoise_group.json").c_str(), "w");
    std::fprintf(fp, "{\"grain_sigma\": %.3f, \"dust_px\": [", a.grainSigma);
    for (size_t i = 0; i < a.dustMasks.size(); ++i) {
        int px = a.dustMasks[i].empty() ? 0 : cv::countNonZero(a.dustMasks[i]);
        std::fprintf(fp, "%d%s", px, i + 1 < a.dustMasks.size() ? ", " : "");
    }
    std::fprintf(fp, "]}\n");
    std::fclose(fp);
    std::printf("denoise-group [%d-%d] sigma=%.3f -> %s\n", start, end,
                a.grainSigma, outDir.c_str());
    return 0;
}

int cmdDefects(int argc, char** argv) {
    // golden/defects ディレクトリの固定ファイル名（scratch_ref_*.png,
    // linenoise.png, scannoise.png）に対して3種の検出器を実行しJSONを出力
    std::string dir = getArg(argc, argv, "--dir");
    std::string output = getArg(argc, argv, "--output", "defects.json");

    std::vector<cv::Mat> refs;
    for (int i = 0;; ++i) {
        cv::Mat f = cv::imread(dir + "/scratch_ref_" + std::to_string(i) + ".png",
                               cv::IMREAD_COLOR);
        if (f.empty()) break;
        refs.push_back(f);
    }
    auto scratch = detectScratchColumns(refs);

    cv::Mat lnImg = cv::imread(dir + "/linenoise.png", cv::IMREAD_COLOR);
    auto rows = detectLineNoise(lnImg, 0);

    cv::Mat snImg = cv::imread(dir + "/scannoise.png", cv::IMREAD_COLOR);
    auto scan = detectScanNoise(snImg, 0);

    FILE* fp = std::fopen(output.c_str(), "w");
    std::fprintf(fp, "{\n \"scratch_columns\": [");
    for (size_t i = 0; i < scratch.size(); ++i)
        std::fprintf(fp, "{\"x\": %d, \"coverage\": %.3f}%s", scratch[i].x,
                     scratch[i].coverage, i + 1 < scratch.size() ? ", " : "");
    std::fprintf(fp, "],\n \"line_noise_rows\": [");
    for (size_t i = 0; i < rows.size(); ++i)
        std::fprintf(fp, "{\"index\": %d, \"offset\": %.2f, \"uniformity\": %.3f}%s",
                     rows[i].index, rows[i].offset, rows[i].uniformity,
                     i + 1 < rows.size() ? ", " : "");
    std::fprintf(fp, "],\n \"scan_noise\": [");
    for (size_t i = 0; i < scan.size(); ++i)
        std::fprintf(fp, "{\"period_px\": %.2f, \"amplitude\": %.3f, \"snr\": %.1f}%s",
                     scan[i].periodPx, scan[i].amplitude, scan[i].snr,
                     i + 1 < scan.size() ? ", " : "");
    std::fprintf(fp, "]\n}\n");
    std::fclose(fp);
    std::printf("defects -> %s (scratch=%zu, line=%zu, scan=%zu)\n",
                output.c_str(), scratch.size(), rows.size(), scan.size());
    return 0;
}

int cmdDenoise(int argc, char** argv) {
    std::string input = getArg(argc, argv, "--input");
    std::string output = getArg(argc, argv, "--output", "denoised.mov");
    std::string modeStr = getArg(argc, argv, "--mode", "texture");
    int extendRadius = std::stoi(getArg(argc, argv, "--extend", "0"));
    double grainReduction = std::stod(getArg(argc, argv, "--grain-reduction", "0"));

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "動画を開けませんでした: %s\n", input.c_str());
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 24.0;
    std::vector<cv::Mat> frames = loadVideo(input);
    if (frames.empty()) return 1;
    const int w = frames[0].cols, h = frames[0].rows;

    // Phase 1：検出（＋オプションのダスト耐性再判定、既定でドリフト検査）
    DetectionThresholds th;
    auto groups = detectHoldGroups(frames, th);
    if (hasFlag(argc, argv, "--dust-robust"))
        groups = refineHoldGroups(frames, groups, th);
    if (!hasFlag(argc, argv, "--no-drift-check"))
        groups = splitDriftingGroups(frames, groups, th);
    std::printf("%zu groups\n", groups.size());

    DenoiseParams p;
    p.mode = (modeStr == "full") ? DenoiseMode::FullTemporalIntegration
                                 : DenoiseMode::TexturePreserving;
    p.grainReduction = grainReduction;

    // 解析はスライディングウィンドウ（中心±radius）で保持し、
    // 次に必要になるグループを非同期に先読みする（メモリ抑制＋パイプライン重畳）
    const int nGroups = static_cast<int>(groups.size());
    const int radius = std::max(extendRadius, 0);
    std::deque<std::shared_ptr<GroupAnalysis>> window;  // groups[windowBase..) に対応
    int windowBase = 0;
    std::future<std::shared_ptr<GroupAnalysis>> prefetch;
    int prefetchIdx = -1;

    auto analyzeIdx = [&](int idx) {
        std::vector<cv::Mat> gf(frames.begin() + groups[idx].start,
                                frames.begin() + groups[idx].end + 1);
        return std::make_shared<GroupAnalysis>(analyzeHoldGroup(gf, p));
    };
    auto ensure = [&](int idx) {
        while (windowBase + static_cast<int>(window.size()) <= idx) {
            int next = windowBase + static_cast<int>(window.size());
            if (prefetchIdx == next && prefetch.valid()) {
                window.push_back(prefetch.get());
            } else {
                window.push_back(analyzeIdx(next));
            }
            int ahead = windowBase + static_cast<int>(window.size());
            if (ahead < nGroups) {  // 次のグループをレンダリングと重畳して先読み
                prefetchIdx = ahead;
                prefetch = std::async(std::launch::async, analyzeIdx, ahead);
            }
        }
    };

    cv::VideoWriter writer;  // 遅延オープン（SMB上のハンドル失効対策）
    ExtendParams ep;
    ep.radius = radius;

    for (int gi = 0; gi < nGroups; ++gi) {
        ensure(std::min(gi + radius, nGroups - 1));
        while (windowBase < gi - radius) {  // 使い終わった左端を解放
            window.pop_front();
            ++windowBase;
        }
        const GroupAnalysis& center = *window[gi - windowBase];

        std::vector<cv::Mat> outputs;
        if (radius > 0) {
            std::vector<const GroupAnalysis*> neighbors;
            for (int j = gi - radius; j <= gi + radius; ++j) {
                if (j < 0 || j >= nGroups || j == gi) continue;
                neighbors.push_back(window[j - windowBase].get());
            }
            ExtendResult ext = extendReference(center, neighbors, ep);
            outputs = renderHoldGroup(center, p, ext.reference);
            if (p.mode == DenoiseMode::FullTemporalIntegration &&
                p.grainReduction > 0) {
                // fullモードのグループ内部フレームは出力（=拡張R）が完全同一
                // なのに、同一入力へNLMを毎回再実行していた。直前と入力が
                // ビット一致なら結果を再利用する（NLMは決定的なので出力も同一）
                cv::Mat prevIn, prevOut;
                for (auto& o : outputs) {
                    if (!prevIn.empty() && o.size() == prevIn.size() &&
                        std::memcmp(o.data, prevIn.data,
                                    o.total() * o.elemSize()) == 0) {
                        o = prevOut;
                        continue;
                    }
                    prevIn = o.clone();
                    o = blendSpatialFallback(o, ext.effectiveN, center.grainSigma,
                                             p.grainReduction);
                    prevOut = o;
                }
            }
            std::printf("group [%d-%d] ext: neighbors=%d\n", groups[gi].start,
                        groups[gi].end, ext.usedNeighbors);
        } else {
            outputs = renderHoldGroup(center, p);
            std::printf("group [%d-%d]\n", groups[gi].start, groups[gi].end);
        }
        for (const auto& o : outputs) {
            if (!writer.isOpened()) {
                writer.open(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps, cv::Size(w, h));
            }
            writer.write(o);
        }
    }
    writer.release();
    std::printf("完了: %s\n", output.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ar_cli <detect|metrics> [options]\n");
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "detect") return cmdDetect(argc, argv);
    if (cmd == "metrics") return cmdMetrics(argc, argv);
    if (cmd == "motion") return cmdMotion(argc, argv);
    if (cmd == "denoise-group") return cmdDenoiseGroup(argc, argv);
    if (cmd == "denoise") return cmdDenoise(argc, argv);
    if (cmd == "defects") return cmdDefects(argc, argv);
    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    return 1;
}
