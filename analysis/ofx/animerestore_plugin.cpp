// AnimeRestore OpenFX プラグイン v2（Phase 7）
//
// docs/cpp_port_design.md 5章の設計を「まず動く」形に絞った初版：
//   - render(t) ごとに t±TemporalRadius のフレームをホストから取得し、
//     そのウィンドウ内で保持グループ検出 → 現在フレームの属するグループを
//     解析（analyzeHoldGroup）→ 現在フレームの出力だけ返す
//   - 同一グループのフレームが連続して要求されるため、解析結果は
//     インスタンス内キャッシュ（グループ先頭時刻がキー）で再利用する
//   - 時間方向アクセスが失敗するホストでは自動的にパススルーへ退化
//
// v2 での追加：
//   - 第2層（カット間拡張統合、Extend Neighbors パラメータ）＋第3層ブレンド
//   - ラインノイズ / スキャンノイズ除去トグル
//   - クリップ端処理（クランプ複製フレームの除外＋クリップ範囲の尊重）
// 未搭載（今後）：傷除去（ショット横断解析が必要）、byte 深度、Analyze パス

#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include "animerestore/defects.h"
#include "animerestore/denoise.h"
#include "animerestore/extend.h"
#include "animerestore/hold_detection.h"

namespace {

using namespace animerestore;

OfxHost* gHost = nullptr;
const OfxImageEffectSuiteV1* gEffect = nullptr;
const OfxPropertySuiteV1* gProp = nullptr;
const OfxParameterSuiteV1* gParam = nullptr;

void logLine(const char* fmt, ...) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    FILE* fp = std::fopen("/tmp/ar_ofx.log", "a");
    if (!fp) return;
    va_list args;
    va_start(args, fmt);
    std::vfprintf(fp, fmt, args);
    va_end(args);
    std::fprintf(fp, "\n");
    std::fclose(fp);
}

// --- インスタンス状態 ----------------------------------------------------

struct CachedGroup {
    std::shared_ptr<GroupAnalysis> analysis;
    cv::Mat extendedRef;          // 第2層適用後のR（未適用なら empty）
    cv::Mat effectiveN;           // 第2層の実効Nマップ
    cv::Mat blendIn, blendOut;    // 第3層NLMのメモ（同一入力なら再利用）
    std::vector<LineNoise> lineRows, lineCols;
    std::vector<ScanNoise> scanH, scanV;
    bool defectsScanned = false;
};

struct DetectResult {
    std::vector<HoldGroup> groups;
};

struct Instance {
    std::mutex mutex;
    // グループ解析キャッシュ：キーは「グループ先頭の絶対時刻＋画像幅」。
    // DaVinci はサムネイル（プロキシ）レンダーにも同じエフェクトを適用するため
    // （実測：92x46 が来る）、時刻だけをキーにするとサイズ違いの解析が衝突する
    std::map<std::pair<double, int>, std::shared_ptr<CachedGroup>> cache;
    // ウィンドウ検出キャッシュ：同一 t の再レンダー（スクラブ時に同フレームが
    // 連続要求される実測挙動）で pHash/SSIM の検出をやり直さないため。
    // 検出は閾値固定＋ソースフレーム決定的なので同一キーなら結果も同一
    std::map<std::pair<double, int>, std::shared_ptr<DetectResult>> detectCache;
};

Instance* instanceOf(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    void* ptr = nullptr;
    gProp->propGetPointer(props, kOfxPropInstanceData, 0, &ptr);
    return static_cast<Instance*>(ptr);
}

// --- OFX画像 ⇔ cv::Mat 変換（float RGBA 前提） ---------------------------

struct OfxImg {
    OfxPropertySetHandle handle = nullptr;
    float* data = nullptr;
    OfxRectI bounds{};
    int rowBytes = 0;
    bool valid() const { return data != nullptr; }
    int width() const { return bounds.x2 - bounds.x1; }
    int height() const { return bounds.y2 - bounds.y1; }
    float* row(int y) const {  // y は bounds.y1 起点
        return reinterpret_cast<float*>(
            reinterpret_cast<char*>(data) + static_cast<long>(y) * rowBytes);
    }
};

bool fetchImage(OfxImageClipHandle clip, double t, OfxImg& img) {
    if (gEffect->clipGetImage(clip, t, nullptr, &img.handle) != kOfxStatOK)
        return false;
    void* ptr = nullptr;
    gProp->propGetPointer(img.handle, kOfxImagePropData, 0, &ptr);
    gProp->propGetInt(img.handle, kOfxImagePropRowBytes, 0, &img.rowBytes);
    gProp->propGetIntN(img.handle, kOfxImagePropBounds, 4, &img.bounds.x1);
    char* depth = nullptr;
    gProp->propGetString(img.handle, kOfxImageEffectPropPixelDepth, 0, &depth);
    if (!ptr || !depth || std::strcmp(depth, kOfxBitDepthFloat) != 0) {
        gEffect->clipReleaseImage(img.handle);
        img.handle = nullptr;
        return false;  // v1 は float のみ対応
    }
    img.data = static_cast<float*>(ptr);
    return true;
}

void releaseImage(OfxImg& img) {
    if (img.handle) gEffect->clipReleaseImage(img.handle);
    img = {};
}

// float RGBA（0..1想定、範囲外はクリップ）→ BGR 8UC3
cv::Mat toMat(const OfxImg& img) {
    cv::Mat m(img.height(), img.width(), CV_8UC3);
    for (int y = 0; y < img.height(); ++y) {
        const float* src = img.row(y);
        cv::Vec3b* dst = m.ptr<cv::Vec3b>(y);
        for (int x = 0; x < img.width(); ++x) {
            const float* px = src + 4 * x;
            dst[x][2] = cv::saturate_cast<uchar>(px[0] * 255.0f + 0.5f);
            dst[x][1] = cv::saturate_cast<uchar>(px[1] * 255.0f + 0.5f);
            dst[x][0] = cv::saturate_cast<uchar>(px[2] * 255.0f + 0.5f);
        }
    }
    return m;
}

// BGR 8UC3 → 出力 float RGBA（αはソースから引き継ぐ）
void writeMat(const cv::Mat& m, const OfxImg& src, OfxImg& dst) {
    for (int y = 0; y < dst.height(); ++y) {
        float* d = dst.row(y);
        const cv::Vec3b* s =
            (y < m.rows) ? m.ptr<cv::Vec3b>(y) : m.ptr<cv::Vec3b>(m.rows - 1);
        const float* a = (y < src.height()) ? src.row(y) : nullptr;
        for (int x = 0; x < dst.width(); ++x) {
            const cv::Vec3b& px = s[std::min(x, m.cols - 1)];
            float* o = d + 4 * x;
            o[0] = px[2] / 255.0f;
            o[1] = px[1] / 255.0f;
            o[2] = px[0] / 255.0f;
            o[3] = (a && x < src.width()) ? a[4 * x + 3] : 1.0f;
        }
    }
}

// --- パラメータ -----------------------------------------------------------

constexpr const char* kParamMode = "mode";
constexpr const char* kParamRadius = "temporalRadius";
constexpr const char* kParamDust = "dustRemoval";
constexpr const char* kParamGrain = "grainReduction";
constexpr const char* kParamExtend = "extendNeighbors";
constexpr const char* kParamLineNoise = "lineNoiseRemoval";
constexpr const char* kParamScanNoise = "scanNoiseRemoval";

void defineParams(OfxImageEffectHandle effect) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxPropertySetHandle p;

    gParam->paramDefine(params, kOfxParamTypeChoice, kParamMode, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Mode");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 0, "Texture-Preserving");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 1,
                         "Full Temporal Integration");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    gParam->paramDefine(params, kOfxParamTypeInteger, kParamRadius, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Temporal Radius");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 6);
    gProp->propSetInt(p, kOfxParamPropMin, 0, 1);
    gProp->propSetInt(p, kOfxParamPropMax, 0, 24);

    gParam->paramDefine(params, kOfxParamTypeBoolean, kParamDust, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Dust/Dirt Removal");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 1);

    gParam->paramDefine(params, kOfxParamTypeDouble, kParamGrain, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Grain Reduction (per-frame)");
    gProp->propSetDouble(p, kOfxParamPropDefault, 0, 0.0);
    gProp->propSetDouble(p, kOfxParamPropMin, 0, 0.0);
    gProp->propSetDouble(p, kOfxParamPropMax, 0, 1.0);

    gParam->paramDefine(params, kOfxParamTypeInteger, kParamExtend, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Extend Neighbors (Layer 2)");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 2);
    gProp->propSetInt(p, kOfxParamPropMin, 0, 0);
    gProp->propSetInt(p, kOfxParamPropMax, 0, 3);

    gParam->paramDefine(params, kOfxParamTypeBoolean, kParamLineNoise, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Line Noise Removal");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);

    gParam->paramDefine(params, kOfxParamTypeBoolean, kParamScanNoise, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, "Scan Noise Removal");
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);
}

struct PluginParams {
    int extendNeighbors = 2;
    bool lineNoise = false;
    bool scanNoise = false;
};

void getParams(OfxImageEffectHandle effect, double t, DenoiseParams& p,
               int& radius, PluginParams& pp) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxParamHandle h;
    int mode = 0, dust = 1;
    double grain = 0.0;
    radius = 6;
    if (gParam->paramGetHandle(params, kParamMode, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &mode);
    if (gParam->paramGetHandle(params, kParamRadius, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &radius);
    if (gParam->paramGetHandle(params, kParamDust, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &dust);
    if (gParam->paramGetHandle(params, kParamGrain, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &grain);
    int ln = 0, sn = 0;
    if (gParam->paramGetHandle(params, kParamExtend, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &pp.extendNeighbors);
    if (gParam->paramGetHandle(params, kParamLineNoise, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &ln);
    if (gParam->paramGetHandle(params, kParamScanNoise, &h, nullptr) == kOfxStatOK)
        gParam->paramGetValueAtTime(h, t, &sn);
    pp.lineNoise = ln != 0;
    pp.scanNoise = sn != 0;

    p.mode = (mode == 1) ? DenoiseMode::FullTemporalIntegration
                         : DenoiseMode::TexturePreserving;
    p.dustDetection = dust != 0;
    p.grainReduction = grain;
}

// --- アクション -----------------------------------------------------------

OfxStatus onLoad() {
    if (!gHost) return kOfxStatErrMissingHostFeature;
    gEffect = static_cast<const OfxImageEffectSuiteV1*>(
        gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1));
    gProp = static_cast<const OfxPropertySuiteV1*>(
        gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1));
    gParam = static_cast<const OfxParameterSuiteV1*>(
        gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1));
    return (gEffect && gProp && gParam) ? kOfxStatOK
                                        : kOfxStatErrMissingHostFeature;
}

OfxStatus describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetString(props, kOfxPropLabel, 0, "AnimeRestore Denoise");
    gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0,
                         "AnimeRestore");
    gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0,
                         kOfxImageEffectContextFilter);
    gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0,
                         kOfxBitDepthFloat);
    gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0,
                         kOfxImageEffectRenderInstanceSafe);
    return kOfxStatOK;
}

OfxStatus describeInContext(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
    gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0,
                         kOfxImageComponentRGBA);
    gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
    gProp->propSetString(props, kOfxImageEffectPropSupportedComponents, 0,
                         kOfxImageComponentRGBA);
    defineParams(effect);
    return kOfxStatOK;
}

OfxStatus createInstance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetPointer(props, kOfxPropInstanceData, 0, new Instance());
    return kOfxStatOK;
}

OfxStatus destroyInstance(OfxImageEffectHandle effect) {
    delete instanceOf(effect);
    return kOfxStatOK;
}

OfxStatus getFramesNeeded(OfxImageEffectHandle effect,
                          OfxPropertySetHandle inArgs,
                          OfxPropertySetHandle outArgs) {
    double t = 0;
    gProp->propGetDouble(inArgs, kOfxPropTime, 0, &t);
    DenoiseParams p;
    int radius = 6;
    PluginParams pp;
    getParams(effect, t, p, radius, pp);
    double range[2] = {t - radius, t + radius};
    gProp->propSetDoubleN(outArgs, "OfxImageClipPropFrameRange_Source", 2, range);
    return kOfxStatOK;
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    double t = 0;
    gProp->propGetDouble(inArgs, kOfxPropTime, 0, &t);
    double renderScale[2] = {1.0, 1.0};
    gProp->propGetDoubleN(inArgs, kOfxImageEffectPropRenderScale, 2, renderScale);

    DenoiseParams p;
    int radius = 6;
    PluginParams pp;
    getParams(effect, t, p, radius, pp);

    OfxImageClipHandle srcClip = nullptr, dstClip = nullptr;
    gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &srcClip,
                           nullptr);
    gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &dstClip,
                           nullptr);

    OfxImg srcImg, dstImg;
    if (!fetchImage(srcClip, t, srcImg)) return kOfxStatFailed;
    if (!fetchImage(dstClip, t, dstImg)) {
        releaseImage(srcImg);
        return kOfxStatFailed;
    }

    // プロキシ／サムネイルレンダー（実測：92x46等）は解析せずパススルー。
    // 縮小画像では検出・σ推定が意味を持たず、計算の無駄になるだけ
    if (renderScale[0] < 0.999 || srcImg.width() < 512) {
        writeMat(toMat(srcImg), srcImg, dstImg);
        releaseImage(srcImg);
        releaseImage(dstImg);
        return kOfxStatOK;
    }

    // クリップのフレーム範囲を取得（範囲外はホストがクランプ複製を返すため、
    // そもそも要求しない。実測：クリップ先頭で複製がウィンドウに混入していた）
    double clipRange[2] = {-1e18, 1e18};
    {
        OfxPropertySetHandle clipProps = nullptr;
        if (gEffect->clipGetPropertySet(srcClip, &clipProps) == kOfxStatOK &&
            clipProps) {
            gProp->propGetDoubleN(clipProps, kOfxImageEffectPropFrameRange, 2,
                                  clipRange);
        }
    }

    // ウィンドウ収集（取得に失敗した時刻はスキップ＝クリップ端で自然に縮む）
    std::vector<cv::Mat> window;
    std::vector<double> times;
    int centerIdx = -1;
    for (int dt = -radius; dt <= radius; ++dt) {
        double tt = t + dt;
        if (dt == 0) {
            centerIdx = static_cast<int>(window.size());
            window.push_back(toMat(srcImg));
            times.push_back(t);
            continue;
        }
        if (tt < clipRange[0] || tt > clipRange[1]) continue;
        OfxImg img;
        if (!fetchImage(srcClip, tt, img)) continue;
        if (img.width() == srcImg.width() && img.height() == srcImg.height()) {
            window.push_back(toMat(img));
            times.push_back(tt);
        }
        releaseImage(img);
    }

    cv::Mat result;
    if (window.size() < 2 || centerIdx < 0) {
        // 時間方向アクセス不可のホスト：パススルーに退化（安全側）
        logLine("render t=%.1f: temporal access unavailable -> passthrough", t);
        result = window.empty() ? toMat(srcImg)
                                : window[std::max(centerIdx, 0)];
    } else {
        // ウィンドウ内で保持グループ検出（ドリフト検査込み）。
        // 同一 t の再レンダー用にキャッシュする（結果は決定的で同一）
        Instance* instD = instanceOf(effect);
        auto detKey = std::make_pair(t, srcImg.width());
        std::shared_ptr<DetectResult> det;
        if (instD) {
            std::lock_guard<std::mutex> lock(instD->mutex);
            auto it = instD->detectCache.find(detKey);
            if (it != instD->detectCache.end()) det = it->second;
        }
        DetectionThresholds th;
        if (!det) {
            det = std::make_shared<DetectResult>();
            det->groups = detectHoldGroups(window, th);
            det->groups = splitDriftingGroups(window, det->groups, th);
            if (instD) {
                std::lock_guard<std::mutex> lock(instD->mutex);
                if (instD->detectCache.size() > 64) instD->detectCache.clear();
                instD->detectCache[detKey] = det;
            }
        }
        const auto& groups = det->groups;
        int mineIdx = -1;
        for (size_t gi = 0; gi < groups.size(); ++gi)
            if (groups[gi].start <= centerIdx && centerIdx <= groups[gi].end)
                mineIdx = static_cast<int>(gi);
        HoldGroup mine = (mineIdx >= 0) ? groups[mineIdx]
                                        : HoldGroup{centerIdx, centerIdx, 1.0, ""};

        Instance* inst = instanceOf(effect);
        auto getCached = [&](const HoldGroup& g) -> std::shared_ptr<CachedGroup> {
            auto key = std::make_pair(times[g.start], srcImg.width());
            if (inst) {
                std::lock_guard<std::mutex> lock(inst->mutex);
                auto it = inst->cache.find(key);
                if (it != inst->cache.end()) return it->second;
            }
            auto cg = std::make_shared<CachedGroup>();
            std::vector<cv::Mat> gf(window.begin() + g.start,
                                    window.begin() + g.end + 1);
            cg->analysis = std::make_shared<GroupAnalysis>(analyzeHoldGroup(gf, p));
            if (inst) {
                std::lock_guard<std::mutex> lock(inst->mutex);
                if (inst->cache.size() > 16) inst->cache.clear();  // 簡易LRU
                inst->cache[key] = cg;
            }
            return cg;
        };

        std::shared_ptr<CachedGroup> center = getCached(mine);

        // 第2層：カット間拡張統合（ウィンドウ内の隣接グループを統合に参加させる。
        // 1コマ素材＝グループ長1でも時間統合が効くようになる本命機能）
        if (pp.extendNeighbors > 0 && mineIdx >= 0 && center->extendedRef.empty()) {
            std::vector<std::shared_ptr<CachedGroup>> keep;  // 生存保証
            std::vector<const GroupAnalysis*> neighbors;
            for (int j = mineIdx - pp.extendNeighbors;
                 j <= mineIdx + pp.extendNeighbors; ++j) {
                if (j < 0 || j >= static_cast<int>(groups.size()) || j == mineIdx)
                    continue;
                auto nb = getCached(groups[j]);
                keep.push_back(nb);
                neighbors.push_back(nb->analysis.get());
            }
            if (!neighbors.empty()) {
                ExtendParams ep;
                ep.radius = pp.extendNeighbors;
                ExtendResult ext = extendReference(*center->analysis, neighbors, ep);
                center->extendedRef = ext.reference;
                center->effectiveN = ext.effectiveN;
            }
        }

        auto outputs = renderHoldGroup(*center->analysis, p,
                                       center->extendedRef);
        result = outputs[centerIdx - mine.start];

        // 第3層：実効Nの低い画素にだけ空間NR（full モード時）
        if (!center->effectiveN.empty() &&
            p.mode == DenoiseMode::FullTemporalIntegration &&
            p.grainReduction > 0) {
            // 同一グループの内部フレームは入力（=拡張R）が同一のため、
            // NLM結果をメモ化して再利用（決定的処理なので出力もビット同一）
            if (!center->blendIn.empty() &&
                result.size() == center->blendIn.size() &&
                std::memcmp(result.data, center->blendIn.data,
                            result.total() * result.elemSize()) == 0) {
                result = center->blendOut;
            } else {
                center->blendIn = result.clone();
                result = blendSpatialFallback(result, center->effectiveN,
                                              center->analysis->grainSigma,
                                              p.grainReduction);
                center->blendOut = result;
            }
        }

        // 欠陥トグル：ライン／スキャンノイズ（グループRで一度だけ検出し、
        // 出力フレームに補正を適用）
        if ((pp.lineNoise || pp.scanNoise) && !center->defectsScanned) {
            const cv::Mat& ref = center->analysis->reference;
            if (pp.lineNoise) {
                center->lineRows = detectLineNoise(ref, 0);
                center->lineCols = detectLineNoise(ref, 1);
            }
            if (pp.scanNoise) {
                center->scanH = detectScanNoise(ref, 0);
                center->scanV = detectScanNoise(ref, 1);
            }
            center->defectsScanned = true;
        }
        if (pp.lineNoise && (!center->lineRows.empty() || !center->lineCols.empty())) {
            result = correctLineNoise(result, center->lineRows, 0);
            result = correctLineNoise(result, center->lineCols, 1);
        }
        if (pp.scanNoise && (!center->scanH.empty() || !center->scanV.empty())) {
            result = correctScanNoise(result, center->analysis->reference,
                                      center->scanH, 0);
            result = correctScanNoise(result, center->analysis->reference,
                                      center->scanV, 1);
        }

        logLine("render t=%.1f %dx%d window=%zu group=[%d-%d] sigma=%.2f "
                "ext=%s unsafe=%d",
                t, srcImg.width(), srcImg.height(), window.size(), mine.start,
                mine.end, center->analysis->grainSigma,
                center->extendedRef.empty() ? "off" : "on",
                center->analysis->integrationUnsafe ? 1 : 0);
    }

    writeMat(result, srcImg, dstImg);
    releaseImage(srcImg);
    releaseImage(dstImg);
    return kOfxStatOK;
}

OfxStatus mainEntry(const char* action, const void* handle,
                    OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    auto effect =
        reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
    try {
        if (std::strcmp(action, kOfxActionLoad) == 0) return onLoad();
        if (std::strcmp(action, kOfxActionDescribe) == 0) return describe(effect);
        if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
            return describeInContext(effect);
        if (std::strcmp(action, kOfxActionCreateInstance) == 0)
            return createInstance(effect);
        if (std::strcmp(action, kOfxActionDestroyInstance) == 0)
            return destroyInstance(effect);
        if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0)
            return getFramesNeeded(effect, inArgs, outArgs);
        if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
            return render(effect, inArgs);
    } catch (const std::exception& e) {
        logLine("exception in %s: %s", action, e.what());
        return kOfxStatFailed;
    }
    return kOfxStatReplyDefault;
}

void setHost(OfxHost* host) { gHost = host; }

OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    "jp.animerestore.denoise",
    0, 2,
    setHost,
    mainEntry,
};

}  // namespace

extern "C" {
OfxExport int OfxGetNumberOfPlugins() { return 1; }
OfxExport OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
}
