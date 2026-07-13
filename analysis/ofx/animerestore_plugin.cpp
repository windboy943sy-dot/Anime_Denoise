// AnimeRestore OpenFX プラグイン v1（Phase 7）
//
// docs/cpp_port_design.md 5章の設計を「まず動く」形に絞った初版：
//   - render(t) ごとに t±TemporalRadius のフレームをホストから取得し、
//     そのウィンドウ内で保持グループ検出 → 現在フレームの属するグループを
//     解析（analyzeHoldGroup）→ 現在フレームの出力だけ返す
//   - 同一グループのフレームが連続して要求されるため、解析結果は
//     インスタンス内キャッシュ（グループ先頭時刻がキー）で再利用する
//   - 時間方向アクセスが失敗するホストでは自動的にパススルーへ退化
//
// v1 で意図的に省いたもの（本設計5章に沿って後続で追加）：
//   - 第2層（カット間拡張統合）・欠陥検出の個別トグル（ダストのみ実装）
//   - クリップ全体の Analyze パス（現状はウィンドウ内検出で代替）
//   - byte 深度（float RGBA のみ。DaVinci は float supply が既定）

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

#include "animerestore/denoise.h"
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

struct Instance {
    std::mutex mutex;
    // グループ解析キャッシュ：キーは「グループ先頭の絶対時刻＋画像幅」。
    // DaVinci はサムネイル（プロキシ）レンダーにも同じエフェクトを適用するため
    // （実測：92x46 が来る）、時刻だけをキーにするとサイズ違いの解析が衝突する
    std::map<std::pair<double, int>, std::shared_ptr<GroupAnalysis>> cache;
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
}

void getParams(OfxImageEffectHandle effect, double t, DenoiseParams& p,
               int& radius) {
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
    getParams(effect, t, p, radius);
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
    getParams(effect, t, p, radius);

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

    // ウィンドウ収集（取得に失敗した時刻はスキップ＝クリップ端で自然に縮む）
    std::vector<cv::Mat> window;
    std::vector<double> times;
    int centerIdx = -1;
    for (int dt = -radius; dt <= radius; ++dt) {
        if (dt == 0) {
            centerIdx = static_cast<int>(window.size());
            window.push_back(toMat(srcImg));
            times.push_back(t);
            continue;
        }
        OfxImg img;
        if (!fetchImage(srcClip, t + dt, img)) continue;
        if (img.width() == srcImg.width() && img.height() == srcImg.height()) {
            window.push_back(toMat(img));
            times.push_back(t + dt);
        }
        releaseImage(img);
    }

    cv::Mat result;
    if (window.size() < 2) {
        // 時間方向アクセス不可のホスト：パススルーに退化（安全側）
        logLine("render t=%.1f: temporal access unavailable -> passthrough", t);
        result = window[centerIdx >= 0 ? centerIdx : 0];
    } else {
        // ウィンドウ内で保持グループ検出（ドリフト検査込み）→
        // 現在フレームの属するグループを切り出す
        DetectionThresholds th;
        auto groups = detectHoldGroups(window, th);
        groups = splitDriftingGroups(window, groups, th);
        HoldGroup mine{centerIdx, centerIdx, 1.0, ""};
        for (const auto& g : groups)
            if (g.start <= centerIdx && centerIdx <= g.end) mine = g;

        // 解析キャッシュ：グループ先頭の絶対時刻＋画像幅がキー。
        // 同一グループの各フレームの render で再利用される
        auto cacheKey = std::make_pair(times[mine.start], srcImg.width());
        Instance* inst = instanceOf(effect);
        std::shared_ptr<GroupAnalysis> analysis;
        if (inst) {
            std::lock_guard<std::mutex> lock(inst->mutex);
            auto it = inst->cache.find(cacheKey);
            if (it != inst->cache.end()) analysis = it->second;
        }
        if (!analysis) {
            std::vector<cv::Mat> gf(window.begin() + mine.start,
                                    window.begin() + mine.end + 1);
            analysis = std::make_shared<GroupAnalysis>(analyzeHoldGroup(gf, p));
            if (inst) {
                std::lock_guard<std::mutex> lock(inst->mutex);
                if (inst->cache.size() > 8) inst->cache.clear();  // 簡易LRU
                inst->cache[cacheKey] = analysis;
            }
        }
        auto outputs = renderHoldGroup(*analysis, p);
        result = outputs[centerIdx - mine.start];
        logLine("render t=%.1f %dx%d window=%zu group=[%d-%d] sigma=%.2f", t,
                srcImg.width(), srcImg.height(), window.size(), mine.start,
                mine.end, analysis->grainSigma);
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
    0, 1,
    setHost,
    mainEntry,
};

}  // namespace

extern "C" {
OfxExport int OfxGetNumberOfPlugins() { return 1; }
OfxExport OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
}
