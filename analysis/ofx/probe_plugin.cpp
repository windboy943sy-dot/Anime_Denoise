// Phase 7 準備：OpenFX プローブプラグイン（docs/cpp_port_design.md 5章）
//
// 目的は本実装の前に DaVinci Resolve の挙動を実測すること：
//   - getFramesNeeded で要求した前後フレームが本当に供給されるか
//   - render がどの順序・スレッドで呼ばれるか
//   - 供給される画素フォーマット（float/byte、RGBA順）
// 映像はパススルー（Source→Output コピー）し、全アクションを
// /tmp/ar_ofx_probe.log に記録する。
//
// getFramesNeeded では現在フレーム±PROBE_RADIUS を要求する
// （本実装では保持グループ＋第2層±radius に置き換わる）。

#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <sstream>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxProperty.h"

namespace {

constexpr double kProbeRadius = 2.0;  // getFramesNeeded で要求する前後フレーム数

OfxHost* gHost = nullptr;
const OfxImageEffectSuiteV1* gEffectSuite = nullptr;
const OfxPropertySuiteV1* gPropSuite = nullptr;
std::mutex gLogMutex;

void logLine(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    FILE* fp = std::fopen("/tmp/ar_ofx_probe.log", "a");
    if (!fp) return;
    std::ostringstream tid;
    tid << std::this_thread::get_id();
    std::fprintf(fp, "[%ld][th %s] ", static_cast<long>(std::time(nullptr)),
                 tid.str().c_str());
    va_list args;
    va_start(args, fmt);
    std::vfprintf(fp, fmt, args);
    va_end(args);
    std::fprintf(fp, "\n");
    std::fclose(fp);
}

OfxStatus onLoad() {
    if (!gHost) return kOfxStatErrMissingHostFeature;
    gEffectSuite = static_cast<const OfxImageEffectSuiteV1*>(
        gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1));
    gPropSuite = static_cast<const OfxPropertySuiteV1*>(
        gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1));
    if (!gEffectSuite || !gPropSuite) return kOfxStatErrMissingHostFeature;

    char* hostName = nullptr;
    gPropSuite->propGetString(gHost->host, kOfxPropName, 0, &hostName);
    logLine("onLoad host=%s", hostName ? hostName : "(unknown)");
    return kOfxStatOK;
}

OfxStatus describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffectSuite->getPropertySet(effect, &props);

    gPropSuite->propSetString(props, kOfxPropLabel, 0, "AR Probe (passthrough)");
    gPropSuite->propSetString(props, kOfxImageEffectPluginPropGrouping, 0,
                              "AnimeRestore");
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedContexts, 0,
                              kOfxImageEffectContextFilter);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0,
                              kOfxBitDepthFloat);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1,
                              kOfxBitDepthByte);
    // 時間方向アクセスの宣言（保持グループ統合に必須の機能検証）
    gPropSuite->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);
    gPropSuite->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
    gPropSuite->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0,
                              kOfxImageEffectRenderFullySafe);

    logLine("describe");
    return kOfxStatOK;
}

OfxStatus describeInContext(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;

    gEffectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0,
                              kOfxImageComponentRGBA);
    gPropSuite->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);

    gEffectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0,
                              kOfxImageComponentRGBA);

    logLine("describeInContext");
    return kOfxStatOK;
}

OfxStatus getFramesNeeded(OfxImageEffectHandle /*effect*/,
                          OfxPropertySetHandle inArgs,
                          OfxPropertySetHandle outArgs) {
    double time = 0;
    gPropSuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    // Source クリップに現在フレーム±kProbeRadius を要求。
    // outArgs のプロパティ名は「OfxImageClipPropFrameRange_」＋クリップ名
    double range[2] = {time - kProbeRadius, time + kProbeRadius};
    gPropSuite->propSetDoubleN(outArgs, "OfxImageClipPropFrameRange_Source", 2,
                               range);
    logLine("getFramesNeeded t=%.1f -> [%.1f, %.1f]", time, range[0], range[1]);
    return kOfxStatOK;
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
    double time = 0;
    gPropSuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);

    OfxImageClipHandle srcClip = nullptr, dstClip = nullptr;
    gEffectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName,
                                &srcClip, nullptr);
    gEffectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &dstClip,
                                nullptr);

    // 時間方向アクセスの実測：前後フレームの取得を試み、成否をログに残す
    for (double dt = -kProbeRadius; dt <= kProbeRadius; dt += 1.0) {
        if (dt == 0.0) continue;
        OfxPropertySetHandle probeImg = nullptr;
        OfxStatus st = gEffectSuite->clipGetImage(srcClip, time + dt, nullptr,
                                                  &probeImg);
        logLine("render t=%.1f probe(t%+.1f)=%s", time, dt,
                st == kOfxStatOK ? "OK" : "FAILED");
        if (st == kOfxStatOK) gEffectSuite->clipReleaseImage(probeImg);
    }

    OfxPropertySetHandle srcImg = nullptr, dstImg = nullptr;
    if (gEffectSuite->clipGetImage(srcClip, time, nullptr, &srcImg) != kOfxStatOK)
        return kOfxStatFailed;
    if (gEffectSuite->clipGetImage(dstClip, time, nullptr, &dstImg) != kOfxStatOK) {
        gEffectSuite->clipReleaseImage(srcImg);
        return kOfxStatFailed;
    }

    void *srcPtr = nullptr, *dstPtr = nullptr;
    int srcRow = 0, dstRow = 0;
    OfxRectI srcBounds{}, dstBounds{};
    char* depth = nullptr;
    gPropSuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
    gPropSuite->propGetPointer(dstImg, kOfxImagePropData, 0, &dstPtr);
    gPropSuite->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRow);
    gPropSuite->propGetInt(dstImg, kOfxImagePropRowBytes, 0, &dstRow);
    gPropSuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
    gPropSuite->propGetIntN(dstImg, kOfxImagePropBounds, 4, &dstBounds.x1);
    gPropSuite->propGetString(srcImg, kOfxImageEffectPropPixelDepth, 0, &depth);

    logLine("render t=%.1f bounds=(%d,%d)-(%d,%d) depth=%s rowbytes=%d", time,
            srcBounds.x1, srcBounds.y1, srcBounds.x2, srcBounds.y2,
            depth ? depth : "?", srcRow);

    // パススルー：交差領域を行コピー
    if (srcPtr && dstPtr) {
        int y1 = std::max(srcBounds.y1, dstBounds.y1);
        int y2 = std::min(srcBounds.y2, dstBounds.y2);
        int copyBytes = std::min(srcRow, dstRow);
        if (copyBytes > 0) {
            for (int y = y1; y < y2; ++y) {
                std::memcpy(static_cast<char*>(dstPtr) +
                                static_cast<long>(y - dstBounds.y1) * dstRow,
                            static_cast<char*>(srcPtr) +
                                static_cast<long>(y - srcBounds.y1) * srcRow,
                            copyBytes);
            }
        }
    }

    gEffectSuite->clipReleaseImage(srcImg);
    gEffectSuite->clipReleaseImage(dstImg);
    return kOfxStatOK;
}

OfxStatus mainEntry(const char* action, const void* handle,
                    OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
    auto effect =
        reinterpret_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
    if (std::strcmp(action, kOfxActionLoad) == 0) return onLoad();
    if (std::strcmp(action, kOfxActionDescribe) == 0) return describe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return describeInContext(effect);
    if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0)
        return getFramesNeeded(effect, inArgs, outArgs);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
        return render(effect, inArgs);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0 ||
        std::strcmp(action, kOfxActionDestroyInstance) == 0 ||
        std::strcmp(action, kOfxActionUnload) == 0)
        return kOfxStatOK;
    return kOfxStatReplyDefault;
}

void setHost(OfxHost* host) { gHost = host; }

OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    "jp.animerestore.probe",
    1, 0,
    setHost,
    mainEntry,
};

}  // namespace

extern "C" {
OfxExport int OfxGetNumberOfPlugins() { return 1; }
OfxExport OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
}
