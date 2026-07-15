#include <iostream>
#include <chrono>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "animerestore/denoise.h"

using namespace animerestore;

int main() {
    std::cout << "=== AnimeRestore 性能検証ベンチマーク ===" << std::endl;

    // テスト用の特徴点豊富な画像を生成 (1920x1080)
    int width = 1920;
    int height = 1080;
    cv::Mat img1 = cv::Mat::zeros(height, width, CV_8UC3);
    // 図形を描画して特徴点を作る
    cv::circle(img1, cv::Point(width/2, height/2), 200, cv::Scalar(255, 0, 0), -1);
    cv::rectangle(img1, cv::Point(100, 100), cv::Point(500, 500), cv::Scalar(0, 255, 0), -1);
    cv::line(img1, cv::Point(0, 0), cv::Point(width, height), cv::Scalar(0, 0, 255), 10);
    // ノイズを追加してグレインを模す
    cv::Mat noise(img1.size(), img1.type());
    cv::randn(noise, cv::Scalar(0,0,0), cv::Scalar(15, 15, 15));
    img1 = img1 + noise;

    // わずかにずらした第2フレームを作成 (並進 dx=2, dy=1)
    cv::Mat img2;
    cv::Mat rot = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 2.0, 0.0, 1.0, 1.0);
    cv::warpAffine(img1, img2, rot, img1.size());

    // -------------------------------------------------------------
    // 1. アライメント性能の検証
    // -------------------------------------------------------------
    std::cout << "\n[1] アライメント (1920x1080 -> 640x360 縮小下での位置合わせ) 計測中..." << std::endl;

    DenoiseParams params_old;
    params_old.align = true;
    params_old.alignWorkWidth = 640;
    params_old.eccIterations = 30; // 従来

    DenoiseParams params_new = params_old;
    params_new.eccIterations = 10; // 新規（粗アライメント成功時/削減）

    // 複数回実行して平均を測る
    const int ALIGN_ITERS = 10;
    
    // 従来方式 (ECCのみ 30 iters)
    double total_time_old_align = 0;
    cv::Mat smallRef, smallMov;
    cv::resize(img1, smallRef, cv::Size(640, 360));
    cv::cvtColor(smallRef, smallRef, cv::COLOR_BGR2GRAY);
    cv::resize(img2, smallMov, cv::Size(640, 360));
    cv::cvtColor(smallMov, smallMov, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < ALIGN_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 1e-5);
        try {
            cv::findTransformECC(smallRef, smallMov, warp, cv::MOTION_EUCLIDEAN, criteria, cv::noArray(), 5);
        } catch(...) {}
        auto end = std::chrono::high_resolution_clock::now();
        total_time_old_align += std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::cout << "  従来方式 (ECC 30回直接): 平均 " << (total_time_old_align / ALIGN_ITERS) << " ms" << std::endl;

    // 新規方式 (ORB粗アライメント + ECC 10回)
    double total_time_new_align = 0;
    std::vector<cv::Mat> frames = {img1, img2};
    for (int i = 0; i < ALIGN_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto aligned = alignGroupFrames(frames, params_new);
        auto end = std::chrono::high_resolution_clock::now();
        total_time_new_align += std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::cout << "  新方式 (ORB粗アライメント + ECC 10回削減): 平均 " << (total_time_new_align / ALIGN_ITERS) << " ms" << std::endl;
    std::cout << "  -> アライメント高速化比率: " << (total_time_old_align / total_time_new_align) << "x" << std::endl;

    // -------------------------------------------------------------
    // 2. 空間デノイズ性能の検証
    // -------------------------------------------------------------
    std::cout << "\n[2] 空間デノイズ (1920x1080, grain_sigma=10.0, strength=1.0) 計測中..." << std::endl;
    
    const int DENOISE_ITERS = 10;
    double grainSigma = 10.0;

    // 従来 (NLM)
    double total_time_nlm = 0;
    for (int i = 0; i < DENOISE_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat out = spatialDenoiseEdgePreserving(img1, grainSigma, 1.0, true, true); // useNlm=true
        auto end = std::chrono::high_resolution_clock::now();
        total_time_nlm += std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::cout << "  従来方式 (NLM Denoise): 平均 " << (total_time_nlm / DENOISE_ITERS) << " ms" << std::endl;

    // 新規 (Guided Filter)
    double total_time_gf = 0;
    for (int i = 0; i < DENOISE_ITERS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat out = spatialDenoiseEdgePreserving(img1, grainSigma, 1.0, true, false); // useNlm=false
        auto end = std::chrono::high_resolution_clock::now();
        total_time_gf += std::chrono::duration<double, std::milli>(end - start).count();
    }
    std::cout << "  新方式 (Guided Filter Denoise): 平均 " << (total_time_gf / DENOISE_ITERS) << " ms" << std::endl;
    std::cout << "  -> 空間デノイズ高速化比率: " << (total_time_nlm / total_time_gf) << "x" << std::endl;

    return 0;
}
