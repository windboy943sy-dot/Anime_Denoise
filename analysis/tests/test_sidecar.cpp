#include "animerestore/sidecar.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "=== Sidecar IPC 結合テスト ===" << std::endl;

    animerestore::SidecarClient client;

    // 接続テスト
    std::cout << "[1] 接続試行 (127.0.0.1:9090)..." << std::endl;
    if (!client.connect("127.0.0.1", 9090)) {
        std::cerr << "エラー: サイドカーサーバーに接続できませんでした。" << std::endl;
        std::cerr << "サイドカーサーバーが起動しているか確認してください。" << std::endl;
        return 1;
    }
    std::cout << "接続成功！" << std::endl;

    // テスト画像作成 (320x240, 8UC3, グラデーションパターン)
    cv::Mat src(240, 320, CV_8UC3);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            src.at<cv::Vec3b>(y, x) = cv::Vec3b(x & 0xFF, y & 0xFF, (x + y) & 0xFF);
        }
    }

    // 共有メモリを介したフレーム処理リクエスト
    std::cout << "[2] フレーム処理リクエスト送信..." << std::endl;
    cv::Mat dst;
    if (!client.processFrame(src, dst, "real-cugan")) {
        std::cerr << "エラー: フレーム処理に失敗しました。" << std::endl;
        client.disconnect();
        return 1;
    }
    std::cout << "処理完了！" << std::endl;

    // データ一致性検証 (パススルー検証)
    std::cout << "[3] データ一致性（パリティ）検証..." << std::endl;
    if (dst.empty() || dst.rows != src.rows || dst.cols != src.cols || dst.type() != src.type()) {
        std::cerr << "エラー: 出力画像のサイズ・型が不一致です。" << std::endl;
        return 1;
    }

    double diff = cv::norm(src, dst, cv::NORM_L1);
    std::cout << "L1 誤差: " << diff << std::endl;
    assert(diff == 0.0);

    std::cout << "すべてのテストが正常に通過しました！" << std::endl;

    client.disconnect();
    return 0;
}
