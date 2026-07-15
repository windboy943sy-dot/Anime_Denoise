#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace animerestore {

struct SidecarParams {
    bool enabled = false;
    std::string host = "127.0.0.1";
    int port = 9090;
    std::string modelName = "real-cugan";
};

class SidecarClient {
public:
    SidecarClient();
    ~SidecarClient();

    // 接続（サイドカープロセスとのTCPハンドシェイク）
    bool connect(const std::string& host, int port);
    void disconnect();
    bool isConnected() const { return connected; }

    // AI推論リクエスト（共有メモリを介したフレームのやり取り）
    bool processFrame(const cv::Mat& src, cv::Mat& dst, const std::string& modelName);

private:
    int socketFd = -1;
    bool connected = false;
    std::string serverHost;
    int serverPort = 9090;

    // 共有メモリの作成・解放 (macOS POSIX互換)
    bool createShm(const std::string& name, size_t size, int& fd, void*& ptr);
    void destroyShm(const std::string& name, int fd, void* ptr, size_t size);
};

} // namespace animerestore
