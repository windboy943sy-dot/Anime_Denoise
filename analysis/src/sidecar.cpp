#include "animerestore/sidecar.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace animerestore {

SidecarClient::SidecarClient() : socketFd(-1), connected(false) {}

SidecarClient::~SidecarClient() {
    disconnect();
}

bool SidecarClient::connect(const std::string& host, int port) {
    if (connected) disconnect();

    serverHost = host;
    serverPort = port;

    socketFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd < 0) {
        std::cerr << "SidecarClient: ソケットを作成できませんでした。" << std::endl;
        return false;
    }

    // タイムアウト設定 (送信・受信ともに2秒)
    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    ::setsockopt(socketFd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    ::setsockopt(socketFd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (::inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
        std::cerr << "SidecarClient: 無効なアドレスです。" << std::endl;
        ::close(socketFd);
        socketFd = -1;
        return false;
    }

    if (::connect(socketFd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        ::close(socketFd);
        socketFd = -1;
        return false;
    }

    connected = true;
    return true;
}

void SidecarClient::disconnect() {
    if (socketFd >= 0) {
        ::close(socketFd);
        socketFd = -1;
    }
    connected = false;
}

bool SidecarClient::createShm(const std::string& name, size_t size, int& fd, void*& ptr) {
    ::shm_unlink(name.c_str());

    fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        std::cerr << "SidecarClient: shm_open に失敗しました。name=" << name << " errno=" << errno << std::endl;
        return false;
    }

    if (::ftruncate(fd, size) < 0) {
        std::cerr << "SidecarClient: ftruncate に失敗しました。" << std::endl;
        ::close(fd);
        ::shm_unlink(name.c_str());
        return false;
    }

    ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "SidecarClient: mmap に失敗しました。" << std::endl;
        ::close(fd);
        ::shm_unlink(name.c_str());
        return false;
    }

    return true;
}

void SidecarClient::destroyShm(const std::string& name, int fd, void* ptr, size_t size) {
    if (ptr && ptr != MAP_FAILED) {
        ::munmap(ptr, size);
    }
    if (fd >= 0) {
        ::close(fd);
    }
    ::shm_unlink(name.c_str());
}

bool SidecarClient::processFrame(const cv::Mat& src, cv::Mat& dst, const std::string& modelName) {
    if (!connected || socketFd < 0) {
        return false;
    }

    std::string shmName = "/ar_shm_" + std::to_string(::getpid()) + "_" + std::to_string(reinterpret_cast<uint64_t>(this));
    size_t bytes = src.total() * src.elemSize();

    int shmFd = -1;
    void* shmPtr = nullptr;

    if (!createShm(shmName, bytes, shmFd, shmPtr)) {
        return false;
    }

    std::memcpy(shmPtr, src.data, bytes);

    std::stringstream ss;
    ss << "infer " << shmName << " " << src.cols << " " << src.rows << " " << src.channels() << " " << modelName << "\n";
    std::string req = ss.str();

    if (::send(socketFd, req.c_str(), req.length(), 0) < 0) {
        std::cerr << "SidecarClient: リクエストの送信に失敗しました。" << std::endl;
        destroyShm(shmName, shmFd, shmPtr, bytes);
        disconnect();
        return false;
    }

    char buf[256];
    std::memset(buf, 0, sizeof(buf));
    ssize_t received = ::recv(socketFd, buf, sizeof(buf) - 1, 0);
    if (received <= 0) {
        std::cerr << "SidecarClient: レスポンスの受信に失敗しました。" << std::endl;
        destroyShm(shmName, shmFd, shmPtr, bytes);
        disconnect();
        return false;
    }

    std::string resp(buf);
    if (!resp.empty() && resp.back() == '\n') resp.pop_back();

    if (resp != "ok") {
        std::cerr << "SidecarClient: AI推論サーバーがエラーを返しました: " << resp << std::endl;
        destroyShm(shmName, shmFd, shmPtr, bytes);
        return false;
    }

    dst.create(src.rows, src.cols, src.type());
    std::memcpy(dst.data, shmPtr, bytes);

    destroyShm(shmName, shmFd, shmPtr, bytes);
    return true;
}

} // namespace animerestore
