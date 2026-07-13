# アニメ特化ノイズ復元 解析エンジン（Phase 6）
# ビルド：
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# 依存：OpenCV 4.x（core, imgproc, imgcodecs, videoio）
cmake_minimum_required(VERSION 3.16)
project(animerestore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 5 でモジュール名が再編された（features2d→features、calib3d→calib）。
# ソース側は互換ヘッダ（opencv2/calib3d.hpp 等）を使うので変更不要
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs videoio calib features video photo)

add_library(animerestore
    src/hold_detection.cpp
    src/motion.cpp
    src/denoise.cpp
    src/extend.cpp
    src/defects.cpp
)
target_include_directories(animerestore PUBLIC include)
target_link_libraries(animerestore PUBLIC ${OpenCV_LIBS})
target_include_directories(animerestore PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(ar_cli tools/ar_cli.cpp)
target_link_libraries(ar_cli PRIVATE animerestore)

# OpenFX プローブプラグイン（Phase 7 準備。ホスト挙動の実測用パススルー）
# インストール：build/AnimeRestoreProbe.ofx.bundle を /Library/OFX/Plugins/ に置く
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openfx/include)
    add_library(ArProbe MODULE ofx/probe_plugin.cpp)
    target_include_directories(ArProbe PRIVATE third_party/openfx/include)
    set_target_properties(ArProbe PROPERTIES
        PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "AnimeRestoreProbe")
    add_custom_command(TARGET ArProbe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_BINARY_DIR}/AnimeRestoreProbe.ofx.bundle/Contents/MacOS
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ArProbe>
            ${CMAKE_BINARY_DIR}/AnimeRestoreProbe.ofx.bundle/Contents/MacOS/AnimeRestoreProbe.ofx
        COMMENT "OFX bundle を生成: build/AnimeRestoreProbe.ofx.bundle")

    # 本体プラグイン v1（解析エンジンをリンク。開発機では /opt/homebrew の
    # OpenCV dylib に動的リンクする。配布時は静的リンク化が必要）
    add_library(ArDenoise MODULE ofx/animerestore_plugin.cpp)
    target_include_directories(ArDenoise PRIVATE third_party/openfx/include)
    target_link_libraries(ArDenoise PRIVATE animerestore)
    set_target_properties(ArDenoise PROPERTIES
        PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "AnimeRestore")
    add_custom_command(TARGET ArDenoise POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_BINARY_DIR}/AnimeRestore.ofx.bundle/Contents/MacOS
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ArDenoise>
            ${CMAKE_BINARY_DIR}/AnimeRestore.ofx.bundle/Contents/MacOS/AnimeRestore.ofx
        COMMENT "OFX bundle を生成: build/AnimeRestore.ofx.bundle")
endif()

enable_testing()
add_test(NAME golden_parity
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_golden.py
                 --ar-cli $<TARGET_FILE:ar_cli>
                 --golden ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden)
