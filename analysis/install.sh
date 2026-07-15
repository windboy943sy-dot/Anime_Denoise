#!/bin/bash
set -e

echo "=== AnimeRestore OFX プラグイン インストーラー (macOS) ==="

# スクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

# 1. ビルドの実行
echo "[1] 最新のコードをビルドしています..."
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. インストールパスの設定
OFX_DIR="/Library/OFX/Plugins"
TARGET_BUNDLE="AnimeRestore.ofx.bundle"
SOURCE_PATH="build/${TARGET_BUNDLE}"
DEST_PATH="${OFX_DIR}/${TARGET_BUNDLE}"

echo "[2] システムの OFX プラグインディレクトリを確認しています..."
if [ ! -d "${OFX_DIR}" ]; then
    echo "ディレクトリ ${OFX_DIR} が存在しないため作成します..."
    sudo mkdir -p "${OFX_DIR}"
fi

# 3. コピー処理
echo "[3] プラグインをコピーしています（管理者パスワードの入力が必要です）..."
if [ -d "${DEST_PATH}" ]; then
    echo "既存のプラグインのバックアップ（.bak）を作成しています..."
    sudo mv "${DEST_PATH}" "${DEST_PATH}.bak_$(date +%Y%m%d%H%M%S)"
fi

sudo cp -R "${SOURCE_PATH}" "${OFX_DIR}/"

# 過去に、コピー元の権限（root:700 等）を引き継いでプラグインが OFX ホストから
# 不可視になった。全ユーザーの読取＋ディレクトリ実行権を明示付与し、
# AppleDouble（._*）と隔離属性（com.apple.quarantine 等）を除去しておく
echo "[4] 権限と拡張属性を正規化しています..."
sudo chmod -R a+rX "${DEST_PATH}"
sudo find "${DEST_PATH}" -name '._*' -delete
sudo xattr -rc "${DEST_PATH}" 2>/dev/null || true

echo ""
echo "=== インストールが成功しました！ ==="
echo "配置先: ${DEST_PATH}"
echo ""
echo "【使用方法】"
echo "1. DaVinci Resolve などの OFX ホストを起動（または再起動）してください。"
echo "2. エフェクトライブラリ（OpenFX フィルター）から 'AnimeRestore' を適用できます。"
echo ""
echo "※ AIサイドカー（外部推論）機能を有効にする場合："
echo "   OFX パラメータで 'Enable AI Sidecar' にチェックを入れ、"
echo "   ターミナルで以下のコマンドを実行して推論サーバーを起動してください："
echo "   ~/.venvs/anime_denoise/bin/python $(pwd)/../prototype/sidecar/server.py"
echo ""
