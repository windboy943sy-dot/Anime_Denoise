#!/bin/bash
set -e

echo "=== AnimeRestore OFX プラグイン アンインストーラー (macOS) ==="

OFX_DIR="/Library/OFX/Plugins"
TARGET_BUNDLE="AnimeRestore.ofx.bundle"
DEST_PATH="${OFX_DIR}/${TARGET_BUNDLE}"

if [ -d "${DEST_PATH}" ]; then
    echo "[1] プラグインを削除しています（管理者パスワードの入力が必要です）..."
    sudo rm -rf "${DEST_PATH}"
    echo "削除完了: ${DEST_PATH}"
else
    echo "プラグインはインストールされていませんでした。"
fi

echo ""
echo "=== アンインストールが完了しました ==="
echo ""
