#!/bin/bash
# AnimeRestore Denoise OFX プラグイン 配布用インストーラービルダー（macOS）
#
# やること：
#   1. Release ビルド（未ビルドなら）
#   2. バンドルを自己完結化：依存する Homebrew 由来の dylib を
#      Contents/Libraries/ へ再帰収集し、参照を @loader_path へ書き換え
#      （＝OpenCV 等が入っていない別の Mac でも動く）
#   3. ad-hoc コード署名（install_name_tool 後に必要）
#   4. .pkg（postinstall で権限正規化）と .dmg を生成
#
# 使い方：  ./make_installer.sh [バージョン]        （既定 1.0.0）
# 前提：    ビルド機に Homebrew OpenCV があること。配布先には不要。
# 署名：    Apple Developer 証明書があれば DEV_ID 環境変数に Developer ID を
#           指定すると正規署名する。無ければ ad-hoc（配布先で Gatekeeper 警告）。

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ANALYSIS="$(cd "$HERE/.." && pwd)"
BUILD="$ANALYSIS/build"
BUNDLE_SRC="$BUILD/AnimeRestore.ofx.bundle"

VERSION="${1:-1.0.0}"
PKG_ID="com.animerestore.denoise"
SIGN_ID="${DEV_ID:--}"   # 既定は ad-hoc（"-"）

DIST="$HERE/dist"
PKG_ROOT="$DIST/root"                       # pkg ペイロードのルート（/ 相当）
BUNDLE_DST="$PKG_ROOT/Library/OFX/Plugins/AnimeRestore.ofx.bundle"
LIBDIR="$BUNDLE_DST/Contents/Libraries"
BIN="$BUNDLE_DST/Contents/MacOS/AnimeRestore.ofx"

echo "=== AnimeRestore インストーラービルド v$VERSION ==="

# --- 0) ビルド確認 --------------------------------------------------------
if [ ! -f "$BUNDLE_SRC/Contents/MacOS/AnimeRestore.ofx" ]; then
    echo "[0] バンドル未検出。Release ビルドを実行します..."
    cmake -S "$ANALYSIS" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$BUILD" -j >/dev/null
fi

# --- 1) クリーンステージング ---------------------------------------------
echo "[1] ステージングを準備..."
rm -rf "$DIST"
mkdir -p "$LIBDIR"
cp -R "$BUNDLE_SRC/Contents" "$BUNDLE_DST/"

# --- 2+3) 依存 dylib を再帰収集し @loader_path へ書き換え -----------------
# dylibbundler が絶対パス／@rpath 双方の依存を再帰解決し Libraries/ へ集約、
# 参照を @loader_path/../Libraries/ に統一する。自前の @rpath 解決だと libomp
# 等が「同梱パス」と「@rpath 経由の /opt/homebrew」で二重ロードされ
# OMP Error #15 で abort する。dylibbundler は同一ライブラリを1つに正規化する。
# -s で OpenCV の lib を検索パスに与え @rpath/libopencv_*.dylib を解決させる。
echo "[2] 依存ライブラリを収集し @loader_path へ集約 (dylibbundler)..."
command -v dylibbundler >/dev/null || {
    echo "エラー: dylibbundler が必要です（brew install dylibbundler）。" >&2
    exit 1
}
dylibbundler --fix-file "$BIN" \
             --bundle-deps \
             --dest-dir "$LIBDIR" \
             --install-path "@loader_path/../Libraries/" \
             --search-path /opt/homebrew/opt/opencv/lib \
             --overwrite-files --create-dir >/dev/null
echo "    集約: $(ls "$LIBDIR"/*.dylib 2>/dev/null | wc -l | tr -d ' ') 個 ($(du -sh "$LIBDIR" | awk '{print $1}'))"

# --- 3.5) 別名依存の補完と LC_RPATH 重複の除去 ----------------------------
# OpenBLAS 等は複数の install name（libopenblas.0 と libopenblasp-r0.3.33）を
# 持ち、dylibbundler が一方しか集約しないことがある。参照名の実体を Homebrew
# から探してコピーし、その依存も @loader_path 化する（見つかるまで反復）。
echo "[3.5] 別名依存の補完と rpath 正規化..."
find_real() { find /opt/homebrew/opt /opt/homebrew/Cellar -name "$1" -type f 2>/dev/null | head -1; }
changed=1
while [ $changed -eq 1 ]; do
    changed=0
    for f in "$BIN" "$LIBDIR"/*.dylib; do
        while IFS= read -r ref; do
            case "$ref" in @loader_path/../Libraries/*) ;; *) continue;; esac
            name="${ref##*/}"
            [ -f "$LIBDIR/$name" ] && continue
            real="$(find_real "$name")"
            [ -z "$real" ] && continue
            cp -f "$real" "$LIBDIR/$name"; chmod u+w "$LIBDIR/$name"
            install_name_tool -id "@loader_path/../Libraries/$name" "$LIBDIR/$name" 2>/dev/null || true
            while IFS= read -r d; do
                case "$d" in /opt/homebrew/*|/usr/local/*)
                    install_name_tool -change "$d" "@loader_path/../Libraries/${d##*/}" "$LIBDIR/$name" 2>/dev/null || true;;
                esac
            done < <(otool -L "$LIBDIR/$name" | tail -n +2 | awk '{print $1}')
            changed=1
        done < <(otool -L "$f" | tail -n +2 | awk '{print $1}')
    done
done
# LC_RPATH に @loader_path/../Libraries/ が複数あると dyld が解決に失敗するため1つに
for f in "$BIN" "$LIBDIR"/*.dylib; do
    n=$(otool -l "$f" | grep -c "path @loader_path/../Libraries/ " || true)
    while [ "${n:-0}" -gt 1 ]; do
        install_name_tool -delete_rpath "@loader_path/../Libraries/" "$f" 2>/dev/null || break
        n=$((n-1))
    done
done

# --- 4) ad-hoc / Developer ID 署名 ---------------------------------------
# install_name_tool でバイナリを書き換えると既存署名が無効になるため再署名。
echo "[4] コード署名 (id=$SIGN_ID)..."
for lib in "$LIBDIR"/*.dylib; do codesign --force --timestamp=none -s "$SIGN_ID" "$lib"; done
codesign --force --timestamp=none -s "$SIGN_ID" "$BIN"

# 自己完結の検証：Homebrew 参照が本体から消えたことを確認
if otool -L "$BIN" | grep -q "/opt/homebrew\|/usr/local"; then
    echo "警告: 本体にまだ外部参照が残っています:" >&2
    otool -L "$BIN" | grep "/opt/homebrew\|/usr/local" >&2
fi

# --- 5) component pkg（postinstall で権限正規化）--------------------------
echo "[5] component pkg を生成..."
SCRIPTS="$DIST/scripts"
mkdir -p "$SCRIPTS"
cat > "$SCRIPTS/postinstall" <<'POST'
#!/bin/bash
# OFX ホストから読めるよう権限を正規化し、隔離属性・AppleDouble を除去
BUNDLE="/Library/OFX/Plugins/AnimeRestore.ofx.bundle"
chmod -R a+rX "$BUNDLE"
find "$BUNDLE" -name '._*' -delete 2>/dev/null || true
xattr -rc "$BUNDLE" 2>/dev/null || true
exit 0
POST
chmod +x "$SCRIPTS/postinstall"

# HFS 上での cp が生成する AppleDouble（._*）を除去してからパッケージ化
# （残すと pkg ペイロードに ._Libraries 等が混入する）
dot_clean "$PKG_ROOT" 2>/dev/null || true
find "$PKG_ROOT" -name '._*' -delete 2>/dev/null || true

pkgbuild --root "$PKG_ROOT" \
         --identifier "$PKG_ID" \
         --version "$VERSION" \
         --scripts "$SCRIPTS" \
         --install-location / \
         "$DIST/AnimeRestore-component.pkg" >/dev/null

# --- 6) product pkg（ウェルカム/結論画面つき）----------------------------
echo "[6] 配布 pkg を生成..."
sed "s/__VERSION__/$VERSION/g" "$HERE/distribution.xml" > "$DIST/distribution.xml"
PKG_OUT="$DIST/AnimeRestore-$VERSION.pkg"
productbuild --distribution "$DIST/distribution.xml" \
             --resources "$HERE/resources" \
             --package-path "$DIST" \
             "$PKG_OUT" >/dev/null

# --- 7) .dmg（pkg + README を同梱）---------------------------------------
echo "[7] .dmg を生成..."
DMGROOT="$DIST/dmg"
mkdir -p "$DMGROOT"
cp "$PKG_OUT" "$DMGROOT/"
cat > "$DMGROOT/はじめにお読みください.txt" <<TXT
AnimeRestore Denoise  v$VERSION  （DaVinci Resolve 等 OpenFX ホスト向け）

【インストール】
  AnimeRestore-$VERSION.pkg をダブルクリックして指示に従ってください。
  プラグインは /Library/OFX/Plugins/ に配置されます。

【未署名パッケージの警告が出た場合】
  このパッケージは Apple 公証されていないため、初回は警告が出ることがあります。
  pkg を「右クリック →『開く』」を選ぶか、
  システム設定 → プライバシーとセキュリティ →「このまま開く」で許可してください。

【使い方】
  DaVinci Resolve を再起動し、OpenFX の Filters から
  "AnimeRestore Denoise" を適用します。

【対応環境】
  Apple Silicon (arm64) macOS。OpenCV 等の別途インストールは不要です。
TXT

DMG_OUT="$DIST/AnimeRestore-$VERSION.dmg"
rm -f "$DMG_OUT"
hdiutil create -volname "AnimeRestore $VERSION" -srcfolder "$DMGROOT" \
               -ov -format UDZO "$DMG_OUT" >/dev/null

echo ""
echo "=== 完了 ==="
echo "  pkg: $PKG_OUT"
echo "  dmg: $DMG_OUT"
du -sh "$PKG_OUT" "$DMG_OUT" | sed 's/^/       /'
