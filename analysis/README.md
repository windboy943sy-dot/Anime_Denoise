# 解析エンジン（C++、Phase 6）

Python プロトタイプ（`../prototype/`）の C++ 移植。設計と移植方針は
`../docs/cpp_port_design.md` を参照。

## 状態

| モジュール | 移植 | パリティ検証 |
|---|---|---|
| Phase 1（pHash / blockSSIM / 検出 / ドリフト検査） | ✅ | **✓ 合格**（golden＋実素材25グループ完全一致） |
| Phase 2（動き分類：ORB+RANSAC / ECCゲート / ワープ改善チェック / シェイク判定） | ✅ | **✓ 合格**（pan/zoom/static 13遷移、type＋数値） |
| Phase 3（位置合わせ / R生成 / ダスト4条件 / モーションガード / 2モード出力） | ✅ | **✓ 合格**（R:PSNR>45dB、σ一致、ダストマスクIoU） |
| Phase 1 ダスト耐性再判定（refine） | ✅ | パイプライン経由（--dust-robust） |
| 第2層拡張統合（extend）＋第3層ブレンド | ✅ | パイプライン品質指標で間接検証（下記） |
| 欠陥検出（scratch / linenoise / scannoise、補正込み） | ✅ | **✓ 合格**（合成欠陥の検出値一致） |
| パイプラインCLI（ar_cli denoise：動画→デノイズ動画） | ✅ | 品質指標がPythonと同等（ノイズ-75.9%/-77.0%） |

**性能**：フルパイプライン（full＋extend2＋grain0.3、2560x1920×82f）
Python 約10〜15分 → C++ 2分42秒 → **ウィンドウ化＋非同期先読みで 1分55秒（Python比 約6〜8倍）**。

## OpenFX プラグイン（Phase 7）

2つのプラグインをビルドする（`ofx/`、OpenFXヘッダは third_party/openfx）：

1. **AR Probe (passthrough)**（`probe_plugin.cpp`）：DaVinci の挙動
   （getFramesNeeded の尊重、時間方向アクセスの成否、画素フォーマット、
   レンダースレッド）を実測する調査用。ログ `/tmp/ar_ofx_probe.log`
2. **AnimeRestore Denoise v1**（`animerestore_plugin.cpp`）：本体。
   render(t) ごとに t±TemporalRadius をホストから取得 → ウィンドウ内で
   保持グループ検出（ドリフト検査込み）→ グループ解析（インスタンス内
   キャッシュで再利用）→ 現在フレームを出力。時間方向アクセス不可の
   ホストではパススルーに退化。パラメータ：Mode / Temporal Radius /
   Dust Removal / Grain Reduction。ログ `/tmp/ar_ofx.log`。
   v1の制約：float RGBA のみ、第2層と欠陥トグル群は未搭載、
   OpenCV は /opt/homebrew に動的リンク（配布時は静的化が必要）

**インストール（手動・要管理者権限）**：
```bash
sudo mkdir -p /Library/OFX/Plugins
sudo cp -r build/AnimeRestoreProbe.ofx.bundle build/AnimeRestore.ofx.bundle /Library/OFX/Plugins/
# 重要：SMBボリュームからのコピーは root:700 ＋ AppleDouble を引き継ぐため必ず実行
sudo chmod -R a+rX /Library/OFX/Plugins
sudo find /Library/OFX/Plugins -name "._*" -delete
```
※権限を直さないと「プラグイン管理画面に名前は出るが OpenFX パネルに
表示されない（ロード失敗）」状態になる（実際に踏んだ）。

DaVinci Resolve を再起動 → OpenFX > AnimeRestore 配下に2つ表示される。
まず AR Probe で挙動確認 → 次に AnimeRestore Denoise を実素材に適用。

## 未実装・今後

- プローブの実測結果を踏まえた本プラグイン実装（Analyze/Render 分離、
  ../docs/cpp_port_design.md 5章）
- ar_cli denoise の入力ストリーミング化（解析側はウィンドウ＋先読み済み。
  入力フレームの全読みだけが残る）

移植時に踏んだ差異の記録：偶数個の中央値は np.median＝「中央2値の平均」に
合わせること（nth_element の上側を取るとダスト検出の時間的単発性条件が
過度に厳しくなり、円盤が欠けて形状フィルタを通らなくなる。実測でIoU=0になった）。

性能の初期値：Phase 1 検出 2560x1920×82フレームで **11.1秒**（Python 約22秒、最適化前）。

## ビルド環境（導入済み）

- CMake 4.4.0・OpenCV **5.0.0**（Homebrew、`/opt/homebrew`）
- OpenCV 5 の注意：CMakeコンポーネント名が再編されている
  （`features2d`→`features`、`calib3d`→`calib`）。ソースの include は
  互換ヘッダ（`opencv2/calib3d.hpp` 等）がそのまま使える
- brew が PATH に無い環境ではフルパス `/opt/homebrew/bin/cmake` を使う

## ビルドとテスト

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure   # ゴールデンパリティテスト
```

ゴールデンデータは Python 側から生成する（生成済み・tests/golden/）：

```bash
cd ../prototype
~/.venvs/anime_denoise/bin/python tools/generate_golden.py --out ../analysis/tests/golden
```
