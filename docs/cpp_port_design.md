# Phase 6：解析エンジン C++ 移植設計（2026-07-12 初版）

Python プロトタイプ（Phase 1〜5、定量評価済み）を C++ に移植し、
Phase 7 以降の OpenFX プラグインの土台にするための設計。
設計提案書6章・実装ロードマップ Phase 6〜9 に対応する。

## 1. 移植の前提と方針

- **アルゴリズムは凍結して移植する**。Phase 2 の大域動き 93.3%・Phase 1 の 87.5% を
  達成した Python 実装（閾値・処理順序込み）を仕様とし、C++ は「同じ入力に対して
  同じ出力」を返すことを目標にする（ゴールデンテストで担保、後述）
- 依存は **OpenCV (C++)** のみに揃える。プロトタイプが使う機能はすべて
  OpenCV C++ に存在する：
  - ORB / BFMatcher / estimateAffinePartial2D / findTransformECC / phaseCorrelate
  - GaussianBlur / Sobel / Canny / warpAffine / connectedComponentsWithStats / inpaint
  - pHash は自前実装（imagehash 相当は DCT 8x8 → 中央値二値化。50行程度）
  - SSIM も自前実装（skimage の gaussian_weights なし版と一致させる。
    ブロック単位なので単純な公式で良い）
  - scipy.ndimage.median_filter → cv2::medianBlur ＋ 1次元は自前
- **ビット深度**：内部処理は float32 に統一する。プロトタイプは uint8 だが、
  OpenFX からは float RGBA が来る（DaVinci は 32bit float）。閾値は
  0-255 スケールのまま保持し、入出力で正規化変換する
- 言語規格 C++17。ビルドは CMake（macOS: Xcode / Windows: MSVC の両対応を最初から）

## 2. 構成：解析エンジンをライブラリとして切り出す

```
analysis/                         # C++ 解析エンジン（OpenFXから独立）
├── CMakeLists.txt
├── include/animerestore/
│   ├── types.h            # HoldGroup, GlobalMotion, LocalMotion, DenoiseParams ...
│   ├── hold_detection.h   # Phase 1（+ dust-robust refine + drift check）
│   ├── motion.h           # Phase 2（+ ワープ改善チェック + シェイク判定）
│   ├── denoise.h          # Phase 3 第1層（align / reference / dust / render）
│   ├── extend.h           # 第2層 + 第3層ブレンド
│   ├── defects.h          # scratch / linenoise / scannoise / flicker
│   └── analysis_result.h  # 共通データフォーマット（JSONシリアライズ込み）
├── src/ ...
├── tests/
│   ├── golden/            # Pythonプロトタイプの出力（JSON・PNG）を正解とする
│   └── test_*.cpp
└── tools/
    └── ar_cli.cpp         # run_detection / run_denoise 相当のCLI（パリティ検証用）
```

- 「解析エンジンとプラグイン本体の分離」（設計提案書6章）をディレクトリ境界で強制する。
  `animerestore` ライブラリは OpenFX の型を一切 include しない
- 共通データフォーマット（ロードマップ2章の JSON スキーマ）は
  `analysis_result.h` の構造体＋ nlohmann/json でシリアライズし、
  Python 側の JSON と互換にする（→ Python の評価スクリプトが C++ 出力を
  そのまま検証できる）

## 3. パリティ（同一出力）の担保：ゴールデンテスト

移植の正しさは「Python と同じ答えを返すか」で機械的に検証する：

1. Python 側で `run_detection.py` / `run_motion_classification.py` /
   `run_denoise.py` を代表素材（quality_check の4本＋合成データ）に実行し、
   出力 JSON・出力フレーム（PNG）を `tests/golden/` に固定する
2. C++ の `ar_cli` で同じ入力を処理し、
   - JSON：hold group 境界の完全一致、動き分類の一致、数値は許容誤差付き比較
   - フレーム：PSNR > 50dB（浮動小数の丸め差のみ許容）
   で比較する
3. OpenCV のバージョン差・プラットフォーム差で ORB/RANSAC が非決定になる点は、
   RANSAC のシードを固定し、それでも残る差は「分類ラベルの一致」までを
   合格条件にする（画素完全一致は要求しない）

## 4. 移植順序（依存の少ない順）

| 順 | モジュール | 内容 | 検証 |
|---|---|---|---|
| 1 | types / pHash / blockSSIM / blurredDiff | 純粋関数群 | 合成データの単体テスト |
| 2 | hold_detection | detect + refine + drift split | golden JSON 一致 |
| 3 | denoise 第1層 | ECC align / trimmed mean / dust / 2モード | golden PNG（PSNR>50dB） |
| 4 | motion | ORB+RANSAC+ECCゲート+改善チェック+シェイク | 分類ラベル一致（93.3%の再現） |
| 5 | extend 第2層＋第3層ブレンド | ワープ受け入れ統合 | golden PNG |
| 6 | defects | scratch / line / scan / flicker | 合成欠陥の検出一致 |
| 7 | ar_cli | CLI パリティ | run_full_evaluation を C++ で再実行 |

並列化はモジュール移植が終わってから：保持グループ単位が自然な並列単位
（グループ間に依存がなく、第2層のみ前後 ±radius の R を参照）。

## 5. OpenFX プラグイン（Phase 7）との接続設計

設計提案書6章の要点に対応する：

- **getFramesNeeded**：現在フレームが属する保持グループ＋第2層の
  ±radius グループ分のフレーム範囲を要求する。グループ境界は解析結果に
  依存するため、**解析パス（Analyze）と描画パス（Render）を分離**する：
  1. ユーザーが Analyze を実行（またはシーケンシャルな先行パス）
     → 共通データフォーマットをクリップ単位でキャッシュ
  2. Render は キャッシュを参照して getFramesNeeded / 出力を決める
- **パラメータ**：3.2節の対象別ON/OFF（run_denoise の CLI フラグと1対1）
  ＋ Mode（Full Integration / Texture-Preserving）＋ Defect Removal Strength
  ＋ Grain Reduction。CLI と同名にしてプリセット互換にする
- **キャッシュの無効化**：入力クリップ・解析に影響するパラメータ（閾値系）が
  変わったら解析キャッシュを破棄。補正のON/OFF・強度は解析キャッシュを
  保持したまま Render のみ再実行（3.2節「解析は共通、適用だけ切替」）
- DaVinci Resolve 固有の挙動（getFramesNeeded の尊重範囲、レンダースレッド数）は
  Phase 7 の最初に小さなテストプラグインで実測してから本実装に入る

## 6. 性能目標と見積もり

- 現状 Python：2560x1920・82フレームで約10分（≒7.3秒/フレーム）
- 内訳の支配項は ECC 位置合わせ・ブロックSSIM・NLM。いずれも C++ 化＋
  マルチスレッド（グループ並列）で 10〜20倍 が既存事例の相場
- **Phase 6 目標：1〜2 フレーム/秒（CPU、フルHD超の実素材）**。
  リアルタイムは狙わない（オフラインレンダー前提）。GPU化（Phase 8）で
  さらに 5〜10倍を見込む
- メモリ：保持グループ＋±2グループのフレームを float32 で保持
  （2560x1920x4ch x 4byte ≒ 79MB/フレーム。窓15フレームで約1.2GB）。
  第2層の参照像キャッシュは half float 化も検討

## 7. 未決事項

- OpenFX SDK のバージョンと配布形態（DaVinci同梱ヘッダ vs 公式SDK）
- Windows ビルドの OpenCV 静的リンク（プラグイン配布サイズ）
- 解析キャッシュの永続化形式（プロジェクトファイル横のサイドカーJSONを想定）
- Metal/CUDA どちらを Phase 8 の一次ターゲットにするか（開発機は macOS）
