# アニメ特化ノイズ復元 OpenFXプラグイン

設計提案書・実装ロードマップに基づくプロジェクト。現在は **Phase 1（保持フレーム検出）・Phase 2（動き分類）・Phase 3（参照像R・ダスト検出・デノイズ出力）のプロトタイプ実装済み** の段階。
詳細は `docs/phase1_phase2_status.md`（Phase 1・2）と `docs/denoise_method_survey.md`（手法調査とデノイズ方式・Phase 3）を参照。

## ディレクトリ構成

```
anime-restoration-project/
├── analysis/            # （Phase 6以降）解析エンジンのC++実装
├── prototype/            # Phase 1-5 用Pythonプロトタイプ
│   ├── requirements.txt
│   ├── hold_frame_detection/          # Phase 1：保持フレーム検出コア
│   ├── motion_classification/         # Phase 2：動き分類コア
│   ├── denoise/                       # Phase 3：参照像R・ダスト検出・デノイズ出力コア
│   ├── run_pipeline.py                # 一括実行（Phase 1→欠陥スキャン→2→3）
│   ├── run_detection.py               # Phase 1 CLI（--dust-robust でダスト耐性再判定）
│   ├── run_motion_classification.py   # Phase 2 CLI（Phase 1のJSONを入力）
│   ├── run_denoise.py                 # Phase 3 CLI（対象別ON/OFF機構つき）
│   ├── run_defect_scan.py             # 傷・ラインノイズ検出レポートCLI
│   ├── run_full_evaluation.py         # 全素材の定量評価一括実行
│   ├── evaluate_hold_frames.py        # Phase 1 精度評価（ラベルCSVと突き合わせ）
│   ├── evaluate_motion.py             # Phase 2 精度評価（ラベルCSVと突き合わせ）
│   ├── verify_denoise_quality.py      # デノイズ前後の品質計測（ノイズ量・鮮鋭度・ちらつき）
│   └── tools/
│       └── bootstrap_shot_labels.py   # カット候補の下書きCSVを自動生成
├── openfx-plugin/        # （Phase 7以降）OpenFXプラグイン本体
├── test-assets/
│   ├── raw/               # テスト素材（25本収集済み）
│   └── labels/            # ラベリングCSV・検出結果JSON
│       ├── shot_labels_template.csv
│       ├── detections/    # 全素材の一括検出結果（ラベリングの下書きに使える）
│       └── README.md      # 列定義・記入ルール
└── docs/
    ├── phase0_checklist.md        # Phase 0 の作業チェックリスト
    ├── phase1_phase2_status.md    # Phase 1・2 の進捗・定量評価結果・既知の課題
    ├── denoise_method_survey.md   # デノイズ手法調査と方式提案（Phase 3 以降の実装経緯）
    ├── cpp_port_design.md         # Phase 6：C++移植・OpenFX接続の設計
    ├── design_integration_review.md  # 特化設計書（reference/）と実装の統合レビュー
    └── reference/                 # 別セッション成果物（特化設計書・調査レポート）
```

## 実行方法

Python 環境は `~/.venvs/anime_denoise`（3.9 ベース、requirements.txt 導入済み）。

```bash
cd prototype
# Phase 1：保持フレーム検出
~/.venvs/anime_denoise/bin/python run_detection.py \
    --input ../test-assets/raw/05_3coma_01.mov \
    --output ../test-assets/labels/detections/05_3coma_01_hold_groups.json
# Phase 2：動き分類（Phase 1 の出力を入力にする）
~/.venvs/anime_denoise/bin/python run_motion_classification.py \
    --input ../test-assets/raw/05_3coma_01.mov \
    --hold-groups ../test-assets/labels/detections/05_3coma_01_hold_groups.json \
    --output ../test-assets/labels/detections/05_3coma_01_motion.json
# Phase 3：デノイズ（--side-by-side で左右比較動画も出力）
# --extend N：第2層（前後Nグループのカット間拡張統合。パン等の1コマ素材に必須）
# --flicker-correction：保持区間内の輝度ゆらぎ正規化（既定OFF）
~/.venvs/anime_denoise/bin/python run_denoise.py \
    --input ../test-assets/raw/05_3coma_01.mov \
    --hold-groups ../test-assets/labels/detections/05_3coma_01_hold_groups.json \
    --output ../test-assets/denoised/05_3coma_01_texpres.mov \
    --mode texture_preserving --extend 2 --side-by-side
```

ダスト・白ゴミの多い素材で Phase 1 が毎フレーム分裂する場合は、
`run_detection.py --dust-robust` で再判定統合を有効にする。

全フェーズを1コマンドで通す場合：

```bash
~/.venvs/anime_denoise/bin/python run_pipeline.py \
    --input ../test-assets/raw/05_3coma_01.mov \
    --work-dir ../test-assets/pipeline/05_3coma_01 \
    --mode texture_preserving --extend 2 --dust-robust --side-by-side
```

run_denoise.py の対象別ON/OFF（設計提案書3.2節）：
`--dust-sigma`（ダスト）／`--scratch-defects <defects.json>`（傷、要目視確認）／
`--grain-reduction`（グレイン）／`--flicker-correction`（フリッカー）／
`--remove-line-noise`（ラインノイズ）／`--remove-scan-noise`（周期スキャンノイズ）

## 進め方（次のステップ）

1. `shot_labels_template.csv` を 25 本分埋める（`detections/` の JSON を下書きに）
2. `evaluate_hold_frames.py` で Phase 1 の適合率を測定し閾値を最終調整
3. Phase 2 の評価スクリプトを作成、Phase 3（参照像R・欠陥検出）に着手

## 関連ドキュメント

本雛形は以下2つの設計文書に基づく（別途共有済み）：

- アニメ特化ノイズ復元_OpenFX設計提案.md（アルゴリズム設計・全10章＋参考文献）
- 実装ロードマップ_詳細仕様書.md（フェーズ構成・完了基準・データフォーマット）
