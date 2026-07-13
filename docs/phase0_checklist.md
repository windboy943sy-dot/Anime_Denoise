# Phase 0 チェックリスト（環境準備・テスト素材収集）

実装ロードマップ「Phase 0：環境準備・テスト素材収集」に対応。完了したら✅を付けていく。

## 環境準備

- [ ] Python 3.10+ をインストール
- [ ] `prototype/requirements.txt` から仮想環境を構築
      ```
      cd prototype
      python -m venv venv
      source venv/bin/activate   # Windowsは venv\Scripts\activate
      pip install -r requirements.txt
      ```
- [ ] OpenFX / DaVinci Resolve のビルド環境を用意（プロジェクト内 `OpenFX/README.txt` の手順に従う）
      - Mac: Xcode
      - Windows: Visual Studio（.slnファイルを使用）
      - Linux: Makefile
      - ※ Phase 6以降まで急いで用意する必要はない。Phase 1〜5はPythonのみで進められる

## テスト素材収集

以下のカテゴリごとに、最低数カットずつ素材をそろえ `test-assets/raw/` に配置する。

### コマ打ち・動きの組み合わせ（Phase 1, 2 用）

- [ ] 静止画パン
- [ ] 静止画ズーム
- [ ] 静止画回転（あれば）
- [ ] 2コマ打ち・静止
- [ ] 3コマ打ち・静止
- [ ] 4コマ打ち・静止
- [ ] キャラのみ動く（口パクのみ含む）
- [ ] エフェクトのみ動く
- [ ] 画面全体が動く
- [ ] 激しいアクション（保持フレームなし）

### 欠陥の種類（Phase 3, 4 用。フィルムスキャン素材中心）

- [ ] フィルムグレインが明瞭な素材
- [ ] ダスト・白ゴミを含む素材
- [ ] 黒ゴミを含む素材
- [ ] 縦傷を含む素材
- [ ] フィルムのチラつき（フリッカー）を含む素材
- [ ] スキャン由来のライン状ノイズを含む素材

### 誤除去防止の検証用（Phase 4, 5 用）

- [ ] グロー・発光エフェクトを含むカット
- [ ] 撮影ブレ（意図的なモーションブラー）を含むカット
- [ ] セル境界線（輪郭線）がはっきりしたカット
- [ ] ハーフトーン・スクリーントーン風の演出を含むカット
- [ ] 意図的なグレイン演出を後乗せしていると思われるデジタル制作カット

## ラベリング

- [ ] `prototype/tools/bootstrap_shot_labels.py` で各動画からカット候補の下書きCSVを生成
      ```
      python tools/bootstrap_shot_labels.py --input ../test-assets/raw/xxx.mov --output ../test-assets/labels/xxx_draft.csv
      ```
- [ ] 下書きCSVを目視確認しながら `test-assets/labels/shot_labels_template.csv` の列定義（`test-assets/labels/README.md` 参照）に沿って埋める
- [ ] 各カテゴリ（上記チェックリスト）が最低1カットはラベル済みであることを確認

## 完了基準（実装ロードマップより）

ラベル付きテスト素材が最低でも数十カット分そろっていること。ラベルは以降のPhase 1〜4の精度評価（適合率・再現率の算出）における正解データとして使う。
