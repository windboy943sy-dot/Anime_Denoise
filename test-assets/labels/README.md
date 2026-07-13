# テスト素材ラベリングルール

`shot_labels_template.csv` にカット単位で正解ラベルを記入する。Phase 1〜4の精度評価（適合率・再現率の算出）の正解データとして使う。

## 列定義

| 列名 | 内容 | 記入例 |
|---|---|---|
| shot_id | カットの通し番号（4桁） | `0001` |
| source_file | 元ファイルパス（test-assets/raw/以下の相対パス） | `raw/sample01_scan.mov` |
| frame_start / frame_end | カットの開始・終了フレーム番号 | `1000` / `1240` |
| koma_pattern | 支配的なコマ打ち。`1koma`（保持なし）/`2koma`/`3koma`/`4koma`/`mixed`（カット内で混在） | `3koma` |
| global_motion | 大域動き。`static`/`pan`/`zoom`/`rotation`/`mixed` | `pan` |
| local_motion | 局所動き。`none`/`character_only`/`effect_only`/`all_moving`/`heavy_action` | `character_only` |
| defects_present | 含まれる欠陥（カンマ区切り、複数可）。`dust`/`scratch`/`grain`/`flicker`/`scan_noise`/`line_noise`/`none` | `dust,grain` |
| intentional_effects | 誤除去回避の検証対象となる意図的演出（カンマ区切り、複数可）。`glow`/`camera_shake`/`halftone`/`intentional_grain`/`none` | `glow` |
| source_type | 素材の種類。`film_scan`（フィルムスキャン）/`digital`（デジタル制作） | `film_scan` |
| notes | 自由記述（判定が難しい理由、特記事項など） | 境界が曖昧、口パクのみ動く |

## 記入時の注意

- **1カット（＝動きが一様な区間）に1行**。1ファイル内で動きが切り替わる素材
  （例：ズームアウト→パン→ズームアップ）は、**区間ごとに行を分けて**
  同じ `source_file` に別の `shot_id`・`frame_start`/`frame_end` を書くのが推奨。
  区間を分けたくない場合のみ `mixed` ＋ notes に詳細でも可
- フレーム番号は **0始まり**（解析ツールと同じ基準）。プレイヤーが1始まりで
  表示する場合は 1 引くこと。各区間の切り替わり位置の当たりを付けるには
  `test-assets/labels/detections/` の JSON（自動検出結果）が参考になる
- `global_motion` は評価時に表記ゆれを吸収する：`zoomout`/`zoomup`/`zoomin`/
  `ズームアップ`/`引き`/`寄り` → `zoom`、`tilt`/`panup` → `pan` など。
  ズームの方向はそのまま書いてよい（4クラス評価では zoom に正規化される）
- 記入用の作業ファイルは **`shot_labels.csv`**（ファイル名から下書きを自動生成済み。
  全25本が1行ずつ入っているので、区間分割が必要な行だけ複製して分ける）。
  `shot_labels_template.csv` は空テンプレートとして残す
- 判定に迷うものは無理に断定せず、notesに迷った理由を書いておく（後で閾値調整の参考になる）
- 実装ロードマップPhase 0の完了基準（各カテゴリの代表カットを最低数十カット分そろえる）を満たすよう、以下の組み合わせが最低1カットずつは含まれるようにする：
  - `koma_pattern` × `global_motion` × `local_motion` の代表的な組み合わせ
  - `defects_present` の各項目を含むカット（`film_scan`素材中心）
  - `intentional_effects` の各項目を含むカット（誤除去防止の検証用）
