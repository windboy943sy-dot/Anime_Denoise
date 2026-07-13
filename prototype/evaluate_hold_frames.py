"""
Phase 1 評価スクリプト：run_detection.py の出力（hold_groups JSON）と
人手ラベル（shot_labels_template.csv の koma_pattern 列）を比較し、
実装ロードマップ Phase 1 の完了基準（適合率）を評価する。

比較対象：
  - 予測：JSON中の各カットの dominant_koma_pattern
  - 正解：ラベルCSVの koma_pattern 列（1koma/2koma/3koma/4koma/mixed）

注意：
  - "mixed" ラベルに対する予測の "mixed" 判定は、閾値（dominant_pattern_for_shotの70%基準）
    が正解ラベル側の判断基準と完全には一致しないため、参考値として扱うこと。
  - source_type（film_scan / digital）別に集計することで、
    設計提案書1章で指摘されている「グレインありの素材で精度が下がりやすい」問題を
    定量的に確認できるようにしている。

使い方：
  python evaluate_hold_frames.py \
      --predictions ../test-assets/labels/sample01_hold_groups.json \
      --labels ../test-assets/labels/sample01_labels.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd


def load_predictions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {shot["shot_id"]: shot["dominant_koma_pattern"] for shot in data["shots"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="run_detection.py の出力JSON")
    parser.add_argument("--labels", required=True, help="人手ラベルCSV（shot_labels_template.csv形式）")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    labels_df = pd.read_csv(args.labels)
    labels_df["shot_id"] = labels_df["shot_id"].astype(str)

    rows = []
    for _, row in labels_df.iterrows():
        shot_id = row["shot_id"]
        if shot_id not in predictions or pd.isna(row.get("koma_pattern")):
            continue
        rows.append({
            "shot_id": shot_id,
            "source_type": row.get("source_type", "unknown"),
            "label": row["koma_pattern"],
            "prediction": predictions[shot_id],
            "correct": row["koma_pattern"] == predictions[shot_id],
        })

    if not rows:
        print("比較可能な行がありませんでした。shot_id・koma_pattern列の記入状況を確認してください。")
        return

    result_df = pd.DataFrame(rows)

    overall_acc = result_df["correct"].mean()
    print(f"全体の正答率: {overall_acc:.1%}  (n={len(result_df)})")
    print()

    print("素材種別ごとの正答率:")
    for source_type, group in result_df.groupby("source_type"):
        acc = group["correct"].mean()
        print(f"  {source_type}: {acc:.1%}  (n={len(group)})")
    print()

    print("誤判定の内訳（正解 → 予測）:")
    mismatches = result_df[~result_df["correct"]]
    if mismatches.empty:
        print("  なし")
    else:
        for _, row in mismatches.iterrows():
            print(f"  shot {row['shot_id']} ({row['source_type']}): "
                  f"{row['label']} → {row['prediction']}")

    out_path = args.predictions.replace(".json", "_eval.csv")
    result_df.to_csv(out_path, index=False)
    print()
    print(f"詳細結果を書き出しました: {out_path}")


if __name__ == "__main__":
    main()
