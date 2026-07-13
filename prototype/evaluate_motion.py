"""
Phase 2 精度評価スクリプト：動き分類の結果（run_motion_classification.py の出力）と
ラベルCSV（shot_labels_template.csv 形式）の global_motion / local_motion 列を
突き合わせて正答率を算出する。

ロードマップ Phase 2 完了基準：
  - 大域動き（static/pan/zoom/rotation）4クラス分類で正答率85%以上
  - 局所動きはまず「動きあり／なし」の2値で評価

使い方：
  python evaluate_motion.py \
      --predictions ../test-assets/labels/detections/sample_motion.json \
      --labels ../test-assets/labels/sample_labels.csv
複数ファイルまとめて評価する場合は --predictions にディレクトリを渡す
（*_motion.json を走査し、labels CSV の source_file 列と突き合わせる）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

GLOBAL_CLASSES = ["static", "pan", "zoom", "rotation"]


def normalize_global_label(raw: str) -> str | None:
    """ラベルCSVの表記ゆれを吸収する（日本語表記も許容）。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    aliases = {
        "static": "static", "静止": "static", "fix": "static", "止め": "static",
        "pan": "pan", "パン": "pan", "tilt": "pan", "ティルト": "pan",
        "panup": "pan", "pandown": "pan", "パンアップ": "pan", "パンダウン": "pan",
        "zoom": "zoom", "ズーム": "zoom",
        "zoomin": "zoom", "zoomout": "zoom", "zoomup": "zoom", "zoomback": "zoom",
        "ズームイン": "zoom", "ズームアウト": "zoom",
        "ズームアップ": "zoom", "ズームバック": "zoom", "寄り": "zoom", "引き": "zoom",
        "rotation": "rotation", "回転": "rotation", "rotate": "rotation",
    }
    return aliases.get(s)


def normalize_local_label(raw: str) -> str | None:
    """局所動きは「動きあり／なし」の2値に落とす（完了基準の初期段階）。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().lower()
    if s in ("none", "なし", "no", "静止"):
        return "none"
    return "moving"  # character_only / effects_only / full / 全体 など


def predicted_local_binary(shot: dict) -> str:
    types = [seg["type"] for seg in shot.get("local_motion", [])]
    if not types:
        return "none"
    moving = sum(1 for t in types if t != "none")
    return "moving" if moving / len(types) >= 0.3 else "none"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="motion JSON ファイル、またはそれらを含むディレクトリ")
    parser.add_argument("--labels", required=True, help="ラベルCSV")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    files = sorted(pred_path.glob("*_motion.json")) if pred_path.is_dir() else [pred_path]

    df = pd.read_csv(args.labels)
    # source_file 列（ファイル名）→ 行 の索引。shot_id 併用も可
    labels_by_file = defaultdict(list)
    for _, row in df.iterrows():
        key = Path(str(row.get("source_file", ""))).stem
        labels_by_file[key].append(row)

    global_results = []   # (正解, 予測)
    local_results = []

    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        stem = Path(data["source_file"]).stem
        rows = labels_by_file.get(stem, [])
        if not rows:
            print(f"[skip] ラベルなし: {stem}")
            continue
        for shot in data["shots"]:
            # shot_id が一致する行、なければ最初の行を使う
            row = next((r for r in rows if str(r.get("shot_id")) == shot["shot_id"]),
                       rows[0])
            gt_g = normalize_global_label(row.get("global_motion"))
            if gt_g:
                global_results.append((gt_g, shot["dominant_global_motion"]))
            gt_l = normalize_local_label(row.get("local_motion"))
            if gt_l:
                local_results.append((gt_l, predicted_local_binary(shot)))

    if global_results:
        correct = sum(1 for gt, pred in global_results if gt == pred)
        print(f"\n大域動き 4クラス: {correct}/{len(global_results)} "
              f"= {correct/len(global_results):.1%}（完了基準 85%）")
        confusion = Counter((gt, pred) for gt, pred in global_results)
        for gt in GLOBAL_CLASSES:
            row_str = "  ".join(f"{pred}:{confusion.get((gt, pred), 0)}"
                                for pred in GLOBAL_CLASSES)
            print(f"  正解={gt:9s} → {row_str}")
    else:
        print("global_motion のラベルが1件もありません")

    if local_results:
        correct = sum(1 for gt, pred in local_results if gt == pred)
        print(f"\n局所動き 2値（あり/なし）: {correct}/{len(local_results)} "
              f"= {correct/len(local_results):.1%}")
    else:
        print("local_motion のラベルが1件もありません")


if __name__ == "__main__":
    main()
