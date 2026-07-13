"""
傷（縦スクラッチ）検出プロトタイプ（設計提案書 4章 対応）

実写フィルムの傷は「細い縦線が複数フレーム・複数カットにわたり
ほぼ同じスキャナx位置に持続する」のが特徴。保持グループ内では
傷は参照像Rにも入ってしまう（全フレームに存在するため残差に出ない）ので、
ダストとは逆に**グループ間の持続性**で検出する：

  1. 各グループの参照像Rから「細い縦線応答」を計算
     （水平方向メディアンとの差＝細い縦構造の強調 ＋ 垂直方向の連続性チェック）
  2. 全グループにわたる応答の最小値（＝どのグループにも存在する縦線）を取る
  3. 列ごとの被覆率が高い列を傷候補とする

限界（設計提案書5章の誤除去回避に関わる）：
  - 静止カットでは「絵柄の縦線」と物理的に区別できない。カメラワークのある
    カットでは絵柄の線は動き傷は固定なので分離できる
  - よって本検出は**候補の提示まで**とし、補正はユーザー確認を前提とする
    （--apply指定時のみ水平方向inpaintで補修）
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .core import _luma


def vertical_line_response(reference: np.ndarray,
                           max_line_width: int = 5,
                           min_vertical_extent: int = 25) -> np.ndarray:
    """細い縦線の応答マップ（大きいほど「細い縦構造」らしい）を返す。"""
    y = cv2.GaussianBlur(_luma(reference), (3, 3), 0)
    # 水平方向メディアンとの差：幅 max_line_width 以下の縦構造だけが残る
    med_h = median_filter(y, size=(1, max_line_width * 2 + 1))
    resp = np.abs(y - med_h)
    # 垂直方向の最小値フィルタ：縦に min_vertical_extent 連続していない応答を消す
    kernel = np.ones((min_vertical_extent, 1), np.uint8)
    return cv2.erode(resp, kernel)


def detect_scratch_columns(references: list[np.ndarray],
                           response_threshold: float = 6.0,
                           min_column_coverage: float = 0.25,
                           active_area_crop: float = 0.10) -> dict:
    """複数グループの参照像から、持続する縦スクラッチ候補の列を検出する。

    返り値：{
      "columns": [{"x": 列, "coverage": 縦方向被覆率, "strength": 平均応答}],
      "persist_map": グループ間持続応答マップ (H, W) float32,
    }
    """
    responses = [vertical_line_response(r) for r in references]
    # グループ間の最小値＝全グループに存在する縦線だけが残る。
    # 絵柄の縦線はカメラワークで位置が動けばここで消える
    persist = np.minimum.reduce(responses) if len(responses) > 1 else responses[0]

    h, w = persist.shape
    my, mx = int(h * active_area_crop), int(w * active_area_crop)
    inner = persist[my:h - my, mx:w - mx]

    hit = inner > response_threshold
    coverage = hit.mean(axis=0)  # 列ごとの縦方向被覆率

    columns = []
    for x in np.where(coverage > min_column_coverage)[0]:
        columns.append({
            "x": int(x + mx),
            "coverage": round(float(coverage[x]), 3),
            "strength": round(float(inner[hit[:, x], x].mean()) if hit[:, x].any() else 0.0, 2),
        })

    return {"columns": columns, "persist_map": persist}


def build_scratch_mask(shape: tuple, columns: list[dict],
                       width: int = 3) -> np.ndarray:
    """検出列から補修用マスク（255=傷）を作る。"""
    mask = np.zeros(shape[:2], np.uint8)
    for c in columns:
        x = c["x"]
        mask[:, max(0, x - width // 2):x + width // 2 + 1] = 255
    return mask


def repair_scratches(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """傷マスク部を水平方向優先のinpaintで補修する（プロトタイプはTelea法）。"""
    return cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
