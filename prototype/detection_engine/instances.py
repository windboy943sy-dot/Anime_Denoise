"""連結成分によるインスタンス化と特徴抽出・ルール分類(サーベイ §1.3, §4.5)。

画素マスク → 連結成分 → 特徴量(面積・細長度・円形度・極性・方向) →
ルールベース分類(ダスト/スクラッチ/カビ/大型ゴミ)。0.3 のインスタンス層を作る。

[考察] 候補画素は通常全体の <1% と疎なので、連結成分と特徴抽出は CPU が現実的
(サーベイ §1.3 GPU 適性)。本実装は scipy.ndimage.label。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .contracts import DefectInstance, DefectType, DetectorSource


def _shape_features(ys: np.ndarray, xs: np.ndarray):
    """成分の画素座標から (elongation, orientation_deg) を慣性モーメントで算出。"""
    if xs.size < 2:
        return 1.0, 0.0
    cx, cy = xs.mean(), ys.mean()
    dx, dy = xs - cx, ys - cy
    cxx = float(np.mean(dx * dx))
    cyy = float(np.mean(dy * dy))
    cxy = float(np.mean(dx * dy))
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l_max = tr / 2.0 + disc
    l_min = max(tr / 2.0 - disc, 1e-9)
    elong = float(np.sqrt(l_max / l_min))
    # 主軸方向(度)。0=水平, 90=垂直。
    theta = 0.5 * np.arctan2(2 * cxy, (cxx - cyy))
    orient = float(np.mod(np.degrees(theta), 180.0))
    return elong, orient


def build_instances(mask: np.ndarray, luma: np.ndarray,
                    polarity_map: np.ndarray | None = None,
                    strength_map: np.ndarray | None = None,
                    source: DetectorSource = DetectorSource.NONE,
                    min_area: int = 2, max_area: int = 20000):
    """二値マスクからインスタンス群と label 画像を生成。分類はまだ行わない。"""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    instances: list[DefectInstance] = []
    if n == 0:
        return instances, labels.astype(np.int32)

    objs = ndimage.find_objects(labels)
    # 背景輝度推定(成分周辺の膨張リング平均)用に軽くぼかした画像
    bg = ndimage.uniform_filter(luma.astype(np.float32), size=9)

    keep_labels = np.zeros(n + 1, np.int32)
    next_id = 1
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        comp = labels[sl] == i
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        y0, x0 = sl[0].start, sl[1].start
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        perim = _perimeter(comp)
        circ = float(4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        elong, orient = _shape_features(ys.astype(np.float64), xs.astype(np.float64))

        comp_luma = float(luma[ys, xs].mean())
        bg_luma = float(bg[ys, xs].mean())
        contrast = comp_luma - bg_luma
        if polarity_map is not None:
            pvals = polarity_map[ys, xs]
            pol = int(np.sign(np.sum(pvals))) if pvals.size else int(np.sign(contrast))
        else:
            pol = int(np.sign(contrast))
        # 半透明度 alpha: コントラストが背景比で浅いほど半透明とみなす [考察]
        alpha = float(np.clip(1.0 - min(abs(contrast) / (abs(bg_luma) + 1e-3), 1.0), 0.0, 1.0))

        strength = float(strength_map[ys, xs].mean()) if strength_map is not None else abs(contrast)

        ins = DefectInstance(
            id=next_id,
            bbox=(int(x0), int(y0), int(w), int(h)),
            centroid=(float(xs.mean()), float(ys.mean())),
            area=area,
            elongation=elong,
            circularity=circ,
            orientation_deg=orient,
            polarity=pol,
            contrast=float(contrast),
            translucency=alpha,
            confidence=float(np.clip(strength, 0.0, 1.0)),
            sources=source,
        )
        instances.append(ins)
        keep_labels[i] = next_id
        next_id += 1

    relabeled = keep_labels[labels]
    return instances, relabeled.astype(np.int32)


def _perimeter(comp: np.ndarray) -> float:
    """成分の周囲長(境界画素数の近似)。"""
    eroded = ndimage.binary_erosion(comp, border_value=0)
    return float((comp & ~eroded).sum())


def classify_instance(ins: DefectInstance, is_impulse: bool) -> DefectType:
    """ルールベース分類(§1.3)。is_impulse=時間系検知由来(=ダスト候補)。

    分類軸:
      細長度が高く主軸が垂直      → 縦スクラッチ
      細長度が高く主軸が水平      → 横傷/ドロップアウト
      円形・小面積・時間的単発    → 白/黒ダスト(極性で分岐)
      大面積・不定形             → 大型ゴミ/パーティクル
    """
    elong = ins.elongation
    orient = ins.orientation_deg
    # 垂直: orientation ~ 90 度、水平: ~ 0 or 180 度
    near_vert = min(abs(orient - 90.0), abs(orient - 90.0)) <= 20.0
    near_horiz = (orient <= 20.0) or (orient >= 160.0)

    if elong >= 4.0:
        if near_vert:
            return DefectType.SCRATCH_VERTICAL
        if near_horiz:
            return DefectType.SCRATCH_HORIZONTAL
        return DefectType.SCRATCH_CURVED

    if is_impulse:
        if ins.area >= 200:
            return DefectType.PARTICLE
        return DefectType.DUST_WHITE if ins.polarity > 0 else DefectType.DUST_BLACK

    # 空間のみ由来の点候補は種別を確定できない(時間確認前)
    if ins.area >= 200:
        return DefectType.PARTICLE
    return DefectType.UNKNOWN


def classify_and_filter(instances: list[DefectInstance], is_impulse: bool,
                        max_circularity_for_scratch: float = 0.55):
    """全インスタンスを分類。円形度が高すぎる線候補は棄却(誤検知抑制)。"""
    out: list[DefectInstance] = []
    for ins in instances:
        t = classify_instance(ins, is_impulse)
        # 線と判定したが円形度が高い(=実は塊)なら格下げ
        if t in (DefectType.SCRATCH_VERTICAL, DefectType.SCRATCH_HORIZONTAL,
                 DefectType.SCRATCH_CURVED) and ins.circularity > max_circularity_for_scratch:
            t = DefectType.PARTICLE if ins.area >= 200 else DefectType.UNKNOWN
        ins.type = t
        out.append(ins)
    return out
