"""
C++ 移植（Phase 6）のゴールデンテストデータ生成。

Python プロトタイプを「仕様」とみなし、その出力を正解データとして
analysis/tests/golden/ に固定する。C++ 実装は同じ入力に対して
同じ出力を返すことをテストで担保する（docs/cpp_port_design.md 3章）。

生成物：
  golden/synthetic/frame_%03d.png   入力フレーム（合成: 3コマ×5グループ＋グレイン＋ダスト）
  golden/synthetic/hold_groups.json Phase 1 出力（drift check込み）
  golden/synthetic/phash.json       各フレームのpHash 16進表現
  golden/synthetic/pair_metrics.json 隣接フレームの diff / block_ssim 値
  golden/synthetic/reference_g2.png グループ2の参照像R（trimmed mean）
  golden/synthetic/dust_mask_f7.png グループ2のダストマスク

使い方：
  python tools/generate_golden.py --out ../analysis/tests/golden
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hold_frame_detection import (  # noqa: E402
    DetectionThresholds, detect_hold_groups, estimate_koma_pattern,
    split_drifting_groups, dominant_pattern_for_shot,
)
from hold_frame_detection.core import (  # noqa: E402
    blurred_mean_abs_diff, block_ssim, compute_phash,
)
from denoise import DenoiseParams, analyze_hold_group  # noqa: E402
from motion_classification import MotionThresholds, estimate_global_motion  # noqa: E402
from denoise.scratch import detect_scratch_columns  # noqa: E402
from denoise.linenoise import detect_line_noise  # noqa: E402
from denoise.scannoise import detect_scan_noise  # noqa: E402


def make_synthetic_frames() -> list:
    """3コマ×5グループ、グレイン、1枚だけダスト付きの合成クリップ（320x240）。"""
    rng = np.random.default_rng(1234)
    frames = []
    for content_idx in range(5):
        r = np.random.default_rng(content_idx)
        base = np.full((240, 320, 3), 190, np.uint8)
        x, y = int(r.integers(50, 270)), int(r.integers(50, 190))
        base[max(0, y - 35):y + 35, max(0, x - 35):x + 35] = \
            r.integers(30, 220, 3, np.uint8)
        cv2.line(base, (20, 200), (300, 200 - content_idx * 8), (40, 40, 40), 2)
        for k in range(3):
            noisy = base.astype(np.int16) + rng.integers(-10, 11, base.shape, np.int16)
            frame = np.clip(noisy, 0, 255).astype(np.uint8)
            if content_idx == 2 and k == 1:  # フレーム7にダスト2粒
                cv2.circle(frame, (80, 60), 4, (255, 255, 255), -1)
                cv2.circle(frame, (250, 180), 3, (10, 10, 10), -1)
            frames.append(frame)
    return frames


def make_motion_sequences(out_root: Path) -> dict:
    """動き分類用の合成シーケンス（pan / zoom / static）を生成し、
    Python の estimate_global_motion の出力を golden として返す。"""
    rng = np.random.default_rng(7)
    # 特徴点が十分取れるテクスチャ背景（960x720、work_width=640に縮小される）
    base = np.full((720, 960, 3), 170, np.uint8)
    r = np.random.default_rng(99)
    for _ in range(60):
        x, y = int(r.integers(30, 930)), int(r.integers(30, 690))
        c = tuple(int(v) for v in r.integers(20, 235, 3))
        cv2.circle(base, (x, y), int(r.integers(6, 28)), c, -1)
    for _ in range(20):
        p1 = (int(r.integers(0, 960)), int(r.integers(0, 720)))
        p2 = (int(r.integers(0, 960)), int(r.integers(0, 720)))
        cv2.line(base, p1, p2, (30, 30, 30), 2)

    def noisy(img):
        n = img.astype(np.int16) + rng.integers(-6, 7, img.shape, np.int16)
        return np.clip(n, 0, 255).astype(np.uint8)

    def warp_seq(name, warps):
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, m in enumerate(warps):
            f = cv2.warpAffine(base, m, (960, 720), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            f = noisy(f)
            cv2.imwrite(str(d / f"frame_{i:03d}.png"), f)
            frames.append(f)
        th = MotionThresholds()
        golden = []
        for i in range(1, len(frames)):
            g = estimate_global_motion(frames[i - 1], frames[i], th)
            golden.append({
                "type": g.type,
                "tx": round(g.tx, 2), "ty": round(g.ty, 2),
                "scale": round(g.scale, 4),
                "rotation_deg": round(g.rotation_deg, 3),
            })
        return golden

    ident = np.float32([[1, 0, 0], [0, 1, 0]])
    seqs = {}
    seqs["motion_pan"] = warp_seq("motion_pan", [
        np.float32([[1, 0, 8 * i], [0, 1, 3 * i]]) for i in range(6)])
    zooms = []
    for i in range(6):
        s = 1.012 ** i
        zooms.append(np.float32([[s, 0, (1 - s) * 480], [0, s, (1 - s) * 360]]))
    seqs["motion_zoom"] = warp_seq("motion_zoom", zooms)
    seqs["motion_static"] = warp_seq("motion_static", [ident] * 4)
    return seqs


def make_defects_golden(out_root: Path) -> dict:
    """傷・ラインノイズ・スキャンノイズの合成入力と Python 検出結果の golden。"""
    d = out_root / "defects"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    r = np.random.default_rng(5)
    base = np.full((480, 640, 3), 175, np.float32)
    for _ in range(40):
        x, y = int(r.integers(20, 620)), int(r.integers(20, 460))
        c = [float(v) for v in r.integers(40, 230, 3)]
        cv2.circle(base, (x, y), int(r.integers(8, 30)), c, -1)

    def noisy(img):
        return np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.float32)

    # (a) 傷：内容が横に動く3枚の参照像に、固定x=400の暗い縦線
    refs = []
    for i in range(3):
        m = np.float32([[1, 0, 30 * i], [0, 1, 0]])
        f = cv2.warpAffine(base, m, (640, 480), borderMode=cv2.BORDER_REFLECT)
        f = noisy(f)
        f[:, 400:402] *= 0.5
        cv2.imwrite(str(d / f"scratch_ref_{i}.png"),
                    np.clip(f, 0, 255).astype(np.uint8))
        refs.append(f)
    scratch = detect_scratch_columns(refs)

    # (b) ラインノイズ：行150に+7、行300に-5
    ln_img = noisy(base.copy())
    ln_img[150, :, :] += 7.0
    ln_img[300, :, :] -= 5.0
    cv2.imwrite(str(d / "linenoise.png"), np.clip(ln_img, 0, 255).astype(np.uint8))
    rows = detect_line_noise(np.clip(ln_img, 0, 255).astype(np.float32), axis=0)

    # (c) スキャンノイズ：周期16px・振幅2.0の水平縞
    sn_img = base + 2.0 * np.sin(2 * np.pi * np.arange(480) / 16.0)[:, None, None]
    sn_img = noisy(sn_img)
    cv2.imwrite(str(d / "scannoise.png"), np.clip(sn_img, 0, 255).astype(np.uint8))
    scan = detect_scan_noise(np.clip(sn_img, 0, 255).astype(np.float32), axis=0)

    golden = {
        "scratch_columns": scratch["columns"],
        "line_noise_rows": rows,
        "scan_noise": [{k: v for k, v in s.items() if k != "bin"} for s in scan],
    }
    with open(d / "expected.json", "w") as fp:
        json.dump(golden, fp, indent=1, ensure_ascii=False)
    return golden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out) / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    frames = make_synthetic_frames()
    for i, f in enumerate(frames):
        cv2.imwrite(str(out / f"frame_{i:03d}.png"), f)

    thresholds = DetectionThresholds()

    # pHash（16進文字列。imagehash の __str__ と同じ表現）
    phashes = [str(compute_phash(f)) for f in frames]
    with open(out / "phash.json", "w") as fp:
        json.dump(phashes, fp, indent=1)

    # 隣接ペアの diff / block_ssim（C++の数値パリティ検証用）
    pair_metrics = []
    for i in range(1, len(frames)):
        pair_metrics.append({
            "pair": [i - 1, i],
            "blurred_diff": round(
                blurred_mean_abs_diff(frames[i - 1], frames[i], thresholds.blur_ksize), 4),
            "block_ssim": round(
                block_ssim(frames[i - 1], frames[i], thresholds.block_size,
                           blur_ksize=thresholds.blur_ksize), 5),
        })
    with open(out / "pair_metrics.json", "w") as fp:
        json.dump(pair_metrics, fp, indent=1)

    # Phase 1 出力（drift check 込み）
    groups = detect_hold_groups(frames, thresholds)
    groups = split_drifting_groups(frames, groups, thresholds)
    groups = estimate_koma_pattern(groups)
    result = {
        "dominant_koma_pattern": dominant_pattern_for_shot(groups),
        "hold_groups": [
            {"start": g.start, "end": g.end, "pattern": g.pattern,
             "confidence": round(g.confidence, 3)}
            for g in groups
        ],
    }
    with open(out / "hold_groups.json", "w") as fp:
        json.dump(result, fp, indent=1)

    # グループ2（frames 6-8）の参照像とダストマスク
    p = DenoiseParams()
    an = analyze_hold_group(frames[6:9], p)
    cv2.imwrite(str(out / "reference_g2.png"),
                np.clip(an["reference"], 0, 255).astype(np.uint8))
    if an["dust_masks"][1] is not None:
        cv2.imwrite(str(out / "dust_mask_f7.png"), an["dust_masks"][1])

    # グループ2のグレインσ（denoise パリティの数値比較用）
    with open(out / "denoise_g2.json", "w") as fp:
        json.dump({"grain_sigma": round(an["grain_sigma"], 3),
                   "dust_px_f7": int((an["dust_masks"][1] > 0).sum())}, fp, indent=1)

    # 動き分類の golden（pan / zoom / static の合成シーケンス）
    motion_golden = make_motion_sequences(Path(args.out))
    with open(Path(args.out) / "motion_golden.json", "w") as fp:
        json.dump(motion_golden, fp, indent=1)

    defects_golden = make_defects_golden(Path(args.out))

    print(f"golden データを生成: {out}")
    print(f"  hold groups: {len(groups)} / dominant={result['dominant_koma_pattern']}")
    print(f"  dust f7: {int((an['dust_masks'][1] > 0).sum())} px")
    for name, g in motion_golden.items():
        print(f"  {name}: {[x['type'] for x in g]}")
    print(f"  scratch: {[c['x'] for c in defects_golden['scratch_columns']]}")
    print(f"  line: {[(r['index'], r['offset']) for r in defects_golden['line_noise_rows']]}")
    print(f"  scan: {[(s['period_px'], s['amplitude']) for s in defects_golden['scan_noise']]}")


if __name__ == "__main__":
    main()
