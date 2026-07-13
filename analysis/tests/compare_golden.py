"""
ゴールデンパリティテスト：C++ 実装（ar_cli）の出力を Python プロトタイプの
出力（tests/golden/）と比較する（docs/cpp_port_design.md 3章）。

合格条件：
  - hold groups：グループ境界の完全一致、pattern の一致
  - blurred_diff / block_ssim：許容誤差 |Δ| ≤ 0.05 / 0.005
  - pHash：ハミング距離 ≤ 4（PILとのリサイズ差を許容。粗選別用途のため）
  - motion：type の完全一致、tx/ty ±1px、scale ±0.005、rot ±0.3°
    （ORB/RANSACの乱数系列がPythonと異なるため数値は許容誤差付き）
  - denoise：参照像R の PSNR > 45dB、grain_sigma ±10%、
    ダストマスクの IoU ≥ 0.7

使い方：
  python tests/compare_golden.py --ar-cli build/ar_cli --golden tests/golden
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar-cli", required=True)
    parser.add_argument("--golden", required=True)
    args = parser.parse_args()

    golden = Path(args.golden) / "synthetic"
    frames_dir = str(golden)
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        det_out = Path(tmp) / "hold_groups.json"
        met_out = Path(tmp) / "metrics.json"
        subprocess.run([args.ar_cli, "detect", "--frames-dir", frames_dir,
                        "--output", str(det_out)], check=True)
        subprocess.run([args.ar_cli, "metrics", "--frames-dir", frames_dir,
                        "--output", str(met_out)], check=True)

        got_det = json.loads(det_out.read_text())
        got_met = json.loads(met_out.read_text())

    exp_det = json.loads((golden / "hold_groups.json").read_text())
    exp_pairs = json.loads((golden / "pair_metrics.json").read_text())
    exp_phash = json.loads((golden / "phash.json").read_text())

    # 1) hold groups
    exp_g = [(g["start"], g["end"], g["pattern"]) for g in exp_det["hold_groups"]]
    got_g = [(g["start"], g["end"], g["pattern"]) for g in got_det["hold_groups"]]
    if exp_g != got_g:
        failures.append(f"hold_groups 不一致:\n  期待={exp_g}\n  実際={got_g}")
    if exp_det["dominant_koma_pattern"] != got_det["dominant_koma_pattern"]:
        failures.append("dominant_koma_pattern 不一致")

    # 2) ペア指標
    for exp, got in zip(exp_pairs, got_met["pairs"]):
        dd = abs(exp["blurred_diff"] - got["blurred_diff"])
        ds = abs(exp["block_ssim"] - got["block_ssim"])
        if dd > 0.05:
            failures.append(f"pair {exp['pair']}: blurred_diff Δ={dd:.4f}")
        if ds > 0.005:
            failures.append(f"pair {exp['pair']}: block_ssim Δ={ds:.5f}")

    # 3) pHash（ハミング距離 ≤ 4）
    for i, (e, g) in enumerate(zip(exp_phash, got_met["phash"])):
        d = hamming_hex(e, g)
        if d > 4:
            failures.append(f"phash frame {i}: hamming={d}")

    # 4) motion（pan / zoom / static の合成シーケンス）
    golden_root = Path(args.golden)
    motion_golden = json.loads((golden_root / "motion_golden.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for name, expected in motion_golden.items():
            mo = Path(tmp) / f"{name}.json"
            subprocess.run([args.ar_cli, "motion",
                            "--frames-dir", str(golden_root / name),
                            "--output", str(mo)], check=True)
            got = json.loads(mo.read_text())
            for k, (e, g) in enumerate(zip(expected, got)):
                if e["type"] != g["type"]:
                    failures.append(f"{name}[{k}]: type {e['type']} != {g['type']}")
                    continue
                if abs(e["tx"] - g["tx"]) > 1.0 or abs(e["ty"] - g["ty"]) > 1.0:
                    failures.append(f"{name}[{k}]: 並進 Δ=({abs(e['tx']-g['tx']):.2f},"
                                    f"{abs(e['ty']-g['ty']):.2f})")
                if abs(e["scale"] - g["scale"]) > 0.005:
                    failures.append(f"{name}[{k}]: scale Δ={abs(e['scale']-g['scale']):.4f}")
                if abs(e["rotation_deg"] - g["rotation_deg"]) > 0.3:
                    failures.append(f"{name}[{k}]: rot Δ")

    # 5) denoise（参照像R・σ・ダストマスク）
    import numpy as np
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        exp_dn = json.loads((golden / "denoise_g2.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([args.ar_cli, "denoise-group",
                            "--frames-dir", frames_dir,
                            "--start", "6", "--end", "8",
                            "--out-dir", tmp], check=True)
            got_dn = json.loads((Path(tmp) / "denoise_group.json").read_text())
            ref_cpp = cv2.imread(str(Path(tmp) / "reference.png"))
            ref_py = cv2.imread(str(golden / "reference_g2.png"))
            mse = float(np.mean((ref_cpp.astype(np.float64) -
                                 ref_py.astype(np.float64)) ** 2))
            psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
            if psnr < 45.0:
                failures.append(f"denoise: 参照像R PSNR={psnr:.1f}dB (<45)")
            sig_e, sig_g = exp_dn["grain_sigma"], got_dn["grain_sigma"]
            if abs(sig_e - sig_g) > 0.1 * max(sig_e, 0.5):
                failures.append(f"denoise: grain_sigma {sig_e} vs {sig_g}")
            mask_cpp = cv2.imread(str(Path(tmp) / "dust_mask_1.png"), 0)
            mask_py = cv2.imread(str(golden / "dust_mask_f7.png"), 0)
            if mask_cpp is None:
                failures.append("denoise: dust_mask_1.png が出力されていない")
            else:
                a, b = mask_cpp > 0, mask_py > 0
                union = np.logical_or(a, b).sum()
                iou = np.logical_and(a, b).sum() / max(union, 1)
                if iou < 0.7:
                    failures.append(f"denoise: ダストマスク IoU={iou:.2f} (<0.7)")

    # 6) defects（傷・ラインノイズ・スキャンノイズ）
    defects_dir = golden_root / "defects"
    if (defects_dir / "expected.json").exists():
        exp_df = json.loads((defects_dir / "expected.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            df_out = Path(tmp) / "defects.json"
            subprocess.run([args.ar_cli, "defects", "--dir", str(defects_dir),
                            "--output", str(df_out)], check=True)
            got_df = json.loads(df_out.read_text())

        exp_cols = sorted(c["x"] for c in exp_df["scratch_columns"])
        got_cols = sorted(c["x"] for c in got_df["scratch_columns"])
        if abs(len(exp_cols) - len(got_cols)) > 1 or not exp_cols or not got_cols or \
                abs(exp_cols[0] - got_cols[0]) > 1 or abs(exp_cols[-1] - got_cols[-1]) > 1:
            failures.append(f"scratch: 期待列={exp_cols} 実際={got_cols}")

        exp_rows = {r["index"]: r["offset"] for r in exp_df["line_noise_rows"]}
        got_rows = {r["index"]: r["offset"] for r in got_df["line_noise_rows"]}
        for idx, off in exp_rows.items():
            if idx not in got_rows:
                failures.append(f"line: 行{idx} 未検出")
            elif abs(got_rows[idx] - off) > 0.5:
                failures.append(f"line: 行{idx} offset {off} vs {got_rows[idx]}")
        for idx in got_rows:
            if idx not in exp_rows:
                failures.append(f"line: 行{idx} は偽検出")

        exp_scan = exp_df["scan_noise"]
        got_scan = got_df["scan_noise"]
        if len(exp_scan) != len(got_scan):
            failures.append(f"scan: 検出数 {len(exp_scan)} vs {len(got_scan)}")
        else:
            for e, g in zip(exp_scan, got_scan):
                if abs(e["period_px"] - g["period_px"]) > 0.5 or \
                        abs(e["amplitude"] - g["amplitude"]) > 0.3:
                    failures.append(f"scan: {e} vs {g}")

    if failures:
        print("ゴールデンパリティ NG:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"ゴールデンパリティ OK "
          f"(groups={len(exp_g)}, pairs={len(exp_pairs)}, phash={len(exp_phash)}, "
          f"motion={sum(len(v) for v in motion_golden.values())}遷移, denoise=R/σ/mask, defects=3種)")


if __name__ == "__main__":
    main()
