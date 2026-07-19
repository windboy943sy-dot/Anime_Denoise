"""検知エンジンの精度検証(合成GTに対する適合率/再現率)。

pytest 不要。`python tests/test_detection_engine.py` で自己完結して走り、
測定値を表示し、しきい値(下記 ASSERT)を割ったら異常終了する。
CI(analysis/tests/compare_golden.py と同じ思想)に組み込める。

依存: numpy, scipy のみ。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_engine import DefectAnalyzer, AnalyzerConfig, DefectType
from detection_engine.synth import make_clip


def _pixel_prf(pred_mask, gt_mask):
    tp = int((pred_mask & gt_mask).sum())
    fp = int((pred_mask & ~gt_mask).sum())
    fn = int((~pred_mask & gt_mask).sum())
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


_DUST_TYPES = (DefectType.DUST_WHITE, DefectType.DUST_BLACK,
               DefectType.PARTICLE, DefectType.UNKNOWN)


def evaluate_dust(analysis, gt):
    """ダストのインスタンス級検出率(再現率)＋画素適合率(誤検知の低さ)。

    サーベイ §0.3 のインスタンス層の趣旨に沿う。再現率は「各GTダスト塊が
    検知インスタンスで捉えられたか」、適合率は「検知ダスト画素のうちGT近傍の
    割合」で測る(中間フレームのみ。端は時間検知不可)。
    """
    from scipy import ndimage
    inst_recalls, pixel_precs = [], []
    n = len(analysis.defect_maps)
    for i in range(1, n - 1):
        fmap = analysis.defect_maps[i]
        dust_ids = [ins.id for ins in fmap.instances if ins.type in _DUST_TYPES]
        pred = np.isin(fmap.labels, dust_ids) if dust_ids else np.zeros_like(fmap.labels, bool)
        det_pts = [ins.centroid for ins in fmap.instances if ins.type in _DUST_TYPES]

        # インスタンス級再現率
        gtlist = gt["dust"][i]
        hits = 0
        for (cy, cx, r, pol) in gtlist:
            if any(abs(px - cx) <= r + 3 and abs(py - cy) <= r + 3 for (px, py) in det_pts):
                hits += 1
        inst_recalls.append(hits / len(gtlist) if gtlist else 1.0)

        # 画素適合率(GT を少し膨張した近傍を正解とする)
        gt_mask = ndimage.binary_dilation(gt["dust_mask"][i], iterations=2)
        tp = int((pred & gt_mask).sum())
        fp = int((pred & ~gt_mask).sum())
        pixel_precs.append(tp / (tp + fp) if (tp + fp) else 1.0)

    return np.mean(pixel_precs), np.mean(inst_recalls), 0.0


def evaluate_scratch(analysis, gt, tol=3):
    """確定スクラッチ x 位置の検出率(GT の各傷が台帳に含まれるか)。"""
    ledger_x = np.array([tr.x for tr in analysis.scratch_ledger])
    hits = 0
    for (x, pol) in gt["scratch_x"]:
        if ledger_x.size and np.min(np.abs(ledger_x - x)) <= tol:
            hits += 1
    recall = hits / len(gt["scratch_x"]) if gt["scratch_x"] else 1.0
    # 誤検知: GT に無い確定傷の数
    false_tracks = 0
    gt_x = np.array([x for (x, _) in gt["scratch_x"]])
    for x in ledger_x:
        if not (gt_x.size and np.min(np.abs(gt_x - x)) <= tol):
            false_tracks += 1
    return recall, int(false_tracks), len(analysis.scratch_ledger)


def run_case(name, **clip_kwargs):
    frames, gt = make_clip(**clip_kwargs)
    analyzer = DefectAnalyzer(AnalyzerConfig())
    analysis = analyzer.analyze_clip(frames, color_space="synthetic_linear")
    dp, dr, df = evaluate_dust(analysis, gt)
    sr, sfp, sconf = evaluate_scratch(analysis, gt)
    prof = analysis.noise_profile
    print(f"\n=== case: {name} ===")
    print(f"  noise: model={prof.dominant_model} global_sigma={prof.global_sigma:.4f} "
          f"(GT sigma={gt['sigma']:.4f}) white={prof.is_white} "
          f"corr_len={prof.spatial_correlation_length:.2f}")
    print(f"  dust  : precision={dp:.3f} recall={dr:.3f} f1={df:.3f}")
    print(f"  scratch: recall={sr:.3f} false_tracks={sfp} confirmed={sconf} "
          f"(GT={len(gt['scratch_x'])})")
    print(f"  diagnostics: {analysis.diagnostics['dominant_defect']}, "
          f"density={analysis.diagnostics['mean_defect_density']:.4f}")
    return dict(dust_p=dp, dust_r=dr, dust_f1=df, scr_r=sr, scr_fp=sfp,
               sigma_est=prof.global_sigma, sigma_gt=gt["sigma"])


def main():
    results = {}
    results["static_lowNoise"] = run_case("static_lowNoise", seed=1, sigma=0.015,
                                          motion=False)
    results["static_midNoise"] = run_case("static_midNoise", seed=2, sigma=0.03,
                                          dust_per_frame=8, motion=False)
    results["denser"] = run_case("denser", seed=3, sigma=0.02, dust_per_frame=12,
                                 n_scratches=3, motion=False)

    print("\n================ ASSERTIONS ================")
    ok = True

    def check(cond, msg):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        ok = ok and cond

    for name, r in results.items():
        # ダスト: 再現率(検出漏れの少なさ)を最優先で担保
        check(r["dust_r"] >= 0.95, f"{name}: dust recall >= 0.95 (got {r['dust_r']:.3f})")
        # ダスト: 適合率(誤検知の低さ)。正確性優先の要件。
        check(r["dust_p"] >= 0.90, f"{name}: dust precision >= 0.90 (got {r['dust_p']:.3f})")
        # スクラッチ: 全 GT 傷を検出
        check(r["scr_r"] >= 0.99, f"{name}: scratch recall == 1.0 (got {r['scr_r']:.3f})")
        # スクラッチ: 誤検知トラック 0
        check(r["scr_fp"] == 0, f"{name}: scratch false tracks == 0 (got {r['scr_fp']})")
        # ノイズ sigma 推定の相対誤差
        rel = abs(r["sigma_est"] - r["sigma_gt"]) / r["sigma_gt"]
        check(rel <= 0.6, f"{name}: sigma rel-error <= 0.6 (got {rel:.3f})")

    print("============================================")
    if not ok:
        print("RESULT: FAILED")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
