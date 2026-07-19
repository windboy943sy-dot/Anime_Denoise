"""検知エンジンのオーケストレータ(カスケード)。

ダスト/スクラッチサーベイ §7.1(高感度候補生成 → 高特異度検証)と §9.1
(2系統×2パス)を実装する。除去は一切行わない(Phase 1 の責務)。

パイプライン:
  A. 解析パス(クリップ全体) : ノイズプロファイル → 傷系統(射影+追跡)
  B. フレーム毎           : ダスト系統(SDI∧ROD + 空間候補) → インスタンス化
                            → 分類・棄却 → 傷台帳を適用 → DefectMap 合成

誤検知抑制の3本柱(§7.1):
  (a) 時間確認(ダスト=単発 / 傷=持続)
  (b) 動き非追従性
  (c) 追跡持続数のヒステリシス
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import spatial, temporal
from .contracts import (DefectInstance, DefectMap, DefectType, DetectorSource,
                        NoiseProfile)
from .instances import build_instances, classify_and_filter
from .noise_profile import estimate_noise_profile, to_luma
from .scratch import ScratchTracker, vertical_projection_scratches


@dataclass
class AnalyzerConfig:
    """検知パラメータ。既定は「正確性優先」(誤検知を抑える保守的な値)。"""
    # ダスト系(時間)
    dust_k: float = 3.0
    use_rod: bool = True
    neighbor_majority: bool = True
    # 空間候補
    tophat_size: int = 7
    tophat_k: float = 4.0
    dog_sigmas: tuple = (1.0, 2.0, 4.0)
    require_spatial_support: bool = True  # 時間∧空間 の AND 強化(§9.1 B2)
    # スクラッチ系
    scratch_k: float = 4.0
    scratch_bands: int = 4
    scratch_min_persistence: int = 5   # 数フレーム未満の検知は棄却(§4.2 最重要ルール)
    scratch_min_confidence: float = 0.7
    # 形状フィルタ
    min_area: int = 2
    max_area: int = 20000
    # 出力
    dust_dilation: int = 1                # マスク膨張(除去段の要求から §7.6)


@dataclass
class ClipAnalysis:
    """クリップ全体の解析結果。"""
    defect_maps: list[DefectMap] = field(default_factory=list)
    noise_profile: NoiseProfile | None = None
    scratch_ledger: list = field(default_factory=list)  # 確定した ScratchTrack
    diagnostics: dict = field(default_factory=dict)


class DefectAnalyzer:
    """クリップ(フレーム列)を解析し DefectMap 列を返す中核クラス。"""

    def __init__(self, config: AnalyzerConfig | None = None):
        self.cfg = config or AnalyzerConfig()

    # ---- フレーム単位: ダスト系統 -------------------------------------
    def _detect_dust_frame(self, frame_t, frame_prev, frame_next,
                           sigma_map, source_acc):
        cfg = self.cfg
        lt = to_luma(frame_t)
        if frame_prev is None or frame_next is None:
            # 端フレーム: 時間検知不可 → 空間候補のみ(感度低下を明示)。§9.2 縮退
            return np.zeros(lt.shape, bool), np.zeros(lt.shape, np.int8), \
                np.zeros(lt.shape, np.float32), DetectorSource.NONE

        t_mask, t_pol, t_str = temporal.detect_dust_temporal(
            frame_t, frame_prev, frame_next, sigma_map,
            k=cfg.dust_k, use_rod=cfg.use_rod, majority=cfg.neighbor_majority)
        src = DetectorSource.SDI | (DetectorSource.ROD if cfg.use_rod
                                    else DetectorSource.TEMPORAL_MEDIAN)

        if cfg.require_spatial_support:
            # 空間候補(Top-Hat)と AND を取り、動き誤差起因の孤立点を弾く
            white, black, _ = spatial.tophat_candidates(
                lt, sigma_map, k=cfg.tophat_k, size=cfg.tophat_size)
            spatial_cand = white | black
            # 空間候補を少し膨張して時間検知の縁を許容
            spatial_cand = ndimage.binary_dilation(spatial_cand, iterations=1)
            t_mask = t_mask & spatial_cand
            src = src | DetectorSource.TOPHAT

        return t_mask, t_pol, t_str, src

    # ---- クリップ単位: 傷系統(解析パス A1) ----------------------------
    def _detect_scratches(self, frames, profile):
        cfg = self.cfg
        tracker = ScratchTracker(min_persistence=cfg.scratch_min_persistence)
        per_frame_cols = []
        for i, f in enumerate(frames):
            cols = vertical_projection_scratches(
                f, sigma=profile.global_sigma, n_bands=cfg.scratch_bands,
                k=cfg.scratch_k)
            per_frame_cols.append(cols)
            tracker.update(cols, i)
        # 持続数＋信頼度ヒステリシスの二重ゲート(§7.1 誤検知抑制の3本柱(c))
        ledger = [tr for tr in tracker.confirmed_tracks()
                  if tr.confidence >= cfg.scratch_min_confidence]
        return ledger, per_frame_cols

    def _apply_scratch_ledger(self, ledger, per_frame_cols, frame_idx, shape,
                              next_id_start):
        """確定傷台帳のうち当該フレームに存在する列を DefectInstance 化。"""
        instances = []
        labels = np.zeros(shape, np.int32)
        nid = next_id_start
        cols = per_frame_cols[frame_idx]
        col_x = np.array([c.x for c in cols]) if cols else np.zeros(0)
        H, W = shape
        for tr in ledger:
            if not (tr.last_frame >= frame_idx and tr.persistence >= self.cfg.scratch_min_persistence):
                pass
            # このフレームに近接する検知列があるか(台帳とライブ検知の一致)
            if col_x.size and np.min(np.abs(col_x - tr.x)) > 4.0:
                continue
            xw = max(1, int(round(tr.width)))
            x0 = int(np.clip(round(tr.x - xw / 2), 0, W - 1))
            x1 = int(np.clip(x0 + xw, 1, W))
            labels[:, x0:x1] = nid
            ins = DefectInstance(
                id=nid,
                type=(DefectType.SCRATCH_VERTICAL),
                bbox=(x0, 0, x1 - x0, H),
                centroid=(float(tr.x), H / 2.0),
                area=int(H * (x1 - x0)),
                elongation=float(H),
                orientation_deg=90.0,
                polarity=int(tr.polarity),
                translucency=0.3,
                first_frame=frame_idx,
                track_id=tr.track_id,
                persistence=tr.persistence,
                confidence=float(tr.confidence),
                sources=DetectorSource.PROJECTION | DetectorSource.PERSISTENCE,
            )
            instances.append(ins)
            nid += 1
        return instances, labels, nid

    # ---- 公開 API ----------------------------------------------------
    def analyze_clip(self, frames: list[np.ndarray],
                     color_space: str = "unknown") -> ClipAnalysis:
        """フレーム列を解析。frames は HxWx3 (0..1 float) を想定。"""
        cfg = self.cfg
        n = len(frames)
        assert n >= 1
        H, W = to_luma(frames[0]).shape

        # A0/A? ノイズプロファイル(数フレームの中央値で安定化)
        prof = estimate_noise_profile(frames[0],
                                      frames[1] if n > 1 else None,
                                      color_space=color_space)
        # 傷系統(解析パス)
        ledger, per_frame_cols = self._detect_scratches(frames, prof)

        out = ClipAnalysis(noise_profile=prof, scratch_ledger=ledger)
        sigma_map = prof.sigma_map(to_luma(frames[0]))

        for i in range(n):
            fmap = DefectMap(width=W, height=H, frame_index=i,
                             noise_profile=prof)
            prob = np.zeros((H, W), np.float32)
            alpha = np.zeros((H, W), np.float32)
            labels = np.zeros((H, W), np.int32)

            fp = frames[i - 1] if i > 0 else None
            fn = frames[i + 1] if i < n - 1 else None
            lt = to_luma(frames[i])
            smap = prof.sigma_map(lt)

            # ダスト系統
            d_mask, d_pol, d_str, src = self._detect_dust_frame(
                frames[i], fp, fn, smap, None)
            dust_instances, dust_labels = build_instances(
                d_mask, lt, polarity_map=d_pol.astype(np.float32),
                strength_map=d_str, source=src,
                min_area=cfg.min_area, max_area=cfg.max_area)
            dust_instances = classify_and_filter(dust_instances, is_impulse=True)

            # 傷系統(台帳適用)
            next_id = (max([ins.id for ins in dust_instances]) + 1
                       if dust_instances else 1)
            scr_instances, scr_labels, _ = self._apply_scratch_ledger(
                ledger, per_frame_cols, i, (H, W), next_id)

            # マスク合成(§7.6: dilation + ソフトエッジ)
            dust_bin = dust_labels > 0
            if cfg.dust_dilation > 0:
                dust_bin = ndimage.binary_dilation(dust_bin, iterations=cfg.dust_dilation)
            prob = np.maximum(prob, dust_bin.astype(np.float32))
            scr_bin = scr_labels > 0
            prob = np.maximum(prob, scr_bin.astype(np.float32) * 0.9)

            # 半透明マップ: インスタンスの alpha を画素へ
            for ins in dust_instances:
                x, y, w, h = ins.bbox
                sub = dust_labels[y:y + h, x:x + w] == ins.id
                alpha[y:y + h, x:x + w][sub] = ins.translucency

            labels = np.where(dust_labels > 0, dust_labels, scr_labels)

            fmap.prob = prob
            fmap.alpha = alpha
            fmap.labels = labels.astype(np.int32)
            fmap.instances = dust_instances + scr_instances
            fmap.compute_frame_stats()
            out.defect_maps.append(fmap)

        out.diagnostics = self._diagnostics(out)
        return out

    def _diagnostics(self, analysis: ClipAnalysis) -> dict:
        """劣化診断(§5.2/§9.1 A3): 種別分布・欠陥密度時系列 → プリセット選択。"""
        hist: dict[str, int] = {}
        densities = []
        for fmap in analysis.defect_maps:
            for t, c in fmap.type_histogram().items():
                hist[t] = hist.get(t, 0) + c
            densities.append(fmap.frame_stats.get("defect_pixel_ratio", 0.0))
        dust_ct = sum(hist.get(t, 0) for t in
                      ("dust_white", "dust_black", "particle"))
        scr_ct = sum(hist.get(t, 0) for t in
                     ("scratch_v", "scratch_h", "scratch_curved"))
        dominant = "dust" if dust_ct > scr_ct else ("scratch" if scr_ct else "clean")
        return {
            "type_histogram": hist,
            "mean_defect_density": float(np.mean(densities)) if densities else 0.0,
            "dominant_defect": dominant,
            "confirmed_scratches": len(analysis.scratch_ledger),
            "noise_model": analysis.noise_profile.dominant_model
            if analysis.noise_profile else "unknown",
            "global_sigma": analysis.noise_profile.global_sigma
            if analysis.noise_profile else 0.0,
        }
