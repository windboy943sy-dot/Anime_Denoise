# 検知・解析エンジン（Phase 1 リファレンス実装）

映像中の **ノイズ・ダスト・スクラッチを高精度に検知・分類・解析**する中核
モジュール。**除去は行わない**（Phase 2-4 の責務）。設計の根拠と採否理由は
`../../docs/detection_engine_architecture.md`、判断・失敗・ベンチは
`../../docs/detection_engine_decision_log.md` を参照。

依存: **numpy, scipy のみ**（OFX/OpenCV 非依存）。C++/OpenFX へ素直に移植できる
よう、画像演算は scipy.ndimage の分離可能フィルタ・モルフォロジー・label に限定。

## モジュール
| ファイル | 役割 |
|---|---|
| `contracts.py` | `DefectMap`/`DefectInstance`/`NoiseProfile`/種別 enum（唯一の対外契約） |
| `noise_profile.py` | ノイズ4軸推定（強度依存σ/空間相関/時間σ・FPN/クロマ） |
| `spatial.py` | DoG・White/Black Top-Hat・Hessian vesselness |
| `temporal.py` | SDI・ROD・時間メディアン・シーンチェンジ・近傍多数決 |
| `instances.py` | 連結成分・形状特徴・ルール分類 |
| `scratch.py` | 帯別垂直射影・ridge/edge 弁別・持続性追跡 |
| `analyzer.py` | オーケストレータ（2系統×2パス、カスケード、診断） |
| `visualize.py` / `io_png.py` | 検知のみオーバーレイ・PNG 出力（stdlib） |
| `synth.py` | GT 付き合成欠陥クリップ（精度検証用） |

## 使い方

```python
from detection_engine import DefectAnalyzer, AnalyzerConfig
analyzer = DefectAnalyzer(AnalyzerConfig())
analysis = analyzer.analyze_clip(frames, color_space="host_working_space")
#   frames: list[H x W x 3 float(0..1)]
for fmap in analysis.defect_maps:          # DefectMap（画素層＋インスタンス層）
    mask = fmap.binary_mask(0.5)           # 除去段はこれを使う
    for ins in fmap.instances:             # UI/分類/追跡はこれを使う
        print(ins.type, ins.bbox, ins.confidence)
print(analysis.diagnostics)                # 劣化診断（プリセット選択用）
print(analysis.noise_profile.global_sigma)
```

CLI（合成デモ・cv2 不要 / 実素材は cv2 必要）:
```bash
python ../run_detection_engine.py --demo --output /tmp/detect_demo
python ../run_detection_engine.py --input clip.mov --output out_dir --max-frames 60
```

## テスト（合成 GT で適合率/再現率を実測）
```bash
python ../tests/test_detection_engine.py     # 全アサーション合格で exit 0
```
2026-07-19 実測: dust 適合率 1.00 / 再現率 ~1.00、傷 recall 1.00・誤検知トラック 0、
ノイズσ相対誤差 <6%。

## 既知の限界
動き補償（MC）未実装＝現状は静止シーン or MC 供給済み入力で正確。動体素材は
`docs/detection_engine_decision_log.md` の H-001/H-002 を参照。
