# GPU化とメモリ管理の設計（Phase 8 準備）

対象環境：**Apple M4**（10コアGPU・Metal 4）、OpenCV 5.0.0（OpenCL: YES / CUDA: なし）。
本書は実装前の設計提案。ユーザー承認後に着手する。

---

## 1. OFXメモリ管理：「無限に増えて見える」問題の解決

### 1.1 現状と、なぜ青天井に見えるか

キャッシュは [animerestore_plugin.cpp](../analysis/ofx/animerestore_plugin.cpp) の `Instance` に2つ：

| キャッシュ | キー | 現状の追い出し | 1エントリのサイズ感（2560×1920）|
|---|---|---|---|
| `cache`（CachedGroup） | (グループ先頭の絶対時刻, 幅) | `size()>16` で**全消し** | 100〜250MB（下記内訳）|
| `detectCache`（DetectResult） | (中心時刻, 幅) | `size()>64` で**全消し** | 数KB（グループ境界のメタのみ）|

CachedGroup 1個の内訳（[animerestore_plugin.cpp:63](../analysis/ofx/animerestore_plugin.cpp)）：
- `analysis->aligned`：位置合わせ済みフレーム群（8UC3, 14MB/枚）。3コマ=42MB、長いホールド10枚=140MB
- `analysis->reference`：32FC3 = 56MB
- `extendedRef`：32FC3 = 56MB、`effectiveN`：32F = 19MB、`blendIn`/`blendOut`：各14MB
- → **合計 100〜250MB/グループ**

**厳密には上限（16個）があるので無限ではない。が、ユーザーが青天井と感じる理由は次の4つ：**

1. **ノコギリ波**：タイムラインをスクラブ／再生すると中心時刻 `t` が動き、グループ先頭時刻ベースの新キーが次々生成される。16個に達するまで単調増加 → 全消し → また増加。監視ツールでは「増え続けて時々ドスンと落ちる」＝実質増加基調に見える。
2. **全消しによるスラッシング**：16に達した瞬間に近傍グループも含めて全破棄するため、直後のレンダーで第2層（近傍±N統合）が再解析を強制され、**破棄と再構築のピークが重なって**瞬間ピークが上限の何倍にもなる。
3. **サイズ別に別キー**：DaVinci はプロキシ／本サイズ／別解像度出力で幅が変わり、`(時刻, 幅)` が別扱いになるため実効エントリ数が増える（プロキシは幅512未満パススルーで一部緩和済みだが、中間解像度は乗る）。
4. **上限が「個数」ゆえ内容量が不定**：16個でも中身が長いホールドグループばかりだと 16×250MB = **4GB**。個数上限はメモリ量を保証しない。

### 1.2 提案：バイト上限つき真のLRU

**方針**：追い出し基準を「個数」から「推定合計バイト数」へ変更し、全消しをやめて**最も長く使われていないエントリから1個ずつ**追い出す。

```cpp
// CachedGroup / DetectResult 共通の推定サイズ（各 cv::Mat の total()*elemSize() 合計）
size_t estimateBytes(const CachedGroup& cg);

struct LruCache {
    // list front = 最近使用、back = 最古。map は key→list iterator
    std::list<std::pair<Key, std::shared_ptr<CachedGroup>>> order;
    std::map<Key, decltype(order)::iterator> index;
    size_t bytes = 0, limitBytes = 512ull << 20;  // 既定512MB（パラメータ化可）
    // get: ヒットなら front へ移動、bytes 据え置き
    // put: 追加後、bytes>limit の間 back を pop（in-flight の shared_ptr は生存）
};
```

- **上限は幅ではなくバイトで**：長いホールドが多くても総量が一定に収まる。
- **1個ずつ追い出す**ので全消しスラッシングが消える。第2層で今まさに使っている近傍グループは直前に `get` されて front にいるため追い出されない（`keep` の shared_ptr も生存を二重に保証）。
- **既定512MBは控えめ**。OFXパラメータ「Cache Budget (MB)」を追加して 256〜4096 で可変にすると、非力なマシンと大容量マシンの両対応。
- `detectCache` は軽量なので個数上限（例128）LRUで十分。

### 1.3 CachedGroup 内の不要データ解放（副次策）

- `blendIn`/`blendOut` は「直前フレームと入力ビット一致なら再利用」のための**1組のメモ**。グループ内で使い回すので保持は正当。ただし NLM を通さないモード（texture / grain=0）では常に空にできる。
- `analysis->aligned` は `renderHoldGroup` が全フレーム出力に使うため保持必須。ただし出力生成が終わったグループを近傍参照にだけ使う場合、aligned は不要で reference/effectiveN だけあればよい → **「参照専用に格下げした隣接グループは aligned を解放」**する軽量化が可能（メモリを最大 6割削減）。第2段階の最適化として。

### 1.4 出力への影響：なし

キャッシュはあくまで**計算結果の再利用**であり、追い出しても次アクセス時に同一入力から再計算されるだけ。追い出し戦略を変えても最終ピクセルは不変。→ ゴールデンパリティ・品質検証に影響しない（この作業は「結果を変えない」制約を自動的に満たす）。

---

## 2. GPU化：NLMの OpenCL(T-API) オフロード

### 2.1 実測（本設計の根拠）

実素材 2560×1920 での `fastNlMeansDenoisingColored`（h=5, template=7, search=21）：

| 処理 | CPU | OpenCL(M4 GPU) | 速度比 |
|---|---|---|---|
| **NLM Colored** | 897 ms/f | **508 ms/f** | **1.76x** |
| NLM Gray（参考） | 329 ms/f | 226 ms/f | 1.46x |

NLM は全処理時間の **79%**（[performance_audit.md](performance_audit.md)）。NLM 単体1.76x → **全体で約1.5x**の高速化見込み。

### 2.2 提案：T-API（UMat）化 ― 第一段階

対象は [denoise.cpp:191](../analysis/src/denoise.cpp) の1呼び出し（`spatialDenoiseEdgePreserving` 内）だけ。

```cpp
// プラグイン onLoad / CLI 起動時に一度：
cv::ocl::setUseOpenCL(true);   // OpenCL が無い環境では自動で false のまま

// spatialDenoiseEdgePreserving 内：
if (cv::ocl::useOpenCL()) {
    cv::UMat uSrc = frame.getUMat(cv::ACCESS_READ), uDst;
    cv::fastNlMeansDenoisingColored(uSrc, uDst, h, h, 7, 21);
    uDst.copyTo(denoised);
} else {
    cv::fastNlMeansDenoisingColored(frame, denoised, h, h, 7, 21);  // 従来
}
```

- **数行・低リスク**。OpenCL無効環境（CI等）は従来CPUパスに自動フォールバック。
- 続く Canny/GaussianBlur/mul も UMat チェーンにすれば CPU↔GPU 転送を1往復に減らせる（第2段階の詰め）。

### 2.3 出力への影響：**ビット非一致になりうる（要方針決定）**

OpenCLカーネルは浮動小数の丸め・加算順序がCPUと異なるため、**NLM出力が±1階調ずれる画素が出る**可能性が高い。これは今までの「結果を変えない」制約に抵触する唯一の論点。対処案：

- **A. golden/パリティはCPU固定**：`generate_golden.py`・`compare_golden.py` は `setUseOpenCL(false)` で走らせ、GPUは実運用時のみ。パリティは従来どおり厳密一致を維持し、GPUパスは別途「CPU出力との差 ≤ 1階調が画素の99.9%以上」という**許容誤差テスト**を新設。
- **B. 品質指標で担保**：GPU出力を verify_denoise_quality.py にかけ、線画IoU・エッジ鮮鋭度・ノイズ削減率がCPU版と同等（線画IoU差<0.02等）であることを確認。数値ビット一致ではなく**知覚品質の同等性**で受け入れる。

推奨は **A+B併用**：パリティ基盤は壊さず（CPU固定）、GPUパスは許容誤差＋品質同等で受け入れる。

### 2.4 スレッド安全

OFXはマルチスレッドレンダー。OpenCLの既定コマンドキューへの同時投入は競合しうる。NLM呼び出しは既に重い単位なので、**NLM区間を専用mutexで直列化**しても並列性の損失は小さい（GPUは元々1つ）。CLIは単一ワーカーなので不要。

### 2.5 Metal native は将来（第3段階）

- macOSのOpenCLはApple非推奨（deprecated）。将来のmacOSで削除されるリスクがある。
- Metal Performance Shaders なら M4 のGPUを最大限使え、1.76x以上が狙える。ただし NLM相当を自前 Metal カーネルで書く必要があり実装コスト大。
- **判断**：まずT-APIで1.76xを確定的に取り、OpenCL廃止が現実味を帯びた時点で MPS 移行を検討。今は着手しない。

---

## 3. 実装順序と工数感（承認後）

| 順 | 作業 | リスク | 出力への影響 | 目安 |
|---|---|---|---|---|
| 1 | OFX cache をバイト上限LRU化（1.2） | 低 | なし | 中 |
| 2 | Cache Budget パラメータ追加（1.2） | 低 | なし | 小 |
| 3 | NLM の T-API 化（2.2） | 中 | 要許容誤差方針（2.3） | 小 |
| 4 | GPU許容誤差テスト＋品質同等確認（2.3） | 低 | ― | 中 |
| 5 | UMatチェーン化・参照専用グループの aligned 解放（2.2/1.3） | 中 | なし/軽微 | 中 |

第3段階（Metal）は本ロードマップ外。
