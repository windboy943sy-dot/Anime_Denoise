# 映像復元・画質向上技術 総合調査レポート
### DaVinci Resolve向けOpenFXプラグイン開発のための基礎資料（レベル3調査）

作成日: 2026年7月13日

対象: Video/Image Denoising, Video/Image Restoration, Super Resolution, Deblurring, Deartifacting, Film Restoration, Temporal Filtering, Motion Estimation/Optical Flow, Frame Interpolation, Multi-frame Restoration, AI Restoration, Classical Image Processing, Anime Restoration

凡例: 【基礎】= 分野を確立した基礎研究、【最新】= 過去3〜5年の重要研究。実装難易度は OpenFX(C++/CUDA/OpenCL/Metal) 実装を前提に ★☆☆☆☆（易）〜★★★★★（難）で評価。

---

## 1. 学術論文

### 1.1 古典的手法（空間・時間フィルタ系）【基礎】

#### Non-Local Means（NLM）
- **タイトル**: A Non-Local Algorithm for Image Denoising
- **著者**: Antoni Buades, Bartomeu Coll, Jean-Michel Morel
- **発表年**: 2005
- **学会**: IEEE CVPR 2005, Vol.2, pp.60–65
- **DOI**: 10.1109/CVPR.2005.38
- **要旨**: 画像内の類似パッチをブロック単位で探索し、類似度に応じた重み付き平均でノイズ除去を行う。局所平滑化ではなく画像全体の反復構造を利用する点が革新的。
- **主なアルゴリズム**: パッチ類似度（SSD）に基づく非局所加重平均
- **長所**: エッジ・テクスチャ保持性能が高い。実装が比較的単純。GPU並列化が容易（パッチ探索は並列処理向き）。
- **短所**: 探索窓が広いと計算コストが O(N²) 相当で増大。フラットな低テクスチャ領域では効果が限定的。
- **実装難易度**: ★★☆☆☆（CUDA/OpenCLでの高速化例が豊富）
- **OpenFXへの応用可能性**: 非常に高い。KNLMeansCLなど既存OSS実装があり、そのままGPUカーネル移植が可能。アニメのベタ塗り領域にも有効。

#### BM3D
- **タイトル**: Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering
- **著者**: Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, Karen Egiazarian
- **発表年**: 2007
- **学会**: IEEE Transactions on Image Processing, Vol.16, No.8, pp.2080–2095
- **DOI**: 10.1109/TIP.2007.901238
- **要旨**: 類似ブロックをグルーピングし3D配列化、変換領域（DCT/Wavelet）でのハード閾値処理とWienerフィルタで協調的にノイズ除去後、加重平均で再構成。長年にわたり画像denoisingのデファクトスタンダード。
- **主なアルゴリズム**: Block-matching + 3D変換協調フィルタリング（2段階: ハード閾値 → Wiener）
- **長所**: 非常に高いPSNR/視覚品質。ノイズモデル（AWGN）が既知なら理論的裏付けも強い。
- **短所**: 計算コストが高く、リアルタイム処理には工夫が必要。パラメータ（ブロックサイズ・探索窓）のチューニングが必要。
- **実装難易度**: ★★★★☆（GPU実装は bm3d-gpu, bm3dcuda 等の先行実装を参考にできる）
- **OpenFXへの応用可能性**: 高い。特にフィルムスキャン・実写ノイズ除去用の「高品質・低速」プリセットとして有用。CUDA版が既に存在するため移植の敷居はやや低い。

#### VBM3D / VBM4D
- **タイトル**: Video Denoising, Deblocking, and Enhancement Through Separable 4-D Nonlocal Spatiotemporal Transforms
- **著者**: Matteo Maggioni, Giacomo Boracchi, Alessandro Foi, Karen Egiazarian
- **発表年**: 2012
- **学会**: IEEE Transactions on Image Processing, Vol.21, No.9
- **要旨**: BM3Dを時空間（動画）に拡張。空間だけでなく時間方向にも類似ブロック（ボリューム）をグルーピングし4D変換でフィルタリング。VBM3Dは2D+時間、VBM4Dはボリューム単位でグルーピングしノイズレベル推定も内包。
- **主なアルゴリズム**: 時空間ブロックマッチング + 4D協調フィルタリング
- **長所**: 動画特有の時間相関を利用でき、BM3Dフレーム単独処理より高精度。
- **短所**: 計算コストが非常に高い。動きが大きいシーンでは探索精度が低下しフリッカーの原因になり得る。
- **実装難易度**: ★★★★★
- **OpenFXへの応用可能性**: 中〜高。品質重視の「高品質デノイズ」モードの参考実装として有用だが、リアルタイムプレビューには不向きなためタイル分割・非同期処理の設計が必須。

#### Guided Filter
- **タイトル**: Guided Image Filtering
- **著者**: Kaiming He, Jian Sun, Xiaoou Tang
- **発表年**: 2010（TPAMI版は2013）
- **学会**: ECCV 2010 / IEEE TPAMI Vol.35, No.6, 2013
- **要旨**: ガイド画像との局所線形モデルに基づくエッジ保持平滑化フィルタ。Bilateral Filterと似た効果を持ちながら、カーネルサイズに依存しない線形時間アルゴリズムを実現。
- **主なアルゴリズム**: 局所線形回帰（box filterのみで構成、O(N)）
- **長所**: 非常に高速（積分画像で実装可）。ハロー（勾配逆転）アーティファクトが少ない。デノイズ、ディテール強調、マット処理など多用途。
- **短所**: 単体では強いランダムノイズの除去力は弱く、前処理・後処理との組み合わせが前提。
- **実装難易度**: ★☆☆☆☆
- **OpenFXへの応用可能性**: 非常に高い。軽量なリアルタイムプレビュー用エッジ保持平滑化、デノイズ後のディテール復元、Dehaze等の基礎コンポーネントとして直接移植しやすい。

#### Bilateral Filter / Wavelet Denoising【基礎・補足】
- 古典的なエッジ保持平滑化（Bilateral: Tomasi & Manduchi, 1998）とWavelet閾値denoising（Donoho & Johnstone, 1994「Wavelet Shrinkage」）は、BM3D以前のスタンダード。現在でも軽量プレビューや高速プリフィルタとして実務で使われる。実装難易度★☆☆☆☆〜★★☆☆☆で、OFXのCPU/GPUカーネル入門に最適。

---

### 1.2 CNNベースの画像・動画復元【基礎→最新の橋渡し】

#### DnCNN
- **タイトル**: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
- **著者**: Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, Lei Zhang
- **発表年**: 2017
- **学会**: IEEE Transactions on Image Processing, Vol.26, No.7
- **要旨**: 残差学習とBatch Normalizationを用いたCNNでノイズ成分そのものを推定するdenoiser。単一モデルでBlind denoisingを実現し、以降のディープラーニングdenoisingの基礎形となった。
- **長所**: BM3Dを上回る精度、推論が高速。
- **短所**: 学習データ分布外のノイズ（実センサーノイズ等）に弱い。
- **実装難易度**: ★★☆☆☆（ONNX/TensorRT化が容易）
- **OpenFXへの応用可能性**: 高い。軽量でリアルタイム推論可能なため「AIデノイズ・軽量版」の土台に適する。

#### EDVR
- **タイトル**: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
- **著者**: Xintao Wang, Kelvin C.K. Chan, Ke Yu, Chao Dong, Chen Change Loy
- **発表年**: 2019
- **学会**: CVPR Workshops 2019（NTIRE、全4部門優勝）
- **URL**: https://arxiv.org/abs/1905.02716
- **要旨**: Pyramid, Cascading and Deformable (PCD) アライメントモジュールでフレーム間位置ずれを特徴量レベルで補正し、Temporal-Spatial Attention (TSA) で複数フレームを融合。SR・deblur・denoiseに汎用的に使える枠組み。
- **長所**: マルチフレーム復元の汎用フレームワークとして極めて高精度。
- **短所**: 計算・メモリコストが大きく、リアルタイム性に乏しい。Deformable Convのカスタムカーネルが必要。
- **実装難易度**: ★★★★☆
- **OpenFXへの応用可能性**: 中。アーキテクチャそのものの実装よりも「マルチフレーム・アライメント＋融合」という設計思想がプラグイン設計の参考になる。

#### FastDVDnet
- **タイトル**: FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation
- **著者**: Matias Tassano, Julie Delon, Thomas Veit
- **発表年**: 2020
- **学会**: CVPR 2020
- **URL**: https://arxiv.org/abs/1907.01361 / https://github.com/m-tassano/fastdvdnet
- **要旨**: 明示的なオプティカルフロー推定を行わず、2段階のU-Net的ブロックで前後フレーム情報を暗黙的に統合し高速なリアルタイム級ビデオデノイズを実現。
- **長所**: 動き補償を省略することで大幅な高速化。単一モデルで広いノイズレベルに対応。
- **短所**: 明示的なフロー推定を行う手法と比べ、大きな動きのあるシーンでの精度はやや劣る。
- **実装難易度**: ★★☆☆☆（軽量、GPUリアルタイム推論向き）
- **OpenFXへの応用可能性**: 非常に高い。フレーム遅延（前後数フレームの参照）を許容できるOFX Temporal Access APIと相性が良く、実装コストと品質のバランスが良い「実用最有力候補」。

#### SwinIR
- **タイトル**: SwinIR: Image Restoration Using Swin Transformer
- **著者**: Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte
- **発表年**: 2021
- **学会**: ICCVW 2021 (AIM)
- **URL**: https://arxiv.org/abs/2108.10257
- **要旨**: Swin Transformerを基盤としたResidual Swin Transformer Block (RSTB) を用い、SR・denoise・JPEGアーティファクト除去を1つのバックボーンで扱う。CNNベースSOTAをパラメータ数67%削減しつつ上回る。
- **長所**: 汎用性が高く軽量。多数の派生研究のベースラインになっている。
- **短所**: Transformerゆえメモリアクセスパターンが複雑でGPUカーネル最適化の難度が高い。
- **実装難易度**: ★★★★☆
- **OpenFXへの応用可能性**: 中。学習済みモデルをONNX/TensorRT経由で推論のみ組み込む形が現実的（フルスクラッチ実装は非推奨）。

#### Restormer
- **タイトル**: Restormer: Efficient Transformer for High-Resolution Image Restoration
- **著者**: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang
- **発表年**: 2022（CVPR Oral）
- **URL**: https://github.com/swz30/Restormer
- **要旨**: チャンネル方向の転置Attention（MDTA）とGated-Dconv FFN（GDFN）により、計算量を線形に抑えつつ高解像度画像に適用可能なTransformerを構築。deblur・derain・denoiseでSOTA。
- **長所**: 高解像度画像に対して現実的な計算コストでTransformerの表現力を活用。
- **短所**: 学習コストが高い。動画の時間一貫性は考慮外（フレーム単体処理）。
- **実装難易度**: ★★★★☆
- **OpenFXへの応用可能性**: 中。静止画・単フレーム処理としての推論組み込みは現実的。時間方向の一貫性は別途フリッカー抑制フィルタと組み合わせる必要あり。

#### NAFNet
- **タイトル**: Simple Baselines for Image Restoration
- **著者**: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun
- **発表年**: 2022
- **学会**: ECCV 2022
- **URL**: https://arxiv.org/abs/2204.04676
- **要旨**: 非線形活性化関数（ReLU/GELU等）を排除しSimpleGate等の乗算演算で置換した「Nonlinear Activation Free Network」を提案。UNetバックボーンでdeblur/denoiseにおいてSOTAかつ計算コストは大幅減（GoPro deblurで従来比8.4%の計算量でSOTA超え）。
- **長所**: シンプルな構造で高精度・高速。実装・移植が容易。
- **短所**: 動画向けの時間モデリングは含まれない（画像復元がベース）。
- **実装難易度**: ★★☆☆☆（設計がシンプルでOFX/CUDA移植の学習コストが低い）
- **OpenFXへの応用可能性**: 非常に高い。「シンプルな構造＝実装しやすい＝プラグイン開発の最初のAIモデル」として最有力。

---

### 1.3 動画復元 Transformer / 大規模モデル【最新】

#### VRT: A Video Restoration Transformer
- **著者**: Jingyun Liang, Jiezhang Cao, Yuchen Fan, Kai Zhang, Rakesh Ranjan, Yawei Li, Radu Timofte, Luc Van Gool
- **発表年**: 2022
- **URL**: https://arxiv.org/abs/2201.12288 / https://github.com/JingyunLiang/VRT
- **要旨**: Temporal Mutual Self Attention (TMSA) とParallel Warpingにより長距離時間依存性を並列にモデル化。VSR・deblur・denoise・frame interpolation・space-time SRを単一アーキテクチャで達成するマルチタスクモデル。
- **長所**: 動画復元タスクを横断する汎用性、長距離時間依存の捕捉力。
- **短所**: 巨大なメモリ使用量、推論コストが非常に高い。
- **実装難易度**: ★★★★★
- **OpenFXへの応用可能性**: 低〜中（研究参照用）。フル実装は非現実的だが、後述のRVRTと合わせてアーキテクチャ設計思想の参考になる。

#### RVRT: Recurrent Video Restoration Transformer with Guided Deformable Attention
- **著者**: Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, Luc Van Gool
- **発表年**: 2022
- **学会**: NeurIPS 2022
- **URL**: https://arxiv.org/abs/2206.02146 / https://github.com/JingyunLiang/RVRT
- **要旨**: VRTの並列処理の限界（メモリ）とRecurrentモデルの長距離依存性の弱さを両立するため、クリップ単位のRecurrent構造＋Guided Deformable Attentionでクリップ間アライメントを実現。VSR/deblur/denoiseでバランスの取れたSOTA。
- **長所**: VRTよりメモリ・速度効率が良く、長距離依存もある程度維持。
- **短所**: 依然として重量級。リアルタイム性は低い。
- **実装難易度**: ★★★★★
- **OpenFXへの応用可能性**: 低〜中。オフライン「最高品質モード」向けの推論エンジン組み込みが現実的な範囲。

#### BasicVSR++
- **著者**: Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy
- **発表年**: 2022
- **学会**: CVPR 2022, pp.5972–5981
- **URL**: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- **要旨**: 二次伝播（Second-order grid propagation）とフロー誘導Deformable Alignmentにより、時間方向の情報伝播とフレーム間アライメントを強化したRecurrent VSRモデル。NTIRE2021優勝。
- **長所**: Transformer系より軽量で実装・学習が現実的。VSR分野の事実上の標準ベースライン。
- **短所**: オプティカルフロー推定への依存があり、フロー精度が低いシーン（アニメ等）で性能低下しやすい。
- **実装難易度**: ★★★☆☆
- **OpenFXへの応用可能性**: 高い。既存のRealBasicVSR（実写向け劣化対応版）も含め、学習済みモデルの推論パイプライン化が比較的現実的。

#### Real-ESRGAN
- **タイトル**: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
- **著者**: Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan
- **発表年**: 2021
- **学会**: ICCVW 2021 (AIM)
- **URL**: https://arxiv.org/abs/2107.10833
- **要旨**: 実世界の複雑な劣化（ブラー・ノイズ・圧縮・リンギング）を高次（二段階）劣化モデルで合成データ生成し、U-Net判別器で学習。実写「Blind SR」の事実上の標準ツールとなった。
- **長所**: 実運用で安定した汎用性。推論実装（NCNN/Vulkan含む）が極めて豊富。
- **短所**: ディテールを「生成」する性質上、過剰なテクスチャ付加（over-sharpening/hallucination）が起きやすい。
- **実装難易度**: ★★☆☆☆（推論エンジン移植の実績多数）
- **OpenFXへの応用可能性**: 非常に高い。既にNCNN-Vulkan版があり、GPU（Vulkan/CUDA）ベースのOFXプラグインへの移植実績が豊富（Video2X等）。

#### StableSR
- **タイトル**: Exploiting Diffusion Prior for Real-World Image Super-Resolution
- **著者**: Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C.K. Chan, Chen Change Loy
- **発表年**: 2023（IJCV 2024版）
- **URL**: https://arxiv.org/abs/2305.07015
- **要旨**: 事前学習済みText-to-Image拡散モデル（Stable Diffusion）の生成的事前分布を凍結したまま、時間認識エンコーダとControllable Feature Wrappingでブラインド超解像に適用。品質と忠実度をスカラー値で調整可能。
- **長所**: 圧倒的な高精細・高品質なディテール生成、劣化に頑健。
- **短所**: 拡散モデル特有の多ステップ推論による低速さ、リアルタイム性皆無、幻覚（実在しないディテール生成）のリスク。
- **実装難易度**: ★★★★★（Stable Diffusion本体への依存が大きい）
- **OpenFXへの応用可能性**: 低（現行）。将来的なオフラインバッチ処理用「最高品質・低速」オプションとしての採用余地はあるが、GPU VRAM要件と特許・ライセンス確認が必須。

#### 動画復元サーベイ論文【必読】
- **タイトル**: Video Restoration Based on Deep Learning: A Comprehensive Survey
- **学会**: Artificial Intelligence Review (Springer), 2022
- **URL**: https://link.springer.com/article/10.1007/s10462-022-10302-5
- **要旨**: 動画denoising/deblur/SR/圧縮アーティファクト除去に関するディープラーニング手法を体系的に分類し、各タスクのデータセット・評価指標・代表手法を横断比較。研究の全体像を掴む出発点として最適。
- **実装難易度**: — （サーベイのため実装対象ではない）
- **OpenFXへの応用可能性**: 学習ロードマップの起点として非常に有用。

---

### 1.4 オプティカルフロー・フレーム補間【基礎＋最新】

#### RAFT
- **タイトル**: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- **著者**: Zachary Teed, Jia Deng
- **発表年**: 2020
- **学会**: ECCV 2020, pp.402–419
- **要旨**: 全ピクセルペアの4Dコリレーションボリュームを構築し、Recurrentユニットで反復的にフローを更新。以降のオプティカルフロー研究の事実上の標準ベースラインとなった。
- **長所**: 高精度、汎化性能が高い、実装がオープンで移植例が豊富。
- **短所**: 全ペア相関のためメモリ消費が大きい（高解像度で顕著）。
- **実装難易度**: ★★★☆☆
- **OpenFXへの応用可能性**: 高い。動き推定が必要な時間フィルタ（VBM4D的処理、フレーム補間、手ブレ復元）の基盤として組み込み価値が高い。

#### RIFE
- **タイトル**: Real-Time Intermediate Flow Estimation for Video Frame Interpolation
- **著者**: Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou
- **発表年**: 2022（ECCV）/ arXiv 2020
- **URL**: https://arxiv.org/abs/2011.06294
- **要旨**: 事前学習済みフローモデルに頼らず、IFNetで中間フローを直接end-to-end推定。特権蒸留（Privileged Distillation）で訓練を安定化。SuperSlomo/DAINと比較して4〜27倍高速。
- **長所**: 軽量・高速でリアルタイム性が高い。任意タイムステップ補間対応。
- **短所**: 極端な大変位・オクルージョンには弱い。アニメのような非線形・誇張された動きには追加のドメイン適応が必要（後述SAFA/Practical-RIFE）。
- **実装難易度**: ★★☆☆☆（ncnn-vulkan版など移植実績多数）
- **OpenFXへの応用可能性**: 非常に高い。フレームレート変換・スローモーション生成プラグインとして即戦力。

#### FILM: Frame Interpolation for Large Motion
- **著者**: Fitsum Reda, Janne Kontkanen, Eric Tabellion, Deqing Sun, Caroline Pantofaru, Brian Curless（Google Research）
- **発表年**: 2022
- **学会**: ECCV 2022
- **URL**: https://github.com/google-research/frame-interpolation
- **要旨**: 追加のフロー/深度ネットワークに頼らない単一ネットワーク構成。スケール共有特徴抽出器で「粗いスケールの小変位」と「細かいスケールの大変位」を同一に扱うScale-Agnosticなモーション推定を実現し、大変位補間で高品質。
- **長所**: 大きな動きに強い、追加モデル不要でシンプル。
- **短所**: TensorFlowベースで元実装のC++/GPU移植コストがやや高い。
- **実装難易度**: ★★★☆☆
- **OpenFXへの応用可能性**: 高い。RIFEと並ぶフレーム補間の有力候補。特に大きな動きが多い実写素材向け。

#### AnimeInterp（アニメ特化）
- **タイトル**: Deep Animation Video Interpolation in the Wild
- **著者**: Li Siyao, Shiyu Zhao, Weijiang Yu, Wenxiu Sun, Dimitris Metaxas, Chen Change Loy, Ziwei Liu
- **発表年**: 2021
- **学会**: CVPR 2021
- **URL**: https://arxiv.org/abs/2104.02495
- **要旨**: アニメ特有の「線とベタ塗りで構成されテクスチャが乏しい」「誇張表現による非線形・大変位動作」という課題に対し、Segment-Guided Matching（色領域単位のマッチング）とRecurrent Flow Refinementを提案。ATD-12Kという大規模アニメ補間データセットも公開。
- **長所**: アニメに特化した設計で実写向けモデルより高精度。
- **短所**: 3コマ打ち等の非等間隔フレームには追加の考慮が必要。線画の細さにより誤マッチングが起きやすい。
- **実装難易度**: ★★★☆☆
- **OpenFXへの応用可能性**: 高い（アニメ特化プラグインを狙うなら最重要参考実装）。

---

## 2. 書籍

| 書籍名 | 対象読者 | 難易度 | 学べる内容 | おすすめ度 |
|---|---|---|---|---|
| **Digital Image Processing (4th Ed.)** – Rafael C. Gonzalez, Richard E. Woods | 学部上級〜大学院、画像処理入門者 | ★★☆☆☆ | 空間・周波数領域フィルタ、ノイズモデル、復元理論、形態学処理の体系的基礎 | ★★★★★（画像処理の土台として必読） |
| **Computer Vision: Algorithms and Applications (2nd Ed.)** – Richard Szeliski（無料PDF公開: szeliski.org/Book） | CVを本格的に学ぶ学生・実務者 | ★★★☆☆ | 特徴点、光学フロー、ステレオ、深層学習ベースCVまで網羅する現代的教科書 | ★★★★★（現在の版はDeep Learningも大幅加筆） |
| **Digital Video and HD: Algorithms and Interfaces** – Charles Poynton | 映像信号処理・色再現の実務者 | ★★★☆☆ | サンプリング、色空間、ガンマ、映像フォーマットの正確な理解（映像プラグイン開発の必須知識） | ★★★★★（OpenFX/Resolve開発者必携） |
| **Deep Learning** – Ian Goodfellow, Yoshua Bengio, Aaron Courville（無料公開: deeplearningbook.org） | ディープラーニング理論を学ぶ全般 | ★★★★☆ | CNN、最適化、正則化、生成モデルの数学的基礎 | ★★★★☆（CNNベース復元手法の理論的裏付け） |
| **Understanding Digital Signal Processing** – Richard G. Lyons | 信号処理の実務者・組込み開発者 | ★★☆☆☆ | FFT、フィルタ設計、標本化定理などDSPの直感的理解 | ★★★★☆（Wavelet/周波数領域フィルタの理解に有用） |
| **Numerical Recipes** – William H. Press et al. | C++実装を伴う数値計算の実務者 | ★★★☆☆ | FFT・線形代数・最適化のアルゴリズム実装詳細 | ★★★☆☆（OFXでの自前実装時の実用リファレンス） |
| **GPU Gems / GPU Pro シリーズ** – NVIDIA他編 | GPUカーネル開発者 | ★★★☆☆ | CUDA/シェーダでの画像処理最適化パターン | ★★★★☆（OFX GPU実装の実務ノウハウ集） |

---

## 3. 技術記事・ブログ（信頼性の高い一次情報源）

- **NVIDIA Technical Blog / NVIDIA Research** — developer.nvidia.com/blog（tag: Denoising, Super Resolution）。OptiX AI Denoiser、DLSS 4（Transformer化されたSuper Resolution/Ray Reconstruction）、Maxine Video Effects SDKなど、GPU推論最適化の一次情報。OFX開発でのCUDA最適化ノウハウの宝庫。
- **Google Research Blog** — research.google/blog。FILM（フレーム補間）などの解説記事、モデル設計思想が詳しい。
- **Adobe Research（AI & Machine Learning）** — research.adobe.com/research/artificial-intelligence-machine-learning。映像リマスター・コンテンツ理解関連の研究解説。
- **Blackmagic Design Developer Portal** — blackmagicdesign.com/developer、DaVinci Resolve内 Help→Documentation→Developer。OFX/DaVinci Resolve SDK、GPU（CUDA/OpenCL/Metal）カーネルサンプル一次情報。
- **OpenFX公式（Academy Software Foundation）** — openeffects.org、github.com/AcademySoftwareFoundation/openfx。OFX仕様そのものの一次ドキュメント。
- **ResolveCafe** — resolve.cafe/developers/openfx。DaVinci Resolve OFX開発コミュニティの実践的ノウハウ。
- **XPixelGroup（BasicSR）Wiki/Docs** — 実装レベルでのSOTAモデル解説が豊富。
- **NTIRE Workshop（CVPR併設）** — cvlai.net/ntire。超解像・denoise・deblurの毎年のチャレンジ結果と手法サーベイ。最新動向を追う定点観測に最適。
- **AmusementClub / VapourSynthコミュニティ** — github.com/AmusementClub、アニメ・フィルムリストア実務者コミュニティによる高品質OSSフィルタ（denoise/deband/upscale）の一次情報源。

---

## 4. オープンソース実装

| リポジトリ | URL | ライセンス | 実装アルゴリズム | 活発度 | OpenFXへの移植しやすさ |
|---|---|---|---|---|---|
| **BasicSR** | github.com/XPixelGroup/BasicSR | Apache-2.0 | EDSR/RCAN/ESRGAN/EDVR/BasicVSR/SwinIR等の統合フレームワーク | 高（Star多数、継続更新） | 中：PyTorch依存だがONNX変換前提でC++推論部を分離すれば移植可 |
| **Real-ESRGAN** | github.com/xinntao/Real-ESRGAN | BSD-3-Clause | 実写/アニメ向けBlind SR（アニメ専用モデル同梱） | 高 | 高：ncnn-vulkan版が既に存在し直接的な参考実装になる |
| **RIFE (Practical-RIFE)** | github.com/hzwer/Practical-RIFE | MIT | フレーム補間（アニメ最適化オプション有） | 高 | 高：ncnn-vulkan移植済み、リアルタイム性が高い |
| **FILM (frame-interpolation)** | github.com/google-research/frame-interpolation | Apache-2.0 | 大変位フレーム補間 | 中（Google公式、更新は緩やか） | 中：TensorFlow→ONNX変換が必要 |
| **VRT / RVRT** | github.com/JingyunLiang/VRT, /RVRT | CC-BY-NC | Transformer型動画復元 | 中 | 低〜中：研究向け、非商用ライセンスに注意 |
| **RAFT** | github.com/princeton-vl/RAFT | BSD-3-Clause | オプティカルフロー推定 | 中〜高（多数の派生実装） | 高：TensorRT/ONNX移植実績が非常に豊富 |
| **Anime4K** | github.com/bloc97/Anime4K | MIT | GLSL/HLSLベースの軽量リアルタイムアニメ超解像・デノイズ | 高 | 非常に高い：シェーダ実装のためOFX OpenGL/Metalカーネルへほぼそのまま移植可能 |
| **Real-CUGAN** | github.com/bilibili/ailab（Real-CUGANディレクトリ） | MIT | アニメ特化超解像（Waifu2x-CUNet系アーキテクチャ） | 中（bilibili公式、大規模アニメデータで学習） | 中：ONNX配布あり、推論組み込みが現実的 |
| **AnimeInterp** | github.com/lisiyao21/AnimeInterp | MIT | アニメ特化フレーム補間、ATD-12Kデータセット付属 | 中（研究用途、更新頻度は低め） | 中：セグメントマッチングの再実装がやや必要 |
| **KNLMeansCL** | github.com/Khanattila/KNLMeansCL | MIT/GPL系 | OpenCL実装Non-Local Means（VapourSynth/Avisynthプラグイン） | 高（実務コミュニティで継続採用） | 非常に高い：OpenCLカーネルがそのままOFX OpenCLサポートに移植可能 |
| **bm3dcuda / BM3D-GPU** | 各種（VapourSynth系リポジトリ含む） | 実装依存（MIT系が多い） | BM3D CUDA実装 | 中 | 高：CUDAカーネルをOFX CUDAサポートへ直接統合しやすい |
| **mvtools（VapourSynth/Avisynth）** | github.com/dubhater/vapoursynth-mvtools | GPL-2.0 | ブロックマッチング動き推定・動き補償ノイズ除去 | 中（実務での採用実績が長い） | 中：GPLライセンスのため商用プラグインに組み込む場合はライセンス確認必須 |
| **OpenFX公式SDK/サンプル** | github.com/AcademySoftwareFoundation/openfx | BSD-3-Clause | プラグイン仕様・サンプルプラグイン一式（TemporalBlur等の時間フィルタ例を含む） | 高（公式維持） | — 開発の出発点そのもの |

---

## 5. データセット

| データセット | 内容 | ライセンス | 規模 | 用途 | 入手方法 |
|---|---|---|---|---|---|
| **REDS**（NTIRE） | 実カメラ撮影の高解像度・大動き動画 | 研究利用（NTIRE配布規約） | 300シーケンス×100フレーム、1280×720 | Video SR・Deblur学習/評価の事実上の標準 | seungjunnah.github.io/Datasets/reds |
| **Vimeo-90K** | Vimeoから収集した高品質動画（7フレームクリップ） | 研究利用 | 91,701シーケンス、448×256 | フレーム補間・denoise・deblock・SRの汎用ベンチマーク | toflow.csail.mit.edu |
| **DAVIS** | 実世界の物体セグメンテーション動画（動き大） | CC-BY | 数十〜百シーケンス | Optical Flow/動画処理の実世界評価 | davischallenge.org |
| **MPI-Sintel** | オープンソース短編CGアニメ由来の合成データ | 研究利用（一部CC） | 1041ペア（Clean/Final） | Optical Flowの標準ベンチマーク（正解フロー付き） | sintel.is.tue.mpg.de |
| **KITTI Flow/Stereo** | 車載カメラによる実写走行データ | 研究・非商用中心 | 数百シーケンス | 実世界Optical Flow評価（自動運転文脈だが復元研究でも汎用利用） | cvlibs.net/datasets/kitti |
| **SIDD**（Smartphone Image Denoising Dataset） | 実センサーノイズ付き実写画像ペア | 研究利用 | 3万枚以上のノイズ/クリーンペア | 実世界ノイズdenoisingの学習・評価標準 | abdokamel.github.io/sidd |
| **ATD-12K** | アニメフレーム補間専用データセット（AnimeInterp付属） | 研究利用 | 約1.2万トリプレット | アニメ特化フレーム補間の学習/評価 | github.com/lisiyao21/AnimeInterp |
| **VQD-SR用アニメデータセット** | アニメ動画のリアルワールド劣化ペア | 研究利用 | 大規模（論文記載） | アニメ超解像の劣化モデリング学習 | 論文リポジトリ経由 |
| **LinkTo-Anime** | 3DモデルレンダリングによるアニメOptical Flowデータセット | 研究利用 | 論文記載規模 | アニメ特化Optical Flow学習（正解フロー取得困難な問題を解決） | arXiv:2506.02733 記載リポジトリ |
| **BSD68 / Set12 / Kodak24 / Urban100** | 古典的画像denoise/SR評価用の小規模標準画像集 | 研究・パブリックドメイン中心 | 12〜100枚程度 | 古典手法との比較評価に必須の定番セット | 各種GitHubミラー |

---

## 6. アルゴリズム分類と比較

| 手法カテゴリ | 基本原理 | 長所 | 短所 | GPU実装のしやすさ | OpenFX実装の現実性 |
|---|---|---|---|---|---|
| **空間フィルタ**（Bilateral, Median） | 近傍画素の重み付き統計処理 | 実装が単純、超高速 | エッジ/テクスチャ保持力に限界 | 非常に容易 | ★★★★★ すぐ実装可能、入門に最適 |
| **時間方向フィルタ**（単純フレーム平均・EMA） | 複数フレームの重み付き合成 | 実装が単純、静止シーンで高効果 | 動体でゴースト・ブラーが発生 | 非常に容易 | ★★★★★ Temporal Access APIの練習に最適 |
| **Non-local Means** | パッチ類似度に基づく非局所平均 | エッジ保持が高品質 | 探索コストが高い | 容易（並列パッチ探索） | ★★★★☆ KNLMeansCL等の移植実績あり |
| **BM3D / VBM3D** | ブロックマッチング+変換領域協調フィルタ | 最高水準の古典的品質 | 計算コスト大、パラメータ調整が必要 | 中程度（3D変換の並列化が鍵） | ★★★☆☆ 高品質オフラインモード向け |
| **Wavelet Denoising** | 周波数帯域分解＋閾値処理 | 高速、多重解像度解析と親和性 | ブロック/リンギングアーティファクトが出やすい | 容易 | ★★★★☆ |
| **Guided Filter** | 局所線形回帰による平滑化 | 超高速、ハロー抑制 | 単体でのノイズ除去力は限定的 | 非常に容易 | ★★★★★ プレビュー用途の即戦力 |
| **Patch-based（NLM/BM3D系全般）** | パッチ単位の統計的類似性利用 | 高品質 | メモリ・探索コスト | 中〜容易 | ★★★★☆ |
| **Optical Flow（RAFT等）ベース** | 画素対応関係を推定し動き補償 | 動体を正確に扱える | 推定誤りが復元誤りに直結、計算コスト | 中（コリレーションボリュームがVRAM負荷大） | ★★★☆☆ フレーム補間・時間デノイズの基盤として重要 |
| **Multi-frame（EDVR等）** | 複数フレームをアライメント後に融合 | 時間情報を積極活用し高品質 | 巨大なモデル・メモリ | 難しい（Deformable Conv等カスタムop） | ★★☆☆☆ 推論のみ移植が現実的 |
| **CNN（DnCNN, NAFNet等）** | 畳み込みで直接残差/復元を推定 | 高品質かつ比較的軽量 | 学習データ分布に依存（汎化限界） | 容易〜中（推論エンジン化しやすい） | ★★★★☆ TensorRT/ONNX Runtime経由での組込みが現実的 |
| **Transformer（SwinIR, Restormer, VRT）** | Self-Attentionで長距離依存を学習 | 高精度、汎用性が高い | 計算・メモリコスト大、リアルタイム困難 | 中〜難（Attentionのメモリアクセスパターン） | ★★☆☆☆ 高品質オフラインモード向け推論組込み |
| **Diffusion（StableSR等）** | 逐次的ノイズ除去過程で生成的に復元 | 圧倒的なディテール生成力 | 超低速（多ステップ推論）、幻覚リスク | 難（大規模モデル、VRAM要件大） | ★☆☆☆☆ 現状は非現実的、将来的な高品質バッチ処理向け |
| **Hybrid（古典＋AI、例: xClean的多段パイプライン）** | 複数手法を直列/並列に組み合わせ | 実運用でのバランスが良い | パイプライン設計・調整の複雑さ | 手法依存 | ★★★★☆ 実務で最も現実的なアプローチ |

---

## 7. アニメ映像への応用

### 7.1 アニメ特有の技術的課題
- **2コマ・3コマ打ち**: フレーム間で同一絵が複数回連続するため、時間フィルタが「静止」と誤認識しやすい一方、実際の動きが跳躍的（大変位）になる。オプティカルフロー推定は実写を前提に設計されているため精度が落ちやすい。フレーム補間では単純な線形補間ではなく、AnimeInterpのSegment-Guided Matchingのような「意味的に近い色領域の対応付け」が必要。
- **セルアニメ・デジタルアニメの線画/ベタ塗り**: フラットな色面（テクスチャ乏しい領域）と細い線画（1〜2px）が混在。NLMやBM3Dのようなパッチ類似性ベース手法はベタ塗り領域では強力だが、線画のエッジをぼかしやすい。線画保護のためエッジマップ（XDoGなど）を用いたマスク処理や、APISRのようなライン強調（Outlier Filter, Passive Dilate）技術が有効。
- **フィルムスキャン（旧作アニメ・セル撮影）**: ハロゲン化銀フィルム由来の粒状ノイズ、色褪せ、傷、フリッカーが複合的に発生。実写フィルム復元技術（Bringing Old Films Back to Life等）が転用できるが、セルアニメ特有の「単色ベタ塗り＋輪郭線」構造を壊さない配慮が必要。
- **グラデーション**: 圧縮（特に旧世代コーデックやDVD由来）によるバンディング（色階調の縞）が生じやすく、denoise処理だけでなくDeband処理（f3kdb, Neo_f3kdb等）との組み合わせが実務上重要。

### 7.2 優先して参照すべき研究・実装
1. **AnimeInterp / ATD-12K**（CVPR 2021）— アニメ特化フレーム補間の基礎研究。セグメント単位マッチングの発想はアニメ向けオプティカルフロー全般に応用可能。
2. **APISR**（CVPR 2024）— アニメ制作プロセスを模した劣化モデリングとライン強調技術。実運用に近い視点を持つ最新研究。
3. **AnimeSR（NeurIPS 2022）/ VQD-SR（ICCV 2023）** — 実世界アニメ動画の劣化に対するリアルワールドSRのデータ駆動型アプローチ。
4. **Real-CUGAN, Anime4K, waifu2x** — 実務で広く使われる軽量・高速な実装群。学術的新規性よりも「実際に動く・速い」ノウハウの宝庫としてOFXプラグイン設計に直結。
5. **Practical-RIFE（SAFA統合）** — アニメシーンに最適化したフレーム補間の実運用実装。

### 7.3 実写向けアルゴリズムのアニメ転用における課題と改善案
- **課題**: オプティカルフロー・動き補償ベースの手法は、テクスチャの乏しさと非線形な誇張動作により誤対応が頻発し、ゴースト・ブレの原因になる。
- **改善案**: (1) 色域・セグメンテーションベースのマッチングを併用しテクスチャ依存を下げる、(2) 学習データをアニメ特有の劣化（圧縮アーティファクト、セルスキャンノイズ）で再構成する、(3) 線画保護マスクを生成し、denoise/SR処理の強度をエッジ近傍で自動的に弱めるアダプティブ処理を導入する、(4) フレーム補間は等間隔前提を崩し、2コマ/3コマ検出ロジックと組み合わせて過補間を防ぐ。

---

## 8. 学習ロードマップ（OpenFXプラグイン開発を前提に）

### ステップ1: 最初に読むべき資料
- Digital Image Processing (Gonzalez & Woods) 第1〜4章（画像表現、空間フィルタ、周波数領域）
- OpenFX公式ドキュメント（openeffects.org）とAcademySoftwareFoundation/openfxのサンプルプラグイン一式
- Blackmagic Design DaVinci Resolve Developer SDK（Help→Documentation→Developer、既に本セッションのプロジェクト資料に含まれるofxsCore.h/ofxImageEffect.h等）
- Guided Filter論文（He, Sun, Tang, 2010）— 実装が容易でOFXの最初のフィルタ課題に最適

### ステップ2: 次に読むべき資料
- Non-Local Means（Buades et al., 2005）とBM3D（Dabov et al., 2007）— 古典的denoisingの二本柱
- Digital Video and HD (Poynton) — 色空間・ガンマ・映像フォーマットの正確な理解（映像プラグインの落とし穴を避けるため必須）
- OFX GPU拡張（CUDA/OpenCL/Metal Kernel）のサンプル実装（プロジェクト内 CudaKernel.md, OpenCLKernel.cpp, MetalKernel.mm を参照）

### ステップ3: 中級レベルで理解すべき内容
- CNNベース手法の基礎: DnCNN、NAFNet（実装がシンプルで学習コストが低い）
- 動画特有の時間処理: FastDVDnet（動き補償なしの時間統合設計思想）
- オプティカルフロー基礎: RAFTの仕組みと限界（アニメで精度が落ちる理由の理解）
- ONNX Runtime / TensorRTによる学習済みモデルの推論組み込み方法（C++からのAI推論統合はOFXプラグインAI化の中核スキル）

### ステップ4: 上級レベルで読むべき論文
- EDVR、BasicVSR++（マルチフレーム・アライメント設計の到達点）
- VRT、RVRT、SwinIR、Restormer（Transformerベース復元の設計思想）
- StableSR等の拡散モデルベース復元（将来的な高品質バッチ処理オプションとして）
- Video Restoration Based on Deep Learning: A Comprehensive Survey（Artificial Intelligence Review, 2022）で全体像を再整理

### ステップ5: 実装前に理解しておくべき理論
- OpenFX Temporal Access（clipGetFrame等でのマルチフレーム参照方法）とプラグインのフレームキャッシュ設計
- GPU間（CUDA/OpenCL/Metal）でのカーネル移植戦略とメモリ管理（VRAM制約下でのタイル分割処理）
- リアルタイムプレビュー（低解像度・軽量モデル）とレンダリング時高品質モード（重量級モデル）の二段構え設計
- ライセンス確認（GPL系実装を商用プラグインに組み込む場合の制約、CC-BY-NC論文実装の利用範囲）

---

## 付録: 参考リンク一覧（一次情報优先）

- OpenFX公式: https://openeffects.org/ , https://github.com/AcademySoftwareFoundation/openfx
- Blackmagic Design Developer: https://www.blackmagicdesign.com/developer/
- NVIDIA Developer Blog (Denoising): https://developer.nvidia.com/blog/tag/denoising/
- BasicSR: https://github.com/XPixelGroup/BasicSR
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- RAFT: https://github.com/princeton-vl/RAFT
- RIFE: https://github.com/hzwer/ECCV2022-RIFE , Practical-RIFE: https://github.com/hzwer/Practical-RIFE
- FILM: https://github.com/google-research/frame-interpolation
- VRT/RVRT: https://github.com/JingyunLiang/VRT , https://github.com/JingyunLiang/RVRT
- SwinIR: https://github.com/JingyunLiang/SwinIR
- Restormer: https://github.com/swz30/Restormer
- NAFNet: https://arxiv.org/abs/2204.04676
- BasicVSR++: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- StableSR: https://github.com/IceClear/StableSR
- AnimeInterp: https://github.com/lisiyao21/AnimeInterp
- Anime4K: https://github.com/bloc97/Anime4K
- Real-CUGAN: https://github.com/bilibili/ailab
- Bringing Old Films Back to Life: https://github.com/raywzy/Bringing-Old-Films-Back-to-Life
- Video Restoration Survey (2022): https://link.springer.com/article/10.1007/s10462-022-10302-5
- Szeliski「Computer Vision」無料PDF: https://szeliski.org/Book/
