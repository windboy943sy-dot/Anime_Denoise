"""合成欠陥クリップ生成(検証用・GT付き)。

外部素材なしで検知精度(適合率/再現率)を測るための地面真実付きクリップを
numpy だけで生成する。サーベイ §6.1 の「合成劣化学習」の評価版に相当。

生成物:
  frames: list[HxWx3 float 0..1]
  gt: {
    "dust": [ per-frame list of (y, x, radius, polarity) ],
    "dust_mask": [ per-frame bool HxW ],
    "scratch_x": [ (x, polarity) ...  クリップ全体で持続 ],
    "scratch_mask": HxW bool,
    "sigma": 付与ノイズ sigma,
  }
"""
from __future__ import annotations

import numpy as np


def _base_scene(h, w, rng):
    """緩やかなグラデ + いくつかの構造(エッジ・小ハイライト)を持つ背景。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 0.35 + 0.3 * (xx / w) + 0.15 * (yy / h)
    scene = grad.copy()
    # 帯状構造(本物のエッジ。誤検知の罠)。実写の壁・柱の縁は細い傷より太い
    # ため、幅10pxの帯にする(細線スクラッチ検知が太い content を拾わないことの検証)。
    scene[:, w // 3:w // 3 + 10] += 0.15
    scene[h // 2:h // 2 + 8, :] += 0.1
    # 本物の小ハイライト(白ダストと紛らわしい静止点)。時間的に持続するため
    # ダスト時間検知(SDI)は無視できるはず、という誤検知テスト用。
    for _ in range(6):
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        scene[cy - 1:cy + 2, cx - 1:cx + 2] += 0.2
    # 実写・スキャン映像の内容は光学的に帯域制限される(細い傷のような1px段は
    # 作らない)。ここで軽く平滑化し、傷=鋭い欠陥 / 内容=なだらか の弁別性を
    # 現実に合わせる。傷はこの後に鋭く焼き込むので分離できる。
    from scipy import ndimage
    scene = ndimage.gaussian_filter(scene, 1.0)
    return np.clip(scene, 0, 1)


def make_clip(n_frames=12, h=180, w=240, sigma=0.02,
              dust_per_frame=6, n_scratches=2, seed=0,
              motion=False):
    rng = np.random.default_rng(seed)
    base = _base_scene(h, w, rng)

    # 持続スクラッチ(全フレーム同一 x)
    scratch_x = []
    scratch_mask = np.zeros((h, w), bool)
    band_lo, band_hi = w // 3 - 8, w // 3 + 18   # content 帯の近傍は避ける
    attempts = 0
    while len(scratch_x) < n_scratches and attempts < 200:
        attempts += 1
        x = int(rng.integers(w // 6, w - w // 6))
        if band_lo <= x <= band_hi:
            continue
        if any(abs(x - xx) < 6 for xx, _ in scratch_x):  # 傷同士も離す
            continue
        pol = int(rng.choice([-1, 1]))
        scratch_x.append((x, pol))

    frames = []
    dust_gt = []
    dust_masks = []
    for f in range(n_frames):
        scene = base.copy()
        if motion:
            shift = f  # 1px/frame パン
            scene = np.roll(scene, shift, axis=1)

        # スクラッチを焼き込む(持続・同位置)
        for (x, pol) in scratch_x:
            val = 0.25 * pol
            scene[:, x] = np.clip(scene[:, x] + val, 0, 1)
            scene[:, min(x + 1, w - 1)] = np.clip(scene[:, min(x + 1, w - 1)] + val * 0.5, 0, 1)
            scratch_mask[:, x] = True

        # ノイズ(グレイン風)
        noisy = scene + rng.normal(0, sigma, scene.shape).astype(np.float32)

        # ダスト(単一フレームのみ・ランダム位置)
        dmask = np.zeros((h, w), bool)
        dlist = []
        for _ in range(dust_per_frame):
            cy = int(rng.integers(8, h - 8))
            cx = int(rng.integers(8, w - 8))
            r = int(rng.integers(1, 4))
            pol = int(rng.choice([-1, 1]))
            yy, xx = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            ys = np.clip(yy[disk], 0, h - 1)
            xs = np.clip(xx[disk], 0, w - 1)
            noisy[ys, xs] = np.clip(0.5 + 0.5 * pol, 0, 1)  # 白=1, 黒=0 付近
            dmask[ys, xs] = True
            dlist.append((cy, cx, r, pol))

        noisy = np.clip(noisy, 0, 1)
        rgb = np.stack([noisy, noisy, noisy], axis=-1).astype(np.float32)
        frames.append(rgb)
        dust_gt.append(dlist)
        dust_masks.append(dmask)

    gt = {
        "dust": dust_gt,
        "dust_mask": dust_masks,
        "scratch_x": scratch_x,
        "scratch_mask": scratch_mask,
        "sigma": sigma,
    }
    return frames, gt
