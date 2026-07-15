import numpy as np
import cv2
import sys
import subprocess
import json
import tempfile
from pathlib import Path

# Python側のモジュールをインポートできるようにパスを設定
sys.path.append(str(Path(__file__).parent.parent))
from hold_frame_detection.core import detect_hold_groups, DetectionThresholds

def make_multiplane_frames():
    """背景パン（毎フレーム動く）＋前景キャラクター（3コマ打ちで動く）の合成フレームシーケンスを生成する。"""
    rng = np.random.default_rng(42)
    frames = []
    width, height = 320, 240
    
    # 6フレーム生成 (2つの 3コマグループ)
    # キャラクター位置:
    # 0,1,2フレーム目: (100, 100)
    # 3,4,5フレーム目: (120, 110)
    # 背景は毎フレーム (6px, 2px) パンする
    
    # 背景生成用の固定乱数シード
    bg_r = np.random.default_rng(999)
    # 大きな背景用テクスチャを事前に生成 (パンしても同じテクスチャがずれるようにする！)
    bg_large = np.full((height + 100, width + 100, 3), 190, np.uint8)
    # 背景にランダムな矩形と線を大量に描画して特徴点を豊富にする
    for _ in range(120):
        rx = int(bg_r.integers(10, width + 90))
        ry = int(bg_r.integers(10, height + 90))
        rs = int(bg_r.integers(8, 25))
        rc = [int(v) for v in bg_r.integers(50, 160, 3)]
        cv2.rectangle(bg_large, (rx, ry), (rx+rs, ry+rs), rc, -1)
    for _ in range(15):
        rx1 = int(bg_r.integers(0, width + 100))
        ry1 = int(bg_r.integers(0, height + 100))
        rx2 = int(bg_r.integers(0, width + 100))
        ry2 = int(bg_r.integers(0, height + 100))
        rc = [int(v) for v in bg_r.integers(40, 120, 3)]
        cv2.line(bg_large, (rx1, ry1), (rx2, ry2), rc, 2)

    for i in range(6):
        # パン量
        dx, dy = 6 * i, 2 * i
        bg = bg_large[dy:dy+height, dx:dx+width].copy()
        
        # 2. 前景キャラクターの描画 (3コマ打ち)
        char_idx = i // 3
        if char_idx == 0:
            cx, cy = 100, 80
            color = (0, 0, 255) # 赤
        else:
            cx, cy = 150, 100
            color = (0, 255, 0) # 緑
            
        # 前景キャラクター（円）を描画
        cv2.circle(bg, (cx, cy), 25, color, -1)
        
        # 3. ノイズ（グレイン）追加
        noisy = bg.astype(np.int16) + rng.integers(-5, 6, bg.shape, np.int16)
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
        frames.append(frame)
        
    return frames

def main():
    print("=== マルチプレーン領域別ホールド判定テスト ===")
    frames = make_multiplane_frames()
    
    # デバッグ用に 1->2フレーム目の位置合わせと前景マスクを可視化
    gray_a = cv2.GaussianBlur(cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gray_b = cv2.GaussianBlur(cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    from hold_frame_detection.core import estimate_global_motion, compute_foreground_mask
    warp = estimate_global_motion(gray_a, gray_b)
    print(f"  Debug 1->2 warp:\n{warp}")
    if warp is not None:
        mask = compute_foreground_mask(gray_a, gray_b, warp)
        cv2.imwrite("/tmp/fg_mask_1_2.png", mask)
        print(f"  Debug 1->2 mask ratio: {np.count_nonzero(mask) / mask.size:.4f}")
    
    # -------------------------------------------------------------
    # Test 1: Python 側での判定
    # -------------------------------------------------------------
    print("\n[1] Python 側での検出テスト:")
    
    # 従来判定（領域分割なし）
    th_normal = DetectionThresholds(use_region_segment=False)
    groups_normal = detect_hold_groups(frames, th_normal)
    print("  領域分割なし（通常判定）:")
    for g in groups_normal:
        print(f"    - start={g.start}, end={g.end}, length={g.length}")
        
    # 領域別判定
    th_region = DetectionThresholds(use_region_segment=True)
    groups_region = detect_hold_groups(frames, th_region)
    print("  領域分割あり (Region-Segment):")
    for g in groups_region:
        print(f"    - start={g.start}, end={g.end}, length={g.length}")

    # -------------------------------------------------------------
    # Test 2: C++ 側での判定 (ar_cli)
    # -------------------------------------------------------------
    print("\n[2] C++ 側 (ar_cli) での検出テスト:")
    
    # 一時ディレクトリに合成フレームを書き出す
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, f in enumerate(frames):
            cv2.imwrite(str(tmp_path / f"frame_{idx:03d}.png"), f)
            
        cli_path = "./analysis/build/ar_cli"
        
        # 通常方式
        res_normal = subprocess.run([
            cli_path, "detect", "--frames-dir", str(tmp_path), "--output", str(tmp_path / "normal.json")
        ], capture_output=True, text=True)
        with open(tmp_path / "normal.json") as fp:
            out_normal = json.load(fp)
            
        # 領域分割方式
        res_region = subprocess.run([
            cli_path, "detect", "--frames-dir", str(tmp_path), "--region-segment", "--output", str(tmp_path / "region.json")
        ], capture_output=True, text=True)
        with open(tmp_path / "region.json") as fp:
            out_region = json.load(fp)
            
        print("  領域分割なし（通常判定）:")
        for g in out_normal["hold_groups"]:
            print(f"    - start={g['start']}, end={g['end']}, pattern={g['pattern']}")
            
        print("  領域分割あり (Region-Segment):")
        for g in out_region["hold_groups"]:
            print(f"    - start={g['start']}, end={g['end']}, pattern={g['pattern']}")

if __name__ == "__main__":
    main()
