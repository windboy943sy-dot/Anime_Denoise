"""最小 PNG 書き出し(stdlib のみ)。可視化オーバーレイを cv2/PIL 無しで保存する。

C++/OFX 本番では使わない。リファレンス検証・デバッグ用の依存ゼロ出力。
"""
from __future__ import annotations

import struct
import zlib

import numpy as np


def write_png(path: str, rgb: np.ndarray) -> None:
    """rgb: HxWx3 float(0..1) または uint8 を 8bit PNG で書き出す。"""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)                      # filter type 0
        raw.extend(rgb[y].tobytes())
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8bit, colour type 2 (RGB)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))
