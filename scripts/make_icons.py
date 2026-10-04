"""Generate the extension icons (rounded gradient square + sound-wave bars) without any image library."""
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "extension" / "icons"
SS = 4  # supersampling


def render(size: int) -> np.ndarray:
    n = size * SS
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    x /= n; y /= n

    # rounded square mask
    r = 0.22
    cx = np.clip(x, r, 1 - r); cy = np.clip(y, r, 1 - r)
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r * r

    # diagonal gradient indigo -> violet
    t = (x + y) / 2
    c0 = np.array([79, 70, 229], np.float32)
    c1 = np.array([168, 85, 247], np.float32)
    rgb = c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]

    # bars
    heights = [0.26, 0.5, 0.72, 0.5, 0.34]
    bar_w, gap = 0.085, 0.052
    total = len(heights) * bar_w + (len(heights) - 1) * gap
    x0 = 0.5 - total / 2
    bars = np.zeros((n, n), bool)
    for i, h in enumerate(heights):
        bx0 = x0 + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        by0, by1 = 0.5 - h / 2, 0.5 + h / 2
        rr = bar_w / 2
        px = np.clip(x, bx0 + rr, bx1 - rr)
        py = np.clip(y, by0 + rr, by1 - rr)
        bars |= ((x - px) ** 2 + (y - py) ** 2) <= rr * rr
    rgb[bars] = 255

    img = np.zeros((n, n, 4), np.float32)
    img[..., :3] = rgb
    img[..., 3] = mask * 255
    # premultiplied downsample
    img[..., :3] *= (img[..., 3:4] / 255)
    img = img.reshape(size, SS, size, SS, 4).mean(axis=(1, 3))
    a = img[..., 3:4]
    rgb_out = np.where(a > 0, img[..., :3] / np.maximum(a / 255, 1e-6), 0)
    out = np.concatenate([np.clip(rgb_out, 0, 255), a], axis=-1)
    return out.astype(np.uint8)


def png_bytes(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for s in (16, 32, 48, 128):
        (OUT / f"icon{s}.png").write_bytes(png_bytes(render(s)))
        print("wrote", OUT / f"icon{s}.png")
    sys.exit(0)
