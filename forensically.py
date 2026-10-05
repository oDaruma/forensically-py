#!/usr/bin/env python3
"""
forensically.py - command-line photo forensics toolkit.

Re-implements the feature set of Forensically (https://29a.ch/photo-forensics/)
as a scriptable CLI:

    magnifier   Zoom into a region with optional contrast enhancement
    clone       Copy-move (clone) detection via block matching
    ela         Error Level Analysis
    noise       Noise analysis (median-filter residual)
    sweep       Level sweep (steep contrast window; optional animated GIF)
    gradient    Luminance gradient (Sobel)
    pca         Principal Component Analysis of the RGB colour space
    metadata    EXIF / XMP / container metadata dump
    geo         GPS geotag extraction with map links
    thumbnail   Embedded EXIF thumbnail extraction and comparison
    jpeg        JPEG structure, quantization tables and quality estimate
    strings     Printable string extraction from the raw file
    report      Run everything and write an HTML/JSON report

Dependencies: numpy, Pillow.
"""
from __future__ import annotations

import argparse
import colorsys
import html
import io
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFilter, ImageOps

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_image(path: str | Path) -> Image.Image:
    """Load an image as 8-bit RGB (alpha is composited onto black).

    EXIF orientation is deliberately NOT applied so pixel data is analysed
    exactly as stored.
    """
    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def to_f(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.float32)


def to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blend(over: np.ndarray, base: np.ndarray, opacity: float) -> np.ndarray:
    if over.ndim == 2 and base.ndim == 3:
        over = np.repeat(over[..., None], 3, axis=2)
    return over * opacity + base * (1.0 - opacity)


def luminance(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def equalize(arr: np.ndarray) -> np.ndarray:
    return to_f(ImageOps.equalize(to_img(arr)))


def default_out(src: str, tool: str, ext: str = "png") -> str:
    p = Path(src)
    return str(p.with_name(f"{p.stem}_{tool}.{ext}"))


def save(arr_or_img, path: str, quiet: bool = False) -> str:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else to_img(arr_or_img)
    img.save(path)
    if not quiet:
        print(f"[+] wrote {path}")
    return path


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Magnifier
# --------------------------------------------------------------------------- #


def magnifier(img: Image.Image, x: int, y: int, size: int = 64, zoom: int = 8,
              enhance: str = "none") -> Image.Image:
    w, h = img.size
    half = size // 2
    box = (max(0, x - half), max(0, y - half), min(w, x + half), min(h, y + half))
    crop = img.crop(box)
    if enhance == "equalize":
        crop = ImageOps.equalize(crop)
    elif enhance == "autocontrast":
        # global stretch using min/max over all channels
        a = to_f(crop)
        lo, hi = a.min(), a.max()
        crop = to_img((a - lo) * (255.0 / max(hi - lo, 1)))
    elif enhance == "autocontrast-channel":
        crop = ImageOps.autocontrast(crop)  # Pillow stretches each band separately
    return crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)


# --------------------------------------------------------------------------- #
# Error Level Analysis
# --------------------------------------------------------------------------- #


def ela(img: Image.Image, quality: int = 75, scale: float = 20.0,
        opacity: float = 1.0, equalize_hist: bool = False) -> np.ndarray:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(to_f(img) - to_f(resaved)) * scale
    if equalize_hist:
        diff = equalize(diff)
    return blend(diff, to_f(img), opacity)


# --------------------------------------------------------------------------- #
# Noise analysis
# --------------------------------------------------------------------------- #


def noise(img: Image.Image, radius: int = 2, amplitude: float = 5.0,
          opacity: float = 1.0, equalize_hist: bool = False) -> np.ndarray:
    med = img.filter(ImageFilter.MedianFilter(2 * radius + 1))
    res = np.abs(to_f(img) - to_f(med)) * amplitude
    if equalize_hist:
        res = equalize(res)
    return blend(res, to_f(img), opacity)


# --------------------------------------------------------------------------- #
# Level sweep
# --------------------------------------------------------------------------- #


def level_sweep(img: Image.Image, position: float = 128, width: float = 32,
                opacity: float = 1.0) -> np.ndarray:
    a = to_f(img)
    lo = position - width / 2.0
    out = (a - lo) * (255.0 / max(width, 1.0))
    return blend(np.clip(out, 0, 255), a, opacity)


def level_sweep_gif(img: Image.Image, path: str, width: float = 32, frames: int = 32,
                    duration: int = 80, max_size: int = 1024) -> str:
    im = img.copy()
    im.thumbnail((max_size, max_size))
    seq = [to_img(level_sweep(im, p, width)) for p in np.linspace(0, 255, frames)]
    seq = seq + seq[-2:0:-1]  # ping-pong
    seq[0].save(path, save_all=True, append_images=seq[1:], duration=duration, loop=0)
    return path


# --------------------------------------------------------------------------- #
# Luminance gradient
# --------------------------------------------------------------------------- #


def sobel(lum: np.ndarray):
    p = np.pad(lum, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def gradient(img: Image.Image, intensity: float = 1.0, opacity: float = 1.0,
             equalize_hist: bool = False) -> np.ndarray:
    """Red/green encode x/y gradient direction (128 = flat), blue = magnitude."""
    gx, gy = sobel(luminance(to_f(img)))
    k = 4.0 * intensity
    out = np.stack([128 + gx * k, 128 + gy * k, np.hypot(gx, gy) * k], axis=-1)
    out = np.clip(out, 0, 255)
    if equalize_hist:
        out = equalize(out)
    return blend(out, to_f(img), opacity)


# --------------------------------------------------------------------------- #
# Principal Component Analysis
# --------------------------------------------------------------------------- #


def pca_basis(img: Image.Image):
    X = to_f(img).reshape(-1, 3).astype(np.float64)
    mu = X.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((X - mu).T))
    order = np.argsort(w)[::-1]
    return mu, w[order], V[:, order]


def pca(img: Image.Image, component: int = 1, mode: str = "projection",
        scaling: str = "linear", invert: bool = False,
        equalize_hist: bool = False, opacity: float = 1.0) -> np.ndarray:
    a = to_f(img)
    h, w, _ = a.shape
    mu, _, V = pca_basis(img)
    X = a.reshape(-1, 3) - mu
    v = V[:, component - 1]
    proj = X @ v
    if mode == "projection":
        out = proj
    elif mode == "difference":
        out = X - np.outer(proj, v)
    elif mode == "distance":
        out = np.linalg.norm(X - np.outer(proj, v), axis=1)
    else:
        raise ValueError(mode)

    if scaling == "linear":
        lo, hi = out.min(), out.max()
        out = (out - lo) * (255.0 / max(hi - lo, 1e-9))
    elif scaling == "abs":
        out = np.abs(out)
        out = out * (255.0 / max(out.max(), 1e-9))
    else:  # none
        out = out + (0 if mode == "distance" else 128)

    out = out.reshape(h, w, -1)
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    out = np.clip(out, 0, 255)
    if invert:
        out = 255 - out
    if equalize_hist:
        out = equalize(out)
    return blend(out, a, opacity)


# --------------------------------------------------------------------------- #
# Clone detection (copy-move)
# --------------------------------------------------------------------------- #


def _box_sums(a: np.ndarray, s: int) -> np.ndarray:
    """Sum over every s x s window (valid positions)."""
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]


def _vec_colour(dx: int, dy: int):
    hue = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def clone_detect(img: Image.Image, block: int = 8, max_size: int = 512,
                 min_similarity: float = 0.97, min_detail: float = 0.03,
                 min_cluster: int = 50, min_distance: int | None = None,
                 compare: int = 8, step: int = 1, verify: bool = True):
    """Detect duplicated regions.

    Returns (clusters, pairs, scale) where pairs is an (N,4) int array of
    (x1,y1,x2,y2) block origins in the analysed (possibly downscaled) image.
    """
    w0, h0 = img.size
    scale = min(1.0, max_size / max(w0, h0))
    small = img.resize((max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                       Image.LANCZOS) if scale < 1 else img
    g = luminance(to_f(small)).astype(np.float64) / 255.0
    H, W = g.shape
    b = block
    if H < b * 2 or W < b * 2:
        return [], np.zeros((0, 4), int), scale
    min_distance = b if min_distance is None else min_distance

    # Detail (std-dev) for every block
    n = b * b
    s1 = _box_sums(g, b)
    s2 = _box_sums(g * g, b)
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))

    # Reduced feature vector: 4x4 grid of sub-block means
    grid = 4 if b % 4 == 0 else (2 if b % 2 == 0 else 1)
    sub = b // grid
    sub_means = _box_sums(g, sub) / (sub * sub)
    ny, nx = H - b + 1, W - b + 1
    ys, xs = np.mgrid[0:ny:step, 0:nx:step]
    ys, xs = ys.ravel(), xs.ravel()
    keep = std[ys, xs] >= min_detail
    ys, xs = ys[keep], xs[keep]
    if len(ys) < 2:
        return [], np.zeros((0, 4), int), scale
    feats = np.stack([sub_means[ys + u * sub, xs + v * sub]
                      for u in range(grid) for v in range(grid)], axis=1).astype(np.float32)

    # Lexicographic sort on quantized features, compare near neighbours
    q = np.floor(feats * 32).astype(np.int16)
    order = np.lexsort(q.T[::-1])
    feats, ys, xs = feats[order], ys[order], xs[order]
    max_rms = 1.0 - min_similarity

    pa, pb = [], []
    for k in range(1, compare + 1):
        if k >= len(ys):
            break
        d = np.sqrt(np.mean((feats[:-k] - feats[k:]) ** 2, axis=1))
        dist = np.hypot(xs[k:] - xs[:-k], ys[k:] - ys[:-k])
        m = (d <= max_rms) & (dist >= min_distance)
        idx = np.nonzero(m)[0]
        pa.append(idx)
        pb.append(idx + k)
    if not pa or sum(len(p) for p in pa) == 0:
        return [], np.zeros((0, 4), int), scale
    ia, ib = np.concatenate(pa), np.concatenate(pb)

    x1, y1, x2, y2 = xs[ia], ys[ia], xs[ib], ys[ib]
    # canonical direction so (a->b) and (b->a) fall in the same cluster
    swap = (y2 < y1) | ((y2 == y1) & (x2 < x1))
    x1, x2 = np.where(swap, x2, x1), np.where(swap, x1, x2)
    y1, y2 = np.where(swap, y2, y1), np.where(swap, y1, y2)

    if verify:  # full-resolution block RMS check
        o = np.arange(b)
        def blocks(yy, xx):
            return g[yy[:, None, None] + o[None, :, None], xx[:, None, None] + o[None, None, :]]
        ok = np.zeros(len(x1), bool)
        for s in range(0, len(x1), 20000):
            e = slice(s, s + 20000)
            rms = np.sqrt(np.mean((blocks(y1[e], x1[e]) - blocks(y2[e], x2[e])) ** 2, axis=(1, 2)))
            ok[e] = rms <= max_rms
        x1, y1, x2, y2 = x1[ok], y1[ok], x2[ok], y2[ok]

    dx, dy = x2 - x1, y2 - y1
    if len(dx) == 0:
        return [], np.zeros((0, 4), int), scale
    vecs, inv, counts = np.unique(np.stack([dx, dy], 1), axis=0, return_inverse=True,
                                  return_counts=True)
    inv = inv.ravel()
    good = counts >= min_cluster
    mask = good[inv]
    pairs = np.stack([x1, y1, x2, y2], 1)[mask]
    pair_vec = inv[mask]

    clusters = []
    for vi in np.nonzero(good)[0]:
        sel = pairs[pair_vec == vi]
        clusters.append({
            "shift_dx": int(round(vecs[vi][0] / scale)),
            "shift_dy": int(round(vecs[vi][1] / scale)),
            "blocks": int(counts[vi]),
            "source_bbox": [int(sel[:, 0].min() / scale), int(sel[:, 1].min() / scale),
                            int((sel[:, 0].max() + b) / scale), int((sel[:, 1].max() + b) / scale)],
            "target_bbox": [int(sel[:, 2].min() / scale), int(sel[:, 3].min() / scale),
                            int((sel[:, 2].max() + b) / scale), int((sel[:, 3].max() + b) / scale)],
        })
    clusters.sort(key=lambda c: -c["blocks"])
    return clusters, pairs, scale


def render_clones(img: Image.Image, pairs: np.ndarray, scale: float, block: int,
                  lines: bool = True, dim: float = 0.5) -> Image.Image:
    base = to_img(to_f(img) * (1 - dim)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    bs = block / scale
    for x1, y1, x2, y2 in pairs:
        c = _vec_colour(int(x2 - x1), int(y2 - y1))
        for x, y in ((x1, y1), (x2, y2)):
            X, Y = x / scale, y / scale
            d.rectangle([X, Y, X + bs, Y + bs], fill=c + (60,))
    if lines:
        step = max(1, len(pairs) // 400)  # don't drown the image in lines
        for x1, y1, x2, y2 in pairs[::step]:
            c = _vec_colour(int(x2 - x1), int(y2 - y1))
            h = bs / 2
            d.line([x1 / scale + h, y1 / scale + h, x2 / scale + h, y2 / scale + h],
                   fill=c + (200,), width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

IFD_EXIF, IFD_GPS, IFD_INTEROP = 0x8769, 0x8825, 0xA005


def _fmt_val(v, maxlen: int = 200):
    if isinstance(v, bytes):
        txt = v.rstrip(b"\x00")
        if txt and all(32 <= c < 127 or c in (9, 10, 13) for c in txt):
            return txt.decode("ascii")
        s = v[:64].hex(" ")
        return f"<{len(v)} bytes> {s}{' ...' if len(v) > 64 else ''}"
    if isinstance(v, tuple):
        return [_fmt_val(x, maxlen) for x in v]
    if isinstance(v, (bool, int)):
        return v
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        try:
            return float(v) if v.denominator else None
        except Exception:
            return str(v)
    s = v if isinstance(v, (int, float)) else str(v)
    if isinstance(s, str) and len(s) > maxlen:
        s = s[:maxlen] + " ..."
    return s


def extract_xmp(data: bytes) -> str | None:
    m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", data, re.S)
    if m:
        return m.group(0).decode("utf-8", "replace")
    return None


def metadata(path: str) -> dict:
    data = Path(path).read_bytes()
    im = Image.open(path)
    result: dict = {
        "File": {
            "path": str(path),
            "size_bytes": len(data),
            "format": im.format,
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
        },
        "Container": {},
        "IFD0": {},
        "Exif": {},
        "GPS": {},
        "Interop": {},
    }
    for k, v in im.info.items():
        if k in ("exif", "xmp"):
            continue
        if k == "icc_profile":
            result["Container"]["icc_profile"] = f"<{len(v)} bytes>"
            try:
                from PIL import ImageCms
                prof = ImageCms.ImageCmsProfile(io.BytesIO(v))
                result["Container"]["icc_description"] = ImageCms.getProfileDescription(prof).strip()
            except Exception:
                pass
            continue
        result["Container"][str(k)] = _fmt_val(v)

    exif = im.getexif()
    for tag, val in exif.items():
        if tag in (IFD_EXIF, IFD_GPS, IFD_INTEROP):
            continue
        result["IFD0"][ExifTags.TAGS.get(tag, hex(tag))] = _fmt_val(val)
    for ifd, key, names in ((IFD_EXIF, "Exif", ExifTags.TAGS),
                            (IFD_GPS, "GPS", ExifTags.GPSTAGS),
                            (IFD_INTEROP, "Interop", ExifTags.TAGS)):
        try:
            sub = exif.get_ifd(ifd)
        except Exception:
            sub = {}
        for tag, val in sub.items():
            name = names.get(tag, hex(tag))
            if name == "MakerNote":
                result[key][name] = f"<{len(val)} bytes>"
                continue
            result[key][name] = _fmt_val(val)
    xmp = extract_xmp(data)
    if xmp:
        result["XMP"] = xmp
    return {k: v for k, v in result.items() if v}


# --------------------------------------------------------------------------- #
# Geo tags
# --------------------------------------------------------------------------- #


def _dms_to_deg(v, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in v)
    except Exception:
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if str(ref).strip().upper() in ("S", "W"):
        deg = -deg
    return deg


def geotags(path: str) -> dict | None:
    im = Image.open(path)
    try:
        gps = im.getexif().get_ifd(IFD_GPS)
    except Exception:
        gps = {}
    if not gps:
        return None
    named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
    lat = _dms_to_deg(named.get("GPSLatitude"), named.get("GPSLatitudeRef", "N"))
    lon = _dms_to_deg(named.get("GPSLongitude"), named.get("GPSLongitudeRef", "E"))
    out = {"raw": {str(k): _fmt_val(v) for k, v in named.items()}}
    if lat is not None and lon is not None:
        out.update({
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "openstreetmap": f"https://www.openstreetmap.org/?mlat={lat:.7f}&mlon={lon:.7f}#map=16/{lat:.7f}/{lon:.7f}",
            "google_maps": f"https://www.google.com/maps?q={lat:.7f},{lon:.7f}",
        })
    if "GPSAltitude" in named:
        alt = float(named["GPSAltitude"])
        ref = named.get("GPSAltitudeRef", 0)
        if isinstance(ref, bytes):
            ref = ref[0] if ref else 0
        out["altitude_m"] = -alt if ref == 1 else alt
    if "GPSDateStamp" in named or "GPSTimeStamp" in named:
        ts = named.get("GPSTimeStamp")
        t = ":".join(f"{float(x):02.0f}" for x in ts) if ts else ""
        out["timestamp_utc"] = f"{named.get('GPSDateStamp', '')} {t}".strip()
    if "GPSImgDirection" in named:
        out["image_direction_deg"] = float(named["GPSImgDirection"])
    return out


# --------------------------------------------------------------------------- #
# Thumbnail analysis
# --------------------------------------------------------------------------- #


def extract_thumbnail(path: str) -> bytes | None:
    data = Path(path).read_bytes()
    # 1) Ask Pillow for IFD1 offsets (relative to TIFF header in APP1)
    try:
        im = Image.open(path)
        exif = im.getexif()
        ifd1 = exif.get_ifd(ExifTags.IFD.IFD1) if hasattr(ExifTags, "IFD") else {}
        off, ln = ifd1.get(0x0201), ifd1.get(0x0202)
        raw = im.info.get("exif")
        if off and ln and raw:
            tiff = raw[6:] if raw.startswith(b"Exif\x00\x00") else raw
            thumb = tiff[off:off + ln]
            if thumb[:2] == b"\xff\xd8":
                return thumb
    except Exception:
        pass
    # 2) Fallback: scan the APP1/Exif segment for an embedded JPEG
    i = data.find(b"Exif\x00\x00")
    if i < 0:
        return None
    seg_len = struct.unpack(">H", data[i - 2:i])[0] if i >= 2 else len(data)
    seg = data[i:i - 2 + seg_len]
    s = seg.find(b"\xff\xd8", 6)
    e = seg.rfind(b"\xff\xd9")
    if s >= 0 and e > s:
        return seg[s:e + 2]
    return None


def thumbnail_analysis(img: Image.Image, path: str, scale: float = 10.0):
    raw = extract_thumbnail(path)
    if not raw:
        return None, None
    thumb = Image.open(io.BytesIO(raw)).convert("RGB")
    tw, th = thumb.size
    iw, ih = img.size
    ar_img, ar_th = iw / ih, tw / th
    # Fit image into the thumbnail frame (thumbs are often letterboxed)
    fitted = ImageOps.pad(img, (tw, th), method=Image.LANCZOS, color=(0, 0, 0))
    diff = np.abs(to_f(thumb) - to_f(fitted))
    info = {
        "thumbnail_size": [tw, th],
        "thumbnail_bytes": len(raw),
        "image_aspect": round(ar_img, 4),
        "thumbnail_aspect": round(ar_th, 4),
        "aspect_mismatch": abs(ar_img - ar_th) > 0.02 and abs(ar_img - 1 / ar_th) > 0.02,
        "mean_abs_difference": round(float(diff.mean()), 3),
        "max_abs_difference": round(float(diff.max()), 3),
        "note": "Large differences suggest the main image was edited after the "
                "thumbnail was generated (or the thumbnail is from another image).",
    }
    pad = 4
    canvas = Image.new("RGB", (tw * 3 + pad * 2, th), (40, 40, 40))
    canvas.paste(thumb, (0, 0))
    canvas.paste(fitted, (tw + pad, 0))
    canvas.paste(to_img(diff * scale), (2 * (tw + pad), 0))
    zoom = max(1, 256 // max(th, 1))
    canvas = canvas.resize((canvas.width * zoom, canvas.height * zoom), Image.NEAREST)
    return info, (canvas, thumb)


# --------------------------------------------------------------------------- #
# JPEG analysis
# --------------------------------------------------------------------------- #

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
          41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
          15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55,
          62, 63]

STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]).reshape(8, 8)
STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32).reshape(8, 8)

MARKERS = {0xC0: "SOF0 (baseline)", 0xC1: "SOF1 (extended sequential)",
           0xC2: "SOF2 (progressive)", 0xC3: "SOF3 (lossless)", 0xC4: "DHT",
           0xCC: "DAC", 0xD8: "SOI", 0xD9: "EOI", 0xDA: "SOS", 0xDB: "DQT",
           0xDD: "DRI", 0xFE: "COM"}


def libjpeg_table(std: np.ndarray, q: int) -> np.ndarray:
    q = min(max(q, 1), 100)
    s = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((std * s + 50) / 100), 1, 255).astype(int)


def estimate_quality(table: np.ndarray, std: np.ndarray) -> dict:
    s = table.sum() / std.sum() * 100
    q = (200 - s) / 2 if s <= 100 else 5000 / s
    q = float(np.clip(q, 1, 100))
    best = min(range(1, 101), key=lambda k: np.abs(libjpeg_table(std, k) - table).sum())
    exact = bool(np.array_equal(libjpeg_table(std, best), table))
    return {"estimated_quality": round(q, 1), "nearest_libjpeg_quality": best,
            "matches_standard_libjpeg_table": exact}


def parse_jpeg(data: bytes) -> dict:
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    i, n = 2, len(data)
    segs, qtables, frame, huff, comments, apps = [], {}, None, [], [], []
    restart = None
    eoi = None
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n - 1 and data[i + 1] == 0xFF:
            i += 1
        marker = data[i + 1]
        pos = i
        i += 2
        if marker == 0xD9:
            eoi = i
            segs.append({"offset": pos, "marker": "EOI"})
            break
        if 0xD0 <= marker <= 0xD7 or marker in (0x01, 0x00):
            continue
        if i + 2 > n:
            break
        length = struct.unpack(">H", data[i:i + 2])[0]
        seg = data[i + 2:i + length]
        name = MARKERS.get(marker, f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else f"0x{marker:02X}")
        entry = {"offset": pos, "marker": name, "length": length}
        if 0xE0 <= marker <= 0xEF:
            ident = seg[:32].split(b"\x00")[0].decode("latin-1", "replace")
            entry["identifier"] = ident
            apps.append(f"{name}: {ident}")
        elif marker == 0xDB:
            p = 0
            while p < len(seg):
                pq, tq = seg[p] >> 4, seg[p] & 15
                p += 1
                if pq:
                    vals = struct.unpack(">64H", seg[p:p + 128]); p += 128
                else:
                    vals = seg[p:p + 64]; p += 64
                nat = np.zeros(64, int)
                for k, z in enumerate(ZIGZAG):
                    nat[z] = vals[k]
                qtables[tq] = {"precision_bits": 16 if pq else 8, "table": nat.reshape(8, 8)}
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            prec, h, w, nc = struct.unpack(">BHHB", seg[:6])
            comps = []
            for c in range(nc):
                cid, hv, tq = seg[6 + 3 * c:9 + 3 * c]
                comps.append({"id": cid, "h": hv >> 4, "v": hv & 15, "qtable": tq})
            frame = {"type": name, "precision": prec, "width": w, "height": h, "components": comps}
        elif marker == 0xC4:
            p = 0
            while p + 17 <= len(seg):
                tc, th = seg[p] >> 4, seg[p] & 15
                cnt = sum(seg[p + 1:p + 17])
                huff.append(f"{'AC' if tc else 'DC'}{th} ({cnt} codes)")
                p += 17 + cnt
        elif marker == 0xFE:
            comments.append(seg.decode("utf-8", "replace"))
        elif marker == 0xDD:
            restart = struct.unpack(">H", seg[:2])[0]
        segs.append(entry)
        i += length
        if marker == 0xDA:  # skip entropy-coded data
            while i < n - 1:
                if data[i] == 0xFF and data[i + 1] != 0x00 and not (0xD0 <= data[i + 1] <= 0xD7):
                    break
                i += 1

    out = {"segments": segs, "app_segments": apps, "comments": comments,
           "huffman_tables": huff, "restart_interval": restart, "frame": frame}
    if frame:
        comps = frame["components"]
        out["progressive"] = "progressive" in frame["type"]
        if len(comps) >= 3:
            y, cb = comps[0], comps[1]
            hr, vr = y["h"] // max(cb["h"], 1), y["v"] // max(cb["v"], 1)
            out["chroma_subsampling"] = {(1, 1): "4:4:4", (2, 1): "4:2:2", (2, 2): "4:2:0",
                                         (1, 2): "4:4:0", (4, 1): "4:1:1"}.get((hr, vr), f"{hr}x{vr}")
        else:
            out["chroma_subsampling"] = "grayscale"
    out["quantization_tables"] = {}
    for tid, t in sorted(qtables.items()):
        std = STD_LUMA if tid == 0 else STD_CHROMA
        out["quantization_tables"][tid] = {
            "precision_bits": t["precision_bits"],
            "table": t["table"].tolist(),
            **estimate_quality(t["table"], std),
        }
    out["trailing_bytes_after_eoi"] = (n - eoi) if eoi else None
    return out


def format_jpeg_report(r: dict) -> str:
    L = []
    f = r.get("frame") or {}
    L.append(f"Frame        : {f.get('type')}  {f.get('width')}x{f.get('height')}  "
             f"{f.get('precision')}-bit")
    L.append(f"Subsampling  : {r.get('chroma_subsampling')}")
    L.append(f"Components   : " + ", ".join(
        f"#{c['id']} {c['h']}x{c['v']} q{c['qtable']}" for c in f.get("components", [])))
    L.append(f"Restart int. : {r.get('restart_interval')}")
    L.append(f"APP segments : {'; '.join(r['app_segments']) or '-'}")
    L.append(f"Huffman      : {', '.join(r['huffman_tables'])}")
    for c in r["comments"]:
        L.append(f"Comment      : {c!r}")
    tb = r.get("trailing_bytes_after_eoi")
    if tb:
        L.append(f"!! {tb} bytes of data after EOI marker (appended/hidden data?)")
    for tid, t in r["quantization_tables"].items():
        L.append("")
        L.append(f"Quantization table {tid} ({'luma' if tid == 0 else 'chroma'}, "
                 f"{t['precision_bits']}-bit): est. quality {t['estimated_quality']}, "
                 f"nearest libjpeg q{t['nearest_libjpeg_quality']}"
                 f"{' (exact standard table)' if t['matches_standard_libjpeg_table'] else ' (NON-standard / custom table)'}")
        for row in t["table"]:
            L.append("   " + " ".join(f"{v:4d}" for v in row))
    L.append("")
    L.append("Segments:")
    for s in r["segments"]:
        extra = f"  {s['identifier']}" if "identifier" in s else ""
        L.append(f"   @{s['offset']:>8}  {s['marker']:<24} len={s.get('length', '-')}{extra}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# String extraction
# --------------------------------------------------------------------------- #


def extract_strings(data: bytes, min_len: int = 6, encoding: str = "ascii",
                    pattern: str | None = None):
    res = []
    if encoding in ("ascii", "both"):
        for m in re.finditer(rb"[\x20-\x7e\t]{%d,}" % min_len, data):
            res.append((m.start(), "ascii", m.group().decode("ascii")))
    if encoding in ("utf16", "both"):
        for m in re.finditer(rb"(?:[\x20-\x7e]\x00){%d,}" % min_len, data):
            res.append((m.start(), "utf16le", m.group().decode("utf-16le")))
    res.sort()
    if pattern:
        rx = re.compile(pattern, re.I)
        res = [r for r in res if rx.search(r[2])]
    return res


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def run_report(args) -> None:
    src = args.image
    out = Path(args.outdir or (Path(src).stem + "_forensics"))
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(src)
    data = Path(src).read_bytes()
    q = args.quiet = True
    rep: dict = {"image": src, "tool_version": __version__, "images": {}}

    def put(name, arr, caption):
        p = out / f"{name}.png"
        save(arr, str(p), quiet=q)
        rep["images"][name] = {"file": p.name, "caption": caption}
        print(f"[+] {name}")

    put("original", img, "Original")
    put("ela", ela(img), "Error Level Analysis (q=75, scale=20)")
    put("noise", noise(img), "Noise analysis (median r=2, amplitude=5)")
    put("level_sweep", level_sweep(img), "Level sweep (pos=128, width=32)")
    put("gradient", gradient(img), "Luminance gradient")
    for c in (1, 2, 3):
        put(f"pca_{c}", pca(img, component=c), f"PCA component {c} projection")
    put("pca_distance_1", pca(img, 1, mode="distance"), "PCA distance from 1st component")

    clusters, pairs, sc = clone_detect(img)
    put("clone", render_clones(img, pairs, sc, 8), f"Clone detection ({len(clusters)} clusters)")
    rep["clone_clusters"] = clusters

    tinfo, tres = thumbnail_analysis(img, src)
    rep["thumbnail"] = tinfo
    if tres:
        put("thumbnail_compare", tres[0], "Thumbnail | image | diff x10")

    rep["metadata"] = metadata(src)
    rep["geo"] = geotags(src)
    if data[:2] == b"\xff\xd8":
        try:
            jr = parse_jpeg(data)
            rep["jpeg"] = jr
            rep["jpeg_text"] = format_jpeg_report(jr)
        except Exception as e:
            rep["jpeg"] = {"error": str(e)}
    rep["strings_sample"] = [s for _, _, s in extract_strings(data, 8)][:300]

    (out / "report.json").write_text(jdump({k: v for k, v in rep.items() if k != "jpeg_text"}))
    (out / "report.html").write_text(_html_report(rep))
    print(f"[+] report written to {out}/report.html")


def _html_report(rep: dict) -> str:
    e = html.escape
    imgs = "".join(
        f'<figure><a href="{e(v["file"])}"><img src="{e(v["file"])}"></a>'
        f'<figcaption>{e(v["caption"])}</figcaption></figure>'
        for v in rep["images"].values())
    meta = e(jdump(rep.get("metadata")))
    geo = rep.get("geo")
    geo_html = (f'<p>{geo["latitude"]}, {geo["longitude"]} &mdash; '
                f'<a href="{e(geo["openstreetmap"])}">OpenStreetMap</a> | '
                f'<a href="{e(geo["google_maps"])}">Google Maps</a></p>'
                if geo and "latitude" in geo else "<p>No GPS data.</p>")
    clones = e(jdump(rep.get("clone_clusters"))) if rep.get("clone_clusters") else "None detected."
    thumb = e(jdump(rep.get("thumbnail"))) if rep.get("thumbnail") else "No embedded thumbnail."
    jpeg = e(rep.get("jpeg_text", "Not a JPEG."))
    strings = e("\n".join(rep.get("strings_sample", [])))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forensics report - {e(Path(rep['image']).name)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#ddd}}
h1,h2{{font-weight:600}} a{{color:#7bf}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;padding:8px;border-radius:6px}}
img{{width:100%;image-rendering:pixelated}} figcaption{{color:#aaa;padding-top:4px}}
pre{{background:#1c1c1c;padding:12px;border-radius:6px;overflow:auto;max-height:480px}}
</style></head><body>
<h1>Forensics report: {e(rep['image'])}</h1>
<div class="grid">{imgs}</div>
<h2>Clone detection clusters</h2><pre>{clones}</pre>
<h2>Geo tags</h2>{geo_html}
<h2>Thumbnail analysis</h2><pre>{thumb}</pre>
<h2>JPEG analysis</h2><pre>{jpeg}</pre>
<h2>Metadata</h2><pre>{meta}</pre>
<h2>Strings (first 300, len&ge;8)</h2><pre>{strings}</pre>
</body></html>"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forensically",
        description="Photo forensics toolkit (CLI port of 29a.ch Forensically).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run '%(prog)s <command> -h' for command-specific options.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sp = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def cmd(name, help_):
        s = sp.add_parser(name, help=help_, description=help_,
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        s.add_argument("image", help="input image")
        return s

    def outopt(s, ext="png"):
        s.add_argument("-o", "--output", help=f"output file (default: <image>_{s.prog.split()[-1]}.{ext})")

    def opacity(s):
        s.add_argument("--opacity", type=float, default=1.0, help="blend over original (0-1)")

    s = cmd("magnifier", "Zoom into a region with optional enhancement")
    s.add_argument("-x", type=int, required=True, help="centre x")
    s.add_argument("-y", type=int, required=True, help="centre y")
    s.add_argument("--size", type=int, default=64, help="region size in pixels")
    s.add_argument("--zoom", type=int, default=8, help="magnification factor")
    s.add_argument("--enhance", choices=["none", "equalize", "autocontrast", "autocontrast-channel"],
                   default="none")
    outopt(s)

    s = cmd("clone", "Clone (copy-move) detection")
    s.add_argument("--block", type=int, default=8, help="block size (analysed image px)")
    s.add_argument("--max-size", type=int, default=512, help="downscale longest side to this before analysis")
    s.add_argument("--min-similarity", type=float, default=0.97, help="1 - max RMS difference (0-1)")
    s.add_argument("--min-detail", type=float, default=0.03, help="min block std-dev (0-1); skips flat areas")
    s.add_argument("--min-cluster", type=int, default=50, help="min matching blocks sharing a shift vector")
    s.add_argument("--min-distance", type=int, default=None, help="min shift length (default: block size)")
    s.add_argument("--compare", type=int, default=8, help="sorted neighbours compared per block")
    s.add_argument("--step", type=int, default=1, help="block stride (higher = faster, less sensitive)")
    s.add_argument("--no-lines", action="store_true", help="don't draw connecting lines")
    s.add_argument("--json", help="also write cluster summary JSON to this path")
    outopt(s)

    s = cmd("ela", "Error Level Analysis")
    s.add_argument("-q", "--quality", type=int, default=75, help="JPEG re-save quality")
    s.add_argument("-s", "--scale", type=float, default=20.0, help="error scale")
    s.add_argument("--equalize", action="store_true", help="histogram-equalize result")
    opacity(s); outopt(s)

    s = cmd("noise", "Noise analysis (median filter residual)")
    s.add_argument("-r", "--radius", type=int, default=2, help="median filter radius")
    s.add_argument("-a", "--amplitude", type=float, default=5.0, help="noise amplitude multiplier")
    s.add_argument("--equalize", action="store_true", help="histogram-equalize result")
    opacity(s); outopt(s)

    s = cmd("sweep", "Level sweep")
    s.add_argument("-p", "--position", type=float, default=128, help="sweep position (0-255)")
    s.add_argument("-w", "--width", type=float, default=32, help="window width")
    s.add_argument("--animate", type=int, metavar="FRAMES", default=0,
                   help="write an animated GIF sweeping 0-255 with N frames")
    s.add_argument("--duration", type=int, default=80, help="GIF frame duration (ms)")
    opacity(s); outopt(s)

    s = cmd("gradient", "Luminance gradient")
    s.add_argument("-i", "--intensity", type=float, default=1.0)
    s.add_argument("--equalize", action="store_true")
    opacity(s); outopt(s)

    s = cmd("pca", "Principal Component Analysis")
    s.add_argument("-c", "--component", type=int, choices=[1, 2, 3], default=1)
    s.add_argument("-m", "--mode", choices=["projection", "difference", "distance"], default="projection")
    s.add_argument("--scaling", choices=["linear", "abs", "none"], default="linear")
    s.add_argument("--invert", action="store_true")
    s.add_argument("--equalize", action="store_true")
    s.add_argument("--all", action="store_true", help="write all 3 components (suffix _1.._3)")
    opacity(s); outopt(s)

    s = cmd("metadata", "Dump EXIF / XMP / container metadata")
    s.add_argument("--json", action="store_true", help="JSON output")
    s.add_argument("--no-xmp", action="store_true", help="omit XMP packet")

    s = cmd("geo", "Extract GPS geotags")
    s.add_argument("--json", action="store_true", help="JSON output")
    s.add_argument("--open", action="store_true", help="open location in web browser")

    s = cmd("thumbnail", "Embedded thumbnail extraction and comparison")
    s.add_argument("-s", "--scale", type=float, default=10.0, help="difference amplification")
    s.add_argument("--extract", help="also save the raw embedded thumbnail here")
    s.add_argument("--json", action="store_true")
    outopt(s)

    s = cmd("jpeg", "JPEG structure / quantization tables / quality estimate")
    s.add_argument("--json", action="store_true")

    s = cmd("strings", "Extract printable strings from the file")
    s.add_argument("-n", "--min-len", type=int, default=6)
    s.add_argument("-e", "--encoding", choices=["ascii", "utf16", "both"], default="ascii")
    s.add_argument("-g", "--grep", help="only show strings matching this regex (case-insensitive)")
    s.add_argument("--offsets", action="store_true", help="prefix with byte offset")

    s = cmd("report", "Run all analyses and write an HTML + JSON report")
    s.add_argument("-d", "--outdir", help="output directory (default: <image>_forensics)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    src = args.image
    if not Path(src).is_file():
        print(f"error: {src} not found", file=sys.stderr)
        return 2
    c = args.cmd
    out = getattr(args, "output", None)

    if c == "report":
        run_report(args)
        return 0

    if c in ("metadata", "geo", "jpeg", "strings"):
        if c == "metadata":
            r = metadata(src)
            if args.no_xmp:
                r.pop("XMP", None)
            if args.json:
                print(jdump(r))
            else:
                for sec, vals in r.items():
                    print(f"== {sec} ==")
                    if isinstance(vals, dict):
                        for k, v in vals.items():
                            print(f"  {k:<32} {v}")
                    else:
                        print(vals)
        elif c == "geo":
            r = geotags(src)
            if not r:
                print("No GPS data found.")
                return 1
            if args.json:
                print(jdump(r))
            else:
                for k, v in r.items():
                    if k != "raw":
                        print(f"{k:<20} {v}")
            if args.open and "openstreetmap" in r:
                import webbrowser
                webbrowser.open(r["openstreetmap"])
        elif c == "jpeg":
            data = Path(src).read_bytes()
            try:
                r = parse_jpeg(data)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            if args.json:
                r["quantization_tables"] = {str(k): v for k, v in r["quantization_tables"].items()}
                print(jdump(r))
            else:
                print(format_jpeg_report(r))
        else:
            for off, enc, s in extract_strings(Path(src).read_bytes(), args.min_len,
                                               args.encoding, args.grep):
                print(f"{off:>10x} {enc:<7} {s}" if args.offsets else s)
        return 0

    img = load_image(src)

    if c == "magnifier":
        save(magnifier(img, args.x, args.y, args.size, args.zoom, args.enhance),
             out or default_out(src, "magnifier"))
    elif c == "ela":
        save(ela(img, args.quality, args.scale, args.opacity, args.equalize), out or default_out(src, "ela"))
    elif c == "noise":
        save(noise(img, args.radius, args.amplitude, args.opacity, args.equalize),
             out or default_out(src, "noise"))
    elif c == "sweep":
        if args.animate:
            p = out or default_out(src, "sweep", "gif")
            level_sweep_gif(img, p, args.width, args.animate, args.duration)
            print(f"[+] wrote {p}")
        else:
            save(level_sweep(img, args.position, args.width, args.opacity), out or default_out(src, "sweep"))
    elif c == "gradient":
        save(gradient(img, args.intensity, args.opacity, args.equalize), out or default_out(src, "gradient"))
    elif c == "pca":
        comps = [1, 2, 3] if args.all else [args.component]
        for comp in comps:
            res = pca(img, comp, args.mode, args.scaling, args.invert, args.equalize, args.opacity)
            p = out or default_out(src, "pca")
            if args.all:
                pp = Path(p)
                p = str(pp.with_name(f"{pp.stem}_{comp}{pp.suffix}"))
            save(res, p)
    elif c == "clone":
        clusters, pairs, sc = clone_detect(
            img, args.block, args.max_size, args.min_similarity, args.min_detail,
            args.min_cluster, args.min_distance, args.compare, args.step)
        save(render_clones(img, pairs, sc, args.block, lines=not args.no_lines),
             out or default_out(src, "clone"))
        if clusters:
            print(f"[!] {len(clusters)} clone cluster(s) found:")
            for cl in clusters:
                print(f"    shift=({cl['shift_dx']:+d},{cl['shift_dy']:+d})  blocks={cl['blocks']:<6} "
                      f"src={cl['source_bbox']} dst={cl['target_bbox']}")
        else:
            print("[-] no clone clusters found with current settings")
        if args.json:
            Path(args.json).write_text(jdump(clusters))
            print(f"[+] wrote {args.json}")
    elif c == "thumbnail":
        info, res = thumbnail_analysis(img, src, args.scale)
        if not info:
            print("No embedded EXIF thumbnail found.")
            return 1
        canvas, thumb = res
        save(canvas, out or default_out(src, "thumbnail"))
        if args.extract:
            Path(args.extract).write_bytes(extract_thumbnail(src))
            print(f"[+] wrote {args.extract}")
        if args.json:
            print(jdump(info))
        else:
            for k, v in info.items():
                print(f"{k:<22} {v}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # e.g. piped into `head`
        sys.stderr.close()
        sys.exit(0)
