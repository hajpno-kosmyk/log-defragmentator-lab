#!/usr/bin/env python3
"""
slice_chat_pngs.py

Upgraded screenshot slicer for OCR/vision models.

Integrated upgrades (1-7):
1) Auto-detect and crop the chat column (optional; robust in light + dark mode)
2) Auto-tune chunk height based on estimated line height (optional; light + dark mode)
3) Smart contrast pipeline (dark/light mode detection) (optional)
4) Deskew (small-angle) via projection-variance search (optional)
5) Optional binarization (Otsu threshold) (optional)
6) Deduplication of near-identical slices (optional; SAFE: dHash + MAD thumbnail diff)
7) Manifest + sha256 + dhash for each produced slice (optional; recommended)

Also:
- Overlapping slicing
- Optional upscale
- Batch packing for uploads (copies slices into batches with PROMPT.txt in Polish by default)
- Self-test mode (synthetic screenshots) to debug pipeline without touching your real evidence

Dependencies: Pillow (PIL). No OpenCV required.
Python 3.10+ recommended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class SliceConfig:
    # Base slicing defaults (keep your chosen values)
    chunk_h: int = 1600
    overlap: int = 120
    upscale: float = 1.5

    # Manual crop margins (applied before auto-crop)
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0

    # Simple manual contrast (applied after auto-contrast, if any)
    contrast: float = 1.0

    # Output encoding
    out_format: str = "png"  # png or jpg
    jpg_quality: int = 92

    # --- Upgrade toggles ---
    auto_crop: bool = False
    auto_crop_pad: int = 12

    auto_chunk: bool = False
    auto_chunk_target_lines: int = 38
    auto_chunk_min: int = 1000
    auto_chunk_max: int = 2200

    auto_contrast: bool = False
    auto_contrast_strength: float = 1.25
    auto_brightness: float = 1.05
    denoise: bool = False

    deskew: bool = False
    deskew_max_deg: float = 2.0
    deskew_step_deg: float = 0.25

    binarize: bool = False

    dedupe: bool = False
    dedupe_hamming: int = 4  # smaller = stricter (0 means exact hash match)
    dedupe_thumb: int = 64  # thumbnail size for MAD check
    dedupe_mad_threshold: float = 2.0  # mean abs pixel diff (0..255) to consider duplicate

    manifest: bool = True

    # Batching
    batch_size: int = 0
    make_prompt: bool = True
    prompt_lang: str = "pl"


# ----------------------------
# File helpers
# ----------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_images(input_path: Path, recursive: bool) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
        return
    pattern = "**/*" if recursive else "*"
    for p in input_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def open_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


# ----------------------------
# Image preprocessing
# ----------------------------

def apply_margin_crop(img: Image.Image, cfg: SliceConfig) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Manual margin crop. Returns image + crop box (left, top, right, bottom) in original image coords."""
    w, h = img.size
    left = max(0, cfg.crop_left)
    top = max(0, cfg.crop_top)
    right = max(0, w - cfg.crop_right)
    bottom = max(0, h - cfg.crop_bottom)
    if right <= left or bottom <= top:
        raise ValueError(f"Crop too aggressive. Invalid box: {(left, top, right, bottom)} for size {(w, h)}")
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def apply_upscale(img: Image.Image, upscale: float) -> Image.Image:
    if upscale <= 0:
        raise ValueError("Upscale must be > 0")
    if abs(upscale - 1.0) < 1e-6:
        return img
    w, h = img.size
    return img.resize((int(w * upscale), int(h * upscale)), Image.Resampling.LANCZOS)


def is_dark_mode(img: Image.Image) -> bool:
    """Heuristic: mean luminance under ~115/255 => dark."""
    g = img.convert("L")
    g_small = g.resize((200, max(50, int(200 * g.size[1] / max(1, g.size[0])))), Image.Resampling.BILINEAR)
    hist = g_small.histogram()
    total = sum(hist)
    if total == 0:
        return False
    mean = sum(i * c for i, c in enumerate(hist)) / total
    return mean < 115


def apply_auto_enhance(img: Image.Image, cfg: SliceConfig) -> Tuple[Image.Image, dict]:
    """Smart contrast pipeline: detect dark/light mode and apply mild improvements."""
    meta = {"dark_mode": None, "applied": False}
    if not cfg.auto_contrast and not cfg.denoise and abs(cfg.contrast - 1.0) < 1e-6:
        return img, meta

    dark = is_dark_mode(img)
    meta["dark_mode"] = dark

    out = img
    if cfg.denoise:
        out = out.filter(ImageFilter.MedianFilter(size=3))
        meta["applied"] = True

    if cfg.auto_contrast:
        if dark:
            out = ImageEnhance.Brightness(out).enhance(cfg.auto_brightness)
            out = ImageEnhance.Contrast(out).enhance(cfg.auto_contrast_strength)
        else:
            out = ImageEnhance.Contrast(out).enhance(max(1.05, cfg.auto_contrast_strength - 0.15))
        meta["applied"] = True

    if abs(cfg.contrast - 1.0) > 1e-6:
        out = ImageEnhance.Contrast(out).enhance(cfg.contrast)
        meta["applied"] = True

    return out, meta


def apply_binarize_otsu(img: Image.Image) -> Tuple[Image.Image, int]:
    """Global Otsu binarization. Returns binarized image and threshold used."""
    g = img.convert("L")
    hist = g.histogram()
    total = sum(hist)
    if total == 0:
        return img, 0

    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    bw = g.point(lambda p: 255 if p > threshold else 0, mode="L")
    return bw.convert("RGB"), threshold


# ----------------------------
# Auto-crop chat column (Upgrade 1)
# ----------------------------

def auto_crop_chat_column(img: Image.Image, pad: int = 12) -> Tuple[Image.Image, Tuple[int, int, int, int], dict]:
    """
    Robust auto-crop:
    - Downscale to ~500px width
    - Compute per-column EDGE energy (sum of abs vertical differences)
      Text/bubbles create edges; flat margins create few edges.
    - Crop to the central span containing ~96% of edge energy.

    Works in both light and dark mode.
    """
    w, h = img.size
    g = img.convert("L")

    target_w = min(500, w)
    target_h = max(80, int(h * (target_w / max(1, w))))
    g_small = g.resize((target_w, target_h), Image.Resampling.BILINEAR)
    px = g_small.load()

    edges = [0.0] * target_w
    for x in range(target_w):
        s = 0
        prev = px[x, 0]
        for y in range(1, target_h):
            cur = px[x, y]
            s += abs(cur - prev)
            prev = cur
        edges[x] = s / max(1, (target_h - 1))

    total = sum(edges)
    if total <= 1e-6:
        return img, (0, 0, w, h), {"detected": False, "reason": "no_edge_energy"}

    frac = 0.02  # 2% trimmed each side
    left_s = 0
    acc = 0.0
    for i, v in enumerate(edges):
        acc += v
        if acc >= total * frac:
            left_s = i
            break

    right_s = target_w - 1
    acc = 0.0
    for i in range(target_w - 1, -1, -1):
        acc += edges[i]
        if acc >= total * frac:
            right_s = i
            break

    if right_s <= left_s:
        return img, (0, 0, w, h), {"detected": False, "reason": "bad_bounds"}

    left = int(left_s * (w / target_w))
    right = int((right_s + 1) * (w / target_w))

    left = max(0, left - pad)
    right = min(w, right + pad)

    box = (left, 0, right, h)
    cropped = img.crop(box)
    return cropped, box, {
        "detected": True,
        "method": "edge_energy",
        "target_w": target_w,
        "target_h": target_h,
        "trim_frac_each_side": frac,
    }


# ----------------------------
# Auto-chunk (Upgrade 2)
# ----------------------------

def estimate_line_height(img: Image.Image) -> Optional[int]:
    """
    Estimate dominant line spacing via autocorrelation on horizontal projection.
    Works in light mode (dark text) and dark mode (bright text).
    Returns line height in pixels (in current image scale), or None.
    """
    g = img.convert("L")

    w, h = g.size
    top = int(h * 0.15)
    bottom = int(h * 0.85)
    if bottom <= top + 50:
        top, bottom = 0, h

    band = g.crop((0, top, w, bottom))

    target_w = min(600, w)
    target_h = min(1200, band.size[1])
    band_s = band.resize((target_w, target_h), Image.Resampling.BILINEAR)
    px = band_s.load()

    hist = band_s.histogram()
    total = sum(hist)
    mean = sum(i * c for i, c in enumerate(hist)) / max(1, total)

    dark_mode = mean < 115
    thr = int(min(235, max(20, mean + 20))) if dark_mode else int(min(235, max(20, mean - 20)))

    proj = []
    if dark_mode:
        for y in range(target_h):
            c = 0
            for x in range(target_w):
                if px[x, y] > thr:
                    c += 1
            proj.append(c)
    else:
        for y in range(target_h):
            c = 0
            for x in range(target_w):
                if px[x, y] < thr:
                    c += 1
            proj.append(c)

    avg = sum(proj) / max(1, len(proj))
    proj = [p - avg for p in proj]

    best_lag = None
    best_score = -1e18
    for lag in range(8, 61):
        s = 0.0
        for i in range(0, len(proj) - lag):
            s += proj[i] * proj[i + lag]
        if s > best_score:
            best_score = s
            best_lag = lag

    if best_lag is None or best_score <= 0:
        return None

    scale = (band.size[1] / target_h)
    return max(8, int(best_lag * scale))


def choose_chunk_height(img: Image.Image, cfg: SliceConfig) -> Tuple[int, dict]:
    meta = {"auto_chunk_used": False, "line_height": None}
    if not cfg.auto_chunk:
        return cfg.chunk_h, meta

    lh = estimate_line_height(img)
    meta["line_height"] = lh
    if lh is None:
        return cfg.chunk_h, meta

    target = int(cfg.auto_chunk_target_lines * lh)
    target = max(cfg.auto_chunk_min, min(cfg.auto_chunk_max, target))
    meta["auto_chunk_used"] = True
    meta["chosen_chunk_h"] = target
    return target, meta


# ----------------------------
# Deskew (Upgrade 4)
# ----------------------------

def _projection_variance_score(g: Image.Image) -> float:
    """Score image by variance of horizontal projection; higher => more horizontally aligned text."""
    w, h = g.size
    px = g.load()
    proj = [0] * h
    for y in range(h):
        s = 0
        for x in range(w):
            s += px[x, y]
        proj[y] = s / w
    mean = sum(proj) / h
    var = sum((p - mean) ** 2 for p in proj) / h
    return var


def deskew_small_angle(img: Image.Image, max_deg: float = 2.0, step_deg: float = 0.25) -> Tuple[Image.Image, float]:
    """
    Brute-force small angle deskew:
    - downscale
    - rotate over [-max_deg, +max_deg]
    - choose angle maximizing horizontal projection variance
    """
    if max_deg <= 0 or step_deg <= 0:
        return img, 0.0

    g = img.convert("L")
    w, h = g.size
    scale = min(1.0, 800 / max(1, w))
    g_s = g.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    best_angle = 0.0
    best_score = -1.0

    steps = int((2 * max_deg) / step_deg) + 1
    for k in range(steps):
        ang = -max_deg + k * step_deg
        rot = g_s.rotate(ang, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255)
        score = _projection_variance_score(rot)
        if score > best_score:
            best_score = score
            best_angle = ang

    if abs(best_angle) < 1e-6:
        return img, 0.0

    out = img.rotate(best_angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))
    return out, float(best_angle)


# ----------------------------
# Hashing + dedupe (Upgrade 6)
# ----------------------------

def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (dHash) -> 64-bit int by default."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    px = g.load()
    bits = 0
    bitpos = 0
    for y in range(hash_size):
        for x in range(hash_size):
            bits |= (1 if px[x, y] > px[x + 1, y] else 0) << bitpos
            bitpos += 1
    return bits


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def thumb_gray_bytes(img: Image.Image, size: int = 64) -> bytes:
    """Return grayscale thumbnail pixels as bytes for fast diffing."""
    t = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return t.tobytes()


def mean_abs_diff(a: bytes, b: bytes) -> float:
    """Mean absolute difference between two equal-length byte arrays."""
    if len(a) != len(b) or not a:
        return 1e9
    s = 0
    for x, y in zip(a, b):
        s += abs(x - y)
    return s / len(a)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slice_one(img: Image.Image, chunk_h: int, overlap: int) -> list[Image.Image]:
    w, h = img.size
    if chunk_h <= 0:
        raise ValueError("chunk_h must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_h:
        raise ValueError("overlap must be < chunk_h")

    step = chunk_h - overlap
    chunks: list[Image.Image] = []
    y = 0
    while y < h:
        y2 = min(y + chunk_h, h)
        chunks.append(img.crop((0, y, w, y2)))
        if y2 >= h:
            break
        y += step
    return chunks


def save_chunks(
    chunks: list[Image.Image],
    src: Path,
    out_dir: Path,
    cfg: SliceConfig,
) -> Tuple[List[Path], List[dict]]:
    """
    Save chunks with stable names.
    If dedupe enabled, skip near-identical consecutive chunks.
    Returns (saved_paths, slice_records_for_manifest).
    """
    safe_mkdir(out_dir)

    fmt = cfg.out_format.lower()
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError("out_format must be png or jpg")
    ext = ".png" if fmt == "png" else ".jpg"

    saved: list[Path] = []
    records: list[dict] = []

    prev_hash: Optional[int] = None
    prev_thumb: Optional[bytes] = None
    stem = src.stem

    for i, ch in enumerate(chunks, start=1):
        hval = dhash(ch)
        thumb = thumb_gray_bytes(ch, size=cfg.dedupe_thumb)

        dup_of = None
        if cfg.dedupe and prev_hash is not None and prev_thumb is not None:
            dist = hamming64(prev_hash, hval)
            mad = mean_abs_diff(prev_thumb, thumb)
            # Only duplicate if BOTH extremely similar
            if dist <= cfg.dedupe_hamming and mad <= cfg.dedupe_mad_threshold:
                dup_of = i - 1

        if dup_of is not None:
            records.append({
                "index": i,
                "skipped": True,
                "duplicate_of_index": dup_of,
                "dhash": f"{hval:016x}",
                "note": "dedupe_skip",
            })
            continue

        out_path = out_dir / f"{stem}_{i:03d}{ext}"
        if fmt == "png":
            ch.save(out_path, format="PNG", optimize=True)
        else:
            ch.save(out_path, format="JPEG", quality=cfg.jpg_quality, optimize=True)

        saved.append(out_path)
        records.append({
            "index": i,
            "skipped": False,
            "file": out_path.name,
            "dhash": f"{hval:016x}",
        })

        prev_hash = hval
        prev_thumb = thumb

    return saved, records


# ----------------------------
# Batching + prompt
# ----------------------------

def write_batch_prompt(batch_dir: Path, batch_files: list[Path], lang: str = "pl") -> Path:
    safe_mkdir(batch_dir)
    if lang.lower().startswith("pl"):
        prompt = (
            "Zadanie: Przepisz DOKŁADNIE tekst z załączonych obrazów (zrzuty rozmowy).\n"
            "Zasady:\n"
            "1) Bez parafraz, bez streszczeń. Tylko transkrypcja.\n"
            "2) Zachowaj podziały na linie.\n"
            "3) Jeśli widzisz daty/godziny i nadawcę, zachowaj je.\n"
            "4) Jeśli fragment jest nieczytelny, oznacz go jako [NIEPEWNE] lub [???].\n"
            "5) Nie zgaduj brakujących słów.\n\n"
            "Wyjście (format):\n"
            "- Każdy obraz zaczynaj nagłówkiem: ### <nazwa_pliku>\n"
            "- Potem transkrypcja tego obrazu.\n"
        )
    else:
        prompt = (
            "Task: Transcribe EXACT text from the attached images (chat screenshots).\n"
            "Rules:\n"
            "1) No paraphrasing, no summarizing. Transcription only.\n"
            "2) Preserve line breaks.\n"
            "3) If timestamps/sender are visible, keep them.\n"
            "4) Mark unclear parts as [UNCERTAIN] or [???].\n"
            "5) Do not guess missing words.\n\n"
            "Output format:\n"
            "- Start each image with: ### <filename>\n"
            "- Then the transcription.\n"
        )

    prompt_path = batch_dir / "PROMPT.txt"
    files_list = "\n".join(f"- {p.name}" for p in batch_files)
    prompt_path.write_text(prompt + "\nFiles in this batch:\n" + files_list + "\n", encoding="utf-8")
    return prompt_path


def pack_into_batches(
    files: list[Path],
    out_dir: Path,
    batch_size: int,
    make_prompt: bool,
    prompt_lang: str,
) -> dict:
    """
    Copy slice files into out_dir/batches/batch_### for easier uploading.
    Returns mapping: slice_filename -> batch_folder_name
    """
    if batch_size <= 0:
        return {}

    batches_root = out_dir / "batches"
    safe_mkdir(batches_root)

    mapping: dict = {}
    files_sorted = sorted(files)

    batch_idx = 1
    for start in range(0, len(files_sorted), batch_size):
        chunk = files_sorted[start: start + batch_size]
        bdir = batches_root / f"batch_{batch_idx:03d}"
        safe_mkdir(bdir)

        copied: list[Path] = []
        for p in chunk:
            dst = bdir / p.name
            shutil.copy2(p, dst)
            copied.append(dst)
            mapping[p.name] = bdir.name

        if make_prompt:
            write_batch_prompt(bdir, copied, lang=prompt_lang)

        batch_idx += 1

    return mapping


# ----------------------------
# Manifest (Upgrade 7)
# ----------------------------

def write_manifest(
    out_dir: Path,
    source_path: Path,
    source_size: Tuple[int, int],
    cfg: SliceConfig,
    meta: dict,
    slice_records: list[dict],
    saved_paths: list[Path],
    batch_map: dict,
    extra: dict,
) -> Path:
    safe_mkdir(out_dir)
    manifest_path = out_dir / "manifest.json"

    saved_info = []
    for p in saved_paths:
        saved_info.append({
            "file": p.name,
            "sha256": sha256_file(p),
            "batch": batch_map.get(p.name),
        })

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "file": source_path.name,
            "path": str(source_path),
            "size": {"w": source_size[0], "h": source_size[1]},
        },
        "config": asdict(cfg),
        "pipeline_meta": meta,
        "slices": slice_records,
        "saved_files": saved_info,
        "extra": extra,
    }

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


# ----------------------------
# Main per-file pipeline
# ----------------------------

def process_file(path: Path, out_root: Path, cfg: SliceConfig) -> int:
    img0 = open_image_rgb(path)
    source_size = img0.size

    meta: dict = {
        "manual_crop_box": None,
        "auto_crop_box": None,
        "auto_crop": cfg.auto_crop,
        "note": "auto_crop_box is after manual crop (if any)",
    }
    extra: dict = {}

    img, manual_box = apply_margin_crop(img0, cfg)
    meta["manual_crop_box"] = manual_box

    if cfg.auto_crop:
        img, auto_box, ac_meta = auto_crop_chat_column(img, pad=cfg.auto_crop_pad)
        meta["auto_crop_box"] = auto_box
        meta["auto_crop_meta"] = ac_meta

    img = apply_upscale(img, cfg.upscale)

    if cfg.deskew:
        img, angle = deskew_small_angle(img, max_deg=cfg.deskew_max_deg, step_deg=cfg.deskew_step_deg)
        meta["deskew_angle_deg"] = angle

    img, enh_meta = apply_auto_enhance(img, cfg)
    meta["enhance"] = enh_meta

    if cfg.binarize:
        img, thr = apply_binarize_otsu(img)
        meta["binarize_otsu_threshold"] = thr

    chunk_h, ch_meta = choose_chunk_height(img, cfg)
    meta["chunk_meta"] = ch_meta

    chunks = slice_one(img, chunk_h=chunk_h, overlap=cfg.overlap)

    per_file_dir = out_root / path.stem
    saved_paths, slice_records = save_chunks(chunks, path, per_file_dir, cfg)

    batch_map = {}
    if cfg.batch_size and cfg.batch_size > 0:
        batch_map = pack_into_batches(
            files=saved_paths,
            out_dir=per_file_dir,
            batch_size=cfg.batch_size,
            make_prompt=cfg.make_prompt,
            prompt_lang=cfg.prompt_lang,
        )

    if cfg.manifest:
        write_manifest(
            out_dir=per_file_dir,
            source_path=path,
            source_size=source_size,
            cfg=cfg,
            meta=meta,
            slice_records=slice_records,
            saved_paths=saved_paths,
            batch_map=batch_map,
            extra=extra,
        )

    return len(saved_paths)


# ----------------------------
# Self-test (synthetic) for debugging
# ----------------------------

def _make_synth_chat(width: int, height: int, dark: bool = False) -> Image.Image:
    """Generate a synthetic chat-like screenshot for testing (no external fonts needed)."""
    bg = (20, 20, 20) if dark else (245, 245, 245)
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)

    margin = 120
    margin_col = (35, 35, 35) if dark else (230, 230, 230)
    d.rectangle([0, 0, margin, height], fill=margin_col)
    d.rectangle([width - margin, 0, width, height], fill=margin_col)

    col_bg = (15, 15, 15) if dark else (255, 255, 255)
    d.rectangle([margin, 0, width - margin, height], fill=col_bg)

    text_col = (240, 240, 240) if dark else (10, 10, 10)
    y = 40
    for i in range(180):
        x = margin + 20 if i % 2 == 0 else width // 2
        bubble_w = width // 2 - 50
        bubble_h = 22
        bubble_col = (60, 60, 60) if dark else ((220, 235, 255) if i % 2 else (235, 235, 235))
        d.rounded_rectangle([x, y, x + bubble_w, y + bubble_h], radius=8, fill=bubble_col)
        d.text((x + 10, y + 3), f"Linia {i:03d} zażółć gęślą jaźń", fill=text_col)
        y += 28
        if y > height - 60:
            break
    return img


def run_selftest(out_root: Path) -> int:
    """Creates synthetic images and runs the pipeline to catch obvious failures."""
    safe_mkdir(out_root)
    test_dir = out_root / "_selftest"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    safe_mkdir(test_dir)

    light = test_dir / "light.png"
    dark = test_dir / "dark.png"
    skew = test_dir / "skew.png"
    repeat = test_dir / "repeat.png"

    _make_synth_chat(900, 5500, dark=False).save(light)
    _make_synth_chat(900, 5500, dark=True).save(dark)

    img_skew = _make_synth_chat(900, 5500, dark=False).rotate(
        1.25, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255)
    )
    img_skew.save(skew)

    base = _make_synth_chat(900, 3500, dark=False)
    top_chunk = base.crop((0, 0, 900, 1600))
    img_rep = Image.new("RGB", (900, 5100), (255, 255, 255))
    img_rep.paste(base, (0, 0))
    img_rep.paste(top_chunk, (0, 3500))
    img_rep.save(repeat)

    cfg_full = SliceConfig(
        auto_crop=True, auto_chunk=True, auto_contrast=True, denoise=True, deskew=True,
        binarize=False, dedupe=False, batch_size=15, prompt_lang="pl",
    )
    cfg_binarize = SliceConfig(
        auto_crop=True, auto_chunk=True, auto_contrast=True, denoise=False, deskew=False,
        binarize=True, dedupe=False, batch_size=0, manifest=True,
    )
    cfg_dedupe = SliceConfig(
        auto_crop=False, auto_chunk=False, auto_contrast=False, denoise=False, deskew=False,
        binarize=False, dedupe=True, dedupe_hamming=4, dedupe_mad_threshold=1.5,
        batch_size=0, manifest=False, upscale=1.0,
    )

    n1 = process_file(light, out_root, cfg_full)
    n2 = process_file(dark, out_root, cfg_full)
    n3 = process_file(skew, out_root, cfg_full)
    n4 = process_file(repeat, out_root, cfg_dedupe)
    n5 = process_file(light, out_root, cfg_binarize)

    assert n1 > 0 and n2 > 0 and n3 > 0 and n5 > 0
    assert (out_root / "light" / "manifest.json").exists()
    assert (out_root / "dark" / "manifest.json").exists()
    assert (out_root / "skew" / "manifest.json").exists()

    print("[SELFTEST] light slices:", n1)
    print("[SELFTEST] dark slices:", n2)
    print("[SELFTEST] skew slices:", n3)
    print("[SELFTEST] repeat slices with dedupe:", n4)
    print("[SELFTEST] light slices with binarize:", n5)
    print("[SELFTEST] OK")
    return 0


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Slice long chat screenshots into OCR-friendly chunks (with batching + manifest)."
    )
    p.add_argument("--input", required=False, help="Input file or directory containing images")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    p.add_argument("--selftest", action="store_true", help="Run synthetic self-test and exit")

    p.add_argument("--chunk-h", type=int, default=1600, help="Chunk height in pixels (used unless --auto-chunk)")
    p.add_argument("--overlap", type=int, default=120, help="Overlap in pixels between chunks")
    p.add_argument("--upscale", type=float, default=1.5, help="Upscale factor (e.g. 1.5)")

    p.add_argument("--crop-left", type=int, default=0, help="Pixels to crop from left")
    p.add_argument("--crop-right", type=int, default=0, help="Pixels to crop from right")
    p.add_argument("--crop-top", type=int, default=0, help="Pixels to crop from top")
    p.add_argument("--crop-bottom", type=int, default=0, help="Pixels to crop from bottom")

    p.add_argument("--contrast", type=float, default=1.0, help="Manual contrast factor (applied after auto-contrast)")

    p.add_argument("--format", default="png", help="png or jpg (png recommended for OCR)")
    p.add_argument("--jpg-quality", type=int, default=92, help="JPG quality if --format=jpg")

    # Upgrade toggles
    p.add_argument("--auto-crop", action="store_true", help="Auto-detect and crop chat column")
    p.add_argument("--auto-crop-pad", type=int, default=12, help="Padding for auto-crop")
    p.add_argument("--auto-chunk", action="store_true", help="Auto-tune chunk height based on estimated line spacing")
    p.add_argument("--auto-chunk-target-lines", type=int, default=38, help="Target number of text lines per chunk")
    p.add_argument("--auto-contrast", action="store_true", help="Apply smart contrast/brightness depending on dark mode")
    p.add_argument("--auto-contrast-strength", type=float, default=1.25, help="Auto contrast factor baseline")
    p.add_argument("--auto-brightness", type=float, default=1.05, help="Auto brightness factor for dark mode")
    p.add_argument("--denoise", action="store_true", help="Apply mild denoise (median filter)")

    p.add_argument("--deskew", action="store_true", help="Attempt small-angle deskew")
    p.add_argument("--deskew-max-deg", type=float, default=2.0, help="Max absolute degrees for deskew search")
    p.add_argument("--deskew-step-deg", type=float, default=0.25, help="Step degrees for deskew search")

    p.add_argument("--binarize", action="store_true", help="Apply Otsu binarization (can help on noisy backgrounds)")

    p.add_argument("--dedupe", action="store_true", help="Skip near-identical consecutive slices")
    p.add_argument("--dedupe-hamming", type=int, default=4, help="dHash Hamming distance threshold")
    p.add_argument("--dedupe-thumb", type=int, default=64, help="Thumbnail size for MAD dedupe check")
    p.add_argument("--dedupe-mad-threshold", type=float, default=2.0, help="Mean abs pixel diff threshold for dedupe")

    p.add_argument("--no-manifest", action="store_true", help="Disable manifest.json generation")

    # Batching
    p.add_argument("--batch-size", type=int, default=0, help="If >0, copy slices into batches/batch_### folders")
    p.add_argument("--no-prompt", action="store_true", help="Do not write PROMPT.txt inside each batch folder")
    p.add_argument("--prompt-lang", default="pl", help="Language for PROMPT.txt: pl or en")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    out_root = Path(args.out).expanduser().resolve()
    safe_mkdir(out_root)

    if args.selftest:
        return run_selftest(out_root)

    if not args.input:
        print("[ERROR] --input is required unless --selftest is used.", file=sys.stderr)
        return 2

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        return 2

    cfg = SliceConfig(
        chunk_h=args.chunk_h,
        overlap=args.overlap,
        upscale=args.upscale,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        contrast=args.contrast,
        out_format=args.format,
        jpg_quality=args.jpg_quality,
        auto_crop=args.auto_crop,
        auto_crop_pad=args.auto_crop_pad,
        auto_chunk=args.auto_chunk,
        auto_chunk_target_lines=args.auto_chunk_target_lines,
        auto_contrast=args.auto_contrast,
        auto_contrast_strength=args.auto_contrast_strength,
        auto_brightness=args.auto_brightness,
        denoise=args.denoise,
        deskew=args.deskew,
        deskew_max_deg=args.deskew_max_deg,
        deskew_step_deg=args.deskew_step_deg,
        binarize=args.binarize,
        dedupe=args.dedupe,
        dedupe_hamming=args.dedupe_hamming,
        dedupe_thumb=args.dedupe_thumb,
        dedupe_mad_threshold=args.dedupe_mad_threshold,
        manifest=(not args.no_manifest),
        batch_size=args.batch_size,
        make_prompt=(not args.no_prompt),
        prompt_lang=args.prompt_lang,
    )

    files = list(iter_images(input_path, recursive=args.recursive))
    if not files:
        print("[ERROR] No images found (png/jpg/jpeg/webp).", file=sys.stderr)
        return 3

    total = 0
    ok = 0
    skip = 0

    for fp in files:
        try:
            n = process_file(fp, out_root, cfg)
            print(f"[OK] {fp.name} -> {n} slice(s)")
            total += n
            ok += 1
        except Exception as e:
            print(f"[SKIP] {fp.name} -> {e}", file=sys.stderr)
            skip += 1

    print(f"\nDone. Files processed: {ok}, skipped: {skip}, total slices: {total}")
    return 0 if ok > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
