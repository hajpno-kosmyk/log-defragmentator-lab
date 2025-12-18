#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare original vs optimized performance"""

from pathlib import Path
from PIL import Image, ImageDraw
import tempfile
import time

# Import both versions
import png_slicer
import png_slicer_optimized as optimized

def create_test_image(size=(1080, 15000)):
    """Create a realistic test image"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", size, (240, 240, 240))
        draw = ImageDraw.Draw(img)
        for i in range(0, size[1], 30):
            x_offset = 100 if i % 60 == 0 else 600
            draw.rectangle([x_offset, i, x_offset + 400, i + 20], fill=(50, 50, 50))
        img.save(tmp.name)
        return Path(tmp.name)

def benchmark_deskew():
    """Compare deskew performance"""
    print("=== Deskew Benchmark ===")

    img_path = create_test_image((1080, 5000))
    try:
        img = png_slicer.open_image_rgb(img_path)

        # Original deskew
        start = time.perf_counter()
        _, angle1 = png_slicer.deskew_small_angle(img, max_deg=2.0, step_deg=0.25)
        t_original = time.perf_counter() - start

        # Optimized fast deskew
        start = time.perf_counter()
        _, angle2 = optimized.deskew_small_angle_fast(img, max_deg=2.0, coarse_step=0.5, fine_step=0.1)
        t_optimized = time.perf_counter() - start

        print(f"Original deskew: {t_original:.3f}s (angle: {angle1:.3f})")
        print(f"Optimized deskew: {t_optimized:.3f}s (angle: {angle2:.3f})")
        print(f"Speedup: {t_original/t_optimized:.2f}x")
        print(f"Angle difference: {abs(angle1 - angle2):.3f} degrees")

    finally:
        img_path.unlink()

def benchmark_full_pipeline():
    """Compare full pipeline with all features"""
    print("\n=== Full Pipeline Benchmark ===")

    img_path = create_test_image((1080, 15000))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Original
            cfg_orig = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.5,
                auto_crop=True, auto_chunk=True, auto_contrast=True,
                deskew=True, dedupe=True, manifest=True
            )
            start = time.perf_counter()
            n1 = png_slicer.process_file(img_path, tmp_path / "orig", cfg_orig)
            t_original = time.perf_counter() - start

            # Optimized
            cfg_opt = optimized.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.5,
                auto_crop=True, auto_chunk=True, auto_contrast=True,
                deskew=True, deskew_fast=True, dedupe=True, manifest=True,
                show_progress=False
            )
            start = time.perf_counter()
            n2 = optimized.process_file(img_path, tmp_path / "opt", cfg_opt)
            t_optimized = time.perf_counter() - start

            print(f"Original pipeline: {t_original:.3f}s ({n1} slices)")
            print(f"Optimized pipeline: {t_optimized:.3f}s ({n2} slices)")
            print(f"Speedup: {t_original/t_optimized:.2f}x")

    finally:
        img_path.unlink()

if __name__ == "__main__":
    print("Performance Comparison: Original vs Optimized\n")
    benchmark_deskew()
    benchmark_full_pipeline()
    print("\nDone!")
