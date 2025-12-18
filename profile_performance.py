#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance profiling for png_slicer.py"""

from pathlib import Path
from PIL import Image
import png_slicer
import tempfile
import time

def profile_operation(name, func):
    """Time an operation"""
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s")
    return result, elapsed

def create_test_image(size=(1080, 15000)):
    """Create a realistic test image"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        # Create gradient for more realistic file
        img = Image.new("RGB", size, (240, 240, 240))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Add some "text-like" patterns
        for i in range(0, size[1], 30):
            x_offset = 100 if i % 60 == 0 else 600
            draw.rectangle([x_offset, i, x_offset + 400, i + 20], fill=(50, 50, 50))

        img.save(tmp.name)
        return Path(tmp.name)

def profile_full_pipeline():
    """Profile the complete pipeline"""
    print("\n=== Profiling Full Pipeline ===")

    img_path = create_test_image((1080, 15000))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Baseline
            cfg_baseline = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=False, auto_chunk=False, auto_contrast=False,
                deskew=False, binarize=False, dedupe=False, manifest=False
            )
            _, t1 = profile_operation(
                "Baseline (no features)",
                lambda: png_slicer.process_file(img_path, tmp_path / "baseline", cfg_baseline)
            )

            # With upscale
            cfg_upscale = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.5,
                auto_crop=False, auto_chunk=False, auto_contrast=False,
                deskew=False, binarize=False, dedupe=False, manifest=False
            )
            _, t2 = profile_operation(
                "With upscale 1.5x",
                lambda: png_slicer.process_file(img_path, tmp_path / "upscale", cfg_upscale)
            )

            # With auto-crop
            cfg_autocrop = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=True, auto_chunk=False, auto_contrast=False,
                deskew=False, binarize=False, dedupe=False, manifest=False
            )
            _, t3 = profile_operation(
                "With auto-crop",
                lambda: png_slicer.process_file(img_path, tmp_path / "autocrop", cfg_autocrop)
            )

            # With auto-chunk
            cfg_autochunk = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=False, auto_chunk=True, auto_contrast=False,
                deskew=False, binarize=False, dedupe=False, manifest=False
            )
            _, t4 = profile_operation(
                "With auto-chunk",
                lambda: png_slicer.process_file(img_path, tmp_path / "autochunk", cfg_autochunk)
            )

            # With deskew
            cfg_deskew = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=False, auto_chunk=False, auto_contrast=False,
                deskew=True, binarize=False, dedupe=False, manifest=False
            )
            _, t5 = profile_operation(
                "With deskew",
                lambda: png_slicer.process_file(img_path, tmp_path / "deskew", cfg_deskew)
            )

            # With binarize
            cfg_binarize = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=False, auto_chunk=False, auto_contrast=False,
                deskew=False, binarize=True, dedupe=False, manifest=False
            )
            _, t6 = profile_operation(
                "With binarize",
                lambda: png_slicer.process_file(img_path, tmp_path / "binarize", cfg_binarize)
            )

            # With dedupe
            cfg_dedupe = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.0,
                auto_crop=False, auto_chunk=False, auto_contrast=False,
                deskew=False, binarize=False, dedupe=True, manifest=False
            )
            _, t7 = profile_operation(
                "With dedupe",
                lambda: png_slicer.process_file(img_path, tmp_path / "dedupe", cfg_dedupe)
            )

            # All features
            cfg_all = png_slicer.SliceConfig(
                chunk_h=1600, overlap=120, upscale=1.5,
                auto_crop=True, auto_chunk=True, auto_contrast=True,
                deskew=True, binarize=False, dedupe=True, manifest=True
            )
            _, t8 = profile_operation(
                "All features enabled",
                lambda: png_slicer.process_file(img_path, tmp_path / "all", cfg_all)
            )

            print(f"\nPerformance Summary:")
            print(f"  Upscale overhead: {t2/t1:.2f}x")
            print(f"  Auto-crop overhead: {t3/t1:.2f}x")
            print(f"  Auto-chunk overhead: {t4/t1:.2f}x")
            print(f"  Deskew overhead: {t5/t1:.2f}x")
            print(f"  Binarize overhead: {t6/t1:.2f}x")
            print(f"  Dedupe overhead: {t7/t1:.2f}x")
            print(f"  All features overhead: {t8/t1:.2f}x")

    finally:
        img_path.unlink()

def profile_component_functions():
    """Profile individual component functions"""
    print("\n=== Profiling Component Functions ===")

    img_path = create_test_image((1080, 15000))

    try:
        img = png_slicer.open_image_rgb(img_path)

        # Test auto_crop_chat_column
        profile_operation(
            "auto_crop_chat_column",
            lambda: png_slicer.auto_crop_chat_column(img, pad=12)
        )

        # Test estimate_line_height
        profile_operation(
            "estimate_line_height",
            lambda: png_slicer.estimate_line_height(img)
        )

        # Test deskew_small_angle
        profile_operation(
            "deskew_small_angle",
            lambda: png_slicer.deskew_small_angle(img, max_deg=2.0, step_deg=0.25)
        )

        # Test apply_binarize_otsu
        profile_operation(
            "apply_binarize_otsu",
            lambda: png_slicer.apply_binarize_otsu(img)
        )

        # Test apply_upscale
        profile_operation(
            "apply_upscale 1.5x",
            lambda: png_slicer.apply_upscale(img, 1.5)
        )

        # Test slice_one
        profile_operation(
            "slice_one (1600px chunks)",
            lambda: png_slicer.slice_one(img, chunk_h=1600, overlap=120)
        )

    finally:
        img_path.unlink()

if __name__ == "__main__":
    print("Starting performance profiling...")
    profile_component_functions()
    profile_full_pipeline()
    print("\n✓ Profiling complete!")
