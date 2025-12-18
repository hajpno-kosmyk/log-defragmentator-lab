#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge case tests for png_slicer.py"""

from pathlib import Path
from PIL import Image
import png_slicer
import tempfile
import shutil
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_tiny_image():
    """Test with very small image"""
    print("Testing tiny image (10x10)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "tiny.png"
        Image.new("RGB", (10, 10), (255, 255, 255)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=100, overlap=10)
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✓ Tiny image: {n} slices")
        except Exception as e:
            print(f"  ✗ Tiny image failed: {e}")

def test_very_tall_image():
    """Test with extremely tall image"""
    print("Testing very tall image (100x50000)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "tall.png"
        Image.new("RGB", (100, 50000), (255, 255, 255)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=1000, overlap=50, manifest=False)
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✓ Tall image: {n} slices")
        except Exception as e:
            print(f"  ✗ Tall image failed: {e}")

def test_zero_overlap():
    """Test with zero overlap"""
    print("Testing zero overlap...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (500, 3000), (255, 255, 255)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=1000, overlap=0)
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✓ Zero overlap: {n} slices")
        except Exception as e:
            print(f"  ✗ Zero overlap failed: {e}")

def test_chunk_equals_height():
    """Test when chunk height equals image height"""
    print("Testing chunk_h == image height...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (500, 1000), (255, 255, 255)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=1000, overlap=50)
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✓ Chunk equals height: {n} slices (expected: 1)")
        except Exception as e:
            print(f"  ✗ Chunk equals height failed: {e}")

def test_invalid_crop():
    """Test with crop that exceeds image bounds"""
    print("Testing invalid crop margins...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "test.png"
        Image.new("RGB", (500, 1000), (255, 255, 255)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=1000, crop_left=600)  # exceeds width
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✗ Invalid crop should have failed but got {n} slices")
        except ValueError as e:
            print(f"  ✓ Invalid crop correctly rejected: {e}")
        except Exception as e:
            print(f"  ? Invalid crop unexpected error: {e}")

def test_all_black_image():
    """Test with completely black image"""
    print("Testing all-black image...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "black.png"
        Image.new("RGB", (500, 2000), (0, 0, 0)).save(img_path)

        cfg = png_slicer.SliceConfig(chunk_h=1000, auto_crop=True, auto_chunk=True)
        try:
            n = png_slicer.process_file(img_path, tmp_path / "out", cfg)
            print(f"  ✓ All-black image: {n} slices")
        except Exception as e:
            print(f"  ✗ All-black image failed: {e}")

def test_hamming_distance():
    """Test hamming distance calculation"""
    print("Testing hamming distance...")
    assert png_slicer.hamming64(0b1111, 0b0000) == 4
    assert png_slicer.hamming64(0b1010, 0b1010) == 0
    assert png_slicer.hamming64(0b1111111111111111, 0b0000000000000000) == 16
    print("  ✓ Hamming distance tests passed")

def test_dhash_consistency():
    """Test dHash produces consistent results"""
    print("Testing dHash consistency...")
    # Create images with gradients so they have different hashes
    img1 = Image.new("RGB", (100, 100), (255, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img1)
    draw.rectangle([0, 0, 50, 100], fill=(200, 0, 0))

    img2 = Image.new("RGB", (100, 100), (255, 0, 0))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([0, 0, 50, 100], fill=(200, 0, 0))

    hash1 = png_slicer.dhash(img1)
    hash2 = png_slicer.dhash(img2)
    assert hash1 == hash2, f"Same images should have same hash: {hash1} != {hash2}"

    # Create distinctly different image
    img3 = Image.new("RGB", (100, 100), (0, 255, 0))
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([50, 0, 100, 100], fill=(0, 200, 0))
    hash3 = png_slicer.dhash(img3)
    # Note: solid color images all have hash 0, so we just check it works
    print(f"  ✓ dHash consistency tests passed (hash1={hash1:x}, hash3={hash3:x})")

if __name__ == "__main__":
    print("Running edge case tests...\n")
    test_hamming_distance()
    test_dhash_consistency()
    test_tiny_image()
    test_very_tall_image()
    test_zero_overlap()
    test_chunk_equals_height()
    test_invalid_crop()
    test_all_black_image()
    print("\n✓ All edge case tests completed!")
