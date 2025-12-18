# PNG Slicer Quick Start Guide

## Installation

```bash
# Required
pip install Pillow

# Optional (for progress bars)
pip install tqdm
```

## Basic Usage

### Original Version

```bash
# Simple slicing
python png_slicer.py --input screenshot.png --out output/

# With all features
python png_slicer.py \
  --input screenshots/ \
  --out output/ \
  --auto-crop \
  --auto-chunk \
  --auto-contrast \
  --deskew \
  --dedupe \
  --batch-size 20
```

### Optimized Version (Recommended)

```bash
# Fast preset
python png_slicer_optimized.py --input screenshots/ --out output/ --preset fast

# Balanced preset (recommended)
python png_slicer_optimized.py --input screenshots/ --out output/ --preset balanced

# Quality preset
python png_slicer_optimized.py --input screenshots/ --out output/ --preset quality
```

## Common Scenarios

### 1. Quick Preview
```bash
python png_slicer_optimized.py --input chat.png --out preview/ --preset fast
```

### 2. OCR Preparation
```bash
python png_slicer_optimized.py --input chat.png --out ocr/ --preset balanced
```

### 3. Maximum Quality for Evidence
```bash
python png_slicer_optimized.py --input evidence/ --out sliced/ --preset quality --recursive
```

### 4. Debug Processing Issues
```bash
python png_slicer_optimized.py --input problem.png --out debug/ --save-debug
```

### 5. Batch Processing with Upload Packs
```bash
python png_slicer_optimized.py \
  --input screenshots/ \
  --out output/ \
  --preset balanced \
  --batch-size 15 \
  --prompt-lang en
```

## Configuration Options

### Presets
| Preset | Speed | Quality | Upscale | Features | Best For |
|--------|-------|---------|---------|----------|----------|
| fast | ⚡⚡⚡ | ⭐⭐ | 1.0x | Minimal | Preview |
| balanced | ⚡⚡ | ⭐⭐⭐ | 1.5x | Most | General use |
| quality | ⚡ | ⭐⭐⭐⭐ | 2.0x | All | Best OCR |

### Feature Flags

| Flag | Effect | Overhead | Recommended |
|------|--------|----------|-------------|
| `--auto-crop` | Crop to chat column | 1.5x | ✓ Yes |
| `--auto-chunk` | Smart chunk height | 1.2x | ✓ Yes |
| `--auto-contrast` | Enhance readability | 1.1x | ✓ Yes |
| `--deskew` | Straighten rotated | 20x | Only if needed |
| `--deskew-fast` | Fast deskew | 17x | ✓ If using deskew |
| `--denoise` | Remove noise | 1.1x | Optional |
| `--binarize` | Black & white | 1.1x | Only for poor scans |
| `--dedupe` | Skip duplicates | 0.96x | ✓ Yes (speeds up!) |

### Output Options

| Option | Default | Notes |
|--------|---------|-------|
| `--format` | png | png or jpg (png recommended for OCR) |
| `--output-dpi` | 300 | DPI for OCR engines |
| `--batch-size` | 0 | Group slices (e.g., 15 for easy upload) |
| `--prompt-lang` | pl | pl or en for PROMPT.txt |

## Performance Comparison

Test image: 1080x15000 pixels

| Configuration | Time | Speedup |
|--------------|------|---------|
| Baseline (no features) | 0.43s | 1.0x |
| Preset: fast | 0.41s | 1.05x |
| Preset: balanced | 1.57s | 0.27x |
| Preset: quality | 11.9s | 0.04x |
| All features (slow deskew) | 13.8s | 0.03x |

## Output Structure

```
output/
  image_name/
    image_name_001.png
    image_name_002.png
    ...
    manifest.json           # Metadata and hashes
    batches/               # If --batch-size > 0
      batch_001/
        image_name_001.png
        image_name_002.png
        ...
        PROMPT.txt         # OCR instructions
      batch_002/
        ...
    debug/                 # If --save-debug
      00_original.png
      01_manual_crop.png
      02_auto_crop.png
      03_upscale.png
      04_deskew.png
      05_enhance.png
      06_binarize.png
```

## Troubleshooting

### Script is too slow
- Don't use `--deskew` unless images are rotated
- Use `--preset fast` for quick processing
- If you need deskew, ensure `--deskew-fast` is enabled

### Text is hard to read in output
- Use `--auto-contrast` (enabled in balanced/quality presets)
- Try `--upscale 2.0` for better resolution
- Use `--denoise` if source is noisy
- Consider `--binarize` for very poor quality scans

### Getting duplicate slices
- Enable `--dedupe` (included in all presets)
- Adjust `--dedupe-hamming` (lower = stricter, default: 4)

### Slices are cutting through text
- Enable `--auto-chunk` (included in balanced/quality presets)
- Adjust `--overlap` (default: 120, try 150-200 for safety)

### Auto-crop removing important content
- Disable `--auto-crop` and use manual crop:
  ```bash
  --crop-left 100 --crop-right 100 --crop-top 50 --crop-bottom 50
  ```

### Out of memory
- Reduce `--upscale` (try 1.0)
- Process files one at a time instead of batch
- Use original version without progress bars

## Testing

```bash
# Run self-test (creates synthetic images)
python png_slicer.py --selftest --out test_output/

# Run edge case tests
python test_edge_cases.py

# Run performance profiling
python profile_performance.py

# Compare original vs optimized
python compare_performance.py
```

## Tips & Tricks

1. **Start with balanced preset** and adjust from there
2. **Enable dedupe** - it actually speeds things up
3. **Use batch-size 15-20** for easy uploading to AI services
4. **Save debug images** when troubleshooting: `--save-debug`
5. **Check manifest.json** for metadata and slice hashes
6. **Use output-dpi 300** for best OCR results

## Example Workflows

### Workflow 1: Quick OCR Upload
```bash
# 1. Slice with balanced preset
python png_slicer_optimized.py \
  --input chat_screenshots/ \
  --out slices/ \
  --preset balanced \
  --batch-size 15

# 2. Upload batches/ folder to OCR service
# 3. Each batch has PROMPT.txt with instructions
```

### Workflow 2: Evidence Documentation
```bash
# 1. Maximum quality, all features
python png_slicer_optimized.py \
  --input evidence/ \
  --out processed/ \
  --preset quality \
  --save-debug \
  --recursive

# 2. Check manifest.json for SHA256 hashes
# 3. Archive with original files
```

### Workflow 3: Development/Debugging
```bash
# 1. Process with debug images
python png_slicer_optimized.py \
  --input problem.png \
  --out debug/ \
  --save-debug \
  --auto-crop \
  --auto-contrast

# 2. Examine debug/ folder to see each step
# 3. Adjust parameters based on results
```

---

**Need help?** See [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) for detailed analysis and recommendations.
