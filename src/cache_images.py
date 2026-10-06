"""
cache_images.py — One-time pre-resize of DDR images to 224×224

DDR fundus photos are ~1 MB JPEGs. Decoding one takes ~50 ms, and every FL
round decodes ~44,000 of them, so the CPU (not the GPU) sets the pace:
~35 min per round on an Arc B580, versus ~3 min with this cache.

This script applies the exact same Resize((224, 224)) once and stores the
pixels losslessly as uint8 arrays:

    data/DDR/DR_grading/cache_224/train.npy          (6260, 224, 224, 3) uint8
    data/DDR/DR_grading/cache_224/train_names.json   image order, checked on load
    (same for valid and test)

DDRDataset uses the cache automatically. The transforms after Resize see
identical pixels, so results stay comparable with the original pipeline.

Safe to re-run after a power cut: finished splits are skipped, and an
interrupted split is rebuilt from scratch (a few minutes).

Usage:
    python src/cache_images.py
"""

import os
import sys
import json
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

# Keep this module light: Pool workers re-import it on Windows, so torch and
# torchvision are only imported inside main().

SIZE    = (224, 224)
SPLITS  = ("train", "valid", "test")
WORKERS = min(8, os.cpu_count() or 1)


def load_resized(img_path):
    """
    Decode one image and resize it to 224×224.

    torchvision's Resize((224, 224)) on a PIL image calls exactly
    img.resize((224, 224), BILINEAR); main() verifies this on a real image.
    """
    try:
        with Image.open(img_path) as img:
            resized = img.convert("RGB").resize(SIZE, Image.BILINEAR)
            return np.asarray(resized, dtype=np.uint8), None
    except Exception as e:
        return None, f"{img_path}: {e}"


def build_split(data_dir, split):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from torchvision import transforms
    from src.dataset import DDRDataset, cache_paths
    from src.run_utils import atomic_write_json

    npy_path, names_path = cache_paths(data_dir, split)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)

    dataset = DDRDataset(root_dir=data_dir, split=split, use_cache=False)
    names   = [name for name, _ in dataset.samples]

    # ── Skip if a complete, matching cache already exists ─────────────────
    if os.path.exists(npy_path) and os.path.exists(names_path):
        with open(names_path, 'r') as f:
            if json.load(f) == names:
                shape = np.load(npy_path, mmap_mode='r').shape
                if shape == (len(names), SIZE[1], SIZE[0], 3):
                    print(f"[{split}] cache already complete ({len(names)} images) — skipping")
                    return True

    paths   = [os.path.join(dataset.img_dir, n) for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[{split}] {len(missing)} images missing, e.g. {missing[0]}")
        print("         Re-run the dataset download command, then this script.")
        return False

    # ── Check our PIL resize matches torchvision's Resize exactly ─────────
    reference = np.asarray(
        transforms.Resize(SIZE)(Image.open(paths[0]).convert("RGB")), dtype=np.uint8
    )
    ours, err = load_resized(paths[0])
    if err or not np.array_equal(reference, ours):
        raise RuntimeError("PIL resize does not match torchvision Resize — do not use this cache.")

    # ── Decode + resize in parallel, write into a memory-mapped .npy ──────
    partial = npy_path + ".partial"
    out     = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.uint8, shape=(len(paths), SIZE[1], SIZE[0], 3)
    )
    errors  = []
    start   = time.time()
    print(f"[{split}] caching {len(paths)} images with {WORKERS} workers...")

    with Pool(WORKERS) as pool:
        for i, (arr, err) in enumerate(pool.imap(load_resized, paths, chunksize=16)):
            if err:
                errors.append(err)
            else:
                out[i] = arr
            if (i + 1) % 500 == 0 or i + 1 == len(paths):
                print(f"  {i + 1}/{len(paths)}  ({time.time() - start:.0f}s)")

    out.flush()
    del out   # close the memory map before renaming (required on Windows)

    if errors:
        os.remove(partial)
        print(f"[{split}] {len(errors)} images could not be read (corrupt download?):")
        for e in errors[:10]:
            print(f"    {e}")
        print("  Delete those files, re-run the dataset download command, then this script.")
        return False

    os.replace(partial, npy_path)
    atomic_write_json(names, names_path)
    print(f"[{split}] done → {npy_path}")
    return True


def main():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.run_utils import use_utf8_output
    use_utf8_output()

    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'DDR', 'DR_grading')
    )
    print("=" * 60)
    print("  Building 224×224 image cache")
    print(f"  {data_dir}")
    print("=" * 60)

    ok = all([build_split(data_dir, split) for split in SPLITS])

    if ok:
        print("\n✅ Cache complete. DDRDataset will use it automatically.")
    else:
        print("\n🚨 Cache incomplete — fix the problems above and run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
