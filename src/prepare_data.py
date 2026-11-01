"""Resize DUTS to 256x256 and cache it as .npy arrays, then split DUTS-TR into train/val.

Photos use cv2.INTER_AREA, masks use cv2.INTER_NEAREST and are binarised with >= 128.
Images are kept in OpenCV's BGR order, which is also what the inference pipeline sees,
so no colour conversion is needed at test time.

Outputs in data/cache/:
    tr_images.npy (N,256,256,3) uint8, tr_masks.npy (N,256,256) uint8 in {0,1}, tr_names.txt
    te_images.npy, te_masks.npy, te_names.txt
    split.npz with integer index arrays `train` and `val` into the DUTS-TR arrays
"""
import argparse

import cv2
import numpy as np

from src.common import CACHE, DATA, SIZE, SPLIT_SEED, VAL_COUNT


def load_pair(img_path, mask_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return img, (mask >= 128).astype(np.uint8)


def build(prefix, img_dir, mask_dir):
    names = sorted(p.stem for p in img_dir.glob("*.jpg"))
    n = len(names)
    images = np.lib.format.open_memmap(CACHE / f"{prefix}_images.npy", "w+", np.uint8, (n, SIZE, SIZE, 3))
    masks = np.lib.format.open_memmap(CACHE / f"{prefix}_masks.npy", "w+", np.uint8, (n, SIZE, SIZE))
    for i, name in enumerate(names):
        images[i], masks[i] = load_pair(img_dir / f"{name}.jpg", mask_dir / f"{name}.png")
        if (i + 1) % 1000 == 0:
            print(f"{prefix}: {i + 1}/{n}")
    images.flush()
    masks.flush()
    (CACHE / f"{prefix}_names.txt").write_text("\n".join(names))
    print(f"{prefix}: {n} images cached")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    sets = {
        "tr": (DATA / "DUTS-TR" / "DUTS-TR-Image", DATA / "DUTS-TR" / "DUTS-TR-Mask"),
        "te": (DATA / "DUTS-TE" / "DUTS-TE-Image", DATA / "DUTS-TE" / "DUTS-TE-Mask"),
    }
    sizes = {}
    for prefix, (img_dir, mask_dir) in sets.items():
        if args.skip_existing and (CACHE / f"{prefix}_names.txt").exists():
            sizes[prefix] = len((CACHE / f"{prefix}_names.txt").read_text().split())
            continue
        sizes[prefix] = build(prefix, img_dir, mask_dir)

    perm = np.random.default_rng(SPLIT_SEED).permutation(sizes["tr"])
    val, train = np.sort(perm[:VAL_COUNT]), np.sort(perm[VAL_COUNT:])
    np.savez(CACHE / "split.npz", train=train, val=val)
    print(f"split: {len(train)} train / {len(val)} val")


if __name__ == "__main__":
    main()
