"""Resize captured/original/* to captured/images_256/<name>.png with the Section 1b rule.

    python -m src.prepare_captured
"""
import cv2

from src.common import CAPTURED, SIZE

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def main():
    out_dir = CAPTURED / "images_256"
    out_dir.mkdir(parents=True, exist_ok=True)
    (CAPTURED / "masks_256").mkdir(exist_ok=True)
    for path in sorted((CAPTURED / "original").iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path.name}: unreadable (convert HEIC to JPG first)")
            continue
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        print(f"{path.name}: {img.shape[1]}x{img.shape[0]} -> {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
