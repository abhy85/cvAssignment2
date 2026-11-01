"""Figures for the report: captured photo overlays, worst validation cases, training curve.

    python -m src.make_figures --run r128_final
Run src.evaluate first (it writes per-image scores and the captured predictions).
"""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common import CACHE, CAPTURED, RESULTS, ROOT  # noqa: E402

FIG_DIR = ROOT / "report" / "figures"
TRUE_COLOUR = np.array([0, 200, 0])     # green (RGB)
PRED_COLOUR = np.array([230, 40, 200])  # magenta (RGB)


def overlay(img_bgr, mask, colour, alpha=0.45):
    rgb = img_bgr[..., ::-1].astype(np.float32)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * colour
    return rgb.clip(0, 255).astype(np.uint8)


def pair_grid(rows, path, title):
    """rows: list of (name, img_bgr, true_mask, pred_mask, iou, bf1)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)), squeeze=False)
    for ax_row, (name, img, gt, pred, iou, bf1) in zip(axes, rows):
        ax_row[0].imshow(img[..., ::-1])
        ax_row[0].set_title(name, fontsize=9)
        ax_row[1].imshow(overlay(img, gt, TRUE_COLOUR))
        ax_row[1].set_title("true mask", fontsize=9)
        ax_row[2].imshow(overlay(img, pred, PRED_COLOUR))
        ax_row[2].set_title(f"predicted  IoU {iou:.2f}  BF1 {bf1:.2f}", fontsize=9)
        for ax in ax_row:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"wrote {path}")


def captured_figure():
    scores = {r["name"]: r for r in json.loads((RESULTS / "per_image_captured.json").read_text())}
    rows = []
    for name, r in scores.items():
        img = cv2.imread(str(CAPTURED / "images_256" / f"{name}.png"))
        gt = cv2.imread(str(CAPTURED / "masks_256" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) >= 128
        pred = cv2.imread(str(RESULTS / "captured_pred" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) >= 128
        rows.append((name, img, gt, pred, r["IoU_obj"], r["BF1"]))
    pair_grid(rows, FIG_DIR / "captured_overlays.png", "Captured photos: true (green) vs predicted (magenta)")


def worst_val_figure(k=6):
    scores = json.loads((RESULTS / "per_image_val.json").read_text())
    worst = sorted(scores, key=lambda r: 0.7 * r["IoU_obj"] + 0.3 * r["BF1"])[:k]
    names = (CACHE / "tr_names.txt").read_text().split()
    index = {n: i for i, n in enumerate(names)}
    images = np.load(CACHE / "tr_images.npy", mmap_mode="r")
    masks = np.load(CACHE / "tr_masks.npy", mmap_mode="r")

    from src.pipeline import Segmenter
    seg = Segmenter(ROOT / "weights" / "model.pt")
    rows = []
    for r in worst:
        i = index[r["name"]]
        img = np.array(images[i])
        rows.append((r["name"], img, masks[i], seg(img), r["IoU_obj"], r["BF1"]))
    pair_grid(rows, FIG_DIR / "worst_val.png", "Lowest scoring validation photos")


def curve_figure(run):
    hist = json.loads((RESULTS / f"train_{run}.json").read_text())["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["loss"] for h in hist])
    ax[0].set_title("training loss (BCE + soft Dice)")
    ax[0].set_xlabel("epoch")
    for key in ("mIoU", "BF1", "Q"):
        ax[1].plot(ep, [h[key] for h in hist], label=key)
    ax[1].set_title("validation")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'training_curve.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="training run name for the curve figure")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULTS / "per_image_captured.json").exists():
        captured_figure()
    if (RESULTS / "per_image_val.json").exists():
        worst_val_figure()
    if args.run:
        curve_figure(args.run)


if __name__ == "__main__":
    main()
