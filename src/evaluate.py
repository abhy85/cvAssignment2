"""Evaluate the deployed pipeline (weights/model.pt, CPU) on validation, DUTS-TE and the
captured photos, and combine with results/timing.json into the final table.

    python -m src.evaluate --weights weights/model.pt [--sets val te captured]

Writes results/metrics.json, per-image scores (results/per_image_<set>.json) and the
predicted masks of the captured photos to results/captured_pred/.
"""
import src.threads  # noqa: F401

import argparse
import json
import os

import cv2
import numpy as np
import torch

from src.common import CACHE, CAPTURED, RESULTS
from src.metrics import SegMeter, boundary_f1, ces
from src.pipeline import Segmenter


def per_image_scores(pred, gt):
    p, g = torch.from_numpy(pred).bool()[None], torch.from_numpy(gt).bool()[None]
    inter, union = (p & g).sum().item(), (p | g).sum().item()
    return {"IoU_obj": inter / union if union else 1.0, "BF1": boundary_f1(p, g).item()}


def run_set(seg, images, masks, names):
    meter, rows, preds = SegMeter(), [], []
    for img, gt, name in zip(images, masks, names):
        pred = seg(np.array(img))
        gt = np.asarray(gt)
        meter.update(pred, gt)
        rows.append({"name": name, **per_image_scores(pred, gt)})
        preds.append(pred)
    return meter.summary(), rows, preds


def load_captured():
    names = sorted(p.stem for p in (CAPTURED / "images_256").glob("*.png"))
    images = [cv2.imread(str(CAPTURED / "images_256" / f"{n}.png"), cv2.IMREAD_COLOR) for n in names]
    masks = [(cv2.imread(str(CAPTURED / "masks_256" / f"{n}.png"), cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
             for n in names]
    return images, masks, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/model.pt")
    parser.add_argument("--sets", nargs="+", default=["val", "te", "captured"])
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    seg = Segmenter(args.weights)

    out_path = RESULTS / "metrics.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for which in args.sets:
        if which == "captured":
            images, masks, names = load_captured()
            if not names:
                print("no captured photos found, skipping")
                continue
        else:
            prefix = "tr" if which == "val" else "te"
            images = np.load(CACHE / f"{prefix}_images.npy", mmap_mode="r")
            masks = np.load(CACHE / f"{prefix}_masks.npy", mmap_mode="r")
            names = (CACHE / f"{prefix}_names.txt").read_text().split()
            if which == "val":
                idx = np.load(CACHE / "split.npz")["val"]
                images, masks, names = images[idx], masks[idx], [names[i] for i in idx]
        summary, rows, preds = run_set(seg, images, masks, names)
        results[which] = summary
        (RESULTS / f"per_image_{which}.json").write_text(json.dumps(rows, indent=1))
        if which == "captured":
            (RESULTS / "captured_pred").mkdir(exist_ok=True)
            for name, pred in zip(names, preds):
                cv2.imwrite(str(RESULTS / "captured_pred" / f"{name}.png"), pred * 255)
        print(f"{which:9s} n={summary['n']:5d} mIoU {summary['mIoU']:.4f} BF1 {summary['BF1']:.4f} Q {summary['Q']:.4f}")

    timing_path = RESULTS / "timing.json"
    if timing_path.exists() and "te" in results:
        timing = json.loads(timing_path.read_text())
        results["timing"] = timing
        results["CES"] = ces(results["te"]["Q"], timing["T_CPU_ms"], os.path.getsize(args.weights))
        print(f"T_CPU {timing['T_CPU_ms']:.2f} ms, S_disk {os.path.getsize(args.weights):,} B, CES {results['CES']:.4f}")
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
