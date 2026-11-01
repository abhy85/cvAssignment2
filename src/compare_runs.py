"""Rank training runs by a validation-only CES estimate, used to choose the final model.

For each checkpoint: export it (fused, float16), time the CPU pipeline on validation photos,
and combine with the best validation Q recorded during training. DUTS-TE is not used.

    python -m src.compare_runs checkpoints/sweep_*.pt
"""
import src.threads  # noqa: F401

import glob
import os
import sys
import tempfile
import time

import numpy as np
import torch

from src.common import CACHE
from src.metrics import ces
from src.model import TinyUNet, count_params, fuse_for_inference
from src.pipeline import Segmenter


def export_tmp(ckpt):
    model = TinyUNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    fuse_for_inference(model)
    path = os.path.join(tempfile.gettempdir(), "compare_runs.pt")
    torch.save({"config": ckpt["config"], "state": {k: v.half() for k, v in model.state_dict().items()}}, path)
    return path, count_params(model)


def main():
    paths = sorted(p for arg in sys.argv[1:] for p in glob.glob(arg))
    images = np.load(CACHE / "tr_images.npy", mmap_mode="r")
    val = np.load(CACHE / "split.npz")["val"][:500]
    photos = [np.array(images[i]) for i in val]
    rows = []
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own training checkpoint
        tmp, params = export_tmp(ckpt)
        seg = Segmenter(tmp)
        seg(photos[0])
        total = 0.0
        for img in photos:
            t0 = time.process_time()
            seg(img)
            total += time.process_time() - t0
        t_ms, size = 1000 * total / len(photos), os.path.getsize(tmp)
        q = ckpt["val"]["Q"]
        rows.append((ces(q, t_ms, size), os.path.basename(path), params, size, t_ms, ckpt["val"], ckpt["epoch"]))
    print(f"{'run':24s} {'params':>8s} {'bytes':>8s} {'T ms':>6s} {'mIoU':>6s} {'BF1':>6s} {'Q':>6s} {'valCES':>7s} ep")
    for c, name, params, size, t_ms, v, ep in sorted(rows, reverse=True):
        print(f"{name:24s} {params:8,d} {size:8,d} {t_ms:6.2f} {v['mIoU']:.4f} {v['BF1']:.4f} {v['Q']:.4f} {c:7.4f} {ep}")


if __name__ == "__main__":
    main()
