"""Quick single-thread CPU timing of candidate architectures with random weights, used to pick
which configurations are worth training. Not the official T_CPU (see src/time_cpu.py).

    python -m src.bench_arch                 # all candidates, all execution modes
    python -m src.bench_arch r128_w48 ...    # selected candidates, deployed mode only
"""
import src.threads  # noqa: F401

import os
import sys
import tempfile
import time

import numpy as np
import torch

from src.model import TinyUNet, count_params, fuse_for_inference
from src.pipeline import Segmenter

CANDIDATES = {
    "r128_w16": dict(widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=128),
    "r128_w24": dict(widths=(24, 32, 64, 96), depths=(2, 2, 3), in_res=128),
    "r160_w16": dict(widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=160),
    "r192_w16": dict(widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=192),
    "r256_w16": dict(widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=256),
    "r192_w24": dict(widths=(24, 32, 64, 96), depths=(2, 2, 3), in_res=192),
    "r128_w32": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=128),
    # second round: wider / deeper at 128
    "r128_w32b": dict(widths=(32, 64, 128, 192), depths=(2, 3, 4), in_res=128),
    "r128_w48": dict(widths=(48, 64, 128, 192), depths=(2, 3, 4), in_res=128),
    "r128_w32d": dict(widths=(32, 48, 96, 160), depths=(2, 4, 6), in_res=128),
    "r128_w24d": dict(widths=(24, 48, 96, 160), depths=(2, 3, 6), in_res=128),
    "r160_w32": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=160),
    # third round: full resolution refinement, wider
    "r128_w32_ref16": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=128, refine=16),
    "r128_w48_ref16": dict(widths=(48, 64, 128, 192), depths=(2, 3, 4), in_res=128, refine=16),
    "r128_w64": dict(widths=(64, 96, 192, 256), depths=(2, 3, 4), in_res=128),
}


def time_segmenter(seg, img, runs=60):
    seg(img)
    t0 = time.process_time()
    for _ in range(runs):
        seg(img)
    return 1000 * (time.process_time() - t0) / runs


def main():
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    tmp = os.path.join(tempfile.gettempdir(), "bench_arch.pt")
    only = sys.argv[1:]
    for name, cfg in CANDIDATES.items():
        if only and name not in only:
            continue
        model = fuse_for_inference(TinyUNet(**cfg))
        torch.save({"config": model.config, "state": {k: v.half() for k, v in model.state_dict().items()}}, tmp)
        size = os.path.getsize(tmp)
        modes = {"freeze_cl": ("freeze", True)} if only else {
            "eager_cl": (None, True), "freeze": ("freeze", False), "freeze_cl": ("freeze", True),
            "optimize_cl": ("optimize", True)}
        times = {mode: time_segmenter(Segmenter(tmp, jit=jit, channels_last=cl), img)
                 for mode, (jit, cl) in modes.items()}
        print(f"{name:10s} params {count_params(model):8,d}  file {size / 1024:7.1f} KiB  "
              + "  ".join(f"{k} {v:6.2f} ms" for k, v in times.items()), flush=True)


if __name__ == "__main__":
    main()
