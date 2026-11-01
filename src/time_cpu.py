"""Official T_CPU measurement (Section 2b) on all 5,019 DUTS-TE photos, single thread.

Each photo is already a 256x256 array in memory before its timer starts; the timer stops
when the 256x256 {0,1} mask is in memory. T_CPU = total process time / number of photos.

    python -m src.time_cpu --weights weights/model.pt
"""
import src.threads  # noqa: F401  (sets single-thread env vars before NumPy/OpenCV/PyTorch)

import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from src.common import CACHE, RESULTS
from src.pipeline import Segmenter


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/model.pt")
    parser.add_argument("--limit", type=int, default=0, help="debug: time only the first N photos")
    args = parser.parse_args()

    images = np.load(CACHE / "te_images.npy", mmap_mode="r")
    n = args.limit or len(images)
    seg = Segmenter(args.weights)
    seg(np.array(images[0]))  # warm-up run, not timed

    total = 0.0
    for i in range(n):
        img = np.array(images[i])  # file I/O, not timed
        t0 = time.process_time()
        mask = seg(img)
        total += time.process_time() - t0
        assert mask.shape == (256, 256)
    t_ms = 1000 * total / n
    s_bytes = os.path.getsize(args.weights)
    info = {"T_CPU_ms": t_ms, "S_disk_bytes": s_bytes, "n": n, "cpu": cpu_name(),
            "torch_threads": torch.get_num_threads(), "torch": torch.__version__}
    print(json.dumps(info, indent=1))
    if not args.limit:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "timing.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
