"""Shared paths and constants.

Import `src.threads` (not this module) first in any script that measures CPU time,
because the thread environment variables must be set before NumPy/OpenCV/PyTorch load.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
CAPTURED = ROOT / "captured"

SIZE = 256            # every photo and mask is resized to SIZE x SIZE (Section 1b)
VAL_COUNT = 1053      # images held out from DUTS-TR for validation (about 10%)
SPLIT_SEED = 2024
