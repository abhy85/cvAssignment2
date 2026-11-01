"""Checks the vectorised metrics against a direct pixel-by-pixel reading of the definitions.

Run: python -m tests.test_metrics
"""
import numpy as np
import torch

from src.metrics import SegMeter, boundary_f1


def slow_boundary(m):
    h, w = m.shape
    b = np.zeros_like(m, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not m[y, x]:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < h and 0 <= xx < w) or not m[yy, xx]:
                    b[y, x] = True
                    break
    return b


def any_in_window(b, y, x):
    return b[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].any()


def slow_bf1(pred, gt):
    pb, gb = slow_boundary(pred), slow_boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    p = np.mean([any_in_window(gb, y, x) for y, x in zip(*np.nonzero(pb))])
    r = np.mean([any_in_window(pb, y, x) for y, x in zip(*np.nonzero(gb))])
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def random_mask(rng, h, w):
    """Blobby random masks: a few random ellipses plus a sprinkle of noise."""
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), bool)
    for _ in range(rng.integers(0, 4)):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(2, h / 2), rng.uniform(2, w / 2)
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    m ^= rng.random((h, w)) < 0.01
    return m


def main():
    rng = np.random.default_rng(0)
    preds = [random_mask(rng, 48, 40) for _ in range(30)]
    gts = [random_mask(rng, 48, 40) for _ in range(30)]
    # edge cases: both empty, one empty, identical, full masks
    preds += [np.zeros((48, 40), bool), np.zeros((48, 40), bool), gts[0].copy(), np.ones((48, 40), bool)]
    gts += [np.zeros((48, 40), bool), gts[1].copy(), gts[0].copy(), np.ones((48, 40), bool)]

    fast = boundary_f1(torch.from_numpy(np.stack(preds)), torch.from_numpy(np.stack(gts))).numpy()
    slow = np.array([slow_bf1(p, g) for p, g in zip(preds, gts)])
    assert np.allclose(fast, slow), np.abs(fast - slow).max()

    meter = SegMeter()
    for p, g in zip(preds, gts):
        meter.update(p, g)
    P, G = np.stack(preds), np.stack(gts)
    iou_o = (P & G).sum() / (P | G).sum()
    iou_b = (~P & ~G).sum() / (~P | ~G).sum()
    assert np.isclose(meter.miou(), (iou_o + iou_b) / 2)
    assert np.isclose(meter.summary()["BF1"], slow.mean())
    print(f"metrics OK on {len(preds)} masks; mean BF1 {slow.mean():.4f}, mIoU {meter.miou():.4f}")


if __name__ == "__main__":
    main()
