"""Evaluation metrics written from scratch (Section 3, "Evaluation Metrics").

mIoU: one 2x2 confusion matrix summed over all images of a set; IoU of background and
      object averaged.
BF1:  per image boundary F1 with a 5x5 tolerance window, averaged over the set.
      The 5x5 window test is vectorised with F.max_pool2d (a 5x5 window max over the
      boundary map tells whether any boundary pixel lies in that window).
"""
import numpy as np
import torch
import torch.nn.functional as F


def boundary(masks):
    """Object pixels with a background pixel directly above, below, left or right.

    masks: (N,H,W) bool tensor. Pixels outside the image count as background.
    """
    m = F.pad(masks.float().unsqueeze(1), (1, 1, 1, 1), value=0.0)
    centre = m[:, :, 1:-1, 1:-1]
    up, down = m[:, :, :-2, 1:-1], m[:, :, 2:, 1:-1]
    left, right = m[:, :, 1:-1, :-2], m[:, :, 1:-1, 2:]
    all_neighbours_object = up * down * left * right
    return ((centre > 0) & (all_neighbours_object == 0)).squeeze(1)


def boundary_f1(pred, gt):
    """Per image BF1 for (N,H,W) bool tensors. Returns an (N,) float64 tensor."""
    pb, gb = boundary(pred), boundary(gt)
    near_gt = F.max_pool2d(gb.float().unsqueeze(1), 5, stride=1, padding=2).squeeze(1) > 0
    near_pred = F.max_pool2d(pb.float().unsqueeze(1), 5, stride=1, padding=2).squeeze(1) > 0

    n_pred = pb.flatten(1).sum(1).double()
    n_gt = gb.flatten(1).sum(1).double()
    correct = (pb & near_gt).flatten(1).sum(1).double()
    found = (gb & near_pred).flatten(1).sum(1).double()

    precision = correct / n_pred.clamp(min=1)
    recall = found / n_gt.clamp(min=1)
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12), torch.zeros_like(denom))
    f1 = torch.where((n_pred == 0) & (n_gt == 0), torch.ones_like(f1), f1)
    f1 = torch.where((n_pred == 0) ^ (n_gt == 0), torch.zeros_like(f1), f1)
    return f1


class SegMeter:
    """Accumulates the confusion matrix and per image BF1 over a set."""

    def __init__(self):
        self.conf = np.zeros((2, 2), dtype=np.int64)  # conf[true, pred]
        self.bf1 = []

    def update(self, pred, gt):
        """pred, gt: (N,H,W) or (H,W) arrays/tensors with values {0,1} (or bool)."""
        pred, gt = torch.as_tensor(pred).bool(), torch.as_tensor(gt).bool()
        if pred.dim() == 2:
            pred, gt = pred[None], gt[None]
        tp = (pred & gt).sum().item()
        fp = (pred & ~gt).sum().item()
        fn = (~pred & gt).sum().item()
        tn = gt.numel() - tp - fp - fn
        self.conf += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        self.bf1.extend(boundary_f1(pred, gt).cpu().tolist())

    def miou(self):
        (tn, fp), (fn, tp) = self.conf
        iou_obj = tp / max(tp + fp + fn, 1)
        iou_bg = tn / max(tn + fn + fp, 1)
        return 0.5 * (iou_obj + iou_bg)

    def summary(self):
        miou = float(self.miou())
        bf1 = float(np.mean(self.bf1)) if self.bf1 else 0.0
        return {"mIoU": miou, "BF1": bf1, "Q": 0.7 * miou + 0.3 * bf1, "n": len(self.bf1)}


def ces(q, t_cpu_ms, s_disk_bytes):
    if t_cpu_ms > 50:
        return 0.0
    return q * (1 - 0.20 * t_cpu_ms / 50 - 0.10 * min(1.0, s_disk_bytes / 4194304))
