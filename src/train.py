"""Train TinyUNet from scratch on the DUTS-TR training split (GPU), selecting the epoch by
validation Q. DUTS-TE is never touched here.

Data stays in memory-mapped uint8 arrays; a background thread reads random batches and all
augmentation (affine warp, flip, colour/lighting jitter, noise, blur) runs on the GPU.

Example:
    python -m src.train --name r128_w1 --in-res 128 --epochs 80
"""
import argparse
import json
import math
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.common import CACHE, RESULTS, ROOT
from src.metrics import SegMeter, boundary
from src.model import TinyUNet, count_params

CKPT_DIR = ROOT / "checkpoints"


def load_split():
    images = np.load(CACHE / "tr_images.npy", mmap_mode="r")
    masks = np.load(CACHE / "tr_masks.npy", mmap_mode="r")
    split = np.load(CACHE / "split.npz")
    return images, masks, split["train"], split["val"]


class BatchReader:
    """Reads shuffled uint8 batches from the memmaps in a background thread."""

    def __init__(self, images, masks, indices, batch_size, seed):
        self.images, self.masks, self.indices = images, masks, indices
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.steps = len(indices) // batch_size

    def epoch(self):
        order = self.rng.permutation(self.indices)
        q = queue.Queue(maxsize=6)

        def work():
            for s in range(self.steps):
                idx = np.sort(order[s * self.batch_size:(s + 1) * self.batch_size])
                q.put((torch.from_numpy(self.images[idx]).pin_memory(),
                       torch.from_numpy(self.masks[idx]).pin_memory()))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            yield item


def to_float(images_u8, masks_u8, device):
    x = images_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    m = masks_u8.to(device, non_blocking=True).unsqueeze(1).float()
    return x, m


def augment(x, m, strength=1.0):
    """x: (B,3,H,W) in [0,1], m: (B,1,H,W) in {0,1}. Same geometry for both."""
    b, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(b, device=dev).uniform_(lo, hi)  # noqa: E731

    # geometry: zoom, aspect jitter, rotation, translation, horizontal flip
    zoom = torch.exp(u(math.log(0.8), math.log(1.45)) * strength)
    aspect = torch.exp(u(-0.15, 0.15) * strength)
    angle = u(-12, 12) * strength * math.pi / 180
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    sx, sy = flip / (zoom * aspect), 1.0 / (zoom / aspect)
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([
        torch.stack([sx * cos, -sy * sin, u(-0.12, 0.12) * strength], 1),
        torch.stack([sx * sin, sy * cos, u(-0.12, 0.12) * strength], 1),
    ], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = (F.grid_sample(m, grid, mode="bilinear", padding_mode="reflection", align_corners=False) > 0.5).float()

    # photometric: brightness, contrast, saturation, colour cast (daylight / tube light), gamma
    v = lambda lo, hi: u(lo, hi).view(b, 1, 1, 1)  # noqa: E731
    x = x * v(0.65, 1.35)
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * v(0.65, 1.35) + mean
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * v(0.5, 1.5) + grey
    x = x * torch.empty(b, 3, 1, 1, device=dev).uniform_(0.85, 1.15)
    x = x.clamp(0, 1) ** v(0.7, 1.4)

    # occasional blur (defocus / motion) and sensor noise
    blur = (torch.rand(b, 1, 1, 1, device=dev) < 0.2).float()
    x = blur * F.avg_pool2d(x, 3, 1, 1, count_include_pad=False) + (1 - blur) * x
    noisy = (torch.rand(b, 1, 1, 1, device=dev) < 0.3).float()
    x = x + noisy * torch.randn_like(x) * v(0.0, 0.04)
    return x.clamp(0, 1), m


def downscale(x, res):
    return x if x.shape[-1] == res else F.interpolate(x, size=(res, res), mode="area")


def loss_fn(logit, m, boundary_weight=0.0):
    """BCE (optionally up-weighted within 2 px of the true outline) + soft Dice."""
    weight = None
    if boundary_weight > 0:
        near = F.max_pool2d(boundary(m[:, 0] > 0.5).float().unsqueeze(1), 5, stride=1, padding=2)
        weight = 1 + boundary_weight * near
    bce = F.binary_cross_entropy_with_logits(logit, m, weight=weight)
    p = torch.sigmoid(logit)
    inter = (p * m).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, images, masks, indices, res, device, batch_size=64):
    model.eval()
    meter = SegMeter()
    for s in range(0, len(indices), batch_size):
        idx = indices[s:s + batch_size]
        x, m = to_float(torch.from_numpy(images[idx]), torch.from_numpy(masks[idx]), device)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(downscale(x, res))
        meter.update(logit[:, 0] > 0, m[:, 0] > 0.5)
    return meter.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--in-res", type=int, default=128)
    parser.add_argument("--widths", type=int, nargs=4, default=[16, 24, 40, 64])
    parser.add_argument("--depths", type=int, nargs=3, default=[2, 2, 3])
    parser.add_argument("--refine", type=int, default=0, help="channels of the full resolution stage (0 = off)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0, help="debug: limit training images")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    images, masks, train_idx, val_idx = load_split()
    if args.max_train:
        train_idx = train_idx[:args.max_train]

    model = TinyUNet(args.widths, args.depths, in_res=args.in_res, refine=args.refine).to(device)
    print(f"{args.name}: {count_params(model):,} parameters, train {len(train_idx)}, val {len(val_idx)}")
    reader = BatchReader(images, masks, train_idx, args.batch_size, args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * reader.steps
    warmup = reader.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))
    scaler = torch.amp.GradScaler()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    history, best_q = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for images_u8, masks_u8 in reader.epoch():
            x, m = to_float(images_u8, masks_u8, device)
            x, m = augment(x, m)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(downscale(x, args.in_res))
            loss = loss_fn(logit.float(), m, args.boundary_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        val = evaluate(model, images, masks, val_idx, args.in_res, device)
        row = {"epoch": epoch, "loss": running / reader.steps, **val, "sec": time.time() - t0}
        history.append(row)
        flag = ""
        if val["Q"] > best_q:
            best_q = val["Q"]
            torch.save({"config": model.config, "state": model.state_dict(), "epoch": epoch, "val": val},
                       CKPT_DIR / f"{args.name}.pt")
            flag = " *"
        print(f"ep {epoch:3d} loss {row['loss']:.4f} val mIoU {val['mIoU']:.4f} BF1 {val['BF1']:.4f} "
              f"Q {val['Q']:.4f} ({row['sec']:.0f}s){flag}", flush=True)

    (RESULTS / f"train_{args.name}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"best val Q {best_q:.4f}")


if __name__ == "__main__":
    main()
