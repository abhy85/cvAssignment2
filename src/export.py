"""Turn a training checkpoint into the deployed parameter file.

Folds BatchNorm and the 1/255 input scale into the convs, stores weights as float16 (or
int8 + per-channel scales with --int8) and checks that the deployed pipeline still agrees
with the original float32 model on validation photos.

    python -m src.export --ckpt checkpoints/final.pt --out weights/model.pt [--int8]
"""
import argparse
import os

import numpy as np
import torch

from src.common import CACHE
from src.model import TinyUNet, fuse_for_inference
from src.pipeline import Segmenter
from src.weights_io import pack_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="weights/model.pt")
    parser.add_argument("--int8", action="store_true", help="int8 conv weights with per-channel scales")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # our own training checkpoint
    model = TinyUNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    reference = TinyUNet(**ckpt["config"])
    reference.load_state_dict(ckpt["state"])
    reference.eval()
    fuse_for_inference(model)

    state = pack_state(model.state_dict(), int8=args.int8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"config": ckpt["config"], "state": state}, args.out)

    # sanity check: deployed pipeline vs. unfused float32 model on some validation photos
    images = np.load(CACHE / "tr_images.npy", mmap_mode="r")
    val = np.load(CACHE / "split.npz")["val"][:50]
    seg = Segmenter(args.out)
    agree = []
    res = ckpt["config"]["in_res"]
    with torch.no_grad():
        for i in val:
            img = np.array(images[i])
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
            if res != 256:
                x = torch.nn.functional.interpolate(x, size=(res, res), mode="area")
            ref = (reference(x)[0, 0] > 0).numpy()
            agree.append((seg(img) == ref).mean())
    print(f"saved {args.out}: {os.path.getsize(args.out):,} bytes (epoch {ckpt['epoch']}, val {ckpt['val']})")
    print(f"pixel agreement deployed pipeline vs original: {np.mean(agree):.5f}")


if __name__ == "__main__":
    main()
