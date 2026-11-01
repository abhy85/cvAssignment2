"""The timed CPU pipeline: 256x256 BGR uint8 photo in memory -> 256x256 {0,1} mask in memory.

Parameter file format (weights/model.pt, plain torch.save, no compression):
    {"config": TinyUNet kwargs, "state": BN-fused state_dict packed by src.weights_io
     (float16, or int8 conv weights with per-channel scales)}
Weights are cast back to float32 once at load time (loading is not timed).

Speed choices (measured with src/bench_arch.py, single thread):
  * BN and the 1/255 input scale are folded into the convs, so the raw uint8 photo goes
    straight into the network; BGR order is kept, so there is no colour conversion.
  * channels_last memory format, which oneDNN runs much faster for these small convs.
  * TorchScript trace + freeze to remove Python overhead between layers.
  * the one-channel logit is upsampled with cv2.resize (bilinear, half-pixel centres like
    F.interpolate(align_corners=False)) and thresholded at 0 in NumPy.
"""
import warnings

import cv2
import numpy as np
import torch

from src.model import build_fused
from src.weights_io import unpack_state

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")


class Segmenter:
    def __init__(self, weights_path, jit="freeze", channels_last=True):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        model = build_fused(ckpt["config"])
        model.load_state_dict(unpack_state(ckpt["state"]))
        self.res = ckpt["config"]["in_res"]
        self.out_size = ckpt["config"]["out_size"]
        self.memory_format = torch.channels_last if channels_last else torch.contiguous_format
        model = model.to(memory_format=self.memory_format)
        if jit:
            with torch.inference_mode():
                example = torch.zeros(1, 3, self.res, self.res).contiguous(memory_format=self.memory_format)
                traced = torch.jit.trace(model, example)
                model = (torch.jit.optimize_for_inference(traced) if jit == "optimize"
                         else torch.jit.freeze(traced))
                # TorchScript's profiling executor re-optimises the graph during the first
                # calls; do that here, at load time, instead of during the first photos
                for _ in range(3):
                    model(example)
        self.model = model

    @torch.inference_mode()
    def __call__(self, img):
        """img: (256,256,3) uint8 BGR. Returns (256,256) uint8 mask of 0s and 1s."""
        if self.res != img.shape[0]:
            img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        x = x.contiguous(memory_format=self.memory_format)
        logit = self.model(x)[0, 0].numpy()
        logit = cv2.resize(logit, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        return (logit > 0).view(np.uint8)
