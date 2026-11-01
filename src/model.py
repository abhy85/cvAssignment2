"""Tiny fully convolutional U-Net for salient object segmentation (Track B).

Encoder: a strided 3x3 stem followed by stages of depthwise separable blocks, each stage
         halving the resolution. A global average pooled context vector is added to the
         deepest features so every pixel sees image-level information cheaply.
Decoder: bilinear x2 upsampling, concatenation with the encoder skip of the same size
         (U-Net style) and a depthwise separable block.
Refine:  optional (refine > 0): one more decoder step at the full input resolution whose
         skip is a single 3x3 conv on the photo, for sharper outlines.
Head:    1x1 conv to one object logit at half (or, with refine, full) input resolution, bilinearly
         upsampled to 256x256 (by F.interpolate in training, by cv2.resize with the same
         half-pixel convention in the CPU pipeline, which is much faster for one channel). A single logit with binary cross entropy is the two-class
         pixel wise cross entropy written with one output channel.

For deployment `fuse_for_inference` folds every BatchNorm into the preceding conv and the
1/255 input scaling into the stem, so the CPU pipeline runs plain convs on raw uint8 values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, groups=1, relu=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )


class DSBlock(nn.Module):
    """Depthwise 3x3 (optionally strided) + pointwise 1x1, residual when shapes match."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(cin, cin, 3, stride, groups=cin)
        self.pw = ConvBNReLU(cin, cout, 1, relu=False)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return F.relu(y + x if self.residual else y)


def make_stage(cin, cout, depth, stride):
    return nn.Sequential(DSBlock(cin, cout, stride), *[DSBlock(cout, cout) for _ in range(depth - 1)])


class TinyUNet(nn.Module):
    def __init__(self, widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=128, out_size=256, refine=0):
        super().__init__()
        self.in_res, self.out_size = in_res, out_size
        self.config = {"widths": list(widths), "depths": list(depths), "in_res": in_res, "out_size": out_size,
                       "refine": refine}
        c0, c1, c2, c3 = widths
        self.stem = ConvBNReLU(3, c0, 3, stride=2)              # 1/2
        self.enc1 = make_stage(c0, c1, depths[0], 2)            # 1/4
        self.enc2 = make_stage(c1, c2, depths[1], 2)            # 1/8
        self.enc3 = make_stage(c2, c3, depths[2], 2)            # 1/16
        self.context = nn.Conv2d(c3, c3, 1)                     # global context, added to enc3
        self.dec2 = DSBlock(c3 + c2, c2)
        self.dec1 = DSBlock(c2 + c1, c1)
        self.dec0 = DSBlock(c1 + c0, c0)
        # optional full input resolution stage: a light conv on the photo itself is the skip
        self.refine = refine
        if refine:
            self.ref_in = ConvBNReLU(3, refine, 3)
            self.ref_dec = DSBlock(c0 + refine, refine)
        self.head = nn.Conv2d(refine or c0, 1, 1)
        # the CPU pipeline upsamples the low resolution logit with cv2.resize instead
        self.final_upsample = True

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

    def forward(self, x):
        """x: (B,3,in_res,in_res) BGR in [0,1] (training) or [0,255] (after fusing)."""
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s3 = s3 + self.context(s3.mean((2, 3), keepdim=True))
        d = self.dec2(self._up_cat(s3, s2))
        d = self.dec1(self._up_cat(d, s1))
        d = self.dec0(self._up_cat(d, s0))
        if self.refine:
            d = self.ref_dec(self._up_cat(d, self.ref_in(x)))
        logit = self.head(d)
        if not self.final_upsample:
            return logit
        return F.interpolate(logit, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


def _fold_bn(conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, groups=conv.groups, bias=True)
    fused.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    fused.bias.data = bn.bias.data - bn.running_mean * scale
    return fused


@torch.no_grad()
def fuse_for_inference(model):
    """In place: fold BN into convs and the 1/255 input scale into the stem conv."""
    model.eval()
    for module in model.modules():
        if isinstance(module, ConvBNReLU):
            module[0] = _fold_bn(module[0], module[1])
            module[1] = nn.Identity()
    model.stem[0].weight.data /= 255.0
    return model


def build_fused(config):
    """Architecture with BN already replaced by biased convs, ready to load fused weights."""
    model = TinyUNet(**config)
    for module in model.modules():
        if isinstance(module, ConvBNReLU):
            conv = module[0]
            module[0] = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                                  conv.padding, groups=conv.groups, bias=True)
            module[1] = nn.Identity()
    model.final_upsample = False
    return model.eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
