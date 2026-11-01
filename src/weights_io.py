"""Compact storage of the fused parameters.

float16: every tensor stored as float16.
int8:    conv weights stored as int8 with one float32 scale per output channel (symmetric,
         round to nearest); biases stay float16. Values are dequantized to float32 once when
         the file is loaded, so inference itself is plain float32.
"""
import torch

SCALE_SUFFIX = ":scale"


def pack_state(state, int8=False):
    packed = {}
    for key, value in state.items():
        value = value.detach().float()
        if int8 and value.dim() == 4:
            scale = (value.abs().amax(dim=(1, 2, 3)) / 127).clamp(min=1e-12)
            packed[key] = torch.round(value / scale.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
            packed[key + SCALE_SUFFIX] = scale
        else:
            packed[key] = value.half()
    return packed


def unpack_state(packed):
    state = {}
    for key, value in packed.items():
        if key.endswith(SCALE_SUFFIX):
            continue
        if value.dtype == torch.int8:
            state[key] = value.float() * packed[key + SCALE_SUFFIX].float().view(-1, 1, 1, 1)
        else:
            state[key] = value.float()
    return state
