"""Figures and tables for report.pdf, generated from the real model and result files.

    python report/make_report_assets.py
Writes report/figures/{architecture,ds_block,pipeline,sweep}.png and report/layers.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from src.model import TinyUNet, count_params, fuse_for_inference  # noqa: E402

FIG = ROOT / "report" / "figures"
FINAL = dict(widths=(48, 64, 128, 192), depths=(2, 3, 4), in_res=128)
ENC, DEC, HEAD, IO, CTX = "#cfe2f3", "#fce5cd", "#d9ead3", "#eeeeee", "#e6d7f0"


def layer_table():
    """One row per block: name, operation, output shape, parameters (with BN, before folding)."""
    model = TinyUNet(**FINAL).eval()
    shapes = {}
    hooks = []
    for name, module in model.named_modules():
        if name.count(".") <= 1 and name:
            hooks.append(module.register_forward_hook(
                lambda m, i, o, n=name: shapes.__setitem__(n, tuple(o.shape[1:]))))
    with torch.no_grad():
        model(torch.zeros(1, 3, 128, 128))
    for h in hooks:
        h.remove()

    def params(module):
        return sum(p.numel() for p in module.parameters())

    def shape(n):
        c, h, w = shapes[n]
        return f"{c} x {h} x {w}"

    rows = [("input", "BGR photo resized to 128 x 128 (cv2.INTER_AREA)", "3 x 128 x 128", 0)]
    rows.append(("stem", "3x3 conv, stride 2, BN, ReLU", shape("stem"), params(model.stem)))
    for stage in ("enc1", "enc2", "enc3"):
        seq = getattr(model, stage)
        for i, block in enumerate(seq):
            stride = "stride 2" if i == 0 else "stride 1"
            res = ", residual" if block.residual else ""
            rows.append((f"{stage}.{i}", f"DS block ({stride}{res})", shape(f"{stage}.{i}"), params(block)))
    rows.append(("context", "global avg pool, 1x1 conv, add to every position", shape("enc3"), params(model.context)))
    rows.append(("dec2", "bilinear x2, concat enc2 (192+128 ch), DS block", shape("dec2"), params(model.dec2)))
    rows.append(("dec1", "bilinear x2, concat enc1 (128+64 ch), DS block", shape("dec1"), params(model.dec1)))
    rows.append(("dec0", "bilinear x2, concat stem (64+48 ch), DS block", shape("dec0"), params(model.dec0)))
    rows.append(("head", "1x1 conv to one object logit", shape("head"), params(model.head)))
    rows.append(("output", "bilinear x4 (cv2.resize), threshold logit > 0", "1 x 256 x 256", 0))
    total = count_params(model)
    fused = count_params(fuse_for_inference(TinyUNet(**FINAL)))
    return {"rows": rows, "total_with_bn": total, "total_fused": fused}


def box(ax, x, y, w, h, color, lines, fontsize=8.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444444", lw=1))
    for k, text in enumerate(lines):
        ax.text(x, y + (len(lines) - 1) * 0.16 - k * 0.32, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if (k == 0 and bold_first) else "normal")


def arrow(ax, p, q, color="#333333", style="-|>", ls="-", lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def architecture_figure():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    ax.set_xlim(-0.9, 15.5)
    ax.set_ylim(-5.0, 2.0)
    ax.axis("off")
    level_y = {-1: 1.1, 0: 0.0, 1: -1.3, 2: -2.6, 3: -3.8}
    h_box, w_box = 0.95, 1.45

    nodes = {
        "input": (0.0, -1, IO, ["Input", "BGR photo", "3 x 128 x 128"]),
        "stem": (1.6, 0, ENC, ["Stem", "3x3 conv s2", "48 x 64 x 64"]),
        "enc1": (3.2, 1, ENC, ["Encoder 1", "2 DS blocks", "64 x 32 x 32"]),
        "enc2": (4.8, 2, ENC, ["Encoder 2", "3 DS blocks", "128 x 16 x 16"]),
        "enc3": (6.6, 3, ENC, ["Encoder 3", "4 DS blocks", "192 x 8 x 8"]),
        "dec2": (8.4, 2, DEC, ["Decoder 2", "1 DS block", "128 x 16 x 16"]),
        "dec1": (10.0, 1, DEC, ["Decoder 1", "1 DS block", "64 x 32 x 32"]),
        "dec0": (11.6, 0, DEC, ["Decoder 0", "1 DS block", "48 x 64 x 64"]),
        "head": (13.1, 0, HEAD, ["Head", "1x1 conv", "1 x 64 x 64"]),
        "out": (14.6, 0, IO, ["Mask", "bilinear x4,", "logit > 0", "256 x 256"]),
    }
    for key, (x, lvl, color, lines) in nodes.items():
        box(ax, x, level_y[lvl], w_box if key != "out" else 1.3, h_box + 0.25 * (len(lines) - 3), color, lines)

    def edge(a, b, label, side):
        (xa, la, *_), (xb, lb, *_) = nodes[a], nodes[b]
        p, q = (xa + w_box / 2, level_y[la]), (xb - w_box / 2, level_y[lb])
        arrow(ax, p, q)
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + (0.12 if side == "right" else -0.12), my, label, fontsize=7.5, color="#555555",
                ha="left" if side == "right" else "right", va="center")

    edge("input", "stem", "stride 2", "right")
    edge("stem", "enc1", "stride 2", "right")
    edge("enc1", "enc2", "stride 2", "right")
    edge("enc2", "enc3", "stride 2", "right")
    edge("enc3", "dec2", "bilinear x2", "left")
    edge("dec2", "dec1", "bilinear x2", "left")
    edge("dec1", "dec0", "bilinear x2", "left")
    for a, b in (("dec0", "head"), ("head", "out")):
        arrow(ax, (nodes[a][0] + w_box / 2, 0.0), (nodes[b][0] - (w_box if b != "out" else 1.3) / 2, 0.0))
    # skip connections (concatenation)
    for a, b, lvl in (("stem", "dec0", 0), ("enc1", "dec1", 1), ("enc2", "dec2", 2)):
        xa, xb = nodes[a][0] + w_box / 2, nodes[b][0] - w_box / 2
        y = level_y[lvl] + 0.2
        arrow(ax, (xa, y), (xb, y), color="#3c78d8", ls="--", lw=1.2)
        ax.text((xa + xb) / 2, y + 0.1, "skip: concatenate", fontsize=7.5, color="#3c78d8", ha="center")
    # global context
    cx, cy = nodes["enc3"][0], level_y[3] - 0.95
    box(ax, cx, cy, 2.9, 0.38, CTX, ["global avg pool -> 1x1 conv -> add"], fontsize=7.5, bold_first=False)
    arrow(ax, (cx - 0.45, level_y[3] - h_box / 2), (cx - 0.45, cy + 0.19), color="#674ea7")
    arrow(ax, (cx + 0.45, cy + 0.19), (cx + 0.45, level_y[3] - h_box / 2), color="#674ea7")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#444444") for c in (ENC, DEC, HEAD, CTX)]
    ax.legend(handles, ["encoder (downsampling)", "decoder (upsampling)", "prediction head", "global context"],
              loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=200)
    plt.close(fig)


def ds_block_figure():
    fig, ax = plt.subplots(figsize=(4.8, 5.4))
    ax.set_xlim(-1.2, 3.6)
    ax.set_ylim(-0.4, 7.4)
    ax.axis("off")
    steps = [("input  C_in x H x W", IO), ("depthwise 3x3 conv\n(one filter per channel, stride s)", ENC),
             ("BatchNorm + ReLU", "#ffffff"), ("pointwise 1x1 conv\n(C_in -> C_out)", ENC),
             ("BatchNorm", "#ffffff"), ("+", "#ffffff"), ("ReLU", "#ffffff"),
             ("output  C_out x H/s x W/s", IO)]
    ys = [7.0, 5.95, 5.0, 4.05, 3.1, 2.25, 1.4, 0.4]
    for (text, color), y in zip(steps, ys):
        if text == "+":
            ax.add_patch(plt.Circle((0.5, y), 0.22, fc="#ffffff", ec="#444444"))
            ax.text(0.5, y, "+", ha="center", va="center", fontsize=13)
            continue
        h = 0.68 if "\n" in text else 0.48
        ax.add_patch(FancyBboxPatch((-0.9, y - h / 2), 2.8, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, ec="#444444"))
        ax.text(0.5, y, text, ha="center", va="center", fontsize=8.5)
    for a, b in zip(ys, ys[1:]):
        top = a - (0.34 if a in (5.95, 4.05) else 0.24 if a != 2.25 else 0.22)
        bot = b + (0.34 if b in (5.95, 4.05) else 0.24 if b != 2.25 else 0.22)
        arrow(ax, (0.5, top), (0.5, bot))
    # residual path, routed outside the boxes on the right
    ax.plot([1.9, 2.35, 2.35], [7.0, 7.0, 2.25], color="#3c78d8", lw=1.2, ls="--")
    arrow(ax, (2.35, 2.25), (0.72, 2.25), color="#3c78d8", ls="--", lw=1.2)
    ax.text(2.45, 4.6, "residual\n(only if\ns = 1 and\nC_in = C_out)", fontsize=7.5, color="#3c78d8", va="center")
    fig.tight_layout()
    fig.savefig(FIG / "ds_block.png", dpi=200)
    plt.close(fig)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(13, 2.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.1)
    ax.axis("off")
    steps = [("256x256 photo\n(BGR uint8)", IO), ("cv2.resize\nINTER_AREA\n-> 128x128", "#ffffff"),
             ("uint8 -> float\ntensor\n(channels-last)", "#ffffff"), ("TinyUNet\nTorchScript, BN folded,\n1/255 folded", ENC),
             ("logit\n1 x 64 x 64", HEAD), ("cv2.resize\nbilinear\n-> 256x256", "#ffffff"),
             ("logit > 0\n(NumPy)", "#ffffff"), ("mask 256x256\n{0, 1}", IO)]
    xs = [0.8 + i * 1.63 for i in range(len(steps))]
    for (text, color), x in zip(steps, xs):
        ax.add_patch(FancyBboxPatch((x - 0.72, 0.45), 1.44, 1.15, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec="#444444"))
        ax.text(x, 1.025, text, ha="center", va="center", fontsize=8)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 0.72, 1.025), (b - 0.72, 1.025))
    ax.annotate("", xy=(xs[-1] + 0.72, 0.25), xytext=(xs[0] - 0.72, 0.25),
                arrowprops=dict(arrowstyle="<->", color="#cc0000"))
    ax.text(6.5, 0.05, "timed with time.process_time() (single thread): 4.79 ms per photo on DUTS-TE",
            ha="center", fontsize=8.5, color="#cc0000")
    fig.tight_layout()
    fig.savefig(FIG / "pipeline.png", dpi=200)
    plt.close(fig)


def sweep_figure():
    runs = [("r128_w16", 1.84, 0.7571, 30), ("r256_w16", 4.00, 0.7612, 30), ("r192_w24", 3.84, 0.7904, 65),
            ("r128_w32", 3.06, 0.7974, 150), ("r128_w32 deep", 3.53, 0.8031, 260),
            ("r128_w32 + bw", 3.00, 0.8045, 150), ("r128_w48", 4.09, 0.8126, 299)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    offsets = {"r128_w32 + bw": (-12, 8), "r128_w32 deep": (10, -10)}
    for name, t, q, p in runs:
        ax.scatter(t, q, s=p * 1.2 + 20, color="#3c78d8", alpha=0.75, edgecolor="#1c4587")
        dx, dy = offsets.get(name, (10, -3))
        ax.annotate(name, (t, q), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    ha="right" if dx < 0 else "left")
    ax.scatter([4.79], [0.8199], s=299 * 1.2 + 20, color="#e69138", edgecolor="#783f04", zorder=3)
    ax.annotate("final (w48 + boundary loss,\n60 epochs, int8)", (4.79, 0.8199), textcoords="offset points",
                xytext=(-16, -4), fontsize=8, color="#783f04", ha="right")
    ax.set_xlabel("CPU time per photo (ms, single thread)")
    ax.set_ylabel("validation Q")
    ax.set_title("Architecture sweep (marker area ~ parameters)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1.2, 5.6)
    fig.tight_layout()
    fig.savefig(FIG / "sweep.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "layers.json").write_text(json.dumps(layer_table(), indent=1))
    architecture_figure()
    ds_block_figure()
    pipeline_figure()
    sweep_figure()
    print("assets written")


if __name__ == "__main__":
    main()
