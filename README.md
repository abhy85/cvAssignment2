# CV Assignment 2: Cost-Efficient Segmentation (Track B, tiny FCN)

A tiny U-Net trained from scratch on DUTS that labels every pixel of a 256x256 photo as
main object or background on one CPU thread. The method, results and discussion are in
[report/report.md](report/report.md).

## Setup
```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install opencv-python numpy matplotlib
```
Put DUTS in `data/DUTS-TR/{DUTS-TR-Image,DUTS-TR-Mask}` and `data/DUTS-TE/{DUTS-TE-Image,DUTS-TE-Mask}`.

## Predict one photo
```
.venv\Scripts\python predict.py --image photo.jpg --out mask.png
```
Prints `T CPU ms: ...` (average over 20 timed runs after the first) and `S disk bytes: ...`.

## Full pipeline (run from the project root)
| step | command |
|---|---|
| cache 256x256 data, train/val split | `python -m src.prepare_data` |
| test the metrics vs. a brute-force reference | `python -m tests.test_metrics` |
| CPU speed of candidate architectures | `python -m src.bench_arch` |
| train (GPU) | `python -m src.train --name final --in-res 128 --widths 48 64 128 192 --depths 2 3 4 --epochs 60 --boundary-weight 2` |
| compare runs by validation CES | `python -m src.compare_runs "checkpoints/*.pt"` |
| export fused int8 parameters | `python -m src.export --ckpt checkpoints/final.pt --out weights/model.pt --int8` |
| official T_CPU on all DUTS-TE | `python -m src.time_cpu` |
| metrics on val, DUTS-TE, captured photos | `python -m src.evaluate` |
| captured photos: resize, annotate | `python -m src.prepare_captured`, `python -m src.annotate` |
| report figures | `python -m src.make_figures --run final` |
| PDF report (needs `pip install reportlab`) | `python report/make_report_assets.py`, then `python report/build_report.py` |

## Layout
```
predict.py              required CLI
weights/model.pt        deployed parameters (BN-fused, int8 conv weights + scales, plain torch.save)
src/weights_io.py       float16 / int8 packing of the parameters
src/common.py           paths and constants          src/threads.py   single-thread setup
src/prepare_data.py     resize + cache + split
src/model.py            TinyUNet, BN folding         src/pipeline.py  timed CPU pipeline
src/metrics.py          mIoU, Boundary F1, CES       src/train.py     training (GPU augmentation)
src/export.py           checkpoint -> weights        src/time_cpu.py  official T_CPU
src/evaluate.py         final metrics table          src/make_figures.py  report figures
src/bench_arch.py       architecture timing          src/compare_runs.py  model selection
src/prepare_captured.py, src/annotate.py             captured photo tools
tests/test_metrics.py   metric correctness check
captured/{original,images_256,masks_256}/            own photos
```

## Rules followed
* No pretrained weights or library segmentation, clustering, morphology or metric code.
  Boundaries use shifted-tensor comparisons; the 5x5 window check uses `F.max_pool2d`.
* DUTS-TE is used only for the final evaluation and timing. All choices use the validation split.
* The timed pipeline runs on the CPU with one thread: the env vars are set before NumPy, OpenCV and PyTorch are imported,
  plus `cv2.setNumThreads(1)` and `torch.set_num_threads(1)`.
