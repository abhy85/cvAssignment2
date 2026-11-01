"""Build report/report.pdf with reportlab (report only; not part of the timed pipeline).

    python report/make_report_assets.py     # diagrams + layer table
    python -m src.make_figures --run final  # overlays, worst cases, training curve
    python report/build_report.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
RES = ROOT / "results"

# ---------------------------------------------------------------- fonts and styles
FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Serif", str(FONT_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("Serif-Bold", str(FONT_DIR / "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("Serif-Italic", str(FONT_DIR / "timesi.ttf")))
pdfmetrics.registerFont(TTFont("Serif-BoldItalic", str(FONT_DIR / "timesbi.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "consola.ttf")))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic",
                              boldItalic="Serif-BoldItalic")

BODY = ParagraphStyle("body", fontName="Serif", fontSize=11, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER)
EQ = ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=12)
CHAP_NUM = ParagraphStyle("chapnum", fontName="Serif-Bold", fontSize=16, leading=20, spaceAfter=10)
CHAP = ParagraphStyle("chap", fontName="Serif-Bold", fontSize=22, leading=27, spaceAfter=22)
SEC = ParagraphStyle("sec", fontName="Serif-Bold", fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=6,
                     keepWithNext=1)
TOC_TITLE = ParagraphStyle("toctitle", fontName="Serif-Bold", fontSize=22, leading=27, spaceAfter=22)
CELL = ParagraphStyle("cell", fontName="Serif", fontSize=9.5, leading=11.5)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="Serif-Bold")
TOC0 = ParagraphStyle("toc0", fontName="Serif-Bold", fontSize=11.5, leading=15, spaceBefore=5, leftIndent=0)
TOC1 = ParagraphStyle("toc1", fontName="Serif", fontSize=10.5, leading=13.5, leftIndent=18)

PAGE_W, PAGE_H = A4
MARGIN = 2.4 * cm
TEXT_W = PAGE_W - 2 * MARGIN


def mono(text):
    return f'<font name="Mono" size="9.5">{text}</font>'


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return ListFlowable([ListItem(p(t), leftIndent=14) for t in items], bulletType="bullet",
                        start="\u2022", leftIndent=14, bulletFontName="Serif", spaceAfter=6)


def numbered(items):
    return ListFlowable([ListItem(p(t), leftIndent=16) for t in items], bulletType="1", leftIndent=16,
                        bulletFontName="Serif", spaceAfter=6)


def figure(path, caption, width=TEXT_W, max_height=None):
    img = Image(str(path))
    scale = width / img.imageWidth
    if max_height and img.imageHeight * scale > max_height:
        scale = max_height / img.imageHeight
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return KeepTogether([Spacer(1, 4), img, Spacer(1, 4), p(caption, CAPTION)])


def table(rows, caption, col_widths=None, header=True, align_right_from=None, bold_rows=(), font_size=9.5):
    """Booktabs-like table: rules above/below the header and at the bottom only."""
    cell = ParagraphStyle("c", parent=CELL, fontSize=font_size, leading=font_size + 2)
    cell_b = ParagraphStyle("cb", parent=CELL_B, fontSize=font_size, leading=font_size + 2)
    data = [[Paragraph(str(c), cell_b if (header and r == 0) or r in bold_rows else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
             ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black))
    if align_right_from is not None:
        for r in range(len(data)):
            for c in range(align_right_from, len(rows[0])):
                data[r][c].style = ParagraphStyle("r", parent=data[r][c].style, alignment=2)
    t.setStyle(TableStyle(style))
    return KeepTogether([p(caption, CAPTION), t, Spacer(1, 12)])


# ---------------------------------------------------------------- document with TOC + running header
class ReportDoc(BaseDocTemplate):
    """Front matter (title, contents) gets roman page numbers; the body is numbered from Chapter 1 = 1."""

    def beforeDocument(self):  # called at the start of every multiBuild pass
        self.chapter, self.chapter_page, self.front, self._chap_no = "", -1, None, ""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style = flowable.style.name
            if style == "chapnum":
                self._chap_no = flowable.getPlainText().replace("Chapter ", "")
            elif style == "chap":
                if self.front is None:
                    self.front = self.page - 1
                self.chapter = flowable.getPlainText()
                self.chapter_page = self.page
                self.notify("TOCEntry", (0, f"{self._chap_no}&nbsp;&nbsp;&nbsp;{self.chapter}",
                                         self.page - self.front))
            elif style == "sec":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page - self.front))


def roman(n):
    return ["", "i", "ii", "iii", "iv", "v"][n]


def on_page_end(canvas, doc):
    canvas.saveState()
    canvas.setFont("Serif", 10)
    if doc.front is None or doc.page <= doc.front:  # front matter
        if doc.page > 1:
            canvas.drawCentredString(PAGE_W / 2, 1.4 * cm, roman(doc.page - 1))
    else:
        canvas.drawCentredString(PAGE_W / 2, 1.4 * cm, str(doc.page - doc.front))
        if doc.page != doc.chapter_page:
            canvas.setFont("Serif", 9)
            canvas.drawString(MARGIN, PAGE_H - 1.5 * cm, f"CHAPTER {doc._chap_no}.  {doc.chapter.upper()}")
            canvas.setLineWidth(0.4)
            canvas.line(MARGIN, PAGE_H - 1.62 * cm, PAGE_W - MARGIN, PAGE_H - 1.62 * cm)
    canvas.restoreState()


def chapter(n, title):
    return [PageBreak(), Spacer(1, 1.2 * cm), p(f"Chapter {n}", CHAP_NUM), p(title, CHAP)]


def sec(title):
    return p(title, SEC)


# ---------------------------------------------------------------- content
def build():
    metrics = json.loads((RES / "metrics.json").read_text())
    layers = json.loads((ROOT / "report" / "layers.json").read_text())
    captured = json.loads((RES / "per_image_captured.json").read_text())
    val, te, cap, timing = metrics["val"], metrics["te"], metrics["captured"], metrics["timing"]
    t_ms, s_bytes, ces_score = timing["T_CPU_ms"], timing["S_disk_bytes"], metrics["CES"]

    story = []
    # title page
    logo = Image(str(FIG / "iitj_logo.jpeg"), width=3.2 * cm, height=3.2 * cm * 494 / 470)
    title_lines = [
        Spacer(1, 1.5 * cm), logo, Spacer(1, 0.8 * cm),
        p("Indian Institute of Technology Jodhpur", ParagraphStyle("t1", parent=CENTER, fontSize=14, leading=20)),
        p("Department of Computer Science and Engineering", ParagraphStyle("t2", parent=CENTER, fontSize=13)),
        Spacer(1, 2.2 * cm),
        p("Assignment 2", ParagraphStyle("t3", parent=CENTER, fontSize=17, leading=24)),
        p("Cost-Efficient Segmentation of the Main Object<br/>with a Tiny Fully Convolutional Network",
          ParagraphStyle("t4", parent=CENTER, fontSize=17, leading=24)),
        Spacer(1, 1.8 * cm),
        p("Abhyudaya Tiwari", ParagraphStyle("t5", parent=CENTER, fontSize=13)),
        p("B23CS1085", ParagraphStyle("t6", parent=CENTER, fontSize=12)),
        Spacer(1, 0.8 * cm),
        p("<i>Supervisor:</i> Deepak Mishra", ParagraphStyle("t7", parent=CENTER, fontSize=11)),
        Spacer(1, 0.8 * cm),
        p("November 1, 2026", ParagraphStyle("t8", parent=CENTER, fontSize=15)),
    ]
    story += title_lines
    story += [PageBreak(), Spacer(1, 1.2 * cm), p("Contents", TOC_TITLE)]
    toc = TableOfContents()
    toc.levelStyles = [TOC0, TOC1]
    toc.dotsMinLevel = 1
    story.append(toc)

    # 1
    story += chapter(1, "Understanding of the Task")
    story += [
        p("The goal is a pipeline that labels every pixel of a photo as <i>main object</i> or <i>background</i> "
          "while using on average at most 50 ms of CPU time on a single core. The score to maximise is the "
          "Cost-Efficient Segmentation (CES) score, which rewards quality and charges a little for time and for "
          "the size of the parameter file:"),
        p("Q = 0.70 &middot; mIoU + 0.30 &middot; BF1", EQ),
        p("CES = Q &middot; [ 1 &minus; 0.20 &middot; T<sub>CPU</sub> / 50 ms &minus; 0.10 &middot; "
          "min(1, S<sub>disk</sub> / 4 MiB) ]&nbsp;&nbsp;&nbsp;if T<sub>CPU</sub> &le; 50 ms, otherwise 0.", EQ),
        p("I followed <b>Track B</b>: a small fully convolutional network (a U-Net) trained from scratch in "
          "PyTorch. The main stages are:"),
        numbered([
            "Resize DUTS photos and masks to 256 &times; 256 and split DUTS-TR into training and validation sets.",
            "Design an encoder-decoder network with skip connections that is small enough for the CPU budget.",
            "Train it on the GPU with pixel-wise cross entropy, soft Dice and a boundary weight.",
            "Choose the architecture and loss on the validation set by an estimate of CES.",
            "Make the CPU pipeline fast: fold BatchNorm, use channels-last TorchScript, store int8 weights.",
            "Measure T<sub>CPU</sub> on all 5,019 DUTS-TE photos, evaluate quality, and analyse failures.",
        ]),
        p("Because time and file size cost only a small fraction of Q in this budget range "
          "(1 ms costs 0.4% of Q and 100 KB costs about 0.24%), the best trade-off is the most accurate "
          "network that still runs in a few milliseconds."),
    ]

    # 2
    story += chapter(2, "Data and Experimental Setup")
    story += [
        sec("2.1 Datasets and Split"),
        p("DUTS masks mark the main object or objects of each photo in white. Every photo is resized to "
          "256 &times; 256 with " + mono("cv2.INTER_AREA") + " and every mask with " + mono("cv2.INTER_NEAREST")
          + "; a mask pixel is object if its value is at least 128. All metrics are computed on these 256 &times; "
          "256 masks. DUTS-TR was split once with a fixed random seed (2024)."),
        table([["Set", "Source", "Photos", "Used for"],
               ["Training", "DUTS-TR", "9,500", "learning the weights"],
               ["Validation", "DUTS-TR (held out)", "1,053", "every design choice and the epoch"],
               ["Test", "DUTS-TE", "5,019", "final metrics and T<sub>CPU</sub> only"],
               ["Captured", "own photos", f"{cap['n']}", "testing only (never trained on)"]],
              "Table 2.1: Data splits.", col_widths=[2.6 * cm, 3.6 * cm, 2.0 * cm, 6.8 * cm]),
        p("The resized data is cached once as memory-mapped uint8 arrays (" + mono("src/prepare_data.py") +
          "). Photos stay in OpenCV's BGR order end to end, so the pipeline needs no colour conversion."),
        sec("2.2 Hardware and Software"),
        table([["Item", "Value"],
               ["CPU (timing)", timing["cpu"] + ", 1 thread"],
               ["GPU (training only)", "NVIDIA GeForce GTX 1650, 4 GB"],
               ["Software", f"Python 3.13, PyTorch {timing['torch']}, OpenCV 5.0, NumPy 2.5"],
               ["Operating system", "Windows 11"]],
              "Table 2.2: Hardware and software.", col_widths=[4.2 * cm, 10.8 * cm]),
        sec("2.3 Rules Followed"),
        bullets([
            "No pretrained weights and no library code for segmentation, clustering, morphology or metrics. "
            "The network, losses and metrics are written from scratch with PyTorch tensors.",
            "DUTS-TE is never used for training, tuning or choosing the epoch.",
            "The timed pipeline runs on the CPU with one thread: the four thread environment variables are set "
            "before NumPy, OpenCV and PyTorch are imported, followed by " + mono("cv2.setNumThreads(1)") +
            " and " + mono("torch.set_num_threads(1)") + ".",
            "All learned values are in the parameter file " + mono("weights/model.pt") +
            " (plain " + mono("torch.save") + ", no extra compression).",
        ]),
    ]

    # 3
    rows = [["Layer", "Operation", "Output (C &times; H &times; W)", "Params"]]
    for name, op, shape, n in layers["rows"]:
        rows.append([name, op, shape.replace(" x ", " &times; "), f"{n:,}" if n else "&ndash;"])
    rows.append(["", "<b>Total</b> (with BatchNorm / after folding BN into convs)", "",
                 f"<b>{layers['total_with_bn']:,} / {layers['total_fused']:,}</b>"])
    story += chapter(3, "Network Architecture")
    story += [
        p("The network is a small U-Net. The <b>encoder</b> shrinks the 128 &times; 128 input four times by "
          "strided convolutions, the <b>decoder</b> brings the features back to 64 &times; 64 with bilinear "
          "upsampling, and <b>skip connections</b> concatenate encoder features of the same size into the "
          "decoder (as in U-Net). Figure 3.1 shows the whole network."),
        figure(FIG / "architecture.png", "Figure 3.1: TinyUNet. Boxes give the output shape (channels &times; "
               "height &times; width) for a 128 &times; 128 input. Blue dashed arrows are skip connections."),
        sec("3.1 Depthwise Separable Block"),
        p("Every encoder and decoder stage is built from the depthwise separable (DS) block of Figure 3.2. A "
          "3 &times; 3 <i>depthwise</i> convolution filters each channel on its own and a 1 &times; 1 "
          "<i>pointwise</i> convolution mixes channels. For C<sub>in</sub> = C<sub>out</sub> = C this costs "
          "9C + C<super>2</super> weights instead of 9C<super>2</super> for a normal 3 &times; 3 convolution, "
          "which is what makes the network fast enough. When the shape does not change, a residual connection "
          "adds the block input to its output."),
        figure(FIG / "ds_block.png", "Figure 3.2: Depthwise separable block (stride s is 2 in the first block "
               "of each encoder stage, otherwise 1).", width=7.2 * cm),
        sec("3.2 Global Context"),
        p("Deciding which object is the <i>main</i> one needs image-level information. The deepest features "
          "(192 &times; 8 &times; 8) are averaged over the whole image, passed through a 1 &times; 1 "
          "convolution and added back to every position. This costs about 37K weights and almost no time."),
        KeepTogether([
            sec("3.3 Layer by Layer"),
            table(rows, "Table 3.1: Layers of the final network for a 128 &times; 128 input (shapes and parameter "
                  "counts read from the PyTorch model).",
                  col_widths=[1.5 * cm, 7.6 * cm, 3.4 * cm, 3.4 * cm], align_right_from=3, font_size=9)]),
        p("The head predicts one object logit per pixel of a 64 &times; 64 grid. It is upsampled bilinearly "
          "to 256 &times; 256 and thresholded at 0 (probability 0.5). With BatchNorm folded into the "
          "convolutions (Chapter 6) the deployed network has 299,489 parameters."),
    ]

    # 4
    story += chapter(4, "Training")
    story += [
        sec("4.1 Loss"),
        p("Let p<sub>i</sub> be the predicted object probability of pixel i (after upsampling to "
          "256 &times; 256) and y<sub>i</sub> (0 or 1) the true label. The loss is a weighted pixel-wise "
          "binary cross entropy plus soft Dice:"),
        p("L = (1/N) &Sigma;<sub>i</sub> w<sub>i</sub> [ &minus;y<sub>i</sub> log p<sub>i</sub> &minus; "
          "(1 &minus; y<sub>i</sub>) log(1 &minus; p<sub>i</sub>) ] &nbsp;+&nbsp; 1 &minus; "
          "(2 &Sigma;<sub>i</sub> p<sub>i</sub> y<sub>i</sub> + 1) / (&Sigma;<sub>i</sub> p<sub>i</sub> + "
          "&Sigma;<sub>i</sub> y<sub>i</sub> + 1)", EQ),
        p("w<sub>i</sub> = 1 + 2 &middot; [ pixel i lies within 2 px of the true outline ]", EQ),
        p("A single logit with binary cross entropy is the two-class pixel-wise cross entropy written with one "
          "output channel. Soft Dice (computed per image) balances small and large objects. The boundary "
          "weight w<sub>i</sub> triples the cost of mistakes near the outline, because Boundary F1 only rewards "
          "outlines within a 5 &times; 5 window; the near-outline region is found with the same boundary map "
          "and 5 &times; 5 max-pool used by the metric."),
        sec("4.2 Optimisation and Augmentation"),
        table([["Setting", "Value"],
               ["Optimiser", "AdamW, learning rate 3 &times; 10<super>&minus;3</super>, weight decay "
                             "10<super>&minus;4</super>"],
               ["Schedule", "1 warm-up epoch, then cosine decay to 0"],
               ["Batch size / epochs", "32 / 60 (about 25 s per epoch, 25 min in total)"],
               ["Precision", "mixed float16 on the GPU"],
               ["Input to the network", "256 &times; 256 photo area-downscaled to 128 &times; 128"],
               ["Model selection", "epoch with the best validation Q (epoch 59)"]],
              "Table 4.1: Training settings.", col_widths=[4.2 * cm, 10.8 * cm]),
        p("All augmentation runs on the GPU on whole batches, with the same geometric warp for photo and mask:"),
        bullets([
            "<b>Geometric:</b> random zoom 0.8&ndash;1.45, aspect ratio &plusmn;15%, rotation &plusmn;12&deg;, "
            "shift &plusmn;12%, horizontal flip.",
            "<b>Photometric:</b> brightness, contrast and saturation &plusmn;35%, a per-channel colour cast "
            "(to mimic tube light versus daylight), gamma 0.7&ndash;1.4, occasional blur and sensor noise.",
        ]),
        figure(FIG / "training_curve.png", "Figure 4.1: Training loss and validation metrics per epoch."),
    ]

    # 5
    story += chapter(5, "Evaluation Metrics")
    story += [
        sec("5.1 Mean Intersection over Union"),
        p("One 2 &times; 2 confusion matrix (TP, FP, FN, TN) is summed over all images of a set. Then"),
        p("IoU<sub>obj</sub> = TP / (TP + FP + FN), &nbsp;&nbsp; IoU<sub>bg</sub> = TN / (TN + FN + FP), "
          "&nbsp;&nbsp; mIoU = (IoU<sub>obj</sub> + IoU<sub>bg</sub>) / 2.", EQ),
        sec("5.2 Boundary F1"),
        p("A boundary pixel is an object pixel with a background pixel directly above, below, left or right "
          "(pixels outside the image count as background). It is found by comparing the zero-padded mask with "
          "its four shifted copies. A predicted boundary pixel is correct if a true boundary pixel lies in its "
          "5 &times; 5 window, and a true boundary pixel is found if a predicted one lies in its window:"),
        p("P = (correct predicted boundary pixels) / (predicted boundary pixels), &nbsp;&nbsp; "
          "R = (found true boundary pixels) / (true boundary pixels), &nbsp;&nbsp; "
          "BF1 = 2PR / (P + R).", EQ),
        p("Looping over every pixel and its window in Python is far too slow, so the window test is vectorised: "
          + mono("F.max_pool2d(kernel 5, stride 1, padding 2)") + " applied to a boundary map gives, at every "
          "pixel, whether any boundary pixel lies in its 5 &times; 5 window. BF1 is computed per image on whole "
          "batches and averaged over the set. If neither mask has boundary pixels BF1 is 1, and if only one does "
          "it is 0."),
        p("<b>Correctness check:</b> " + mono("tests/test_metrics.py") + " compares both metrics with a direct "
          "pixel-by-pixel implementation of the definitions on 34 random and edge-case masks (empty, identical, "
          "full); all values match exactly."),
    ]

    # 6
    story += chapter(6, "Keeping the Pipeline Fast")
    story += [
        p("The timed part starts with a 256 &times; 256 photo in memory and ends with its 256 &times; 256 mask "
          "of 0s and 1s in memory (Figure 6.1). Every step in between counts, including the downscaling."),
        figure(FIG / "pipeline.png", "Figure 6.1: The timed CPU pipeline."),
        sec("6.1 Speed Measures"),
        bullets([
            "<b>BatchNorm folding.</b> Each BN after a convolution is folded into it: with BN scale &gamma;, "
            "shift &beta;, running mean &mu; and variance &sigma;<super>2</super>, the new weights are "
            "W' = W &middot; &gamma; / &radic;(&sigma;<super>2</super> + &epsilon;) and bias "
            "b' = &beta; &minus; &mu; &gamma; / &radic;(&sigma;<super>2</super> + &epsilon;). The 1/255 input "
            "scaling is folded into the stem weights too, so raw pixel values go straight into the network.",
            "<b>Low working resolution.</b> The network sees 128 &times; 128 and predicts 64 &times; 64, about "
            "4&times; less work than 256 &times; 256. The validation sweep (Chapter 7) showed that a 256 input "
            "does not pay for its cost.",
            "<b>Channels-last memory layout.</b> oneDNN runs small depthwise and 1 &times; 1 convolutions much "
            "faster in this layout (for example 21.6 ms to 13.5 ms for one sweep model in an early, loaded test).",
            "<b>TorchScript trace and freeze</b> remove the Python overhead between layers. Three warm-up calls "
            "on a blank input are made while loading, because TorchScript re-optimises the graph during its "
            "first calls.",
            "<b>Upsampling outside PyTorch.</b> Profiling showed that " + mono("F.interpolate") + " of the "
            "one-channel logit to 256 &times; 256 plus the comparison took about 1.5 ms. " + mono("cv2.resize") +
            " (bilinear, same half-pixel convention) and a NumPy threshold take about 0.1 ms.",
        ]),
        sec("6.2 Parameter File"),
        p("Convolution weights are stored as int8 with one float32 scale per output channel c, and biases as "
          "float16:"),
        p("s<sub>c</sub> = max |W<sub>c</sub>| / 127, &nbsp;&nbsp; q = round(W<sub>c</sub> / s<sub>c</sub>), "
          "an integer from &minus;127 to 127, &nbsp;&nbsp; W<sub>c</sub> &asymp; q &middot; s<sub>c</sub>.", EQ),
        p("The weights are converted back to float32 once at load time (not timed), so inference itself is "
          "ordinary float32. On the validation set this gives the same Q as float16 storage (0.8128 versus "
          "0.8127) while the file shrinks from 615 KB to 341 KB. The final file is "
          f"<b>{s_bytes:,} bytes</b>."),
        sec("6.3 Timing"),
        p(f"T<sub>CPU</sub> is the sum of " + mono("time.process_time()") + " over the 5,019 DUTS-TE photos, "
          f"one at a time after one untimed warm-up run, divided by 5,019: <b>{t_ms:.2f} ms</b> per photo. "
          "Two practical notes: (1) on Windows " + mono("time.process_time()") + " advances in 15.6 ms ticks, "
          "so the 20-run average printed by " + mono("predict.py") + " is coarse; the 5,019-photo average is the "
          "reliable figure. (2) The laptop CPU is about twice as slow on battery power: a repeat of the timing "
          "on battery gave 9.84 ms per photo. Both are far below the 50 ms limit."),
    ]

    # 7
    sweep = [["Run", "Input", "Widths", "Params", "Loss", "T (ms)", "mIoU", "BF1", "Q", "valCES"],
             ["r128_w16", "128", "16-24-40-64", "30K", "BCE+Dice", "1.84", "0.840", "0.564", "0.757", "0.750"],
             ["r256_w16", "256", "16-24-40-64", "30K", "BCE+Dice", "4.00", "0.837", "0.584", "0.761", "0.748"],
             ["r192_w24", "192", "24-32-64-96", "65K", "BCE+Dice", "3.84", "0.860", "0.628", "0.790", "0.776"],
             ["r128_w32", "128", "32-48-96-128", "150K", "BCE+Dice", "3.06", "0.865", "0.640", "0.797", "0.782"],
             ["r128_w32 deep", "128", "32-48-96-160", "260K", "BCE+Dice", "3.53", "0.869", "0.650", "0.803",
              "0.781"],
             ["r128_w32", "128", "32-48-96-128", "150K", "+boundary", "3.00", "0.868", "0.656", "0.805", "0.789"],
             ["r128_w48", "128", "48-64-128-192", "299K", "BCE+Dice", "4.09", "0.874", "0.670", "0.813", "0.787"],
             ["<b>final</b>", "128", "48-64-128-192", "299K", "+boundary", "&ndash;", "0.877", "0.687", "0.820",
              "&ndash;"]]
    story += chapter(7, "Design Choices on the Validation Set")
    story += [
        p("Each candidate was trained for 50 epochs with the same split. T is the deployed pipeline timed on "
          "500 validation photos with the CPU otherwise idle, and valCES uses validation Q in place of DUTS-TE "
          "Q. DUTS-TE played no part in these choices."),
        table(sweep, "Table 7.1: Architecture and loss sweep (validation set). \u201cdeep\u201d uses 2-4-6 "
              "blocks per encoder stage instead of 2-3-4.",
              col_widths=[2.4 * cm, 1.1 * cm, 2.6 * cm, 1.3 * cm, 1.8 * cm, 1.2 * cm, 1.3 * cm, 1.3 * cm,
                          1.3 * cm, 1.5 * cm], align_right_from=5, bold_rows=(8,), font_size=8.5),
        figure(FIG / "sweep.png", "Figure 7.1: Validation Q against CPU time; marker area grows with the number "
               "of parameters.", width=12 * cm),
        p("<b>Findings.</b> (1) Width matters much more than input resolution: running at 256 instead of 128 "
          "doubles the time for +0.004 Q, while doubling the width gains +0.04 Q. (2) More depth helps less "
          "than more width. (3) The boundary weight adds about +0.007 Q, almost all of it through BF1. (4) At "
          "these sizes time and size cost only 1&ndash;3% of Q, so the final model is the widest one tested, with "
          "the boundary loss, trained for 60 epochs and stored as int8."),
    ]

    # 8
    story += chapter(8, "Results")
    story += [
        table([["Set", "Images", "mIoU", "Boundary F1", "Q"],
               ["Validation (held out from DUTS-TR)", f"{val['n']:,}", f"{val['mIoU']:.4f}", f"{val['BF1']:.4f}",
                f"{val['Q']:.4f}"],
               ["<b>DUTS-TE</b>", f"{te['n']:,}", f"<b>{te['mIoU']:.4f}</b>", f"<b>{te['BF1']:.4f}</b>",
                f"<b>{te['Q']:.4f}</b>"],
               ["Captured photos", f"{cap['n']}", f"{cap['mIoU']:.4f}", f"{cap['BF1']:.4f}", f"{cap['Q']:.4f}"]],
              "Table 8.1: Segmentation quality.", col_widths=[6.4 * cm, 1.8 * cm, 2.0 * cm, 2.4 * cm, 2.0 * cm],
              align_right_from=1),
        table([["T<sub>CPU</sub> (DUTS-TE)", "S<sub>disk</sub>", "Q (DUTS-TE)", "CES"],
               [f"{t_ms:.2f} ms", f"{s_bytes:,} B ({s_bytes / 2**20:.2f} MiB)", f"{te['Q']:.4f}",
                f"<b>{ces_score:.4f}</b>"]],
              "Table 8.2: Cost and final score.", col_widths=[3.6 * cm, 4.6 * cm, 3.0 * cm, 2.6 * cm],
              align_right_from=0),
        p(f"CES = {te['Q']:.4f} &times; (1 &minus; 0.20 &middot; {t_ms:.2f}/50 &minus; 0.10 &middot; "
          f"{s_bytes:,}/4,194,304) = {te['Q']:.4f} &times; "
          f"{1 - 0.2 * t_ms / 50 - 0.1 * s_bytes / 4194304:.4f} = <b>{ces_score:.4f}</b>.", EQ),
        sec("8.1 Validation versus DUTS-TE"),
        p("Q falls from 0.82 on validation to 0.66 on DUTS-TE. The cached DUTS-TE masks were checked against "
          "the raw files and are aligned with their photos, so the gap is a real change of distribution:"),
        bullets([
            "Objects cover on average <b>27%</b> of a DUTS-TR photo but only <b>15%</b> of a DUTS-TE photo.",
            "<b>18.5%</b> of DUTS-TE photos get object IoU below 0.3, against 3% on validation.",
            "DUTS-TE comes from different image sources, with more clutter, small objects and several objects.",
        ]),
        p("Validation comes from the same distribution as training, so it overestimates performance on new "
          "data; this is why the assignment asks for both numbers."),
    ]

    # 9
    cap_rows = [["Photo", "Conditions", "Object IoU", "BF1"]]
    conditions = {"bottle": "indoor, tube light, near, cluttered desk",
                  "cat": "outdoor, harsh daylight, grass and shadows",
                  "chair": "indoor, near, cluttered background",
                  "shoes": "indoor, warm light, very near, cut by the frame"}
    for r in captured:
        cap_rows.append([r["name"], conditions.get(r["name"], ""), f"{r['IoU_obj']:.2f}", f"{r['BF1']:.2f}"])
    story += chapter(9, "Captured Photos")
    story += [
        p("Each photo was resized with the rule of Section 2.1 and its mask drawn on the resized photo with a "
          "small polygon tool (" + mono("src/annotate.py") + "). The photos are used for testing only. In "
          "Figure 9.1 the true mask is green and the prediction magenta."),
        table(cap_rows, "Table 9.1: Captured photos.", col_widths=[2.2 * cm, 8.0 * cm, 2.3 * cm, 1.8 * cm],
              align_right_from=2),
        figure(FIG / "captured_overlays.png", "Figure 9.1: Captured photos: true mask (green) and predicted mask "
               "(magenta) as semi-transparent overlays.", max_height=13.5 * cm),
    ]

    # 10
    story += chapter(10, "Failure Cases")
    story += [
        sec("10.1 Camouflage in a Cluttered Outdoor Scene"),
        p("In the <b>cat</b> photo (IoU 0.12) the network marks the pink flowers along the wall and only the "
          "cat's face. The grey-brown fur has the same colours and texture as the dry grass and soil, under "
          "harsh daylight with hard shadows, while the flowers are the most colourful, high-contrast region. A "
          "network of about 300K parameters trained from scratch on 9,500 photos learns mostly colour and "
          "contrast cues and little about object categories, so it cannot tell that the cat matters more."),
        KeepTogether([
            sec("10.2 Small Object on a Large Textured Surface"),
            figure(FIG / "worst_val.png", "Figure 10.1: The six lowest-scoring validation photos.",
                   max_height=19 * cm)]),
        p("In all six worst validation photos the true object is <b>small</b> (an ant on a tree trunk, a bird on "
          "a roof, a nail in bark, a ladybird on a flower, a pretzel in front of a face) and the network "
          "segments the <b>large surface</b> instead. Its strongest cue is a big, central region that differs "
          "from the border, and at a 64 &times; 64 output a 10-pixel object shrinks to two or three cells. The "
          "same cause explains most of the DUTS-TE gap, since DUTS-TE contains many more small objects."),
        sec("10.3 Object Touching the Border under a Colour Cast"),
        p("In the <b>shoes</b> photo (IoU 0.70, BF1 0.38) the shoes fill the frame and are cut by its left and "
          "bottom edges. The network misses the dark heel and the part that touches the border. Most DUTS "
          "objects lie away from the border, so the network has learned that border regions are usually "
          "background, and the dark leather in strong warm light is close in colour to the wooden floor."),
        sec("10.4 Fine Structures and Boundary Accuracy"),
        p("Even when the region is right, the outline can be off: the <b>chair</b> has IoU 0.86 but BF1 0.39. "
          "The gaps between the arm-rests and the seat and the cane mesh are thinner than one cell of the "
          "64 &times; 64 output grid, and hand-drawn polygon masks are themselves only accurate to a few "
          "pixels."),
    ]

    # 11
    story += chapter(11, "Conclusion")
    story += [
        p("A tiny U-Net with depthwise separable convolutions, global context and concatenated skip connections, "
          "trained from scratch on 9,500 DUTS photos, segments the main object with validation Q = "
          f"{val['Q']:.2f} and DUTS-TE Q = {te['Q']:.2f}. The CPU pipeline needs {t_ms:.2f} ms per photo on one "
          f"thread and a {s_bytes / 1000:.0f} KB parameter file, which gives <b>CES = {ces_score:.3f}</b>."),
        p("The most useful lessons were that width matters more than input resolution, that a cheap boundary "
          "weight in the loss improves Boundary F1, and that most of the CPU time was overhead (upsampling, "
          "memory layout, Python) rather than convolutions, so profiling paid off more than shrinking the "
          "network. The main weakness is small objects and objects that look like their background, which is "
          "also why DUTS-TE scores lower than validation."),
        Spacer(1, 10),
        table([["File", "Purpose"],
               [mono("predict.py"), "segment one photo, print T CPU ms and S disk bytes"],
               [mono("weights/model.pt"), "parameter file (int8 conv weights + scales)"],
               [mono("src/model.py"), "TinyUNet and BatchNorm folding"],
               [mono("src/train.py"), "training with GPU augmentation"],
               [mono("src/metrics.py"), "mIoU, Boundary F1, CES"],
               [mono("src/pipeline.py"), "the timed CPU pipeline"],
               [mono("src/time_cpu.py"), "official T<sub>CPU</sub> on DUTS-TE"],
               [mono("src/evaluate.py"), "metrics for validation, DUTS-TE and captured photos"]],
              "Table 11.1: Main files of the submission.", col_widths=[4.6 * cm, 10.4 * cm]),
    ]

    doc = ReportDoc(str(ROOT / "report" / "report.pdf"), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                    topMargin=2.3 * cm, bottomMargin=2.2 * cm, title="Assignment 2: Cost-Efficient Segmentation",
                    author="Abhyudaya Tiwari (B23CS1085)")
    frame = Frame(MARGIN, 2.2 * cm, TEXT_W, PAGE_H - 4.5 * cm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPageEnd=on_page_end)])
    doc.multiBuild(story)
    print(f"wrote {ROOT / 'report' / 'report.pdf'}")


if __name__ == "__main__":
    build()
