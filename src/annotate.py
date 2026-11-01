"""Minimal polygon mask annotator for the captured photos (OpenCV window).

    python -m src.annotate [name ...]      # default: every photo in captured/images_256

The 256x256 photo is shown enlarged (x3) with the current mask tinted green.
    left click   add a polygon vertex
    right click  close the polygon and ADD it to the mask (or CUT it, in cut mode)
    x            toggle add / cut mode (cut removes regions, e.g. holes between handles)
    u            undo last vertex          c   clear the whole mask
    s            save and go to the next photo
    q            quit without saving this photo
Masks are written to captured/masks_256/<name>.png with object 255 and background 0.
"""
import sys

import cv2
import numpy as np

from src.common import CAPTURED, SIZE

ZOOM = 3
SHIFT = 4  # sub-pixel precision for cv2.fillPoly


def annotate(name):
    img = cv2.imread(str(CAPTURED / "images_256" / f"{name}.png"), cv2.IMREAD_COLOR)
    mask_path = CAPTURED / "masks_256" / f"{name}.png"
    existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (existing >= 128).astype(np.uint8) * 255 if existing is not None else np.zeros((SIZE, SIZE), np.uint8)
    state = {"points": [], "cut": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append((x / ZOOM - 0.5, y / ZOOM - 0.5))
        elif event == cv2.EVENT_RBUTTONDOWN and len(state["points"]) >= 3:
            pts = np.round(np.array(state["points"]) * (1 << SHIFT)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 0 if state["cut"] else 255, lineType=cv2.LINE_8, shift=SHIFT)
            state["points"] = []

    window = f"annotate {name}"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = img.copy()
        view[mask > 0] = (0.55 * view[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        view = cv2.resize(view, (SIZE * ZOOM, SIZE * ZOOM), interpolation=cv2.INTER_NEAREST)
        pts = [(int((px + 0.5) * ZOOM), int((py + 0.5) * ZOOM)) for px, py in state["points"]]
        colour = (0, 0, 255) if state["cut"] else (255, 0, 255)
        for a, b in zip(pts, pts[1:]):
            cv2.line(view, a, b, colour, 1)
        for p in pts:
            cv2.circle(view, p, 3, colour, -1)
        cv2.putText(view, "CUT" if state["cut"] else "ADD", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and state["points"]:
            state["points"].pop()
        elif key == ord("x"):
            state["cut"] = not state["cut"]
        elif key == ord("c"):
            mask[:] = 0
        elif key == ord("s"):
            cv2.imwrite(str(mask_path), mask)
            print(f"saved {mask_path}")
            break
        elif key == ord("q"):
            break
    cv2.destroyWindow(window)


def main():
    (CAPTURED / "masks_256").mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(p.stem for p in (CAPTURED / "images_256").glob("*.png"))
    for name in names:
        annotate(name)


if __name__ == "__main__":
    main()
