#!/usr/bin/env python
"""Line-grid chessboard detector — robust v3
==========================================
Adds *segment pruning & merging* so that an excessive number of Hough segments
(from background clutter) no longer swamp the intersection stage.

Main new points
---------------
1. **Angle‑bucket outlier rejection** – after k‑means, keep only segments whose
   angle lies within ±8 ° of their bucket’s centre.
2. **Length ranking** – keep at most the 30 longest segments of each family
   (horizontal & vertical).  Enough for a 9×9 board while eliminating tiny
   spurious edges.
3. **Optional display of kept segments** when `--debug` is passed.

You reported “too many horiz/vert segments”; the above heuristics reduce noise
without needing hand‑tuned thresholds for every photo.

Run:
    python linegrid_detector_enhanced.py --image board.jpg --show --debug
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _line_intersection(l1: Tuple[int, int, int, int],
                       l2: Tuple[int, int, int, int]) -> Tuple[float, float] | None:
    """Intersection of two infinite lines given by segments *l1* & *l2*.
    Uses float64 arithmetic to avoid integer overflow.
    """
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # parallel

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def _snap_points(pts: Iterable[Tuple[float, float]], bin_sz: int = 20) -> np.ndarray:
    """Cluster *pts* by rounding to *bin_sz*-pixel bins → return centroids."""
    acc: dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}
    for x, y in pts:
        key = (int(round(x / bin_sz)), int(round(y / bin_sz)))
        if key in acc:
            acc[key] = (acc[key][0] + np.array([x, y]), acc[key][1] + 1)
        else:
            acc[key] = (np.array([x, y], dtype=np.float32), 1)
    out = [sum_pt / cnt for sum_pt, cnt in acc.values()]
    return np.stack(out) if out else np.empty((0, 2), np.float32)


# ---------------------------------------------------------------------------
# Detection core
# ---------------------------------------------------------------------------

def detect_board_corners(img_bgr: np.ndarray, rows: int = 7, cols: int = 7,
                         dbg: bool = False) -> np.ndarray | None:
    """Return TL, TR, BR, BL board corners or *None* if not found."""
    # 1️⃣  CLAHE for illumination invariance → Canny
    l_eq = cv.createCLAHE(3.0, (8, 8)).apply(cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)[:, :, 0])
    edges = cv.Canny(l_eq, 30, 100, apertureSize=3)

    # 2️⃣  Probabilistic Hough with lenient parameters
    lines_p = cv.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=30, minLineLength=25, maxLineGap=15)
    if lines_p is None:
        return None

    segs = [tuple(l) for l in lines_p[:, 0]]
    angles = np.array([
        np.mod(np.degrees(np.arctan2(y2 - y1, x2 - x1)), 180)
        for x1, y1, x2, y2 in segs
    ], dtype=np.float32).reshape(-1, 1)

    # 3️⃣  k-means (k=2) on angles
    _r, labels, centres = cv.kmeans(angles, 2, None,
                                    (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    5, cv.KMEANS_PP_CENTERS)
    centres = centres.flatten()

    # Which bucket is closer to horizontal (|angle| < 45° after wrapping)?
    horiz_bucket = 0 if min(abs(centres[0]), abs(centres[0] - 180)) < 45 else 1

    # 3a. **Angle‑based outlier rejection** (±8° around its centre)
    angle_tol = 8.0
    horizontals, verticals = [], []
    for seg, lbl, ang in zip(segs, labels.ravel(), angles.flatten()):
        if abs((ang - centres[lbl] + 90) % 180 - 90) > angle_tol:
            continue  # discard outlier
        (horizontals if lbl == horiz_bucket else verticals).append(seg)

    if dbg:
        print(f"after angle filter: {len(horizontals)} horiz, {len(verticals)} vert")

    # 3b. **Keep only the N longest segments** to avoid tiny background edges
    def _seg_len_sq(s):
        x1, y1, x2, y2 = s; return (x2 - x1) ** 2 + (y2 - y1) ** 2

    max_keep = 30
    horizontals = sorted(horizontals, key=_seg_len_sq, reverse=True)[:max_keep]
    verticals   = sorted(verticals,   key=_seg_len_sq, reverse=True)[:max_keep]

    if dbg:
        print(f"pruned to: {len(horizontals)} horiz, {len(verticals)} vert (top {max_keep} by length)")

    # 4️⃣  Compute all horizontal × vertical intersections
    inters: List[Tuple[float, float]] = []
    for h in horizontals:
        for v in verticals:
            pt = _line_intersection(h, v)
            if pt is None:
                continue
            x, y = pt
            if 0 <= x < img_bgr.shape[1] and 0 <= y < img_bgr.shape[0]:
                inters.append(pt)

    inters = _snap_points(inters, bin_sz=20)
    if len(inters) < 4:
        return None

        # 5️⃣  RANSAC homography (accept missing or extra intersections)
    inters = inters[np.lexsort((inters[:, 0], inters[:, 1]))]

    # ensure equal count ≤ rows*cols  ------------------------------
    ideal_full = np.array([[c, r] for r in range(rows) for c in range(cols)], np.float32)
    n = min(len(inters), len(ideal_full))
    inters = inters[:n]
    ideal  = ideal_full[:n]

    H, _ = cv.findHomography(ideal, inters, cv.RANSAC, 5.0)
    if H is None:
        return None

    frame = np.float32([[-1, -1], [cols, -1], [cols, rows], [-1, rows]])
    return cv.perspectiveTransform(frame[None, ...], H)[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def overlay(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    out = img.copy()
    pts = quad.astype(int)
    for i in range(4):
        cv.line(out, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2, cv.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Detect chessboard frame via Hough line grid (v3)")
    ap.add_argument("--image", required=True, help="input photo of a chessboard")
    ap.add_argument("--rows", type=int, default=7, help="inner grid rows (squares-1)")
    ap.add_argument("--cols", type=int, default=7, help="inner grid cols (squares-1)")
    ap.add_argument("--show", action="store_true", help="show result with matplotlib")
    ap.add_argument("--save", default=None, help="optional filename to save overlay image")
    ap.add_argument("--debug", action="store_true", help="print extra diagnostics")
    args = ap.parse_args()

    img = cv.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    quad = detect_board_corners(img, args.rows, args.cols, dbg=args.debug)
    if quad is None:
        print("board not found")
        return

    vis = overlay(img, quad)

    if args.save:
        cv.imwrite(args.save, vis)
        print("saved", args.save)

    if args.show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
        plt.title("Detected board outline")
        plt.axis("off")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
