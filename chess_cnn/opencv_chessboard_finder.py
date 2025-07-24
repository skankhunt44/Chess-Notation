"""
OpenCV Chessboard Corner Finder
==============================
Quick utility to detect the inner corner grid of a chessboard using OpenCV’s
`findChessboardCorners` / `findChessboardCornersSB` and visualise the result.

Usage example
-------------
python opencv_chessboard_finder.py --image data/0046.png --rows 7 --cols 7 --show

Arguments
---------
--image   path to the input render or photo that has a board
--rows    number of **internal** rows (squares − 1)  [default: 7]
--cols    number of **internal** cols (squares − 1)  [default: 7]
--show    pop up a matplotlib window with the detected corners overlaid
--save    optional output filename to write the overlay image

Notes
-----
• `rows × cols` must match the number of inner corner intersections, not
  squares. A standard 8×8 board has 7×7 inner intersections.
• The script first tries the modern `findChessboardCornersSB`. If that fails it
  falls back to the classic `findChessboardCorners` + sub‑pixel refinement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors


def detect_corners(img_bgr: np.ndarray, rows: int, cols: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # try the modern detector first
    ret, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=0)
    if ret:
        return corners.reshape(-1, 2)

    # fallback: classic detector + sub‑pixel refinement
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not ret:
        raise RuntimeError("No chessboard detected. Try adjusting rows/cols or lighting.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def overlay(img_rgb: np.ndarray, pts: np.ndarray):
    over = img_rgb.copy()
    for p in pts:
        cv2.circle(over, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    return over

def outer_corners_from_inner(pts, rows=7, cols=7):
    """
    pts: (rows*cols,2) array from findChessboardCorners,
         ordered row-major TL➜BR.
    returns: 4×2 array TL, TR, BR, BL of the outer frame corners.
    """
    pts = pts.reshape(rows, cols, 2)
    # unit vectors along a rank (→) and a file (↓)
    u = (pts[0, -1] - pts[0, 0]) / (cols - 1)
    v = (pts[-1, 0] - pts[0, 0]) / (rows - 1)

    tl = pts[0, 0]     - 0.5 * (u + v)
    tr = pts[0, -1]    + 0.5 * (u - v)
    br = pts[-1, -1]   + 0.5 * (u + v)
    bl = pts[-1, 0]    + 0.5 * (v - u)

    return np.stack([tl, tr, br, bl])

def overlay(img: np.ndarray, pts: np.ndarray, colours=None):
    out = img.copy()
    if colours is None:
        colours = [(0, 255, 0)] * len(pts)           # default green
    for (x, y), col in zip(pts, colours):
        if isinstance(col, str):                     # matplotlib colour names
            col = tuple(int(c * 255) for c in mcolors.to_rgb(col))[::-1]  # to BGR
        cv2.circle(out, (int(x), int(y)), 4, col, -1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--rows", type=int, default=7)     # inner grid
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    inner = detect_corners(img_bgr, args.rows, args.cols)   # N×2, N≤rows*cols
    print(f"found {len(inner)} inner intersections")

    # ------------------------------------------------------------
    # 1️⃣  Build corresponding ideal grid coordinates (0…7,0…7)
    # ------------------------------------------------------------
    rows, cols = args.rows, args.cols
    ideal = np.array(
        [[c, r] for r in range(rows) for c in range(cols)], np.float32
    )
    ideal = ideal[: len(inner)]  # keep same N rows as detected order

    # ------------------------------------------------------------
    # 2️⃣  Solve H with RANSAC (pixel ← ideal)
    # ------------------------------------------------------------
    H, mask = cv2.findHomography(ideal, inner, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("Homography solve failed")

    # ------------------------------------------------------------
    # 3️⃣  Map the four outer frame corners
    # ------------------------------------------------------------
    
    # frame_ideal = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    frame_ideal = np.float32([
    [-1,        -1],           # TL
    [cols,  -1],           # TR
    [cols,  rows],     # BR
    [-1,        rows],     # BL
    ])

    frame_px    = cv2.perspectiveTransform(frame_ideal[None, ...], H)[0]

    # visualise
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    vis = overlay(img_rgb, inner)               # green inner dots
    vis = overlay(vis, frame_px, colours=["r", "g", "b", "y"])  # frame

    if args.save:
        cv2.imwrite(args.save, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print("saved", args.save)

    if args.show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(vis)
        plt.title("Inner (green) & outer frame (TL blue, TR green, BR red, BL yellow)")
        plt.axis("off")
        plt.tight_layout(); plt.show()



if __name__ == "__main__":
    main()
