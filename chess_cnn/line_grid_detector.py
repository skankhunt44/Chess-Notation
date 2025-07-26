"""Line-grid chessboard detector
===============================
Detect board corners using Hough line intersections.
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

def _line_intersection(l1: Tuple[int, int, int, int], l2: Tuple[int, int, int, int]) -> Tuple[float, float] | None:
    """Return the intersection point of two line segments if it exists."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _snap_points(pts: Iterable[Tuple[float, float]], bin_sz: int = 20) -> np.ndarray:
    """Cluster *pts* by rounding to the nearest *bin_sz* pixels."""
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
# Detection
# ---------------------------------------------------------------------------

def detect_board_corners(img_bgr: np.ndarray, rows: int = 7, cols: int = 7) -> np.ndarray | None:
    """Return TL, TR, BR, BL board corners using line intersections.

    Intersections are snapped to the nearest grid points then sorted
    top-to-bottom and left-to-right.  At most ``rows * cols`` points are
    matched against an ideal grid before estimating the homography.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150, apertureSize=3)

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=10)
    if lines is None:
        return None

    horizontals: List[Tuple[int, int, int, int]] = []
    verticals: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            horizontals.append((x1, y1, x2, y2))
        elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
            verticals.append((x1, y1, x2, y2))

    inters: List[Tuple[float, float]] = []
    for h in horizontals:
        for v in verticals:
            pt = _line_intersection(h, v)
            if pt is None:
                continue
            x, y = pt
            if 0 <= x < img_bgr.shape[1] and 0 <= y < img_bgr.shape[0]:
                inters.append((x, y))

    if not inters:
        return None

    inters = _snap_points(inters, bin_sz=20)
    if len(inters) < 4:
        return None

    n = min(len(inters), rows * cols)
    inters = inters[np.lexsort((inters[:, 0], inters[:, 1]))][:n]

    ideal = np.array([[c, r] for r in range(rows) for c in range(cols)], np.float32)
    ideal = ideal[:n]

    H, _ = cv.findHomography(ideal, inters, cv.RANSAC, 5.0)
    if H is None:
        return None

    frame = np.float32([
        [-1, -1],
        [cols, -1],
        [cols, rows],
        [-1, rows],
    ])
    outer = cv.perspectiveTransform(frame[None, ...], H)[0]
    return outer.astype(np.float32)


def overlay(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = img.copy()
    for i in range(4):
        pt1 = tuple(map(int, pts[i]))
        pt2 = tuple(map(int, pts[(i + 1) % 4]))
        cv.line(out, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)
    return out


def main(img_path: str) -> None:
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    corners = detect_board_corners(img)
    if corners is None:
        print("board not found")
        return
    vis = overlay(img, corners)
    out_name = Path(img_path).with_name("linegrid_overlay.jpg")
    cv.imwrite(str(out_name), vis)
    print("saved", out_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input photo of a chessboard")
    main(ap.parse_args().image)
