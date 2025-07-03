import cv2 as cv
import numpy as np
from typing import Tuple, Optional

"""vision.py – camera‑side utilities
-------------------------------------
•  Calibrate the chessboard once per session (detect 4 corners)
•  Warp every frame into an 800 × 800 top‑down board image
•  Return an 8 × 8 occupancy matrix (0 = empty, 1 = white, 2 = black)

Designed to stay dependency‑light (just OpenCV + NumPy).  If HSV colour
segmentation proves unreliable on your set, swap `occupancy_hsv()` with a CNN
classifier trained on 64 × 64 crops – the public API remains identical.
"""

# ---------------------------------------------------------------------------
# Constants & tunables
# ---------------------------------------------------------------------------
CALIB_SIZE = 800                 # output board width/height in px

# HSV thresholds – tweak for your lighting & pieces; mask ∈ [0, 255]
WHITE_LOWER = (0, 0, 170)
WHITE_UPPER = (180, 40, 255)
BLACK_LOWER = (0, 0, 0)
BLACK_UPPER = (180, 255, 70)

# Morphology kernel for noise suppression
_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))


# ---------------------------------------------------------------------------
# Board calibration
# ---------------------------------------------------------------------------

def find_board(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4 outer‑corner points of the board if detected, else ``None``.

    The corners are **ordered TL, TR, BR, BL** (clockwise, starting top‑left)
    and expressed in the *input‑frame* coordinate space.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 1. Edge map
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # 2. Probabilistic Hough – find candidate long lines (board border)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150,
                           minLineLength=0.5 * min(frame.shape[:2]),
                           maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None  # not enough structure

    # 3. Gather all endpoints; fit a minimum‑area rectangle around them
    pts = np.concatenate([[l[0][:2], l[0][2:]] for l in lines])
    rect = cv.minAreaRect(pts)          # ((cx, cy), (w, h), angle)
    box  = cv.boxPoints(rect)           # 4 × 2 float32

    # 4. Ensure top‑left is first, then clockwise order
    box = sorted(box, key=lambda p: (p[1], p[0]))        # sort by y, then x
    tl, tr = sorted(box[:2], key=lambda p: p[0])          # leftmost of top two
    bl, br = sorted(box[2:], key=lambda p: p[0])
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered


def warp_board(frame: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective‑warp *frame* so the board is a flat square image.

    Parameters
    ----------
    frame : BGR image (H × W × 3)
    corners : 4 × 2 float32 array, TL‑TR‑BR‑BL order.

    Returns
    -------
    tuple(img, M)
        *img* – warped board (CALIB_SIZE × CALIB_SIZE × 3, BGR)
        *M*   – 3 × 3 homography matrix (frame → board)
    """
    dst = np.array([[0, 0], [CALIB_SIZE, 0],
                    [CALIB_SIZE, CALIB_SIZE], [0, CALIB_SIZE]], dtype=np.float32)
    M = cv.getPerspectiveTransform(corners, dst)
    board_img = cv.warpPerspective(frame, M, (CALIB_SIZE, CALIB_SIZE))
    return board_img, M


# ---------------------------------------------------------------------------
# Occupancy detection (HSV threshold version)
# ---------------------------------------------------------------------------

def occupancy_hsv(board_img: np.ndarray) -> np.ndarray:
    """Return 8 × 8 matrix: 0 = empty, 1 = white piece, 2 = black piece."""
    hsv = cv.cvtColor(board_img, cv.COLOR_BGR2HSV)

    w_mask = cv.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    b_mask = cv.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    w_mask = cv.morphologyEx(w_mask, cv.MORPH_CLOSE, _KERNEL, iterations=2)
    b_mask = cv.morphologyEx(b_mask, cv.MORPH_CLOSE, _KERNEL, iterations=2)

    occ = np.zeros((8, 8), np.int8)
    sq = CALIB_SIZE // 8

    for r in range(8):
        for c in range(8):
            y0, y1 = r * sq, (r + 1) * sq
            x0, x1 = c * sq, (c + 1) * sq
            roi_w = w_mask[y0:y1, x0:x1]
            roi_b = b_mask[y0:y1, x0:x1]

            # proportion of coloured pixels inside this square
            if roi_w.mean() > 30:      # 30 ≈ (0.12 × 255)
                occ[r, c] = 1
            elif roi_b.mean() > 30:
                occ[r, c] = 2
    return occ


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def draw_board_overlay(frame: np.ndarray, corners: np.ndarray, colour=(0, 255, 0)) -> np.ndarray:
    """Return *frame* with the detected board outline overlaid (for debugging)."""
    out = frame.copy()
    if corners is not None:
        for i in range(4):
            pt1 = tuple(map(int, corners[i]))
            pt2 = tuple(map(int, corners[(i + 1) % 4]))
            cv.line(out, pt1, pt2, colour, 2, cv.LINE_AA)
    return out
