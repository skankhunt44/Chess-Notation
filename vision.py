import cv2 as cv
import numpy as np
from typing import Tuple, Optional

from chess_cnn.opencv_chessboard_finder import (
    detect_corners,
    outer_corners_from_inner,
)
import torch
import torch.nn as nn
from torchvision import models, transforms

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
CALIB_SIZE = 800                 # output board width/height in px

# HSV thresholds – tweak for your lighting & pieces; mask ∈ [0, 255]
WHITE_LOWER = (0, 0, 170)
WHITE_UPPER = (180, 40, 255)
BLACK_LOWER = (0, 0, 0)
BLACK_UPPER = (180, 255, 70)

# Morphology kernel for noise suppression
_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# Lazy loaded CNN model
_CNN_MODEL = None


# ---------------------------------------------------------------------------
# Board calibration
# ---------------------------------------------------------------------------

def find_board(frame: np.ndarray) -> Optional[np.ndarray]:
    """Return the 4 outer‑corner points of the board if detected, else ``None``.

    The corners are **ordered TL, TR, BR, BL** (clockwise, starting top‑left)
    and expressed in the *input‑frame* coordinate space.
    """
    try:
        inner = detect_corners(frame, rows=7, cols=7)
    except Exception:
        return None

    corners = outer_corners_from_inner(inner, rows=7, cols=7)
    return corners.astype(np.float32)


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
    dst = np.array([[0, 0], [CALIB_SIZE, 0], [CALIB_SIZE, CALIB_SIZE], [0, CALIB_SIZE]], dtype=np.float32)
    H, _ = cv.findHomography(corners.astype(np.float32), dst)
    board_img = cv.warpPerspective(frame, H, (CALIB_SIZE, CALIB_SIZE))
    return board_img, H


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


def _load_cnn_model(device=None) -> nn.Module:
    """Lazy-load the CNN piece classifier."""
    global _CNN_MODEL
    if '_CNN_MODEL' in globals() and _CNN_MODEL is not None:
        return _CNN_MODEL

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(num_classes: int = 3) -> nn.Module:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    model = build_model()
    state = torch.load('assets/model.pt', map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _CNN_MODEL = model
    return model


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def occupancy_cnn(board_img: np.ndarray) -> np.ndarray:
    """Return occupancy matrix using the trained CNN classifier."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _load_cnn_model(device)
    sq = CALIB_SIZE // 8
    occ = np.zeros((8, 8), np.int8)
    with torch.no_grad():
        for r in range(8):
            for c in range(8):
                crop = board_img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq]
                crop = cv.resize(crop, (64, 64))
                crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                inp = _TRANSFORM(crop).unsqueeze(0).to(device)
                pred = model(inp).argmax(1).item()
                occ[r, c] = pred
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


def ascii_occ(mat):
    chars = {0: '.', 1: 'W', 2: 'B'}
    return '\n'.join(' '.join(chars[x] for x in row) for row in mat)
