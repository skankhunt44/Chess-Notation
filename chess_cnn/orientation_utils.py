# orientation_utils.py
# -------------------------------------------------------------
# Decide if the warped 800×800 board needs to be rotated 180°.
# Idea: after the homography, squares on the A-file have file-index 0,
# squares on the H-file have file-index 7.
#   • If x-coordinate  ↑ with file-index  → camera above White   (no flip)
#   • If x-coordinate ↓ with file-index  → camera behind Black  (flip)
# -------------------------------------------------------------
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import cv2

def _file_idx(square: str) -> int:
    return ord(square[0].upper()) - ord('A')      # 'A'→0 … 'H'→7

def _bbox_center(box: List[int]) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w * 0.5, y + h * 0.5

def needs_flip_180(meta: dict, H: np.ndarray) -> bool:
    """
    Parameters
    ----------
    meta : dict
        The JSON dictionary (must contain "pieces").
    H : np.ndarray
        3×3 homography that maps original px ➜ 800×800 warp.

    Returns
    -------
    bool
        True  → rotate the warp 180° so that A8 is top-left.
        False → leave warp as is.
    """
    if "pieces" not in meta or len(meta["pieces"]) < 2:
        # not enough info – assume no flip
        return False

    files, xs = [], []

    for p in meta["pieces"]:
        cx, cy = _bbox_center(p["box"])
        pt     = np.array([cx, cy, 1.0])
        bx, by, w = H @ pt
        bx /= w;  by /= w            # board-space coordinates
        files.append(_file_idx(p["square"]))
        xs.append(bx)

    files = np.array(files, dtype=np.float32)
    xs    = np.array(xs,    dtype=np.float32)

    if xs.std() < 1e-3:
        # degenerate (all x equal) – assume no flip
        return False

    corr = np.corrcoef(files, xs)[0, 1]   # Pearson correlation
    return corr < 0                       # negative → mirror → flip
