"""
Visualise the 64 crops for ONE board image.
===========================================

$ python view_one_board.py --image data/0046.jpg
"""

import argparse
from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Normalize, ToTensor
from torchvision.utils import make_grid

# ---- import helpers from your classifier -------------------------
from chess_square_classifier2 import (compute_homography, warp_board,
                                      square_from_row_col)
# ------------------------------------------------------------------

IMG_SIZE  = 800        # same as ChessSquareDataset.board_size
CROP_SIZE = 64         # same as ChessSquareDataset.square_size

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DENORM = Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                   std=[1/s for s in STD])

def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Undo normalisation → HWC uint8 for plt.imshow."""
    t = DENORM(t).clamp(0, 1)
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# ---- universal meta→dict helper (same as in classifier) ----------
def make_config(meta: dict) -> dict[str, str]:
    if "config" in meta:
        return meta["config"]
    if "pieces" in meta:
        return {p["square"].upper(): p["piece"] for p in meta["pieces"]}
    # derive from FEN
    cfg, rows = {}, meta["fen"].split()[0].split("/")
    for r, row in enumerate(rows):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                cfg[f"{chr(ord('A')+f)}{8-r}"] = ch
                f += 1
    return cfg
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True,
                    help="board photo (.jpg/.png) that has a matching .json")
    args = ap.parse_args()

    img_path  = Path(args.image)
    meta_path = img_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    # ---- load + warp exactly as in training -----------------------
    img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    with meta_path.open() as f:
        meta = json.load(f)

    H = compute_homography(meta["corners"],
                           img.shape[1], img.shape[0],  # width, height
                           size=IMG_SIZE)
    board_img = warp_board(img, H, size=IMG_SIZE)
    cfg = make_config(meta)          # per-square dict
    # ---------------------------------------------------------------

    # Cut into 64 crops
    cell  = IMG_SIZE // 8
    crops, labels = [], []

    for r in range(8):
        for c in range(8):
            crop = board_img[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                              interpolation=cv2.INTER_LINEAR)

            sq  = square_from_row_col(r, c)
            if sq in cfg:
                val = cfg[sq]
                is_white = str(val).endswith("white") or str(val).isupper()
                lab = "white" if is_white else "black"
            else:
                lab = "empty"
            labels.append(lab)

            # show network input (normalised tensor) if you like
            crop_t = Normalize(MEAN, STD)(ToTensor()(crop))
            crops.append(crop_t)

    # Build a mosaic for display
    grid = make_grid(torch.stack(crops),            # (64,3,H,W)
                     nrow=8, padding=1)             # nice 8×8 layout
    grid_np = tensor_to_img(grid)                   # CHW → HWC uint8

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_np); plt.axis("off")

    # overlay tiny labels
    for r in range(8):
        for c in range(8):
            plt.text(c*CROP_SIZE + 2, r*CROP_SIZE + 10,
                     labels[r*8+c], fontsize=6, color="white",
                     bbox=dict(facecolor="black", alpha=0.6, pad=1))
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
