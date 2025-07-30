# view_square_crops.py
"""
Quick visual sanity-check for ChessSquareDataset
------------------------------------------------
$ python view_square_crops.py --data-dir data/0046.png --rows 8 --cols 8 --shuffle
"""

from pathlib import Path
import argparse, random, math

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# ------- import your dataset class directly ----------
from chess_square_classifier2 import ChessSquareDataset, CLASS_MAP
# -----------------------------------------------------

def tensor_to_img(t):
    """Undo normalisation -> HWC uint8 for matplotlib."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t * std + mean
    t = (t.clamp(0, 1) * 255).byte()
    return t.permute(1, 2, 0).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--rows", type=int, default=8,
                    help="number of rows in figure grid")
    ap.add_argument("--cols", type=int, default=8,
                    help="number of cols in figure grid")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    ds = ChessSquareDataset(args.data_dir)
    idxs = list(range(len(ds)))
    if args.shuffle:
        random.shuffle(idxs)

    # pick exactly rows*cols samples
    sel = idxs[: args.rows * args.cols]
    imgs  = []
    titles = []
    for i in sel:
        x, y = ds[i]          # x = Tensor C×H×W (normalised); y = int label
        imgs.append(x)
        lab = [k for k, v in CLASS_MAP.items() if v == y][0]
        titles.append(lab)

    # torchvision.make_grid expects BCHW tensor
    grid = make_grid(torch.stack(imgs), nrow=args.cols, padding=2)
    grid_img = tensor_to_img(grid)

    # --- plot ---
    plt.figure(figsize=(args.cols, args.rows))
    plt.imshow(grid_img)
    plt.axis("off")

    # add per-cell titles
    # (matplotlib doesn't support per-image titles in a single axis,
    # so we annotate manually)
    h, w = grid_img.shape[:2]
    cell_h = h / args.rows
    cell_w = w / args.cols
    for r in range(args.rows):
        for c in range(args.cols):
            idx = r * args.cols + c
            plt.text(c * cell_w + 2, r * cell_h + 10,
                     titles[idx], color="white", fontsize=6,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
