"""
Chess Square Occupancy Classifier
================================
Full pipeline to **train**, **pre‑process**, **preview**, and **infer** square‑level
occupancy (empty / white / black) on chessboards rendered in the *Synthetic
Chess Board Images* Kaggle dataset.

Sub‑commands
------------
train     – train MobileNet‑V2 on 64×64 crops generated on‑the‑fly or from a
            pre‑processed folder
preprocess – one‑shot offline extractor that warps every board once and saves
            64×64 PNG crops into class folders (MUCH faster training)
preview   – warp one board and draw the 8×8 grid so you can visually inspect
            the result (optionally saves an image)
infer     – run the trained model on a board image and output the 8×8 matrix

Example usage
-------------
# ① One‑shot preprocessing (recommended for speed)
python chess_square_classifier.py preprocess \
       --data-dir data/SyntheticChessBoards \
       --out-dir preproc64

# ② Train
python chess_square_classifier.py train \
       --data-dir preproc64 \
       --epochs 10 --batch-size 256 \
       --model-out ../assets/model.pt

# ③ Preview a single board warp and grid overlay
python chess_square_classifier.py preview --image data/42.png --show

# ④ Inference
python chess_square_classifier.py infer --model ../assets/model.pt --image data/42.png

Dependencies
------------
* Python 3.9+
* PyTorch 2.x (pip install torch torchvision)
* OpenCV‑Python (pip install opencv-python)
* tqdm, matplotlib

Tested on macOS M1 (MPS) and Linux + CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

# --------------------------
# 1.  Utility functions
# --------------------------

def compute_homography(corners: List[List[float]], size: int = 800, src_res: int = 1280) -> np.ndarray:
    """Return H that warps board to a size×size square."""
    pts_src = np.float32(corners) * src_res  # denormalise 0‑1 → px
    pts_dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def warp_board(img: np.ndarray, H: np.ndarray, size: int = 800) -> np.ndarray:
    return cv2.warpPerspective(img, H, (size, size))


def square_from_row_col(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{8 - row}"

# --------------------------
# 2.  Dataset – on‑the‑fly crops
# --------------------------

CLASS_MAP = {"empty": 0, "white": 1, "black": 2}
INV_CLASS = {v: k for k, v in CLASS_MAP.items()}


class ChessSquareDataset(Dataset):
    """Yields (3×64×64 Tensor, label) extracted on‑the‑fly from 1280×1280 renders."""

    def __init__(self, root: str | Path, board_size: int = 800, square_size: int = 64, transform=None):
        self.root = Path(root)
        self.board_size = board_size
        self.square_size = square_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Accept both .jpg and .png (dataset uses .png)
        self.image_files = sorted(list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")), key=lambda p: int(p.stem))
        assert self.image_files, "No images found in data‑dir"

        # Pre‑compute index mapping to avoid nested loops during __getitem__
        self.index: List[Tuple[int, int, int]] = [(b, r, c) for b in range(len(self.image_files)) for r in range(8) for c in range(8)]

    def __len__(self):
        return len(self.index)

    def _load_board(self, board_idx: int):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if board_idx not in self._cache:
            img_path = self.image_files[board_idx]
            with open(img_path.with_suffix(".json")) as f:
                meta = json.load(f)
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            H = compute_homography(meta["corners"], size=self.board_size)
            warped = warp_board(img, H, size=self.board_size)
            self._cache[board_idx] = (warped, meta["config"])
        return self._cache[board_idx]

    def __getitem__(self, idx):
        board_idx, row, col = self.index[idx]
        warped, config = self._load_board(board_idx)
        cell_px = self.board_size // 8
        crop = warped[row*cell_px:(row+1)*cell_px, col*cell_px:(col+1)*cell_px]
        crop = cv2.resize(crop, (self.square_size, self.square_size), interpolation=cv2.INTER_LINEAR)

        square_name = square_from_row_col(row, col)
        if square_name in config:
            label = CLASS_MAP["white" if config[square_name].endswith("_w") else "black"]
        else:
            label = CLASS_MAP["empty"]

        crop = self.transform(crop)
        return crop, label

# --------------------------
# 3.  Model & helpers
# --------------------------

def build_model(num_classes=3):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optim, device):
    model.train(); loss_sum = corr = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optim.step()
        loss_sum += loss.item() * x.size(0)
        corr += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, corr / n


def eval_epoch(model, loader, criterion, device):
    model.eval(); loss_sum = corr = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            corr += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, corr / n

# --------------------------
# 4.  Inference helper
# --------------------------

def infer_board(model, image_path: str | Path, device, board_size=800, square_size=64):
    model.eval()
    img_path = Path(image_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    H = compute_homography(meta["corners"], size=board_size)
    warped = warp_board(img, H, size=board_size)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cell_px = board_size // 8
    board = np.zeros((8, 8), dtype=int)
    with torch.no_grad():
        for r in range(8):
            for c in range(8):
                crop = warped[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                crop = cv2.resize(crop, (square_size, square_size))
                pred = model(tform(crop).unsqueeze(0).to(device)).argmax(1).item()
                board[r, c] = pred
    return board

# --------------------------
# 5.  Preview utility
# --------------------------

def preview_board(image_path: str | Path, out_path=None, board_size=800, show=False):
    img_path = Path(image_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    H = compute_homography(meta["corners"], size=board_size)
    warped = warp_board(img, H, size=board_size)
    grid = warped.copy()
    cell_px = board_size // 8
    for i in range(9):
        cv2.line(grid, (0, i*cell_px), (board_size, i*cell_px), (0, 255, 0), 1)
        cv2.line(grid, (i*cell_px, 0), (i*cell_px, board_size), (0, 255, 0), 1)
    if out_path:
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(grid); plt.axis("off")
        plt.show()

# --------------------------
# 6.  Pre‑processing: save crops to disk
# --------------------------

def preprocess_dataset(src_dir: str | Path, out_dir: str | Path, workers: int = 6, board_size=800, square_size=64):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    (out_dir/"empty").mkdir(parents=True, exist_ok=True)
    (out_dir/"white").mkdir(exist_ok=True)
    (out_dir/"black").mkdir(exist_ok=True)

    imgs = sorted(list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")), key=lambda p: int(p.stem))
    from multiprocessing.pool import ThreadPool

    def process(img_path):
        with open(img
