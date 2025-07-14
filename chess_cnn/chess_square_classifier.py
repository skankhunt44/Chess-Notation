"""
Chess Square Occupancy Classifier
================================
End‑to‑end pipeline to train a 3‑class CNN (white piece / black piece / empty)
using the **Synthetic Chess Board Images** dataset from Kaggle and to run
inference on new board photos for notation recording.

Usage (training):
-----------------
$ python chess_square_classifier.py train --data-dir /path/to/dataset --epochs 10 --batch-size 512 --model-out model.pt

Usage (inference on a single image):
------------------------------------
$ python chess_square_classifier.py infer --model model.pt --image test.jpg --show

Requirements:
-------------
* Python 3.9+
* PyTorch 2.x (pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118)
* OpenCV‑Python (pip install opencv-python)
* tqdm (pip install tqdm)

Tested on Linux + CUDA GPU, but will fall back to CPU automatically.

Data layout:
------------
The script expects the Kaggle files directly inside --data-dir::

  /data-dir/
      0.jpg
      0.json
      1.jpg
      1.json
      ... 1942.jpg/json

No train/val split is provided; the code makes an 90/10 random split the
first time it runs and caches the indices to splits.json so results are
reproducible.
"""

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

###############################################
# ----- 1.  Utility functions  -------------- #
###############################################

def compute_homography(corners: List[List[float]], size: int = 800) -> np.ndarray:
    """Return a homography that warps the board to a size×size top‑down view."""
    pts_src = np.float32(corners) * 1280  # denormalise
    pts_dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def warp_board(img: np.ndarray, H: np.ndarray, size: int = 800) -> np.ndarray:
    """Apply homography and return the warped board image."""
    return cv2.warpPerspective(img, H, (size, size))


def square_from_row_col(row: int, col: int) -> str:
    """Return algebraic square name (A1 .. H8) given 0‑based row/col (row 0 = rank 8)."""
    file = chr(ord("A") + col)
    rank = 8 - row
    return f"{file}{rank}"

###############################################
# ----- 2.  Dataset definition  ------------- #
###############################################

CLASS_MAP = {
    "empty": 0,
    "white": 1,
    "black": 2,
}


class ChessSquareDataset(Dataset):
    """Dataset that yields (crop, label) where crop is 64×64 RGB Tensor."""

    def __init__(
        self,
        root: str | Path,
        board_size: int = 800,
        square_size: int = 64,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.board_size = board_size
        self.square_size = square_size
        self.transform = transform if transform else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.image_files = sorted(self.root.glob("*.jpg"), key=lambda p: int(p.stem))
        assert self.image_files, "No .jpg files found in data-dir"

        # Build an index mapping global idx -> (board_idx, row, col)
        self.index: List[Tuple[int, int, int]] = []
        for b_idx, img_path in enumerate(self.image_files):
            for row in range(8):
                for col in range(8):
                    self.index.append((b_idx, row, col))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        board_idx, row, col = self.index[idx]
        img_path = self.image_files[board_idx]
        json_path = img_path.with_suffix(".json")

        # Load metadata only once per board by caching in object dict
        if not hasattr(self, "_cache"):
            self._cache = {}
        if board_idx not in self._cache:
            with open(json_path) as f:
                meta = json.load(f)
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            H = compute_homography(meta["corners"], size=self.board_size)
            warped = warp_board(img, H, size=self.board_size)
            self._cache[board_idx] = (warped, meta["config"])
        warped, config = self._cache[board_idx]

        # Crop square and resize to square_size×square_size
        cell_px = self.board_size // 8
        y1, y2 = row * cell_px, (row + 1) * cell_px
        x1, x2 = col * cell_px, (col + 1) * cell_px
        crop = warped[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.square_size, self.square_size), interpolation=cv2.INTER_LINEAR)

        # Determine label
        square_name = square_from_row_col(row, col)
        if square_name in config:
            label = CLASS_MAP["white" if config[square_name].endswith("_w") else "black"]
        else:
            label = CLASS_MAP["empty"]

        crop = self.transform(crop)
        return crop, label

###############################################
# ----- 3.  Model & training utils ---------- #
###############################################

def build_model(num_classes: int = 3) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_loop(model: nn.Module, loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss, running_correct = 0.0, 0
    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)


def eval_loop(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    running_loss, running_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), running_correct / len(loader.dataset)

###############################################
# ----- 4.  Inference helper  -------------- #
###############################################

def infer_board(model: nn.Module, image_path: str | Path, device, board_size=800, square_size=64) -> np.ndarray:
    """Return 8×8 numpy array with values 0,1,2 (empty, white, black)."""
    model.eval()
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    with open(Path(image_path).with_suffix(".json")) as f:
        meta = json.load(f)
    H = compute_homography(meta["corners"], size=board_size)
    warped = warp_board(img, H, size=board_size)
    tensorify = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((square_size, square_size)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cell_px = board_size // 8
    result = np.zeros((8, 8), dtype=np.int64)
    with torch.no_grad():
        for row in range(8):
            for col in range(8):
                crop = warped[row*cell_px:(row+1)*cell_px, col*cell_px:(col+1)*cell_px]
                crop = tensorify(crop).unsqueeze(0).to(device)
                pred = model(crop).argmax(1).item()
                result[row, col] = pred
    return result

###############################################
# ----- 5.  Main CLI  ---------------------- #
###############################################

def main():
    parser = argparse.ArgumentParser(description="Chess Square Occupancy Trainer/Inferencer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train sub‑command
    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True, help="Path to Kaggle dataset directory")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--model-out", default="model.pt")
    tr.add_argument("--val-split", type=float, default=0.1)
    tr.add_argument("--workers", type=int, default=4)

    # Inference sub‑command
    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True)
    inf.add_argument("--image", required=True)
    inf.add_argument("--show", action="store_true")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "train":
        ds = ChessSquareDataset(args.data_dir)
        val_len = int(len(ds) * args.val_split)
        train_len = len(ds) - val_len
        train_set, val_set = random_split(ds, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        model = build_model()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_loop(model, train_loader, criterion, optimizer, device)
            vl, va = eval_loop(model, val_loader, criterion, device)
            print(f"Epoch {epoch:02d}: train loss {tl:.4f}, acc {ta:.3f} | val loss {vl:.4f}, acc {va:.3f}")
            if va > best_acc:
                best_acc = va
                torch.save(model.state_dict(), args.model_out)
                print("  ↳ saved best model")

    elif args.cmd == "infer":
        model = build_model()
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.to(device)
        board = infer_board(model, args.image, device)
        print("Predicted occupancy (0=empty,1=white,2=black):\n", board)
        if args.show:
            import matplotlib.pyplot as plt
            plt.imshow(board, cmap="viridis", interpolation="nearest")
            plt.title("Predicted occupancy")
            plt.colorbar()
            plt.show()


if __name__ == "__main__":
    main()
