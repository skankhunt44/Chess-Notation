"""Board Corner Regressor
=======================
Train a CNN to predict the four outer corners of a chessboard.

Usage (training):
    python board_corner_regressor.py train --data-dir data --epochs 10 --model-out ../assets/corners.pt

Usage (inference):
    python board_corner_regressor.py infer --model ../assets/corners.pt --image data/0046.png --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


class CornerDataset(Dataset):
    """Dataset yielding (image_tensor, corners)."""

    def __init__(self, root: str | Path, img_size: int = 224, transform=None):
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.image_files = sorted(
            list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")),
            key=lambda p: int(p.stem)
        )
        if not self.image_files:
            raise RuntimeError("no images found in data-dir")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        with open(img_path.with_suffix(".json")) as f:
            meta = json.load(f)
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = self.transform(img)

        corners = meta["corners"]
        norm = all(0 <= v <= 1 for pair in corners for v in pair)
        if norm:  # old dataset BL,BR,TL,TR
            bl, br, tl, tr = corners
            pts = [tl, tr, br, bl]
        else:  # pixel coords TR,BR,BL,TL
            tr_, br_, bl_, tl_ = corners
            pts = [
                [tl_[0] / w, tl_[1] / h],
                [tr_[0] / w, tr_[1] / h],
                [br_[0] / w, br_[1] / h],
                [bl_[0] / w, bl_[1] / h],
            ]
        corners_t = torch.tensor([p for pair in pts for p in pair], dtype=torch.float32)
        return img_t, corners_t


def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    return model


def epoch_loop(model: nn.Module, loader: DataLoader, criterion, optim, device, train: bool):
    model.train() if train else model.eval()
    loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train:
                optim.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward(); optim.step()
            loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)


def predict_corners(model: nn.Module, image: str | Path | np.ndarray, device=None) -> np.ndarray:
    """Return 4×2 array of normalised corner points."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(image, (str, Path)):
        img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (224, 224))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img_r).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(t).cpu().view(4, 2).numpy()
    return out


def main():
    p = argparse.ArgumentParser(description="Board corner regression trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--model-out", default="corners.pt")
    tr.add_argument("--val-split", type=float, default=0.1)
    tr.add_argument("--workers", type=int, default=4)

    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True)
    inf.add_argument("--image", required=True)
    inf.add_argument("--show", action="store_true")

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "train":
        ds = CornerDataset(args.data_dir)
        val_len = int(len(ds) * args.val_split)
        train_len = len(ds) - val_len
        train_set, val_set = random_split(ds, [train_len, val_len])
        tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        vl = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        model = build_model().to(device)
        criterion = nn.MSELoss()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = float("inf")
        for epoch in range(1, args.epochs + 1):
            tloss = epoch_loop(model, tl, criterion, optim, device, True)
            vloss = epoch_loop(model, vl, criterion, optim, device, False)
            print(f"epoch {epoch:02d}: train {tloss:.4f} | val {vloss:.4f}")
            if vloss < best:
                best = vloss
                torch.save(model.state_dict(), args.model_out)
                print("  ↳ saved best model")
    elif args.cmd == "infer":
        model = build_model()
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        corners = predict_corners(model, args.image, device=device)
        print(corners)
        if args.show:
            import matplotlib.pyplot as plt
            img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            pts = corners * np.array([[w, h]])
            plt.imshow(img)
            plt.scatter(pts[:,0], pts[:,1], c=['r','g','b','y'])
            plt.show()


if __name__ == "__main__":
    main()