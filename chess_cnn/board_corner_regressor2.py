"""
Board Corner Regressor  (v2 – 384 px, EfficientNet-B0, aug)
==========================================================
Train a CNN to predict the four outer corners of a chessboard.

Training
--------
python board_corner_regressor.py train \
       --data-dir data \
       --epochs 10 --model-out ../assets/corners2.pt

Inference + overlay
-------------------
python board_corner_regressor.py infer \
       --model ../assets/corners.pt \
       --image data/0046.png --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm


# ------------------------------------------------------------
# Hyper-params
# ------------------------------------------------------------
IMG_SIZE = 384
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


# ------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------
class CornerDataset(Dataset):
    def __init__(self, root: str | Path, train: bool):
        self.root  = Path(root)
        self.train = train
        self.files = sorted(
            list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")),
            key=lambda p: int(p.stem)
        )
        if not self.files:
            raise RuntimeError("no images found")

    def __len__(self): return len(self.files)

    def _load_meta(self, img_path: Path) -> Tuple[np.ndarray, List[List[float]]]:
        with open(img_path.with_suffix(".json")) as f:
            meta = json.load(f)
        img   = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w  = img.shape[:2]

        # Convert corners ➜ normalised TL,TR,BR,BL
        if all(0 <= v <= 1 for p in meta["corners"] for v in p):     # old set
            bl, br, tl, tr = meta["corners"]
            pts = [tl, tr, br, bl]
        else:                                                        # new set
            tr_, br_, bl_, tl_ = meta["corners"]
            pts = [
                [tl_[0] / w, tl_[1] / h],
                [tr_[0] / w, tr_[1] / h],
                [br_[0] / w, br_[1] / h],
                [bl_[0] / w, bl_[1] / h],
            ]
        return img, pts

    def __getitem__(self, idx):
        img, pts = self._load_meta(self.files[idx])

        # -------------------------------- augment --------------------------------
        if self.train:
            # random brightness/contrast
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.9, 1.1),
                                           beta=np.random.randint(-15, 15))

            # random in-plane rotation ±10°
            angle = np.random.uniform(-10, 10)
            center = (img.shape[1] / 2, img.shape[0] / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR)
            pts = [cv2.transform(np.array([[p]], np.float32), M)[0, 0] /
                   np.array([img.shape[1], img.shape[0]]) for p in
                   (np.array(pts) * np.array([img.shape[1], img.shape[0]]))]

            # mild perspective jitter
            if np.random.rand() < 0.5:
                jitter = 0.02 * IMG_SIZE
                src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * IMG_SIZE
                dst = src + np.random.uniform(-jitter, jitter, src.shape)
                P   = cv2.getPerspectiveTransform(src / IMG_SIZE,
                                                  dst / IMG_SIZE)
                img = cv2.warpPerspective(img, P, (img.shape[1], img.shape[0]))
                pts = [cv2.perspectiveTransform(np.array([[p]], np.float32), P)[0, 0] /
                       np.array([img.shape[1], img.shape[0]]) for p in
                       (np.array(pts) * np.array([img.shape[1], img.shape[0]]))]

        # -------------------------------------------------------------------------
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_t = TF.normalize(TF.to_tensor(img), MEAN, STD)

        corners_t = torch.tensor([c for p in pts for c in p], dtype=torch.float32)
        return img_t, corners_t


# ------------------------------------------------------------
#  Model
# ------------------------------------------------------------
def build_model() -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 8)
    return m


# ------------------------------------------------------------
#  Training helpers
# ------------------------------------------------------------
def epoch_loop(model, loader, criterion, optim, device, train: bool):
    model.train() if train else model.eval()
    loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train: optim.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            if train: loss.backward(); optim.step()
            loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)


# ------------------------------------------------------------
#  Inference util
# ------------------------------------------------------------
@torch.no_grad()
def predict_corners(model: nn.Module, img_path: str | Path, device) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    inp = TF.normalize(TF.to_tensor(cv2.resize(img, (IMG_SIZE, IMG_SIZE))), MEAN, STD)
    pred = model(inp.unsqueeze(0).to(device)).cpu().view(4, 2).numpy()
    h, w = img.shape[:2]
    return pred * np.array([w, h])


# ------------------------------------------------------------
#  CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # ------------------------------- train -------------------------------
    if args.cmd == "train":
        ds = CornerDataset(args.data_dir, train=True)
        val_len = int(len(ds) * args.val_split)
        tr_set, vl_set = random_split(ds, [len(ds) - val_len, val_len])
        tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers)
        vl_loader = DataLoader(vl_set, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers)

        model = build_model().to(device)
        crit  = nn.MSELoss()
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best  = float("inf")

        for ep in range(1, args.epochs + 1):
            tl = epoch_loop(model, tr_loader, crit, opt, device, True)
            vl = epoch_loop(model, vl_loader, crit, opt, device, False)
            print(f"epoch {ep:02d}: train {tl:.4f}  val {vl:.4f}")
            if vl < best:
                best = vl
                torch.save(model.state_dict(), args.model_out)
                print("  ↳ saved best")

    # ------------------------------- infer -------------------------------
    else:
        model = build_model().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        pts = predict_corners(model, args.image, device)

        print("TL, TR, BR, BL (px):\n", pts)

        if args.show:
            import matplotlib.pyplot as plt
            img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            # white polyline + coloured dots
            plt.plot(*zip(*(pts.tolist() + [pts[0].tolist()])),
                     color="white", linewidth=2)
            plt.scatter(pts[:, 0], pts[:, 1],
                        c=["red", "green", "blue", "yellow"], s=60)
            plt.axis("off")
            plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
