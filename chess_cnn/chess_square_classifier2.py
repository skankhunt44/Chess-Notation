"""
Chess Square Occupancy Classifier
================================
CLI tool to train, preprocess, preview, and infer square‑level occupancy
(empty / white / black) on chessboards from the Synthetic Chess Board Images
Kaggle dataset.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


def compute_homography(
    corners: list[list[float]],
    img_w: int,
    img_h: int,
    size: int = 800,
) -> np.ndarray:
    """
    Return H that maps the board to a size×size top-down square.

    Works with either
      * normalised corners in [0,1]  (old dataset, order BL,BR,TL,TR)
      * pixel-coords corners         (new dataset, order TR,BR,BL,TL)
    """
    # detect format
    norm = all(0 <= v <= 1 for pair in corners for v in pair)

    if norm:                                      # old dataset
        bl, br, tl, tr = corners                 # BL, BR, TL, TR
        pts_src = np.float32([
            [tl[0]*img_w, tl[1]*img_h],
            [tr[0]*img_w, tr[1]*img_h],
            [br[0]*img_w, br[1]*img_h],
            [bl[0]*img_w, bl[1]*img_h],
        ])
    else:                                         # new dataset
        tr, br, bl, tl = corners                 # TR, BR, BL, TL
        pts_src = np.float32([tl, tr, br, bl])   # reorder to TL,TR,BR,BL

    pts_dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H



def warp_board(img: np.ndarray, H: np.ndarray, size: int = 800) -> np.ndarray:
    return cv2.warpPerspective(img, H, (size, size))


def square_from_row_col(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{8 - row}"


CLASS_MAP = {"empty": 0, "white": 1, "black": 2}
INV_CLASS = {v: k for k, v in CLASS_MAP.items()}


class ChessSquareDataset(Dataset):
    def __init__(self, root: str | Path, board_size: int = 800, square_size: int = 64, transform=None):
        self.root = Path(root)
        self.board_size = board_size
        self.square_size = square_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.image_files = sorted(list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")), key=lambda p: int(p.stem))
        if not self.image_files:
            raise RuntimeError("no images found in data‑dir")
        self.index: List[Tuple[int, int, int]] = [(b, r, c) for b in range(len(self.image_files)) for r in range(8) for c in range(8)]

    def __len__(self):
        return len(self.index)

    
    def _load_board(self, board_idx: int):
        # cache check
        if not hasattr(self, "_cache"):
            self._cache = {}
        if board_idx in self._cache:
            return self._cache[board_idx]

        # 1  read image + JSON
        img_path = self.image_files[board_idx]
        with open(img_path.with_suffix(".json")) as f:
            meta = json.load(f)
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        # 2  warp the board
        h, w = img.shape[:2]
        H = compute_homography(meta["corners"], w, h, size=self.board_size)
        warped = warp_board(img, H, size=self.board_size)

        # 3  build a per-square dict  { "A1": "p", … }
        if "config" in meta:                          # old synthetic set
            config = meta["config"]

        elif "pieces" in meta:                        # new set – list of pieces
            # keep the piece letter exactly as given (upper = white, lower = black)
            config = {p["square"].upper(): p["piece"] for p in meta["pieces"]}

        else:                                         # derive from FEN
            config = {}
            rows = meta["fen"].split()[0].split("/")
            for r, row in enumerate(rows):            # r = 0 is rank 8
                file_idx = 0
                for ch in row:
                    if ch.isdigit():
                        file_idx += int(ch)
                    else:
                        square = f"{chr(ord('A') + file_idx)}{8 - r}"
                        config[square] = ch
                        file_idx += 1

        # 4  store in cache and return
        self._cache[board_idx] = (warped, config)
        return self._cache[board_idx]



    def __getitem__(self, idx):
        board_idx, row, col = self.index[idx]
        warped, config = self._load_board(board_idx)
        cell_px = self.board_size // 8
        crop = warped[row * cell_px : (row + 1) * cell_px, col * cell_px : (col + 1) * cell_px]
        crop = cv2.resize(crop, (self.square_size, self.square_size))
        square = square_from_row_col(row, col)
        if square in config:
            label = CLASS_MAP["white" if config[square].isupper() else "black"]
        else:
            label = CLASS_MAP["empty"]
        crop = self.transform(crop)
        return crop, label


class CropFolderDataset(Dataset):
    def __init__(self, root: str | Path, transform=None):
        self.ds = datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


def build_model(num_classes: int = 3):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def epoch_loop(model, loader, criterion, optim, device, train: bool):
    model.train() if train else model.eval()
    loss_sum = correct = 0
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
            correct += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_sum / n, correct / n


def infer_board(model, image_path: str | Path, device, board_size: int = 800, square_size: int = 64):
    model.eval()
    img_path = Path(image_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    H = compute_homography(meta["corners"], w, h, size=board_size)
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
                crop = warped[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
                crop = cv2.resize(crop, (square_size, square_size))
                pred = model(tform(crop).unsqueeze(0).to(device)).argmax(1).item()
                board[r, c] = pred
    return board


def preview_board(image_path: str | Path, out_path: str | None, board_size: int = 800, show: bool = False):
    img_path = Path(image_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    H = compute_homography(meta["corners"], w, h, size=board_size)
    warped = warp_board(img, H, size=board_size)
    grid = warped.copy()
    cell_px = board_size // 8
    for i in range(9):
        cv2.line(grid, (0, i * cell_px), (board_size, i * cell_px), (0, 255, 0), 1)
        cv2.line(grid, (i * cell_px, 0), (i * cell_px, board_size), (0, 255, 0), 1)
    if out_path:
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(grid); plt.axis("off"); plt.show()


def preprocess_dataset(src: str | Path, dst: str | Path, workers: int = 6, board_size: int = 800, square_size: int = 64):
    src, dst = Path(src), Path(dst)
    for cls in CLASS_MAP:
        (dst / cls).mkdir(parents=True, exist_ok=True)
    imgs = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")), key=lambda p: int(p.stem))

    def process(img_path: Path):
        with open(img_path.with_suffix(".json")) as f:
            meta = json.load(f)
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        H = compute_homography(meta["corners"], w, h, size=board_size)
        warped = warp_board(img, H, size=board_size)
        cell_px = board_size // 8
        for r in range(8):
            for c in range(8):
                crop = warped[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
                crop = cv2.resize(crop, (square_size, square_size))
                sq = square_from_row_col(r, c)
                if sq in meta["config"]:
                    lbl = "white" if meta["config"][sq].endswith("_w") else "black"
                else:
                    lbl = "empty"
                cv2.imwrite(str(dst / lbl / f"{img_path.stem}_{r}{c}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    with ThreadPool(workers) as pool:
        list(tqdm(pool.imap_unordered(process, imgs), total=len(imgs)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--model-out", required=True)
    tr.add_argument("--val-split", type=float, default=0.1)
    tr.add_argument("--workers", type=int, default=4)
    tr.add_argument("--preprocessed", action="store_true")

    pp = sub.add_parser("preprocess")
    pp.add_argument("--src-dir", required=True)
    pp.add_argument("--dst-dir", required=True)
    pp.add_argument("--workers", type=int, default=6)

    pv = sub.add_parser("preview")
    pv.add_argument("--image", required=True)
    pv.add_argument("--out", default=None)
    pv.add_argument("--show", action="store_true")

    inf = sub.add_parser("infer")
    inf.add_argument("--model", required=True)
    inf.add_argument("--image", required=True)

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.cmd == "preprocess":
        preprocess_dataset(args.src_dir, args.dst_dir, args.workers)
        return

    if args.cmd == "preview":
        preview_board(args.image, args.out, show=args.show)
        return

    if args.cmd == "train":
        if args.preprocessed:
            ds = CropFolderDataset(args.data_dir, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
        else:
            ds = ChessSquareDataset(args.data_dir)
        val_len = int(len(ds) * args.val_split)
        train_len = len(ds) - val_len
        train_set, val_set = random_split(ds, [train_len, val_len])
        tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        vl = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        model = build_model().to(device)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = 0.0
        for epoch in range(1, args.epochs + 1):
            tloss, tacc = epoch_loop(model, tl, criterion, optim, device, True)
            vloss, vacc = epoch_loop(model, vl, criterion, optim, device, False)
            print(f"epoch {epoch:02d}: train {tacc:.3f} val {vacc:.3f}")
            if vacc > best:
                best = vacc
                torch.save(model.state_dict(), args.model_out)
        return

    if args.cmd == "infer":
        model = build_model().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        board = infer_board(model, args.image, device)
        print(board)


if __name__ == "__main__":
    main()
