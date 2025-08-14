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
from logging import config
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

from orientation_utils import needs_flip_180


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
    """
    Yields (64×64 RGB Tensor,   label:int)
           where label ∈ {0:empty, 1:white, 2:black}
    """

    def __init__(self,
                 root: str | Path,
                 board_size: int = 800,
                 square_size: int = 64,
                 transform=None):
        self.root         = Path(root)
        self.board_size   = board_size
        self.square_size  = square_size
        # self.transform    = transform or transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406],
        #                          [0.229, 0.224, 0.225]),
        # ])
        self.transform = transforms.Compose([
            transforms.ToPILImage() if using_numpy else (lambda z: z),  # only if needed
            transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1,1.0))], p=0.25),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(.02,.08)),  # lower
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])


        self.image_files  = sorted(
            list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")),
            key=lambda p: int(p.stem)
        )
        if not self.image_files:
            raise RuntimeError("no images found in data-dir")

        # global index → (board_idx, row, col)
        self.index: List[Tuple[int, int, int]] = [
            (b, r, c)
            for b in range(len(self.image_files))
            for r in range(8)
            for c in range(8)
        ]

        self._cache: dict[int, Tuple[np.ndarray, dict]] = {}
        self._flip : dict[int, bool]                    = {}

    # --------------------------------------------------------------
    def __len__(self):  return len(self.index)

    # --------------------------------------------------------------
    def _load_board(self, board_idx: int):
        """
        Warp one board and return (warped_img, square_dict).
        Flip the image 180° (and remember that decision) if JSON says
        the camera was behind Black.
        """
        if board_idx in self._cache:
            return self._cache[board_idx]

        img_path = self.image_files[board_idx]
        with open(img_path.with_suffix(".json")) as f:
            meta = json.load(f)

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # ── 1  warp the board ─────────────────────────────────────
        H  = compute_homography(meta["corners"], w, h, size=self.board_size)
        warped = warp_board(img, H, size=self.board_size)

        # ── 2  decide orientation & maybe rotate ─────────────────
        flip = needs_flip_180(meta, H)
        if flip:
            warped = cv2.rotate(warped, cv2.ROTATE_180)
        self._flip[board_idx] = flip

        # ── 3  build square→piece-letter dict ────────────────────
        if "config" in meta:                         # old schema
            cfg = meta["config"]
        elif "pieces" in meta:                       # new schema
            cfg = {p["square"].upper(): p["piece"] for p in meta["pieces"]}
        else:                                        # derive from FEN
            cfg = {}
            rows = meta["fen"].split()[0].split("/")
            for r, row in enumerate(rows):
                f = 0
                for ch in row:
                    if ch.isdigit():
                        f += int(ch)
                    else:
                        cfg[f"{chr(ord('A')+f)}{8-r}"] = ch
                        f += 1

        self._cache[board_idx] = (warped, cfg)
        return self._cache[board_idx]

    # --------------------------------------------------------------
    def __getitem__(self, idx: int):
        board_idx, row, col = self.index[idx]
        warped, cfg = self._load_board(board_idx)

        # adjust indices if we rotated the warp
        # if self._flip[board_idx]:
        #     row, col = 7 - row, 7 - col

        # crop + resize
        cell = self.board_size // 8
        crop = warped[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
        crop = cv2.resize(crop, (self.square_size, self.square_size),
                          interpolation=cv2.INTER_LINEAR)

        # label
        square = f"{chr(ord('A')+col)}{8-row}"   # algebraic
        if square in cfg:
            val = cfg[square]
            is_white = str(val).isupper() or str(val).endswith("white")
            label = CLASS_MAP["white" if is_white else "black"]
        else:
            label = CLASS_MAP["empty"]

        return self.transform(crop), label


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

    cv2.imwrite("warp_ok.png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)) # debug

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    cell_px = board_size // 8
    print("board size:", board_size, "cell size:", cell_px) # debug
    board = np.zeros((8, 8), dtype=int)
    with torch.no_grad():
        for r in range(8):
            for c in range(8):
                crop = warped[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
                crop = cv2.resize(crop, (square_size, square_size))

                cv2.imwrite(f"crop_{r}_{c}.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))  # debug

                pred = model(tform(crop).unsqueeze(0).to(device)).argmax(1).item()
                board[r, c] = pred
    return board


# python chess_square_classifier2.py infer-warp \
#        --model ../assets/model.pt \
#        --board-img ../calib_warp.jpg

# ────────────────────────────────────────────────────────────────────────────────
# ★  tiny helper – nice to visualise in the terminal
# ────────────────────────────────────────────────────────────────────────────────
_CHR = {0: '.', 1: 'W', 2: 'B'}          # empty / white / black
def ascii_occ(mat: np.ndarray) -> str:
    return '\n'.join(' '.join(_CHR[x] for x in row) for row in mat)

# ────────────────────────────────────────────────────────────────────────────────
# ★  inference on an *already warped* image (no JSON, no corners)
# ────────────────────────────────────────────────────────────────────────────────
def infer_warp_board(model, board_img_path: str | Path,
                     device,
                     square_size: int = 64) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(board_img_path)), cv2.COLOR_BGR2RGB)
    model.eval()

    cv2.imwrite("warped_ok.png", img) # debug
    cv2.imwrite("warped_ok_colour_correct.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) # debug

    if img is None:
        raise FileNotFoundError(board_img_path)
    if img.shape[0] != img.shape[1]:
        raise ValueError("image must be a square top-down warp of the board")

    
    # # quick brightness heuristic (no JSON available)
    # # flip if top-left square is lighter than bottom-right
    # cell = img.shape[0] // 8
    # if img[:cell,:cell].mean() > img[-cell:,-cell:].mean():
    #     img = cv2.rotate(img, cv2.ROTATE_180)

    board_px   = img.shape[0]
    cell_px    = board_px // 8
    print("board px:", board_px, "cell size:", cell_px) # debug
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    occ = np.zeros((8, 8), np.int8)
    with torch.no_grad():
        for r in range(8):
            for c in range(8):
                crop = img[r*cell_px:(r+1)*cell_px,
                           c*cell_px:(c+1)*cell_px]
                crop = cv2.resize(crop, (square_size, square_size))

                cv2.imwrite(f"cropped_{r}_{c}.png", crop)  # debug
                
                pred = model(transform(crop).unsqueeze(0)
                                   .to(device)).argmax(1).item()
                occ[r, c] = pred
    return occ



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
    # for i in range(9):
    #     cv2.line(grid, (0, i * cell_px), (board_size, i * cell_px), (0, 255, 0), 1)
    #     cv2.line(grid, (i * cell_px, 0), (i * cell_px, board_size), (0, 255, 0), 1)
    # if out_path:
    #     cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(grid); plt.axis("off"); plt.show()


def _make_config(meta):
    """Return { 'A1': 'P', ... } mapping for ANY meta format."""
    if "config" in meta:                     # old synthetic set
        return meta["config"]

    if "pieces" in meta:                     # new list-of-pieces set
        cfg = {}
        for p in meta["pieces"]:
            # keep the exact letter: upper = white, lower = black
            cfg[p["square"].upper()] = p["piece"]
        return cfg

    # fallback – derive from FEN
    cfg = {}
    rows = meta["fen"].split()[0].split("/")
    for r, row in enumerate(rows):           # r = 0 is rank 8
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                sq = f"{chr(ord('A')+file_idx)}{8-r}"
                cfg[sq] = ch
                file_idx += 1
    return cfg


# def preprocess_dataset(src: str | Path, dst: str | Path, workers: int = 6, board_size: int = 800, square_size: int = 64):
#     src, dst = Path(src), Path(dst)
#     for cls in CLASS_MAP:
#         (dst / cls).mkdir(parents=True, exist_ok=True)
#     imgs = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")), key=lambda p: int(p.stem))

#     def process(img_path: Path):
#         with open(img_path.with_suffix(".json")) as f:
#             meta = json.load(f)

#         config = _make_config(meta)

#         img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
#         h, w = img.shape[:2]
#         H = compute_homography(meta["corners"], w, h, size=board_size)
#         warped = warp_board(img, H, size=board_size)
#         cell_px = board_size // 8
#         for r in range(8):
#             for c in range(8):
#                 crop = warped[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px]
#                 crop = cv2.resize(crop, (square_size, square_size))
#                 sq = square_from_row_col(r, c)
#                 # if sq in meta["config"]:
#                 #     lbl = "white" if meta["config"][sq].endswith("_w") else "black"
#                 if sq in config:
#                     val = config[sq]
#                     is_white = val.endswith("white") or str(val).isupper()
#                     lbl = "white" if is_white else "black"
#                 else:
#                     lbl = "empty"
#                 cv2.imwrite(str(dst / lbl / f"{img_path.stem}_{r}{c}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

#     with ThreadPool(workers) as pool:
#         list(tqdm(pool.imap_unordered(process, imgs), total=len(imgs)))


def preprocess_dataset(src: str | Path, dst: str | Path, workers: int = 6,
                       board_size: int = 800, square_size: int = 64):
    src, dst = Path(src), Path(dst)
    CLASS_DIRS = {"E":"0_empty", "W":"1_white", "B":"2_black"}  # canonical order
    for d in CLASS_DIRS.values():
        (dst / d).mkdir(parents=True, exist_ok=True)

    # Prefer new-format label JSONs first
    label_jsons = sorted(src.rglob("labels_*.json"), key=lambda p: p.stem)

    # Fallback to image list (old datasets)
    if not label_jsons:
        imgs = sorted(list(src.rglob("*.jpg")) + list(src.rglob("*.png")), key=lambda p: p.stem)
    else:
        imgs = label_jsons  # we'll handle differently in process()

    def process(p: Path):
        # ───────────── NEW: direct-warp samples with labels_XXXX.json ─────────────
        if p.suffix.lower() == ".json" and p.name.startswith("labels_"):
            meta = json.loads(p.read_text())
            img_path = p.parent / meta["image"]   # e.g., img_0001.png
            if not img_path.exists():
                print(f"[warn] missing image for {p.name}: {img_path}")
                return
            warped = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            h, w = warped.shape[:2]
            cell_px = min(h, w) // 8
            lbl_map = {"E": "0_empty", "W": "1_white", "B": "2_black"}
            labels = meta.get("labels", [])
            if len(labels) != 64:
                print(f"[warn] {p.name} has {len(labels)} labels (expected 64); skipping")
                return
            k = 0
            for r in range(8):
                for c in range(8):
                    crop = warped[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                    crop = cv2.resize(crop, (square_size, square_size))
                    cls = lbl_map.get(labels[k], "empty"); k += 1
                    out = dst / cls / f"{img_path.stem}_{r}{c}.png"
                    ok = cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    if not ok:
                        print(f"[warn] failed to write {out}")
            return

        # ───────────── OLD: images with corners/pieces/FEN ─────────────
        img_path = p
        meta_path = img_path.with_suffix(".json")
        if not meta_path.exists():
            # quietly skip if no matching meta for old format
            return
        meta = json.loads(meta_path.read_text())
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        H = compute_homography(meta["corners"], w, h, size=board_size)
        warped = warp_board(img, H, size=board_size)
        config = _make_config(meta)
        cell_px = board_size // 8
        for r in range(8):
            for c in range(8):
                crop = warped[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                crop = cv2.resize(crop, (square_size, square_size))
                sq = square_from_row_col(r, c)
                if sq in config:
                    val = config[sq]
                    is_white = str(val).endswith("white") or str(val).isupper()
                    lbl = "white" if is_white else "black"
                else:
                    lbl = "empty"
                lbl = "white" if is_white else "black" if sq in config else "empty"
                folder = {"empty":"0_empty", "white":"1_white", "black":"2_black"}[lbl]
                out = dst / folder / f"{img_path.stem}_{r}{c}.png"
                cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    with ThreadPool(workers) as pool:
        list(tqdm(pool.imap_unordered(process, imgs), total=len(imgs)))


# # Preprocess your new session to crops
# python chess_square_classifier2.py preprocess \
#   --src-dir data/session_*/samples \
#   --dst-dir ../data/crops_my_cam \
#   --workers 2

# # Train (from scratch)
# python chess_square_classifier2.py train \
#   --data-dir data/crops_my_cam \
#   --preprocessed \
#   --epochs 10 \
#   --batch-size 256 \
#   --lr 3e-4 \
#   --model-out ../assets/model.pt

# # ...or continue from your existing model.pt
# python chess_square_classifier2.py train \
#   --data-dir data/crops_my_cam \
#   --preprocessed \
#   --epochs 6 \
#   --batch-size 256 \
#   --lr 2e-4 \
#   --resume ../assets/model.pt \
#   --model-out ../assets/model.pt

# Quick sanity check on a saved warp
# python chess_square_classifier2.py infer-warp \
#   --model ../assets/model.pt \
#   --board-img ../data/session_2591762550623958/samples/img_0001.png



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
    tr.add_argument("--resume", default=None, help="path to an existing model.pt to continue training")


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

    iw = sub.add_parser("infer-warp", help="infer from an 800×800 warp")
    iw.add_argument("--model", required=True)
    iw.add_argument("--board-img", required=True)


    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if args.cmd == "preprocess":
        preprocess_dataset(args.src_dir, args.dst_dir, args.workers)
        return

    if args.cmd == "preview":
        preview_board(args.image, args.out, show=args.show)
        return

    if args.cmd == "train":
        if args.cmd == "train":
            if args.preprocessed:
                ds = CropFolderDataset(args.data_dir, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]))
            else:
                ds = ChessSquareDataset(args.data_dir)  # <-- this now returns PIL or add ToPILImage

            val_len  = int(len(ds) * args.val_split)
            train_len = len(ds) - val_len
            train_set, val_set = random_split(ds, [train_len, val_len])

            tl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
            vl = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            model = build_model().to(device)

            # >>> LOAD EXISTING WEIGHTS HERE <<<
            if args.resume:
                sd = torch.load(args.resume, map_location=device)
                # handle both "state_dict" checkpoints and raw state_dict
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=True)
                print(f"Resumed weights from {args.resume}")

            criterion = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            best = 0.0

            for epoch in range(1, args.epochs + 1):
                tloss, tacc = epoch_loop(model, tl, criterion, opt, device, True)
                vloss, vacc = epoch_loop(model, vl, criterion, opt, device, False)
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
    
    # -------------------------------------------------- plain JSON-based infer
    if args.cmd == "infer":
        ...

    # -------------------------------------------------- ★ direct warp infer

    if args.cmd == "infer-warp":
        model = build_model().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        board = infer_warp_board(model, args.board_img, device)
        print(board)               # numeric
        print()                    # pretty
        print(ascii_occ(board))
        return
    




if __name__ == "__main__":
    main()


# python chess_square_classifier2.py train \
#        --data-dir data/ \
#        --epochs   5 \
#        --batch-size 256 \
#        --lr 1e-3 \
#        --model-out ../assets/model.pt
