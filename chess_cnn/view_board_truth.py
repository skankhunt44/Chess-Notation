import argparse, json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from chess_square_classifier2 import compute_homography, warp_board
from orientation_utils import needs_flip_180

IMG_SIZE = 800
CELL     = IMG_SIZE // 8

def square_name(r, c): return f"{chr(ord('A')+c)}{8-r}"

def make_cfg(meta):
    if "config" in meta:
        return meta["config"]
    if "pieces" in meta:
        return {p["square"].upper(): p["piece"] for p in meta["pieces"]}
    # derive from FEN
    cfg = {}
    rows = meta["fen"].split()[0].split("/")
    for r,row in enumerate(rows):
        f=0
        for ch in row:
            if ch.isdigit(): f += int(ch)
            else: cfg[f"{chr(ord('A')+f)}{8-r}"] = ch; f += 1
    return cfg

def label(val): return "white" if (val.isupper() or val.endswith("white")) else "black"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    img_p = Path(args.image)
    meta  = json.load(img_p.with_suffix(".json").open())
    img   = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)

    H     = compute_homography(meta["corners"], img.shape[1], img.shape[0], IMG_SIZE)
    brd   = warp_board(img, H, IMG_SIZE)

    flip  = needs_flip_180(meta, H)

    cfg   = make_cfg(meta)

    for r in range(8):
        for c in range(8):
            rr, cc = (7-r, 7-c) if flip else (r, c)
            sq     = square_name(rr, cc)
            txt    = label(cfg[sq]) if sq in cfg else "empty"

            y, x = r*CELL+4, c*CELL+4
            cv2.rectangle(brd, (x-2,y-10), (x+40,y+6), (0,0,0), -1)
            cv2.putText(brd, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (255,255,255), 1, cv2.LINE_AA)

    for i in range(9):                                   # grid
        cv2.line(brd, (0,i*CELL), (IMG_SIZE,i*CELL), (0,255,0),1)
        cv2.line(brd, (i*CELL,0), (i*CELL,IMG_SIZE), (0,255,0),1)

    plt.figure(figsize=(6,6)); plt.imshow(brd); plt.axis("off"); plt.show()

if __name__ == "__main__":
    main()



# python view_board_truth.py --image data/0046.png