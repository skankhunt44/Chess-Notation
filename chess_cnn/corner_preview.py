# corner_preview.py  —  works with the new dataset
# Show the four board-corner points using matplotlib.
# Colours: TL = red, TR = green, BR = blue, BL = yellow.

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main(img_path: str) -> None:
    img_path = Path(img_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    # New dataset order is [TR, BR, BL, TL] with **pixel** coordinates
    tr, br, bl, tl = meta["corners"]

    # Re-order to TL, TR, BR, BL for plotting
    points = np.array([tl, tr, br, bl], dtype=np.int32)

    colours = ["red", "green", "blue", "yellow"]
    labels  = ["TL", "TR", "BR", "BL"]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    # connect the four points
    plt.plot(*zip(*(points.tolist() + [points[0].tolist()])), color="white", linewidth=2)
    # draw coloured dots and annotate
    for (x, y), c, lab in zip(points, colours, labels):
        plt.scatter(x, y, c=c, s=80)
        plt.text(x + 5, y - 5, lab, color=c, fontsize=9, weight="bold")
    plt.axis("off")
    plt.title("Board corners (TL, TR, BR, BL)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="render (.png / .jpg) that has a matching .json")
    main(parser.parse_args().image)





# python corner_preview.py data/0046.png --show
