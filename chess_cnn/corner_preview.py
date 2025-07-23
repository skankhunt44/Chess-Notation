# corner_preview.py
# -----------------
# Visualise the four board-corner points stored in a Kaggle Synthetic-Chess
# JSON.  Colours: TL=red, TR=green, BR=blue, BL=yellow.  A white quadrilateral
# joins the four.

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main(img_path: str) -> None:
    img_path = Path(img_path)
    with open(img_path.with_suffix(".json")) as f:
        meta = json.load(f)

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Dataset order is [BL, BR, TL, TR]
    bl, br, tl, tr = meta["corners"]

    points = [
        (int(tl[0] * w), int(tl[1] * h)),  # TL
        (int(tr[0] * w), int(tr[1] * h)),  # TR
        (int(br[0] * w), int(br[1] * h)),  # BR
        (int(bl[0] * w), int(bl[1] * h)),  # BL
    ]

    # Draw the four points
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for p, c in zip(points, colors):
        cv2.circle(img, p, 10, c, -1)

    # Outline the quadrilateral
    cv2.polylines(img, [np.array(points, dtype=np.int32)], True, (255, 255, 255), 3)

    # Show
    cv2.imshow("Corners", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="render (.png / .jpg) that has a matching .json")
    main(parser.parse_args().image)





# python corner_preview.py data/123.png
