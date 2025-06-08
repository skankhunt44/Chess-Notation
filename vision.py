import cv2 as cv
import numpy as np

CALIB_SIZE = 800  # warp board → 800×800 px

def find_board(frame):
    """Locate outer board contour and return 4 ordered corners."""
    gray   = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges  = cv.Canny(gray, 50, 150, apertureSize=3)
    lines  = cv.HoughLinesP(edges, 1, np.pi/180, 150,
                            minLineLength=200, maxLineGap=20)

    # cluster detected line endpoints; pick largest rectangle
    # (quick heuristic – robust enough for clear boards)
    pts = np.concatenate([[l[0][:2], l[0][2:]] for l in lines])
    rect = cv.minAreaRect(pts)
    box  = cv.boxPoints(rect)          # 4 corners, float32
    box  = np.array(sorted(box, key=lambda p: p[0] + p[1]))  # TL, TR, BR, BL order

    return box.astype(np.float32)

def warp_board(frame, corners):
    dst = np.array([[0,0], [CALIB_SIZE,0],
                    [CALIB_SIZE,CALIB_SIZE], [0,CALIB_SIZE]], np.float32)
    M   = cv.getPerspectiveTransform(corners, dst)
    return cv.warpPerspective(frame, M, (CALIB_SIZE, CALIB_SIZE)), M


WHITE_LOWER = (  0,   0,180)   # tweak for your set / lighting
WHITE_UPPER = (180,  50,255)
BLACK_LOWER = (  0,   0,  0)
BLACK_UPPER = (180,255, 60)

def occupancy_hsv(board_img):
    hsv = cv.cvtColor(board_img, cv.COLOR_BGR2HSV)

    white_mask = cv.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    black_mask = cv.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    # clean up small speckles
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    black_mask = cv.morphologyEx(black_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    occ = np.zeros((8,8), np.int8)               # 0 empty, 1 white, 2 black
    sq   = CALIB_SIZE // 8

    for r in range(8):
        for c in range(8):
            y0, y1 = r*sq, (r+1)*sq
            x0, x1 = c*sq, (c+1)*sq

            w_pixels = white_mask[y0:y1, x0:x1].mean() / 255
            b_pixels = black_mask[y0:y1, x0:x1].mean() / 255

            if w_pixels > 0.12:       occ[r,c] = 1
            elif b_pixels > 0.12:     occ[r,c] = 2

    return occ
