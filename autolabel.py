# autolabel.py
import cv2 as cv
import numpy as np

def _hsv_feat(patch):
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    m = hsv.reshape(-1,3).mean(0); v = hsv.reshape(-1,3).std(0)
    return np.hstack([m, v]).astype(np.float32)

def learn_proto(empty_warp, piece_warp):
    e = cv.cvtColor(empty_warp, cv.COLOR_BGR2GRAY)
    p = cv.cvtColor(piece_warp, cv.COLOR_BGR2GRAY)
    diff = cv.absdiff(e, p)
    s = diff.shape[0] // 8
    # find most changed cell (where you placed the demo piece)
    scores = [diff[r*s:(r+1)*s, c*s:(c+1)*s].mean() for r in range(8) for c in range(8)]
    k = int(np.argmax(scores)); r, c = divmod(k, 8)
    cell = piece_warp[r*s:(r+1)*s, c*s:(c+1)*s]
    h = s // 4; ctr = cell[s//2-h:s//2+h, s//2-h:s//2+h]
    return _hsv_feat(ctr)

def auto_label(empty_warp, frame_warp, proto_w, proto_b, occ_thresh=12.0):
    e = cv.GaussianBlur(cv.cvtColor(empty_warp, cv.COLOR_BGR2GRAY), (5,5), 0)
    f = cv.GaussianBlur(cv.cvtColor(frame_warp, cv.COLOR_BGR2GRAY), (5,5), 0)
    diff = cv.absdiff(e, f)
    s = diff.shape[0] // 8
    labels = []
    for r in range(8):
        for c in range(8):
            dcell = diff[r*s:(r+1)*s, c*s:(c+1)*s].mean()
            if dcell < occ_thresh:
                labels.append("E"); continue
            cell = frame_warp[r*s:(r+1)*s, c*s:(c+1)*s]
            h = s // 4; ctr = cell[s//2-h:s//2+h, s//2-h:s//2+h]
            feat = _hsv_feat(ctr)
            pw = np.linalg.norm(feat - proto_w); pb = np.linalg.norm(feat - proto_b)
            labels.append("W" if pw < pb else "B")
    return labels  # 64
