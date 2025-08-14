from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
import base64
import cv2 as cv
import numpy as np

from vision import find_board, warp_board, occupancy_cnn, draw_board_overlay, ascii_occ
from tracker import SquareTracker

from fastapi.staticfiles import StaticFiles

import json, os
from pathlib import Path
from autolabel import learn_proto, auto_label
import time


app = FastAPI()
# move to the end of the file
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

tracker = SquareTracker()

# Will hold the calibrated 4×2 corner array after /calibrate runs.
_corners: np.ndarray | None = None

# ---------------------------------------------------------------------------
# Utility – base64‑>numpy image
# ---------------------------------------------------------------------------

def _decode_image(data_url: str) -> np.ndarray:
    """Decode a data‑URL or raw base64 string to a BGR (OpenCV) image."""
    if "," in data_url:  # strip leading "data:image/jpeg;base64," etc.
        data_url = data_url.split(",", 1)[1]
    buf = base64.b64decode(data_url)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

# ---------------------------------------------------------------------------
# HTTP endpoint – one‑shot board calibration
# ---------------------------------------------------------------------------


@app.post("/calibrate")
async def calibrate(image_b64: str = Body(..., media_type="text/plain")):
    global _corners
    frame = _decode_image(image_b64)
    corners = find_board(frame)
    if corners is None:
        cv.imwrite("calib_dbg_fail.jpg", frame)
        raise HTTPException(400, "Board not found")

    cv.imwrite("calib_dbg_ok.jpg", draw_board_overlay(frame, corners))
    _corners = corners

    # NEW: every calibration -> new session folder
    sd = _start_new_session()
    # save corners for provenance
    np.save(sd / "corners.npy", np.array(_corners, dtype=np.float32))
    return {"ok": True, "session": str(sd)}



# ---------------------------------------------------------------------------
# HTTP endpoint – current move history
# ---------------------------------------------------------------------------

@app.get("/history")
async def get_history():
    """Return the list of moves seen so far in SAN."""
    return {"moves": tracker.get_history()}

# ---------------------------------------------------------------------------
# WebSocket – streaming frames → moves / FEN
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                frame = _decode_image(data)
            except ValueError:
                await ws.send_json({"err": "Bad image data"})
                continue

            if _corners is None:
                await ws.send_json({"err": "Not calibrated"})
                continue

            cv.imwrite("calib_ok.jpg", frame) # the raw camera frame
            board_img, _ = warp_board(frame, _corners)

            cv.imwrite("calib_warp.jpg", board_img) # the rectified 800 × 800 board
            occ = occupancy_cnn(board_img)

            print(ascii_occ(occ))
            
            move, fen = tracker.update(occ)

            if move:
                await ws.send_json({"move": move, "fen": fen, "history": tracker.get_history()})
            else:
                await ws.send_json({"noop": True})
    except WebSocketDisconnect:
        # Normal disconnect; nothing special to clean up for now.
        pass



STATE = {"H": None, "empty": None, "protoW": None, "protoB": None, "session_dir": None, "nW": 0, "nB": 0}

def _start_new_session():
    ts = int(time.time())
    sd = Path("data") / f"session_{ts}"
    (sd / "samples").mkdir(parents=True, exist_ok=True)
    STATE["session_dir"] = sd
    return sd

def _ensure_session_dir():
    return STATE["session_dir"] or _start_new_session()

@app.post("/capture-empty")
async def capture_empty(image_b64: str = Body(..., media_type="text/plain")):
    if _corners is None:
        raise HTTPException(400, "Not calibrated")
    frame = _decode_image(image_b64)
    board_img, _ = warp_board(frame, _corners)
    STATE["empty"] = board_img
    sd = _ensure_session_dir()
    cv.imwrite(str(sd / "empty.png"), board_img)
    return {"ok": True}


# @app.post("/teach-color/{which}")
# async def teach_color(which: str, image_b64: str = Body(..., media_type="text/plain")):
#     if STATE["empty"] is None:
#         raise HTTPException(400, "Capture empty first")
#     frame = _decode_image(image_b64)
#     board_img, _ = warp_board(frame, _corners)
#     proto = learn_proto(STATE["empty"], board_img)
#     if which.lower() == "white":
#         STATE["protoW"] = proto
#     else:
#         STATE["protoB"] = proto
#     return {"ok": True}



def _ema(old, new, n, alpha=0.3):
    """Exponential moving average; if no old value, return new."""
    if old is None or n == 0:
        return new.astype(np.float32)
    return ((1 - alpha) * old + alpha * new).astype(np.float32)

@app.post("/teach-color/{which}")
async def teach_color(which: str, image_b64: str = Body(..., media_type="text/plain")):
    if STATE["empty"] is None:
        raise HTTPException(400, "Capture empty first")
    frame = _decode_image(image_b64)
    board_img, _ = warp_board(frame, _corners)
    feat = learn_proto(STATE["empty"], board_img).astype(np.float32)

    if which.lower() == "white":
        STATE["protoW"] = _ema(STATE["protoW"], feat, STATE["nW"], alpha=0.35)
        STATE["nW"] += 1
        return {"ok": True, "n_white": STATE["nW"]}
    else:
        STATE["protoB"] = _ema(STATE["protoB"], feat, STATE["nB"], alpha=0.35)
        STATE["nB"] += 1
        return {"ok": True, "n_black": STATE["nB"]}


@app.post("/propose-labels")
async def propose_labels(image_b64: str = Body(..., media_type="text/plain")):
    for k in ("empty","protoW","protoB"):
        if STATE[k] is None:
            raise HTTPException(400, f"Missing {k} – run the previous steps")
    frame = _decode_image(image_b64)
    board_img, _ = warp_board(frame, _corners)
    labels = auto_label(STATE["empty"], board_img, STATE["protoW"], STATE["protoB"])
    _, png = cv.imencode(".png", board_img)
    return {"labels": labels, "warp_png": base64.b64encode(png).decode()}

@app.post("/save-sample")
async def save_sample(payload: dict = Body(...)):
    labels = payload.get("labels")
    image_b64 = payload.get("image_b64")
    if not labels or not image_b64:
        raise HTTPException(400, "Need labels and image_b64")

    sd = _ensure_session_dir()  # <- reuse unless recalibrated
    samples_dir = sd / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # browser sends the 800x800 warp; do NOT re-warp
    img = _decode_image(image_b64)
    next_idx = len(list(samples_dir.glob("img_*.png"))) + 1
    img_path = samples_dir / f"img_{next_idx:04d}.png"
    cv.imwrite(str(img_path), img)

    # keep corners in the session root (overwrite with latest if needed)
    np.save(sd / "corners.npy", np.array(_corners, dtype=np.float32))

    (samples_dir / f"labels_{next_idx:04d}.json").write_text(
        json.dumps({"image": img_path.name, "grid_size": 8, "labels": labels}, indent=2)
    )
    return {"ok": True, "saved": str(img_path), "session": str(sd)}



app.mount("/", StaticFiles(directory="static", html=True), name="static")
