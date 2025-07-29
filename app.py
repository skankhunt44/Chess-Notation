from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
import base64
import cv2 as cv
import numpy as np

from vision import find_board, warp_board, occupancy_cnn, draw_board_overlay, ascii_occ
from tracker import SquareTracker

from fastapi.staticfiles import StaticFiles


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
        dbg = draw_board_overlay(frame, corners)
        cv.imwrite("calib_fail.jpg", dbg)
        raise HTTPException(400, "Board not found")
    dbg = draw_board_overlay(frame, corners)
    cv.imwrite("calib_fail.jpg", dbg)
    _corners = corners
    return {"ok": True}

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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
