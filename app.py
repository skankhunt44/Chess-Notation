from fastapi import FastAPI, WebSocket
import base64, cv2, numpy as np
from vision import find_board, warp_board, occupancy_hsv
from tracker import SquareTracker

app      = FastAPI()
tracker  = SquareTracker()
H        = None                   # homography

@app.post("/calibrate/")
async def calibrate(image_b64: str):
    frame = decode(image_b64)
    corners = find_board(frame)
    global H
    H = corners
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        frame = decode(data)

        if H is None:
            await ws.send_json({"err": "Not calibrated"})
            continue

        board_img = warp_board(frame, H)[0]
        occ = occupancy_hsv(board_img)
        move, fen = tracker.update(occ)

        if move:
            await ws.send_json({"move": move, "fen": fen})
        else:
            await ws.send_json({"noop": True})

def decode(b64):
    buf = base64.b64decode(b64.split(",")[-1])
    img = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)
