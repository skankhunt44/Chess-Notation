from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from vision import find_board, warp_board, occupancy_cnn, draw_board_overlay, ascii_occ
from tracker import SquareTracker
from stockfish import _analyze_moves_with_engine
# from autolabel import learn_proto, auto_label

import asyncio
from dataclasses import asdict

import chess
import chess.engine

# --------------------------------------------------------------------------------------
# App & globals
# --------------------------------------------------------------------------------------

app = FastAPI()
tracker = SquareTracker()

# Calibrated 4x2 corner array (np.ndarray with dtype float32), set by /calibrate
_corners: Optional[np.ndarray] = None

STATE = {
    "H": None,
    "empty": None,
    "protoW": None,
    "protoB": None,
    "session_dir": None,
    "nW": 0,
    "nB": 0,
}

# Debug dump rate limiter
_last_dump = 0.0
DEBUG_DUMP_EVERY_S = 2.0

# WS config
RECEIVE_TIMEOUT_S = 60
HEARTBEAT_EVERY_S = 25

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _decode_image(data_url: str) -> np.ndarray:
    """Decode a data-URL or raw base64 string to a BGR (OpenCV) image."""
    if "," in data_url:  # strip leading "data:image/jpeg;base64," etc.
        data_url = data_url.split(",", 1)[1]
    buf = base64.b64decode(data_url)
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _start_new_session() -> Path:
    ts = int(time.time())
    sd = Path("data") / f"session_{ts}"
    (sd / "samples").mkdir(parents=True, exist_ok=True)
    STATE["session_dir"] = sd
    return sd


def _ensure_session_dir() -> Path:
    return STATE["session_dir"] or _start_new_session()


def _ema(old: Optional[np.ndarray], new: np.ndarray, n: int, alpha: float = 0.3) -> np.ndarray:
    """Exponential moving average; if no old value, return new."""
    if old is None or n == 0:
        return new.astype(np.float32)
    return ((1 - alpha) * old + alpha * new).astype(np.float32)

# --------------------------------------------------------------------------------------
# HTTP endpoints
# --------------------------------------------------------------------------------------

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
    tracker.reset()

    sd = _start_new_session()
    np.save(sd / "corners.npy", np.array(_corners, dtype=np.float32))
    return {"ok": True, "session": str(sd)}


@app.get("/history")
async def get_history():
    """Return the list of moves seen so far in SAN."""
    return {"moves": tracker.get_history()}


# @app.post("/capture-empty")
# async def capture_empty(image_b64: str = Body(..., media_type="text/plain")):
#     if _corners is None:
#         raise HTTPException(400, "Not calibrated")
#     frame = _decode_image(image_b64)
#     board_img, _ = warp_board(frame, _corners)
#     STATE["empty"] = board_img
#     sd = _ensure_session_dir()
#     cv.imwrite(str(sd / "empty.png"), board_img)
#     return {"ok": True}


# @app.post("/teach-color/{which}")
# async def teach_color(which: str, image_b64: str = Body(..., media_type="text/plain")):
#     if STATE["empty"] is None:
#         raise HTTPException(400, "Capture empty first")
#     frame = _decode_image(image_b64)
#     board_img, _ = warp_board(frame, _corners)
#     feat = learn_proto(STATE["empty"], board_img).astype(np.float32)

#     if which.lower() == "white":
#         STATE["protoW"] = _ema(STATE["protoW"], feat, STATE["nW"], alpha=0.35)
#         STATE["nW"] += 1
#         return {"ok": True, "n_white": STATE["nW"]}
#     else:
#         STATE["protoB"] = _ema(STATE["protoB"], feat, STATE["nB"], alpha=0.35)
#         STATE["nB"] += 1
#         return {"ok": True, "n_black": STATE["nB"]}


# @app.post("/propose-labels")
# async def propose_labels(image_b64: str = Body(..., media_type="text/plain")):
#     for k in ("empty", "protoW", "protoB"):
#         if STATE[k] is None:
#             raise HTTPException(400, f"Missing {k} – run the previous steps")
#     frame = _decode_image(image_b64)
#     board_img, _ = warp_board(frame, _corners)
#     labels = auto_label(STATE["empty"], board_img, STATE["protoW"], STATE["protoB"])
#     _, png = cv.imencode(".png", board_img)
#     return {"labels": labels, "warp_png": base64.b64encode(png).decode()}


# @app.post("/save-sample")
# async def save_sample(payload: dict = Body(...)):
#     labels = payload.get("labels")
#     image_b64 = payload.get("image_b64")
#     if not labels or not image_b64:
#         raise HTTPException(400, "Need labels and image_b64")

#     sd = _ensure_session_dir()
#     samples_dir = sd / "samples"
#     samples_dir.mkdir(parents=True, exist_ok=True)

#     img = _decode_image(image_b64)  # browser sends 800x800 warp; do NOT re-warp
#     next_idx = len(list(samples_dir.glob("img_*.png"))) + 1
#     img_path = samples_dir / f"img_{next_idx:04d}.png"
#     cv.imwrite(str(img_path), img)

#     np.save(sd / "corners.npy", np.array(_corners, dtype=np.float32))  # keep latest

#     (samples_dir / f"labels_{next_idx:04d}.json").write_text(
#         json.dumps({"image": img_path.name, "grid_size": 8, "labels": labels}, indent=2)
#     )
#     return {"ok": True, "saved": str(img_path), "session": str(sd)}


def _occ_to_labels(occ: np.ndarray) -> list[str]:
    """
    Map your CNN’s occupancy to ['E','W','B'] in row-major order.
    Assumes occ is (8,8) with 0/1/2 = E/W/B. Adjust if needed.
    """
    if occ.ndim == 3 and occ.shape[-1] in (3,):
        # e.g. per-class logits/probs → argmax
        occ = occ.argmax(axis=-1)
    mapping = {0:'E', 1:'W', 2:'B'}
    flat = occ.reshape(-1)
    return [mapping[int(x)] for x in flat]

@app.post("/propose-labels")
async def propose_labels(image_b64: str = Body(..., media_type="text/plain")):
    if _corners is None:
        raise HTTPException(400, "Not calibrated")
    frame = _decode_image(image_b64)
    board_img, _ = warp_board(frame, _corners)

    # ← your trained model
    occ = occupancy_cnn(board_img)                 # (8,8) ints or (8,8,3) probs
    labels = _occ_to_labels(occ)

    _, png = cv.imencode(".png", board_img)
    return {"labels": labels, "warp_png": base64.b64encode(png).decode()}


# --------------------------------------------------------------------------------------
# WebSocket – streaming frames → moves / FEN
# --------------------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    async def keepalive():
        while True:
            await asyncio.sleep(HEARTBEAT_EVERY_S)
            try:
                await ws.send_json({"type": "ping", "t": time.time()})
            except Exception:
                break

    ka_task = asyncio.create_task(keepalive())

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=RECEIVE_TIMEOUT_S)
            except asyncio.TimeoutError:
                await ws.close(code=1001)
                break

            if data.strip() in ("PING", '{"type":"pong"}'):
                continue

            # Decode image
            try:
                frame = _decode_image(data)
            except ValueError:
                await ws.send_json({"err": "Bad image data"})
                continue

            if _corners is None:
                await ws.send_json({"err": "Not calibrated"})
                continue

            def _process():
                t0 = time.perf_counter()
                board_img, _ = warp_board(frame, _corners)
                occ = occupancy_cnn(board_img)
                move, fen, state = tracker.update(occ)   # ← unpack 3
                ms = int((time.perf_counter() - t0) * 1000)

                # (optional debug)
                log_every = float(os.getenv("LOG_EVERY", "1"))
                if log_every > 0:
                    now = time.time()
                    if int(now / log_every) != int((now - 0.05) / log_every):
                        print(ascii_occ(occ))

                return move, fen, state, ms

            try:
                move, fen, state, ms = await run_in_threadpool(_process)
            except Exception as e:
                await ws.send_json({"err": f"server error: {type(e).__name__}", "detail": str(e)})
                continue

            # Always send state so the UI can show check/claimable-draw flags
            payload = {"ms": ms, "state": state}

            if move:
                payload.update({"move": move, "fen": fen, "history": tracker.get_history()})
                print(move)
            else:
                payload["noop"] = True

            await ws.send_json(payload)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            ka_task.cancel()
        except Exception:
            pass


@app.post("/resign/{color}")
def api_resign(color: str):
    try:
        tracker.resign(color)
    except ValueError as e:
        raise HTTPException(400, str(e))
    st = tracker._state()
    return {"state": asdict(st), "fen": tracker.board.fen(), "history": tracker.get_history()}

@app.post("/draw/agree")
def api_draw_agree():
    tracker.agree_draw()
    st = tracker._state()
    return {"state": asdict(st), "fen": tracker.board.fen(), "history": tracker.get_history()}

@app.post("/draw/claim")
def api_draw_claim():
    try:
        tracker.claim_draw()
    except ValueError as e:
        raise HTTPException(400, str(e))
    st = tracker._state()
    return {"state": asdict(st), "fen": tracker.board.fen(), "history": tracker.get_history()}

@app.post("/reset")
def api_reset():
    tracker.reset()
    return {"fen": tracker.board.fen(), "history": tracker.get_history()}


ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")

@app.post("/analyze")
async def api_analyze(payload: dict = Body(default={})):
    """
    Body (all optional):
      { "moves": ["e4","e5","Nf3", ...], "depth": 14 }  OR  { "movetime_ms": 200 }
    If 'moves' not provided, uses current tracker's move list.
    """
    san_moves = payload.get("moves") or tracker.get_history()  # already SAN per your code
    if not san_moves:
        raise HTTPException(400, "No moves to analyze yet.")

    depth = payload.get("depth")
    movetime_ms = payload.get("movetime_ms")

    def _run():
        return _analyze_moves_with_engine(san_moves, depth=depth, movetime_ms=movetime_ms)

    try:
        result = await run_in_threadpool(_run)  # don’t block the event loop with engine calls
    except FileNotFoundError:
        raise HTTPException(500, f"Stockfish binary not found at '{ENGINE_PATH}'. Set STOCKFISH_PATH.")
    except chess.engine.EngineError as e:
        raise HTTPException(500, f"Engine error: {e}")

    return JSONResponse(result)


# --------------------------------------------------------------------------------------
# Static site
# --------------------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
