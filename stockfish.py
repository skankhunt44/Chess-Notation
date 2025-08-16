# --- add imports at top ---
import os
import chess
import chess.pgn
import chess.engine
import time
from chess.pgn import NAG_GOOD_MOVE

# brew install stockfish
# export STOCKFISH_PATH=stockfish   # or absolute path, e.g. /usr/local/bin/stockfish

ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")

def _score_to_cp(score: chess.engine.PovScore) -> tuple[float | None, int | None]:
    """Return (cp_in_pawns, mate_in) from a PovScore; cp is in pawns (e.g., 0.35)."""
    if score.is_mate():
        return (None, score.mate())  # positive: mate for POV side, negative: mated
    return (score.score() / 100.0, None)

def _classify_delta(delta_cp: float) -> tuple[str, int | None]:
    """
    Classify by eval swing (for the mover). Thresholds roughly align with lichess:
      ≥ -0.50: 'ok' / good
      <  -0.50: inaccuracy
      <  -1.50: mistake
      <  -3.00: blunder
    Returns (tag, NAG) where NAG is chess.pgn.NAG_* or None.
    """
    if delta_cp >= -0.50:
        return ("good", None)
    if delta_cp < -3.00:
        return ("blunder", chess.pgn.NAG_BLUNDER)         # ??
    if delta_cp < -1.50:
        return ("mistake", chess.pgn.NAG_MISTAKE)         # ?
    return ("inaccuracy", chess.pgn.NAG_DUBIOUS_MOVE)          # ?!

def _analyze_moves_with_engine(
    san_moves: list[str],
    depth: int | None = 14,
    movetime_ms: int | None = None,
    engine_path: str = ENGINE_PATH,
) -> dict:
    limit = chess.engine.Limit(depth=depth) if movetime_ms is None else chess.engine.Limit(time=movetime_ms/1000.0)
    board = chess.Board()

    # Prepare PGN scaffold
    game = chess.pgn.Game()
    game.headers["Event"] = "Phone Chess Tracker"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = "-"
    game.headers["White"] = "White"
    game.headers["Black"] = "Black"

    node = game
    results = []
    # ACPL and counts
    sum_loss = {True: 0.0, False: 0.0}   # True: White, False: Black
    cnt_loss = {True: 0,   False: 0}
    counts   = {True: {"best":0, "inaccuracy":0, "mistake":0, "blunder":0},
                False: {"best":0, "inaccuracy":0, "mistake":0, "blunder":0}}

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for idx, san in enumerate(san_moves, start=1):
            mover = board.turn
            board_before = board.copy()

            raw_before = engine.analyse(board_before, limit, multipv=1)
            info_before = raw_before[0] if isinstance(raw_before, list) else raw_before

            best_before = None
            if "pv" in info_before and info_before["pv"]:
                best_before = info_before["pv"][0]

            eval_before_cp, eval_before_mate = _score_to_cp(info_before["score"].pov(mover))

            # The move actually played
            try:
                move = board.parse_san(san)
            except Exception as e:
                raise ValueError(f"Bad SAN at ply {idx}: {san} ({e})")

            # Is it the engine's best?
            is_best = (best_before is not None and move == best_before)

            board.push(move)

            raw_after  = engine.analyse(board, limit, multipv=1)
            info_after = raw_after[0] if isinstance(raw_after, list) else raw_after
            eval_after_cp, eval_after_mate = _score_to_cp(info_after["score"].pov(not board.turn))

            delta_cp = None
            if eval_before_cp is not None and eval_after_cp is not None:
                delta_cp = round((eval_after_cp - eval_before_cp), 2)

            # Tag + NAG
            if is_best:
                tag, nag = "best", NAG_GOOD_MOVE       # "!"
                counts[mover]["best"] += 1
                loss = 0.0
            else:
                tag, nag = _classify_delta(delta_cp) if delta_cp is not None else ("good", None)
                # ACPL/loss bookkeeping only for non-best (or best with negative delta, but that's rare)
                if delta_cp is not None:
                    loss = max(0.0, -delta_cp)
                    sum_loss[mover] += loss
                    cnt_loss[mover] += 1
                    if tag in ("inaccuracy", "mistake", "blunder"):
                        counts[mover][tag] += 1

            # PGN annotate
            node = node.add_variation(move)

            best_san = None
            if best_before is not None:
                try:
                    best_san = board_before.san(best_before)
                except Exception:
                    best_san = best_before.uci()

            if eval_after_mate is not None:
                cm = f"Eval: mate in {eval_after_mate:+d}"
            else:
                cm = f"Eval: {eval_after_cp:+.2f}" if eval_after_cp is not None else "Eval: (n/a)"
            if delta_cp is not None:
                cm += f" | Δ {delta_cp:+.2f}"
            if best_san:
                cm += f" | Best: {best_san}"
            if tag == "best":
                cm += " | BEST"
            elif tag != "good":
                cm += f" | {tag.upper()}"

            node.comment = cm
            if nag is not None:
                node.nags.add(nag)

            results.append({
                "ply": idx,
                "move": san,
                "side": "White" if mover else "Black",
                "eval_after": None if eval_after_cp is None else round(eval_after_cp, 2),
                "mate_after": eval_after_mate,
                "delta": delta_cp,
                "best": best_san,
                "tag": tag
            })
    # with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
    #     for idx, san in enumerate(san_moves, start=1):
    #         mover = board.turn         # side to move before applying SAN
    #         board_before = board.copy()

    #         # Analysis BEFORE the move to get best line and eval reference
    #         raw_before = engine.analyse(board_before, limit, multipv=1)
    #         info_before = raw_before[0] if isinstance(raw_before, list) else raw_before
    #         best_before = None
    #         if "pv" in info_before and info_before["pv"]:
    #             best_before = info_before["pv"][0]
    #         eval_before_cp, eval_before_mate = _score_to_cp(info_before["score"].pov(mover))

    #         # Apply the actual move
    #         try:
    #             move = board.parse_san(san)
    #         except Exception as e:
    #             raise ValueError(f"Bad SAN at ply {idx}: {san} ({e})")
    #         board.push(move)

    #         # Analysis AFTER the move, POV = mover (who just played)
    #         raw_after = engine.analyse(board, limit, multipv=1)
    #         info_after = raw_after[0] if isinstance(raw_after, list) else raw_after
    #         eval_after_cp, eval_after_mate = _score_to_cp(info_after["score"].pov(not board.turn))

    #         # Delta for mover (positive = improved, negative = got worse)
    #         delta_cp = None
    #         tag = "good"
    #         nag = None
    #         if eval_before_cp is not None and eval_after_cp is not None:
    #             delta_cp = round((eval_after_cp - eval_before_cp), 2)
    #             tag, nag = _classify_delta(delta_cp)
    #             # Loss is negative swing only
    #             loss = max(0.0, -delta_cp)
    #             sum_loss[mover] += loss
    #             cnt_loss[mover] += 1
    #             if tag in ("inaccuracy", "mistake", "blunder"):
    #                 counts[mover][tag] += 1

    #         # PGN annotate this move
    #         node = node.add_variation(move)
    #         # Best move SAN (from the *before* position)
    #         best_san = None
    #         if best_before is not None:
    #             try:
    #                 best_san = board_before.san(best_before)
    #             except Exception:
    #                 best_san = best_before.uci()

    #         # Compose a human-readable comment
    #         if eval_after_mate is not None:
    #             cm = f"Eval: mate in {eval_after_mate:+d}"
    #         else:
    #             cm = f"Eval: {eval_after_cp:+.2f}" if eval_after_cp is not None else "Eval: (n/a)"
    #         if delta_cp is not None:
    #             cm += f" | Δ {delta_cp:+.2f}"
    #         if best_san:
    #             cm += f" | Best: {best_san}"
    #         if tag != "good":
    #             cm += f" | {tag.upper()}"

    #         node.comment = cm
    #         if nag is not None:
    #             node.nags.add(nag)

    #         results.append({
    #             "ply": idx,
    #             "move": san,
    #             "side": "White" if mover else "Black",
    #             "eval_after": None if eval_after_cp is None else round(eval_after_cp, 2),
    #             "mate_after": eval_after_mate,                   # e.g. +3 means mate in 3 for mover
    #             "delta": delta_cp,
    #             "best": best_san,
    #             "tag": tag
    #         })

    # Finish PGN result if game ended
    if board.is_game_over():
        game.headers["Result"] = board.result()
    else:
        game.headers["Result"] = "*"

    # ACPL
    acpl_white = round(sum_loss[True] / max(1, cnt_loss[True]) * 100, 1)
    acpl_black = round(sum_loss[False] / max(1, cnt_loss[False]) * 100, 1)

    # Dump PGN string
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_str = game.accept(exporter)

    return {
        "summary": {
            "acpl_white": acpl_white,
            "acpl_black": acpl_black,
            "white": counts[True],
            "black": counts[False],
            "result": game.headers.get("Result", "*"),
        },
        "moves": results,
        "pgn": pgn_str
    }

