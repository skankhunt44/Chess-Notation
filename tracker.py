import numpy as np
import chess
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Helper – map between (row, col) indices and algebraic square names ("e4", …)
# row 0 == rank 8, col 0 == file "a"
# ---------------------------------------------------------------------------
_square_names = np.array([[f"{file}{rank}" for file in "abcdefgh"]
                          for rank in range(8, 0, -1)])


def _board_to_occ(board: chess.Board) -> np.ndarray:
    """Convert a python‑chess Board into an 8 × 8 occupancy matrix.

    Values: 0 = empty, 1 = white piece, 2 = black piece.
    Row 0 corresponds to rank 8, column 0 to file *a* (top‑left from White’s
    perspective – the conventional computer‑vision orientation after warping).
    """
    occ = np.zeros((8, 8), np.int8)
    for square, piece in board.piece_map().items():
        rank = 7 - chess.square_rank(square)  # 0‑indexed row (rank 8 at top)
        file = chess.square_file(square)      # 0‑indexed col (file a at left)
        occ[rank, file] = 1 if piece.color == chess.WHITE else 2
    return occ

@dataclass
class GameState:
    game_over: bool = False
    # checks / mates
    check: bool = False
    checkmate: bool = False
    # automatic draws
    stalemate: bool = False
    insufficient_material: bool = False
    fivefold_repetition: bool = False         # auto
    seventyfive_moves: bool = False           # auto
    # claimable draws
    threefold_claimable: bool = False
    fifty_moves_claimable: bool = False
    # summary/result
    result: str | None = None                 # "1-0", "0-1", "1/2-1/2"
    termination: str | None = None            # e.g. "CHECKMATE", "DRAW_AGREEMENT"
    winner: str | None = None                 # "white" | "black" | None

class SquareTracker:
    def __init__(self, fen: str | None = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.prev = _board_to_occ(self.board)
        self.history: list[str] = []
        self._forced_result: dict | None = None

    def reset(self, fen: str | None = None):
        self.board.set_fen(fen) if fen else self.board.reset()
        self.prev = _board_to_occ(self.board)
        self.history.clear()
        self._forced_result = None

    def get_history(self): return self.history.copy()

    # --- new: force results for resignation/draws ---
    def _force_result(self, *, result: str, termination: str, winner: str | None):
        self._forced_result = {"result": result, "termination": termination, "winner": winner}

    def resign(self, color: str):
        color = color.lower()
        if color not in ("white", "black"):
            raise ValueError("color must be 'white' or 'black'")
        self._force_result(
            result="0-1" if color == "white" else "1-0",
            termination="RESIGNATION",
            winner="black" if color == "white" else "white",
        )

    def agree_draw(self):
        self._force_result(result="1/2-1/2", termination="DRAW_AGREEMENT", winner=None)

    def claim_draw(self):
        # Prefer threefold, else fifty-move, else reject
        if self.board.can_claim_threefold_repetition():
            self._force_result(result="1/2-1/2", termination="THREEFOLD_REPETITION", winner=None)
        elif self.board.can_claim_fifty_moves():
            self._force_result(result="1/2-1/2", termination="FIFTY_MOVES", winner=None)
        else:
            raise ValueError("No draw can be claimed at this position.")

    def _state(self) -> GameState:
        # Safe getattr for older python-chess versions
        fivefold = getattr(self.board, "is_fivefold_repetition", lambda: False)()
        mv75     = getattr(self.board, "is_seventyfive_moves", lambda: False)()
        st = GameState(
            check=self.board.is_check(),
            checkmate=self.board.is_checkmate(),
            stalemate=self.board.is_stalemate(),
            insufficient_material=self.board.is_insufficient_material(),
            fivefold_repetition=fivefold,
            seventyfive_moves=mv75,
            threefold_claimable=self.board.can_claim_threefold_repetition(),
            fifty_moves_claimable=self.board.can_claim_fifty_moves(),
        )
        oc = self.board.outcome(claim_draw=True)  # includes auto draws
        if oc:
            st.game_over = True
            st.result = oc.result()
            st.termination = oc.termination.name
            st.winner = "white" if oc.winner is True else "black" if oc.winner is False else None
        if self._forced_result:  # overrides (agreement / claim / resign)
            st.game_over = True
            st.result = self._forced_result["result"]
            st.termination = self._forced_result["termination"]
            st.winner = self._forced_result["winner"]
        return st

    def update(self, new_occ: np.ndarray):
        if np.array_equal(new_occ, self.prev):
            return None, None, asdict(self._state())

        candidate = None
        tested = set()
        for mv in self.board.legal_moves:
            norm = mv
            if self._is_pawn_promotion(mv):
                norm = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            key = (norm.from_square, norm.to_square, norm.promotion or 0)
            if key in tested: continue
            tested.add(key)

            nb = self.board.copy(stack=False)
            nb.push(norm)
            if np.array_equal(_board_to_occ(nb), new_occ):
                if candidate is not None:
                    return None, None, asdict(self._state())
                candidate = norm

        if candidate is None:
            return None, None, asdict(self._state())

        san_before = self.board.san(candidate)
        self.board.push(candidate)
        self.history.append(san_before)
        self.prev = new_occ.copy()
        return candidate.uci(), self.board.fen(), asdict(self._state())

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_pawn_promotion(move: chess.Move) -> bool:
        """True if *move* is a pawn move that reaches the back rank."""
        return (
            chess.square_rank(move.from_square) in (6, 1) and  # rank 7 or 2 (0‑idx)
            chess.square_rank(move.to_square) in (7, 0)        # promotion rank
        )
