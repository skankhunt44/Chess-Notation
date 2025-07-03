import numpy as np
import chess

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


class SquareTracker:
    """Track moves by *occupancy‑only* board diffs.

    Initialise with the standard starting position (or any FEN).  On every
    call to :pymeth:`update`, feed the newly detected occupancy matrix.
    The method tries to find *exactly one* legal move whose resulting board
    matches the new occupancy.  If found, that move is pushed and returned.

    The tracker assumes **queen promotion** when multiple legal promotions
    would otherwise produce identical occupancy (empty/white/black only).
    """

    def __init__(self, fen: str | None = None):
        self.board: chess.Board = chess.Board(fen) if fen else chess.Board()
        self.prev: np.ndarray = _board_to_occ(self.board)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def reset(self, fen: str | None = None) -> None:
        """Set a fresh game state and sync the occupancy baseline."""
        self.board.set_fen(fen) if fen else self.board.reset()
        self.prev = _board_to_occ(self.board)

    # ---------------------------------------------------------------------
    # Core – consume a new 8 × 8 occupancy matrix
    # ---------------------------------------------------------------------
    def update(self, new_occ: np.ndarray):
        """Process *one* frame.

        Returns
        -------
        tuple | (None, None)
            ``(uci, fen)`` of the recognised move, or ``(None, None)`` if no
            single legal move explains the change (noise frame or ambiguous).
        """
        # Fast‑path: identical occupancy → nothing happened.
        if np.array_equal(new_occ, self.prev):
            return None, None

        # ------------------------------------------------------------------
        # Brute‑force search: try every legal move & compare the resulting
        # occupancy.  Works for captures, promotions, castling, en‑passant,
        # and disambiguates multiple changed squares naturally.
        # ------------------------------------------------------------------
        candidate = None
        for move in self.board.legal_moves:
            # Assume queen on promotion by default (python‑chess uses None → Q)
            if move.promotion is None and self._is_pawn_promotion(move):
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

            next_board = self.board.copy(stack=False)
            next_board.push(move)
            if np.array_equal(_board_to_occ(next_board), new_occ):
                if candidate is not None:
                    # Ambiguous – two moves create identical occupancy with only
                    # white/black info (rare but possible with under‑promotion).
                    return None, None
                candidate = move

        if candidate is None:
            # No legal move matches → likely CV noise, ignore frame.
            return None, None

        # Push the single matching move.
        san_before = self.board.san(candidate)  # for UI if needed
        self.board.push(candidate)
        self.prev = new_occ.copy()
        return candidate.uci(), self.board.fen()

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
