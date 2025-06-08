import numpy as np
import chess

square_names = np.array([
    [f"{file}{rank}" for file in "abcdefgh"]
    for rank in range(8,0,-1)     # row 0 = rank 8
])

class SquareTracker:
    def __init__(self):
        self.board  = chess.Board()        # start position
        self.prev   = self._start_matrix() # occupancy matrix

    def _start_matrix(self):
        # ranks 1-2 white pieces, 7-8 black pieces
        occ = np.zeros((8,8), np.int8)
        occ[6,:] = occ[7,:] = 1            # white pawn / pieces
        occ[0,:] = occ[1,:] = 2            # black pawn / pieces
        return occ

    def update(self, new_occ):
        diff = new_occ - self.prev
        changed = np.argwhere(diff != 0)

        if changed.size == 0:
            return None, None  # no move

        #
        # normal move → exactly two squares change
        #
        if len(changed) == 2:
            src, dst = sorted(changed, key=lambda x: diff[tuple(x)])
            from_sq  = square_names[tuple(src)]
            to_sq    = square_names[tuple(dst)]
            uci      = from_sq + to_sq

            # promotion assumption (always queen)
            if (from_sq[1] == "7" and to_sq[1] == "8") or \
               (from_sq[1] == "2" and to_sq[1] == "1"):
                uci += "q"

        #
        # castling → four squares change
        #
        elif len(changed) == 4:
            # figure out side by board colour & files
            uci = "e1g1" if self.board.turn == chess.WHITE else "e8g8"
            # naive – covers king-side only; extend for queen-side

        #
        # en-passant → three squares change
        #
        else:
            # derive from board legal moves
            uci = self._infer_en_passant(changed)

        move = chess.Move.from_uci(uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.prev = new_occ.copy()
            return move.uci(), self.board.fen()

        # illegal – ignore this frame
        return None, None

    def _infer_en_passant(self, changed):
        # brute-force: try every legal en-passant; keep the one whose
        # resulting occupancy == new matrix. Left as exercise.
        pass
