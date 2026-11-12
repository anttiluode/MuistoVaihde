"""MuistoVaihde: a 5/7-cell coincidence-learning ring switchyard.

Standard library only. Each cell is a labelled location in a rotating ring.
The shared integer tick is a simulator convenience, NOT proof of clock-free physics.
"""
from __future__ import annotations

from dataclasses import dataclass

A_SIZE, B_SIZE, C_SIZE = 5, 7, 35


def crt(a: int, b: int) -> int:
    """Unique c in [0,35) with c % 5 == a and c % 7 == b."""
    return next(c for c in range(C_SIZE) if c % A_SIZE == a % A_SIZE
                and c % B_SIZE == b % B_SIZE)


@dataclass
class Event:
    tick: int
    a: int
    b: int
    c: int
    action: str


class Switchyard:
    """Three rings, 35 plastic meeting weights, no backpropagation.

    Teach: co-occupied A/B cells strengthen their *cell-labelled* pair.
    Recall: an A bit plus a learned pair can write a missing B bit.
    Bind: a learned A/B coincidence writes one corresponding bit in ring C.
    Silent ticks decay *every* weight. All writing rules are explicit design.
    """

    def __init__(self, learning_rate=0.30, decay=0.0002, threshold=0.55):
        if not (0 <= learning_rate <= 1 and 0 <= decay < 1 and 0 < threshold <= 1):
            raise ValueError("Invalid learning parameters")
        self.learning_rate = learning_rate
        self.decay = decay
        self.threshold = threshold
        self.weights = [[0.0] * B_SIZE for _ in range(A_SIZE)]
        self.reset_fast()

    def reset_fast(self, a=(), b=(), phase_a=0, phase_b=0):
        """Replace ring contents and phase; preserve the slow weight matrix."""
        a, b = set(a), set(b)
        if any(not isinstance(i, int) or not 0 <= i < A_SIZE for i in a):
            raise ValueError("Invalid A cell")
        if any(not isinstance(i, int) or not 0 <= i < B_SIZE for i in b):
            raise ValueError("Invalid B cell")
        self.A = [int(i in a) for i in range(A_SIZE)]
        self.B = [int(i in b) for i in range(B_SIZE)]
        self.C = [0] * C_SIZE
        self.t = 0
        self.phase_a, self.phase_b = phase_a % A_SIZE, phase_b % B_SIZE
        # Synchronizing C's phase makes its labels encode A/B phase-pairs.
        self.phase_c = crt(self.phase_a, self.phase_b)

    def erase(self):
        self.weights = [[0.0] * B_SIZE for _ in range(A_SIZE)]
        self.reset_fast()

    def junction(self):
        a = (self.phase_a - self.t) % A_SIZE
        b = (self.phase_b - self.t) % B_SIZE
        c = (self.phase_c - self.t) % C_SIZE
        assert (c % A_SIZE, c % B_SIZE) == (a, b)
        return a, b, c

    def tick(self, mode="idle"):
        if mode not in ("idle", "teach", "recall", "bind"):
            raise ValueError("Mode must be idle, teach, recall or bind")
        a, b, c = self.junction()
        # This is physical-time decay, not decay tied to successful crossings.
        for row in self.weights:
            for j in range(B_SIZE):
                row[j] *= 1.0 - self.decay
        action = ""
        if mode == "teach" and self.A[a] and self.B[b]:
            w = self.weights[a][b]
            self.weights[a][b] = w + self.learning_rate * (1.0 - w)
            action = "learn"
        elif mode == "recall" and self.A[a] and self.weights[a][b] >= self.threshold:
            if not self.B[b]:
                self.B[b] = 1  # COPY/COMPLETE, not a conserved A-to-B move.
                action = "recall"
        elif (mode == "bind" and self.A[a] and self.B[b]
              and self.weights[a][b] >= self.threshold):
            if not self.C[c]:
                self.C[c] = 1
                action = "bind"
        event = Event(self.t, a, b, c, action)
        self.t += 1
        return event

    def run(self, ticks=C_SIZE, mode="idle"):
        return [self.tick(mode) for _ in range(ticks)]

    def teach(self, a, b, laps=10, phase_a=0, phase_b=0):
        self.reset_fast(a, b, phase_a, phase_b)
        return self.run(C_SIZE * laps, "teach")

    def recall(self, a, b=(), phase_a=0, phase_b=0):
        self.reset_fast(a, b, phase_a, phase_b)
        events = self.run(C_SIZE, "recall")
        return {j for j, v in enumerate(self.B) if v}, events

    def bind(self, a, b, phase_a=0, phase_b=0):
        self.reset_fast(a, b, phase_a, phase_b)
        events = self.run(C_SIZE, "bind")
        return {(c % A_SIZE, c % B_SIZE) for c, v in enumerate(self.C) if v}, events

    def active_pairs(self):
        return {(a, b) for a in range(A_SIZE) for b in range(B_SIZE)
                if self.weights[a][b] >= self.threshold}


def demo():
    m = Switchyard()
    x_a, x_b = {1}, {2, 5}
    y_a, y_b = {3}, {1, 4}
    print("MuistoVaihde / two rings + learned junction + ring C")
    print("Untrained recall X:", sorted(m.recall(x_a)[0]))
    m.teach(x_a, x_b)
    m.teach(y_a, y_b)
    print("Learned pair weights:", sorted(m.active_pairs()))
    print("Recall X:", sorted(m.recall(x_a)[0]), "expected", sorted(x_b))
    print("Recall Y:", sorted(m.recall(y_a)[0]), "expected", sorted(y_b))
    print("Unseen A cue:", sorted(m.recall({4})[0]))
    print("Bind X into C:", sorted(m.bind(x_a, x_b)[0]))
    print("Reset phases; recall X:", sorted(m.recall(x_a, phase_a=3, phase_b=4)[0]))
    print("Same A cue trained to two different B targets cannot disambiguate context.")
    print("This is a transparent association table, NOT established neural intelligence.")


if __name__ == "__main__":
    demo()
