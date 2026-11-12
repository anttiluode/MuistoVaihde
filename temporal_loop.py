"""MuistoVaihde V2: a learned event sequence feeding its own seven-tick delay loop.

This is an engineered finite-state sequence learner, not a native neuron, and
not a demonstration that a phase ring alone learns order. Learning uses an
explicit one-event, seven-tick trace of *external* observations. At replay,
self-generated events circulate through an actual seven-slot delay register.

Run: python temporal_loop.py
"""
from __future__ import annotations

from dataclasses import dataclass

TOKENS = ("P", "A", "B", "Q", "C", "END")
LAP = 7


@dataclass(frozen=True)
class Event:
    tick: int
    token: str
    origin: str                  # world or self
    expected: str | None        # what was arriving from the recurrent loop
    mismatch: bool              # only meaningful on world arrivals with expectation
    wrote: tuple[str, str] | None
    sent: str | None             # the NEXT event entering the loop, not yet observed


class TemporalLoop:
    """One recurrent seven-slot delay line and one learned directed transition matrix.

    When a world event arrives, it wins over a predicted event and can update
    the transition from the PREVIOUS WORLD event at exactly one-lap spacing.
    Self-generated events can drive more generation, but can NEVER teach.
    An explicit END token terminates the replay instead of being fed back.
    """

    def __init__(self, learning_rate: float = .40, threshold: float = .55,
                 decay: float = 0.0):
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate outside [0,1]")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold outside (0,1]")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay outside [0,1)")
        self.learning_rate, self.threshold, self.decay = learning_rate, threshold, decay
        self.weights = {a: {b: 0.0 for b in TOKENS} for a in TOKENS}
        self.reset_fast()

    def reset_fast(self):
        """Clear circulation and learning trace; preserve learned weights."""
        self.ring: list[str | None] = [None] * LAP
        self.time = 0
        self.previous_world: tuple[str, int] | None = None

    def clear_weights(self):
        self.__init__(self.learning_rate, self.threshold, self.decay)

    def _check_token(self, token: str):
        if token not in TOKENS:
            raise ValueError(f"Unknown event {token!r}; choose from {TOKENS}")

    def predict(self, token: str) -> str | None:
        self._check_token(token)
        if token == "END":
            return None
        row = self.weights[token]
        strongest = max(row.values())
        if strongest < self.threshold:
            return None
        # Ties abstain; otherwise two incompatible episodes would be hidden.
        if sum(value >= strongest - 1e-12 for value in row.values()) != 1:
            return None
        return max(row, key=row.get)

    def tick(self, world: str | None = None, *, learn: bool = False,
             feedback: bool = True) -> Event | None:
        """Advance one discrete tick. A WORLD event overrides any self event.

        World observations enter directly at the junction. A generated next
        event traverses the seven-slot ring before it becomes a self event.
        No background teaching, no implicit world confirmation, no hidden state.
        """
        if world is not None:
            self._check_token(world)
        if self.decay:
            for row in self.weights.values():
                for key in row:
                    row[key] *= (1.0 - self.decay)

        arriving = self.ring[0]
        current = world if world is not None else arriving
        event = None
        sent = None
        if current is not None:
            origin = "world" if world is not None else "self"
            wrote = None
            if world is not None:
                # A real observation interrupts/overwrites the previous prediction.
                # A missing prediction doesn't count as a mismatch.
                if learn and self.previous_world is not None:
                    previous, previous_tick = self.previous_world
                    if self.time - previous_tick == LAP and previous != "END":
                        old = self.weights[previous][world]
                        self.weights[previous][world] = old + self.learning_rate * (1.0 - old)
                        wrote = (previous, world)
                self.previous_world = (world, self.time) if world != "END" else None
            if feedback and current != "END":
                sent = self.predict(current)
            event = Event(
                tick=self.time, token=current, origin=origin,
                expected=arriving if world is not None else None,
                mismatch=world is not None and arriving is not None and arriving != world,
                wrote=wrote, sent=sent,
            )

        if world is not None:
            self.ring = [None] * LAP  # authoritative sensory input clears pending self events
        self.ring[0] = sent
        self.ring = self.ring[1:] + self.ring[:1]
        self.time += 1
        return event

    def teach(self, sequence: tuple[str, ...] | list[str], repeats: int = 8):
        """Present observed events one lap apart; predictions disabled during teaching."""
        if not sequence or sequence[-1] != "END" or "END" in sequence[:-1]:
            raise ValueError("Teach a nonempty episode ending with END")
        for token in sequence:
            self._check_token(token)
        if repeats < 1:
            raise ValueError("repeats must be positive")
        for _ in range(repeats):
            self.reset_fast()
            for i, token in enumerate(sequence):
                if i:
                    for _ in range(LAP - 1):
                        self.tick(feedback=False)
                self.tick(world=token, learn=True, feedback=False)
        self.reset_fast()

    def replay(self, seed: str = "P", *, feedback: bool = True, max_events: int = 10):
        """One real seed, followed by self-produced arrivals; stop on END or timeout."""
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.reset_fast()
        events = [self.tick(world=seed, feedback=feedback)]
        for _ in range((max_events - 1) * LAP):
            if events[-1].token == "END":
                break
            event = self.tick(feedback=feedback)
            if event is not None:
                events.append(event)
        return events


def demo():
    m = TemporalLoop()
    print("MuistoVaihde V2 / real-world teaching -> self-driven replay")
    print("Untrained:", [e.token for e in m.replay("P")])
    m.teach(("P", "A", "B", "END"))
    print("Trained:", " -> ".join(f"{e.token}({e.origin}@{e.tick})" for e in m.replay("P")))
    print("Feedback cut:", [e.token for e in m.replay("P", feedback=False)])
    q = TemporalLoop()
    q.teach(("P", "B", "A", "END"))
    print("Shuffled teaching:", [e.token for e in q.replay("P")])
    print("NOTE: direct conditional transition memory, not native GeometricNeuron dynamics.")


if __name__ == "__main__":
    demo()
