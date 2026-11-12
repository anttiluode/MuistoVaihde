"""Two rings, one junction. Phase is the address.

Ring A has 5 cells, ring B has 7. Each ring turns one cell per tick.
Both pass through one fixed junction. Ring B has one marked cell, the PORT.
Rule: a bit crosses A -> B only on a tick when the bit and the port are
both at the junction. No clock: the lining-up is the gate.

  python rings.py          run the test and the control
  python rings.py show     watch one bit go round and cross
"""
import sys, random

MA, MB, PORT = 5, 7, 0          # ring sizes, and which B cell is the port


def at_junction(m, t):
    """Index of the cell of an m-cell ring sitting at the junction at tick t."""
    return (-t) % m


def run(start, ticks=1200, gate="align", seed=0, show=False):
    """One bit starts in cell `start` of ring A. Returns (cross tick, B cell, alive)."""
    rng = random.Random(seed)
    A, B = [0] * MA, [0] * MB
    A[start] = 1
    crossed = None
    for t in range(ticks):
        a, b = at_junction(MA, t), at_junction(MB, t)
        if gate == "align":
            is_open = b == PORT
        else:                                   # control: open as often, at random
            is_open = rng.random() < 1 / MB
        if crossed is None and A[a] and is_open and not B[b]:
            A[a], B[b], crossed = 0, 1, (t, b)
        if show and t < 40:
            ring = lambda R, j: "".join(("[%s]" if i == j else " %s ") % ".#"[v] for i, v in enumerate(R))
            print(f"t={t:2d}  A {ring(A, a)}   B {ring(B, b)}" + ("   <- crossed" if crossed and crossed[0] == t else ""))
    return (crossed[0], crossed[1], sum(B) == 1) if crossed else (None, None, False)


def predicted_tick(start):
    """First tick with t = start (mod 5) and t = PORT (mod 7), i.e. bit and port both at the junction."""
    return next(t for t in range(MA * MB) if (-t) % MA == start and (-t) % MB == PORT)


if __name__ == "__main__":
    if sys.argv[1:] == ["show"]:
        run(2, show=True); sys.exit()
    print("1) alignment gate: where the bit starts decides when it crosses")
    ok = True
    for s in range(MA):
        t, cell, alive = run(s)
        ok &= t == predicted_tick(s) and cell == PORT and alive
        print(f"   start cell {s}: crosses at tick {t:2d} (predicted {predicted_tick(s):2d}), lands in B cell {cell}, alive after 1200 ticks: {alive}")
    states = [(at_junction(MA, t), at_junction(MB, t)) for t in range(200)]
    period = next(p for p in range(1, 200) if states[:100] == states[p:p + 100])
    print(f"2) the two rings together repeat every {period} ticks (5 x 7 = {MA * MB})")
    print("3) control: gate opens at random ticks, same average rate")
    for s in range(MA):
        res = [run(s, gate="random", seed=k) for k in range(200)]
        cells = sorted({c for _, c, _ in res if c is not None})
        ticks = {t for t, _, _ in res if t is not None}
        print(f"   start cell {s}: lands in B cells {cells}, {len(ticks)} different crossing ticks in 200 runs")
    print("PASS" if ok and period == MA * MB else "FAIL")
