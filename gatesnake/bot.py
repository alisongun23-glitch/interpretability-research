"""A simple player used to drive tests: heads for the food, avoids walls and its own body
most of the time, makes random moves sometimes, presses reset a few ticks after dying."""
import numpy as np

KEYS = ["right", "down", "left", "up"]
STEP = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def bot_inputs(ref, r, p_random=0.08, p_idle=0.3):
    d = {"up": 0, "down": 0, "left": 0, "right": 0, "rst": 0}
    if ref.go:
        ref._dead = getattr(ref, "_dead", 0) + 1
        if ref._dead >= 3:
            d["rst"] = 1; ref._dead = 0
        return d
    N = ref.N
    if r.random() < p_random:
        d[KEYS[r.integers(4)]] = 1
        return d
    if ref.started and r.random() < p_idle:
        return d
    safe = []
    for k, (dx, dy) in enumerate(STEP):
        if ref.started and (k ^ ref.dir) == 2:
            continue
        x, y = ref.hx + dx, ref.hy + dy
        if 0 <= x < N and 0 <= y < N and ref.cells[y * N + x] <= 1:
            dist = abs(x - ref.fx) + abs(y - ref.fy) if ref.fv else 0
            safe.append((dist + r.random() * 0.5, k))
    if safe:
        d[KEYS[min(safe)[1]]] = 1
    return d


def cycle_dir(N, x, y):
    """Direction along a fixed loop through every cell (N even): along the top row, snake
    down through columns 1..N-1, back up column 0. Following it never crashes, so the
    snake grows until it fills the board."""
    if y == 0:
        return 0 if x < N - 1 else 1
    if x == 0:
        return 3
    if y % 2:                                   # odd rows run left, then step down (or into column 0)
        if x > 1:
            return 2
        return 1 if y < N - 1 else 2
    return 0 if x < N - 1 else 1                # even rows run right, then step down


def cycle_inputs(ref, r=None):
    d = {"up": 0, "down": 0, "left": 0, "right": 0, "rst": 0}
    if ref.go:
        d["rst"] = 1
        return d
    d[KEYS[cycle_dir(ref.N, ref.hx, ref.hy)]] = 1
    return d
