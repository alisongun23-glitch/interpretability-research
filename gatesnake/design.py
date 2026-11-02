"""Snake as a synchronous machine of 2-input gates and D flip-flops. One clock tick = one
game step.

Board N x N (N a power of two), walls kill. Every cell keeps a W-bit "lifetime" counter:
the head cell gets the snake's length, every other occupied cell counts down by one per
step, a cell is snake while its counter is nonzero. Eating skips the countdown for one
step, so the tail stays and the snake grows by one. (The same idea as the third RAM in
stevomitric/Logisim-Snake, but spread over the board instead of stored in RAM.)

Inputs  up, down, left, right (held level for the coming tick), rst
State   head x/y, direction, length, game-over, started, food x/y + valid, 16-bit LFSR,
        one W-bit counter per cell
Outputs body[N*N] (counter nonzero), food[N*N], head x/y, length, game_over, started

Every rule below is mirrored in gatesnake.ref.SnakeRef, and the two are checked against
each other tick by tick.
"""
from .hdl import Circuit

LFSR_BITS = 16
LFSR_TAPS = (15, 14, 12, 3)        # x^16 + x^15 + x^13 + x^4 + 1 (XNOR form: all-zero start is fine)
START_LEN = 3


def counter_bits(N):
    """Enough bits to hold the longest possible snake, the whole board."""
    return max(4, (N * N).bit_length())


def build_snake(N=16):
    k = N.bit_length() - 1
    assert 1 << k == N and k >= 2, "N must be a power of two >= 4"
    W = counter_bits(N)
    c = Circuit()
    c.scope = "control"
    up, down, left, right, rst = (c.input(n) for n in ("up", "down", "left", "right", "rst"))
    nrst = c.NOT(rst)

    hx = c.regs_bus("hx", k, N // 2); hy = c.regs_bus("hy", k, N // 2)
    dr = c.regs_bus("dir", 2, 0)                     # 0 right, 1 down, 2 left, 3 up
    ln = c.regs_bus("len", W, START_LEN)
    go = c.reg("game_over", 0); st = c.reg("started", 0)
    fx = c.regs_bus("fx", k, 0); fy = c.regs_bus("fy", k, 0); fv = c.reg("food_valid", 0)
    lf = c.regs_bus("lfsr", LFSR_BITS, 0)
    cnt = [c.regs_bus(f"cell{i}", W, 0) for i in range(N * N)]   # cell i = row i // N, column i % N

    c.scope = "control"
    # ---- direction: priority right > down > left > up, no 180-degree turn once started ----
    anykey = c.OR(c.OR(up, down), c.OR(left, right))
    req1 = c.AND(c.NOT(right), c.AND(c.NOT(down), c.OR(left, up)))
    req0 = c.AND(c.NOT(right), c.OR(down, c.AND(c.NOT(left), up)))
    opposite = c.AND(c.XNOR(req0, dr[0]), c.XOR(req1, dr[1]))
    allowed = c.AND(anykey, c.NAND(opposite, st))
    nd = c.mux_bus(allowed, dr, [req0, req1])

    c.scope = "head & walls"
    # ---- candidate head: x +- 1 or y +- 1; the carry out is the wall ----
    cx, wall_x = c.incdec(hx, c.NOT(nd[0]), nd[1])
    cy, wall_y = c.incdec(hy, nd[0], nd[1])
    wall = c.OR(wall_x, wall_y)
    row, col = c.decoder(cy), c.decoder(cx)
    head = [c.AND(row[i // N], col[i % N]) for i in range(N * N)]

    # ---- per-cell occupancy; hitting a cell whose counter is > 1 is death (counter 1 = tail, leaves now) ----
    c.scope = "cells"
    nz, gt1 = [], []
    for ci in cnt:
        g = c.or_tree(ci[1:]); gt1.append(g); nz.append(c.OR(g, ci[0]))
    c.scope = "control"
    self_hit = c.or_tree([c.AND(h, g) for h, g in zip(head, gt1)])
    crash = c.OR(wall, self_hit)
    awake = c.OR(st, anykey)
    active = c.AND(awake, c.NOT(go))
    run = c.AND(active, c.NOT(crash))
    c.set_d(go, c.AND(c.OR(go, c.AND(active, crash)), nrst))
    c.set_d(st, c.AND(awake, nrst))

    # ---- food ----
    c.scope = "food"
    eat = c.AND(c.AND(fv, c.equal(cx + cy, fx + fy)), run)
    c.scope = "length"
    ln_inc = c.increment(ln, eat)
    write_val = [c.AND(b, nrst) for b in ln_inc]           # reset writes 0 into every cell
    for i, (q, b) in enumerate(zip(ln, ln_inc)):
        c.set_d(q, c.OR(b, rst) if (START_LEN >> i) & 1 else c.AND(b, nrst))
    c.scope = "control"
    countdown = c.AND(run, c.NOT(eat))

    # ---- per-cell counter update ----
    c.scope = "cells"
    head_w = []
    for ci, h, z in zip(cnt, head, nz):
        hw = c.AND(h, run); head_w.append(hw)
        write = c.OR(hw, rst)
        dec = c.decrement(ci, c.AND(countdown, z))
        for q, v in zip(ci, c.mux_bus(write, dec, write_val)):
            c.set_d(q, v)

    # ---- head, direction ----
    c.scope = "head & walls"
    for bus, cand in ((hx, cx), (hy, cy)):
        for i, (q, v) in enumerate(zip(bus, c.mux_bus(run, bus, cand))):
            c.set_d(q, c.OR(v, rst) if ((N // 2) >> i) & 1 else c.AND(v, nrst))
    c.scope = "control"
    for q, v in zip(dr, nd):
        c.set_d(q, c.AND(v, nrst))

    # ---- food placement: candidate from the LFSR, accepted once it lands on a free cell ----
    c.scope = "food"
    frow, fcol = c.decoder(fy), c.decoder(fx)
    fhot = [c.AND(frow[i // N], fcol[i % N]) for i in range(N * N)]
    on_snake = c.or_tree([c.AND(f, c.OR(z, hw)) for f, z, hw in zip(fhot, nz, head_w)])
    reroll = c.OR(eat, c.AND(c.NOT(fv), on_snake))
    for q, v in zip(fx + fy, c.mux_bus(reroll, fx + fy, lf[:k] + lf[k:2 * k])):
        c.set_d(q, v)
    c.set_d(fv, c.AND(c.AND(c.NOT(eat), c.OR(fv, c.NOT(on_snake))), nrst))

    c.scope = "random (LFSR)"
    # ---- LFSR (free running, not reset, so each game gets different food) ----
    t = LFSR_TAPS
    fb = c.XNOR(c.XNOR(lf[t[0]], lf[t[1]]), c.XNOR(lf[t[2]], lf[t[3]]))
    c.set_d(lf[0], fb)
    for i in range(1, LFSR_BITS):
        c.set_d(lf[i], lf[i - 1])

    # ---- outputs: what the screen shows ----
    c.scope = "food"
    c.output("body", nz)
    c.output("food", [c.AND(f, fv) for f in fhot])
    c.output("hx", hx); c.output("hy", hy); c.output("len", ln)
    c.output("game_over", [go]); c.output("started", [st])
    return c, {"N": N, "k": k, "W": W}


def reg_part(name):
    """Which part of the machine a register belongs to (for cost breakdowns)."""
    base = name.split("[")[0]
    if base.startswith("cell"): return "cells"
    if base in ("hx", "hy"): return "head & walls"
    if base in ("fx", "fy", "food_valid"): return "food"
    if base == "len": return "length"
    if base == "lfsr": return "random (LFSR)"
    return "control"
