"""Plain-Python snake with exactly the rules of gatesnake.design (the gate machine is
checked against this one tick by tick)."""
from .design import LFSR_BITS, LFSR_TAPS, START_LEN, counter_bits


class SnakeRef:
    def __init__(self, N=16):
        self.N, self.k, self.W = N, N.bit_length() - 1, counter_bits(N)
        self.reset_state()
        self.lfsr = 0

    def reset_state(self):
        N = self.N
        self.hx = self.hy = N // 2
        self.dir = 0; self.len = START_LEN; self.go = 0; self.started = 0
        self.fx = self.fy = 0; self.fv = 0
        self.cells = [0] * (N * N)

    def step(self, up=0, down=0, left=0, right=0, rst=0):
        N, k, W = self.N, self.k, self.W
        anykey = up | down | left | right
        if right: req = 0
        elif down: req = 1
        elif left: req = 2
        else: req = 3
        opposite = (req ^ self.dir) == 2
        nd = req if (anykey and not (opposite and self.started)) else self.dir
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][nd]
        cx, cy = self.hx + dx, self.hy + dy
        wall = not (0 <= cx < N and 0 <= cy < N)
        cx, cy = cx % N, cy % N                       # what the gates compute (carry dropped)
        hcell = cy * N + cx
        crash = wall or self.cells[hcell] > 1
        awake = self.started or anykey
        active = awake and not self.go
        run = active and not crash
        eat = self.fv and (cx, cy) == (self.fx, self.fy) and run
        len_inc = (self.len + eat) % (1 << W)
        countdown = run and not eat
        on_snake = (self.cells[self.fy * N + self.fx] > 0) or (run and hcell == self.fy * N + self.fx)
        reroll = eat or (not self.fv and on_snake)
        mask = (1 << k) - 1
        new_fx, new_fy = ((self.lfsr & mask), (self.lfsr >> k) & mask) if reroll else (self.fx, self.fy)
        new_fv = (not eat) and (self.fv or not on_snake)
        t = LFSR_TAPS
        fb = 1 ^ (((self.lfsr >> t[0]) ^ (self.lfsr >> t[1]) ^ (self.lfsr >> t[2]) ^ (self.lfsr >> t[3])) & 1)
        self.lfsr = ((self.lfsr << 1) | fb) & ((1 << LFSR_BITS) - 1)
        if rst:
            self.reset_state(); self.fx, self.fy = new_fx, new_fy   # food position is not reset
            return
        cells = self.cells
        for i in range(N * N):
            if run and i == hcell:
                cells[i] = len_inc
            elif countdown and cells[i] > 0:
                cells[i] -= 1
        if run:
            self.hx, self.hy = cx, cy
        self.dir = nd; self.len = len_inc
        self.go = int(self.go or (active and crash)); self.started = int(awake)
        self.fx, self.fy, self.fv = new_fx, new_fy, int(new_fv)

    def regs(self):
        """State as {register name: bit}, same names as the gate machine."""
        r = {}
        def bus(name, val, n):
            for i in range(n):
                r[f"{name}[{i}]"] = (val >> i) & 1
        bus("hx", self.hx, self.k); bus("hy", self.hy, self.k); bus("dir", self.dir, 2)
        bus("len", self.len, self.W); bus("fx", self.fx, self.k); bus("fy", self.fy, self.k)
        bus("lfsr", self.lfsr, LFSR_BITS)
        r["game_over"] = self.go; r["started"] = self.started; r["food_valid"] = self.fv
        for i, v in enumerate(self.cells):
            bus(f"cell{i}", v, self.W)
        return r
