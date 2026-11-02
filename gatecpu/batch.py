"""Run many copies of a gate netlist at once, in exact Boolean logic.

Bit-sliced: copy m of the machine lives in bit m % 64 of word m // 64, so one uint64 AND
evaluates that gate in 64 machines. Flip-flops are ideal (Q := D each tick), which gives the
same results as the NAND flip-flops (checked in run_cpu.py and tests/check_all.py).

build_table() runs the calculator program (gatecpu/calc.py) on the gate CPU for every
(op, x, y), 4 x 256 x 256 calls, and returns the answers and clock ticks.
"""
import numpy as np

from gatesnake.hdl import GATES


def pack(bits):
    """bool array (n,) -> uint64 words, machine m in bit m % 64 of word m // 64."""
    b = np.asarray(bits, bool)
    pad = (-len(b)) % 64
    if pad:
        b = np.concatenate([b, np.zeros(pad, bool)])
    return np.packbits(b, bitorder="little").view(np.uint64)


def unpack(words, n):
    return np.unpackbits(np.ascontiguousarray(words).view(np.uint8), bitorder="little")[:n].astype(bool)


ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
OPS = {"AND": lambda a, b: a & b, "OR": lambda a, b: a | b, "NAND": lambda a, b: ~(a & b),
       "NOR": lambda a, b: ~(a | b), "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: ~(a ^ b),
       "NOT": lambda a, b: ~a}


class BitMachine:
    def __init__(self, net, n):
        self.net, self.n, self.W = net, n, (n + 63) // 64
        self.V = np.zeros((net.n_wires, self.W), np.uint64)
        self.V[1] = ALL
        self.q = np.array([g["q"] for g in net.regs]); self.d = np.array([g["d"] for g in net.regs])
        self.Q = np.where(np.array([g["init"] for g in net.regs])[:, None] == 1, ALL, np.uint64(0)) * np.ones((1, self.W), np.uint64)
        self.reg_index = {g["name"]: i for i, g in enumerate(net.regs)}
        self.groups = []
        by = {}
        for w, t, a in net.gates:
            by.setdefault((net.level[w], t), []).append((w, a[0], a[-1]))
        for (lv, t) in sorted(by, key=lambda k: (k[0], GATES.index(k[1]))):
            g = by[(lv, t)]
            self.groups.append((OPS[t], np.array([x[0] for x in g]), np.array([x[1] for x in g]), np.array([x[2] for x in g])))

    def step(self, inputs):
        """inputs: name -> 0/1 (same for every machine) or a bool array (n,) (one value per machine)."""
        V = self.V
        for name, w in self.net.inputs.items():
            v = inputs.get(name, 0)
            V[w] = (ALL if v else np.uint64(0)) if np.isscalar(v) else pack(v)
        V[self.q] = self.Q
        for f, out, a, b in self.groups:
            V[out] = f(V[a], V[b])
        self.Q = V[self.d].copy()

    def bus(self, name, nbits):
        """Integer value of register bus name[0..nbits-1] in every machine."""
        val = np.zeros(self.n, np.int64)
        for i in range(nbits):
            val |= unpack(self.Q[self.reg_index[f"{name}[{i}]"]], self.n).astype(np.int64) << i
        return val

    def bit(self, name):
        return unpack(self.Q[self.reg_index[name]], self.n)


def port(addr, data, n=None):
    """Program-port inputs writing data (int, or array of per-machine ints) at addr."""
    d = {"prog": 1, "prog_we": 1}
    for i in range(5):
        d[f"prog_addr{i}"] = (addr >> i) & 1
    for i in range(8):
        d[f"prog_data{i}"] = ((data >> i) & 1) if np.isscalar(data) else ((np.asarray(data) >> i) & 1).astype(bool)
    return d


def build_table(net, cap=None):
    """Answers of the calculator program for every (op, x, y), computed on the gates."""
    from .calc import PROGRAM, ADDR, CAP
    cap = cap or CAP
    n = 256 * 256
    xs, ys = np.arange(n) >> 8, np.arange(n) & 255
    out = np.zeros((4, 256, 256), np.uint8)
    halt_tick = np.zeros((4, 256, 256), np.int32)
    for op in range(4):
        m = BitMachine(net, n)
        for i, byte in enumerate(PROGRAM):                 # load the program once
            m.step(port(i, byte))
        for k, v in (("x", xs), ("y", ys), ("op", op), ("prod", 0)):
            m.step(port(ADDR[k], v))
        first_halt = np.zeros(n, np.int32)
        for t in range(1, cap + 1):
            m.step({})
            if t % 8 == 0 or t == cap:                      # halt tick to the nearest 8 ticks is enough for cost
                h = m.bit("halt")
                first_halt[(first_halt == 0) & h] = t
        out[op] = m.bus("out", 8).reshape(256, 256)
        halt_tick[op] = first_halt.reshape(256, 256)
    return out, halt_tick
