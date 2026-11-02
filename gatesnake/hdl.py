"""A tiny hardware description layer: build a synchronous machine out of 2-input gates.

Wires are ints. Wire 0 is constant 0, wire 1 is constant 1. Every other wire is driven by
an input pin, a register output (Q of a D flip-flop) or a gate.

    c = Circuit()
    a = c.input("a"); q = c.reg("q", init=0)
    c.set_d(q, c.XOR(a, q))          # q toggles while a is high
    c.output("q", [q])

Building folds constants (AND(x, 0) -> 0), removes duplicates (two identical gates on the
same wires become one) and cancels double NOTs, so the counts it reports are for a
reasonably tight netlist, not a naive one.

Registers are D flip-flops. finalize() keeps them abstract for fast simulation, but every
one is accounted for (and simulated by gatesnake.sim) as the textbook master-slave flip-flop
of 8 NAND gates, plus one shared clock inverter. So the whole machine is gates only.
"""
from collections import defaultdict

GATES = ("NOT", "AND", "OR", "NAND", "NOR", "XOR", "XNOR")
EVAL = {"NOT": lambda a, b: 1 - a, "AND": lambda a, b: a & b, "OR": lambda a, b: a | b,
        "NAND": lambda a, b: 1 - (a & b), "NOR": lambda a, b: 1 - (a | b),
        "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: 1 - (a ^ b)}
DFF_NANDS = 8     # gated D latch = 4 NAND; master + slave = 8 NAND


class Circuit:
    def __init__(self):
        self.kind = ["const", "const"]     # per wire: const | in | q | gate type
        self.args = [(), ()]
        self.inputs = {}                   # name -> wire
        self.regs = []                     # list of dicts: name, q, d, init
        self.reg_of = {}                   # q wire -> index in regs
        self.outputs = {}                  # name -> list of wires
        self.cse = {}
        self.not_of = {}                   # wire -> wire that is its NOT (for NOT(NOT x) -> x)
        self.scope = "other"               # label for cost breakdowns; set it before building a part
        self.tag = ["const", "const"]

    # ---------- sources ----------
    def _new(self, kind, args=()):
        self.kind.append(kind); self.args.append(args); self.tag.append(self.scope)
        return len(self.kind) - 1

    def input(self, name):
        w = self._new("in"); self.inputs[name] = w; return w

    def reg(self, name, init=0):
        w = self._new("q")
        self.reg_of[w] = len(self.regs)
        self.regs.append({"name": name, "q": w, "d": None, "init": int(init)})
        return w

    def regs_bus(self, name, n, init=0):
        return [self.reg(f"{name}[{i}]", (init >> i) & 1) for i in range(n)]

    def set_d(self, q, d):
        self.regs[self.reg_of[q]]["d"] = d

    def output(self, name, wires):
        self.outputs[name] = list(wires)

    # ---------- gates with folding ----------
    def NOT(self, a):
        if a in (0, 1):
            return 1 - a
        if a in self.not_of:
            return self.not_of[a]
        key = ("NOT", a)
        if key not in self.cse:
            w = self._new("NOT", (a,)); self.cse[key] = w
            self.not_of[a] = w; self.not_of[w] = a
        return self.cse[key]

    def gate(self, t, a, b):
        if t == "NOT":
            return self.NOT(a)
        if a > b:
            a, b = b, a                    # every 2-input gate here is symmetric
        neg = {"NAND": "AND", "NOR": "OR", "XNOR": "XOR"}
        if t in neg:
            base = self._fold(neg[t], a, b)
            if base is not None:
                return self.NOT(base) if base not in (0, 1) else 1 - base
        else:
            base = self._fold(t, a, b)
            if base is not None:
                return base
        key = (t, a, b)
        if key not in self.cse:
            self.cse[key] = self._new(t, (a, b))
        return self.cse[key]

    def _fold(self, t, a, b):
        """Return a wire if the gate simplifies, else None. Inputs sorted, a <= b."""
        na = self.not_of.get(a)
        if t == "AND":
            if a == 0: return 0
            if a == 1: return b
            if a == b: return a
            if na == b: return 0
        if t == "OR":
            if a == 1: return 1
            if a == 0: return b
            if a == b: return a
            if na == b: return 1
        if t == "XOR":
            if a == 0: return b
            if a == 1: return self.NOT(b)
            if a == b: return 0
            if na == b: return 1
        return None

    def AND(self, a, b): return self.gate("AND", a, b)
    def OR(self, a, b): return self.gate("OR", a, b)
    def NAND(self, a, b): return self.gate("NAND", a, b)
    def NOR(self, a, b): return self.gate("NOR", a, b)
    def XOR(self, a, b): return self.gate("XOR", a, b)
    def XNOR(self, a, b): return self.gate("XNOR", a, b)

    # ---------- building blocks ----------
    def tree(self, t, ws):
        ws = list(ws)
        if not ws:
            return 1 if t == "AND" else 0
        while len(ws) > 1:
            nxt = [self.gate(t, ws[i], ws[i + 1]) for i in range(0, len(ws) - 1, 2)]
            if len(ws) % 2:
                nxt.append(ws[-1])
            ws = nxt
        return ws[0]

    def and_tree(self, ws): return self.tree("AND", ws)
    def or_tree(self, ws): return self.tree("OR", ws)

    def mux(self, s, a, b):
        """s ? b : a"""
        return self.OR(self.AND(a, self.NOT(s)), self.AND(b, s))

    def mux_bus(self, s, A, B):
        return [self.mux(s, a, b) for a, b in zip(A, B)]

    def decoder(self, bits):
        """bits LSB first -> one-hot list of 2**len(bits) wires (index = value)."""
        if len(bits) == 1:
            return [self.NOT(bits[0]), bits[0]]
        lo = self.decoder(bits[:len(bits) // 2]); hi = self.decoder(bits[len(bits) // 2:])
        return [self.AND(l, h) for h in hi for l in lo]

    def incdec(self, x, en, dec):
        """x + en (dec = 0) or x - en (dec = 1). Returns (bits, carry/borrow out)."""
        out, c = [], en
        for xi in x:
            out.append(self.XOR(xi, c))
            c = self.AND(self.XOR(xi, dec), c)
        return out, c

    def increment(self, x, en):
        return self.incdec(x, en, 0)[0]

    def decrement(self, x, en):
        return self.incdec(x, en, 1)[0]

    def equal(self, A, B):
        return self.and_tree([self.XNOR(a, b) for a, b in zip(A, B)])

    # ---------- finish ----------
    def finalize(self):
        """Drop gates nothing uses, order the rest by level. Returns a Netlist."""
        for r in self.regs:
            if r["d"] is None:
                raise ValueError(f"register {r['name']} has no D input")
        roots = [r["d"] for r in self.regs] + [w for ws in self.outputs.values() for w in ws]
        live = set(); stack = list(roots)
        while stack:
            w = stack.pop()
            if w in live:
                continue
            live.add(w)
            stack.extend(self.args[w])
        return Netlist(self, live)


class Netlist:
    """A finished circuit. Gates are renumbered in level order (level 1 = fed only by
    sources). Sources: wire 0/1, input pins, register outputs."""

    def __init__(self, c, live):
        self.inputs = dict(c.inputs)
        self.regs = [dict(r) for r in c.regs]
        self.outputs = dict(c.outputs)
        level = {}
        for w in range(len(c.kind)):          # wires are created after their inputs
            if c.kind[w] in GATES:
                level[w] = 1 + max(level.get(a, 0) for a in c.args[w])
            else:
                level[w] = 0
        gates = sorted((w for w in live if c.kind[w] in GATES), key=lambda w: (level[w], w))
        self.n_wires = len(c.kind)
        self.gates = [(w, c.kind[w], c.args[w]) for w in gates]   # (out wire, type, input wires)
        self.tag = {w: c.tag[w] for w in gates}
        self.level = {w: level[w] for w, _, _ in self.gates}
        self.depth = max(self.level.values()) if self.level else 0

    def counts(self):
        cnt = defaultdict(int)
        for _, t, _ in self.gates:
            cnt[t] += 1
        return dict(cnt)

    def breakdown(self, reg_group):
        """Gate counts per part. reg_group(name) -> part label for a register's 8 NAND gates."""
        out = defaultdict(lambda: {"comb": 0, "dff": 0})
        for w, _, _ in self.gates:
            out[self.tag[w]]["comb"] += 1
        for r in self.regs:
            out[reg_group(r["name"])]["dff"] += DFF_NANDS
        return {k: dict(v) for k, v in out.items()}

    def stats(self, library=None):
        """Gate, neuron, parameter and multiply-add counts. library: gate type -> cost dict."""
        cnt = self.counts()
        n_reg = len(self.regs)
        s = {"comb_gates": sum(cnt.values()), "gates_by_type": cnt, "registers": n_reg,
             "dff_gates": n_reg * DFF_NANDS + (1 if n_reg else 0), "logic_depth": self.depth,
             "inputs": len(self.inputs), "output_wires": sum(len(v) for v in self.outputs.values())}
        s["total_gates"] = s["comb_gates"] + s["dff_gates"]
        if library:
            for key in ("neurons", "params", "macs"):
                comb = sum(cnt[t] * library[t][key] for t in cnt)
                mem = n_reg * DFF_NANDS * library["NAND"][key] + (library["NOT"][key] if n_reg else 0)
                s[key] = comb + mem
                s[f"{key}_comb"] = comb
            # neuron layers on the longest path: each gate adds its own depth in neurons
            ndepth = {}
            for w, t, a in self.gates:
                ndepth[w] = library[t]["depth"] + max(ndepth.get(x, 0) for x in a)
            s["neuron_depth_comb"] = max(ndepth.values()) if ndepth else 0
        return s

    def eval_logic(self, values):
        """Exact Boolean evaluation. values: dict wire -> 0/1 for every source."""
        v = dict(values); v[0] = 0; v[1] = 1
        for w, t, a in self.gates:
            v[w] = EVAL[t](v[a[0]], v[a[1]] if len(a) > 1 else 0)
        return v


def lower_to_nand(netlist_builder):
    """Rebuild a circuit with NAND gates only (NOT x = NAND(x, x) is drawn as a 1-input NAND
    neuron, so it stays one neuron). Returns a new Circuit with the same inputs, registers and
    outputs."""
    src = netlist_builder
    c = Circuit()
    m = {0: 0, 1: 1}
    for name, w in src.inputs.items():
        m[w] = c.input(name)
    for r in src.regs:
        m[r["q"]] = c.reg(r["name"], r["init"])

    def nand(a, b): return c.NAND(a, b)
    def inv(a): return c.NAND(a, a) if a not in (0, 1) else 1 - a

    for w in range(2, len(src.kind)):
        t = src.kind[w]
        if t not in GATES:
            continue
        c.scope = src.tag[w]
        a = [m[x] for x in src.args[w]]
        if t == "NOT": m[w] = inv(a[0])
        elif t == "NAND": m[w] = nand(a[0], a[1])
        elif t == "AND": m[w] = inv(nand(a[0], a[1]))
        elif t == "OR": m[w] = nand(inv(a[0]), inv(a[1]))
        elif t == "NOR": m[w] = inv(nand(inv(a[0]), inv(a[1])))
        elif t == "XOR":
            n = nand(a[0], a[1]); m[w] = nand(nand(a[0], n), nand(a[1], n))
        elif t == "XNOR":
            n = nand(a[0], a[1]); m[w] = inv(nand(nand(a[0], n), nand(a[1], n)))
    for r in src.regs:
        c.set_d(m[r["q"]], m[r["d"]])
    for name, ws in src.outputs.items():
        c.output(name, [m[w] for w in ws])
    return c
