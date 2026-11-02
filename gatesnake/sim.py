"""Run a gatesnake Netlist, either as exact Boolean logic or as a network of neurons.

mode="logic"   every gate is its truth table (the reference for the netlist itself)
mode="neural"  every gate is a trained network from results_minigates/library.json;
               each gate instance draws its own network from that gate type's pool.
               Signals are real numbers in [0, 1]; nothing is rounded unless snap=True.
dff="gates"    every register is a master-slave flip-flop of 8 NAND gates (neural NANDs in
               neural mode), clocked low then high every tick; the latch loops are iterated
               until they settle. This is the "gates only" machine.
dff="ideal"    Q := D at the tick (same result when the flip-flops work; faster).
"""
import json
import numpy as np

from .hdl import GATES, DFF_NANDS

TT = {"NOT": [1, 1, 0, 0], "AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0],
      "NOR": [1, 0, 0, 0], "XOR": [0, 1, 1, 0], "XNOR": [1, 0, 0, 1]}   # indexed by 2a + b (NOT: b = a)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def to_common(p):
    """One trained gate (gatesnake.gates param dict) -> common 2-input form:
    U (2, H), c (H), w (2), v (H), bo. A 1-input gate gets a zero weight on input 2."""
    wx = np.array(p["wx"], float)
    Wh = np.array(p["Wh"], float).reshape(wx.size, np.array(p["bh"]).size)
    if Wh.shape[0] == 1:
        Wh = np.vstack([Wh, np.zeros_like(Wh)]); wx = np.array([wx[0], 0.0])
    return {"U": Wh, "c": np.array(p["bh"], float).ravel(), "w": wx,
            "v": np.array(p["wh"], float).ravel(), "bo": float(np.asarray(p["bo"]))}


def make_library(pools, nand_only=False, margin=None):
    """pools: gate type -> list of trained param dicts. With nand_only every gate type (NOT
    included, as NAND(x, x)) uses the NAND pool."""
    out = {t: [to_common(p) for p in pools["NAND" if nand_only else t]] for t in GATES}
    out["_meta"] = {"margin": margin, "nand_only": nand_only}
    return out


def load_library(path="results_minigates/library.json", nand_only=False):
    lib = json.load(open(path))
    out = make_library({t: lib["gates"][t]["pool"] for t in GATES}, nand_only, lib["margin"])
    out["_meta"]["cost"] = {t: lib["gates"]["NAND" if nand_only else t]["cost"] for t in GATES}
    return out


class Group:
    """Gates of one level that share a hidden width: evaluated in one vectorised shot."""

    def __init__(self, out, a, b, tt=None, params=None):
        self.out, self.a, self.b, self.tt = out, a, b, tt
        if params:
            self.U = np.stack([p["U"] for p in params])        # (n, 2, H)
            self.c = np.stack([p["c"] for p in params])        # (n, H)
            self.w = np.stack([p["w"] for p in params])        # (n, 2)
            self.v = np.stack([p["v"] for p in params])        # (n, H)
            self.bo = np.array([p["bo"] for p in params])
            self.H = self.U.shape[2]

    def neural(self, x, y):
        z = self.w[:, 0] * x + self.w[:, 1] * y + self.bo
        if self.H:
            h = sig(x[:, None] * self.U[:, 0, :] + y[:, None] * self.U[:, 1, :] + self.c)
            z = z + (h * self.v).sum(1)
        return sig(z)


class Machine:
    def __init__(self, net, mode="logic", library=None, dff="ideal", snap=False, seed=0, latch_iters=6, noise=0.0):
        """noise: std of Gaussian noise added to every neuron-gate output (neural mode)."""
        self.net, self.mode, self.dff, self.snap, self.latch_iters = net, mode, dff, snap, latch_iters
        self.noise = noise
        self.rng = np.random.default_rng(seed + 12345)
        self.lib = library
        r = np.random.default_rng(seed)
        self.choice = {}
        if mode == "neural":                                    # which trained net each gate instance uses
            for w, t, _ in net.gates:
                self.choice[w] = int(r.integers(len(library[t])))
        self.in_w = {n: w for n, w in net.inputs.items()}
        self.q_w = np.array([g["q"] for g in net.regs])
        self.d_w = np.array([g["d"] for g in net.regs])
        self.groups = self._compile([g for g in net.gates])
        cone = self._cone([w for ws in net.outputs.values() for w in ws])
        self.out_groups = self._compile([g for g in net.gates if g[0] in cone])
        self.V = np.zeros(net.n_wires, float if mode == "neural" else np.int64)
        self.V[1] = 1
        init = np.array([g["init"] for g in net.regs], float)
        self.Q = init.copy() if mode == "neural" else init.astype(np.int64)
        self.worst = 0.0                                        # closest any signal came to 0.5
        if dff == "gates":
            n = len(net.regs)
            if mode == "neural":
                nand = library["NAND"]
                self.ff = Group(None, None, None, params=[nand[int(r.integers(len(nand)))] for _ in range(n * DFF_NANDS)])
                self.clk_inv = Group(None, None, None, params=[library["NOT"][int(r.integers(len(library["NOT"])))]])
            # flip-flop internal wires, settled for clock high: master closed, slave open
            self.L = {"m1": np.ones(n), "m2": np.ones(n), "M": init.copy(), "Mb": 1 - init,
                      "s1": 1 - init, "s2": init.copy(), "Q": init.copy(), "Qb": 1 - init}
            self._settle(1.0)
            self.Q = self.L["Q"].copy() if mode == "neural" else np.round(self.L["Q"]).astype(np.int64)

    # ---------- compile ----------
    def _cone(self, roots):
        args = {w: a for w, _, a in self.net.gates}
        seen, stack = set(), list(roots)
        while stack:
            w = stack.pop()
            if w in seen or w not in args:
                continue
            seen.add(w); stack.extend(args[w])
        return seen

    def _compile(self, gates):
        by_level = {}
        for w, t, a in gates:
            by_level.setdefault(self.net.level[w], []).append((w, t, a))
        groups = []
        for L in sorted(by_level):
            gs = by_level[L]
            if self.mode == "logic":
                groups.append(Group(np.array([w for w, _, _ in gs]), np.array([a[0] for _, _, a in gs]),
                                    np.array([a[-1] for _, _, a in gs]), tt=np.array([TT[t] for _, t, _ in gs])))
            else:
                byH = {}
                for w, t, a in gs:
                    p = self.lib[t][self.choice[w]]
                    byH.setdefault(p["U"].shape[1], []).append((w, a, p))
                for H, lst in byH.items():
                    groups.append(Group(np.array([w for w, _, _ in lst]), np.array([a[0] for _, a, _ in lst]),
                                        np.array([a[-1] for _, a, _ in lst]), params=[p for _, _, p in lst]))
        return groups

    # ---------- evaluate ----------
    def _eval(self, groups, inputs=None):
        V = self.V
        V[self.q_w] = self.Q
        for name, w in self.in_w.items():
            V[w] = (inputs or {}).get(name, 0)
        if self.mode == "logic":
            for g in groups:
                V[g.out] = g.tt[np.arange(len(g.out)), 2 * V[g.a] + V[g.b]]
        else:
            worst = 0.5
            for g in groups:
                o = g.neural(V[g.a], V[g.b])
                if self.noise:
                    o = np.clip(o + self.rng.normal(0, self.noise, o.shape), 0, 1)
                if self.snap:
                    o = np.round(o)
                V[g.out] = o
                worst = min(worst, np.abs(o - 0.5).min())
            self.worst = max(self.worst, 0.5 - worst)
        return V

    def _nand(self, x, y, sl):
        if self.mode == "logic":
            return 1 - np.round(x).astype(np.int64) * np.round(y).astype(np.int64)
        g = self.ff
        o = sig(g.w[sl, 0] * x + g.w[sl, 1] * y + g.bo[sl] +
                ((sig(x[:, None] * g.U[sl, 0, :] + y[:, None] * g.U[sl, 1, :] + g.c[sl]) * g.v[sl]).sum(1) if g.H else 0))
        if self.noise:
            o = np.clip(o + self.rng.normal(0, self.noise, o.shape), 0, 1)
        return np.round(o) if self.snap else o

    def _settle(self, clk, D=None):
        """Iterate every flip-flop's 8 NAND gates (all at once, each step = one gate delay)."""
        n = len(self.net.regs); L = self.L
        if self.mode == "neural":
            clkb = float(self.clk_inv.neural(np.array([clk]), np.array([clk]))[0])
            if self.snap: clkb = float(np.round(clkb))
        else:
            clkb = 1.0 - clk
        E = np.full(n, clkb); C = np.full(n, clk)
        D = L["M"] if D is None else D
        sl = [slice(i, None, DFF_NANDS) for i in range(DFF_NANDS)]
        for _ in range(self.latch_iters):
            m1 = self._nand(D, E, sl[0]); m2 = self._nand(L["m1"], E, sl[1])
            M = self._nand(L["m1"], L["Mb"], sl[2]); Mb = self._nand(L["m2"], L["M"], sl[3])
            s1 = self._nand(L["M"], C, sl[4]); s2 = self._nand(L["s1"], C, sl[5])
            Q = self._nand(L["s1"], L["Qb"], sl[6]); Qb = self._nand(L["s2"], L["Q"], sl[7])
            L.update(m1=m1, m2=m2, M=M, Mb=Mb, s1=s1, s2=s2, Q=Q, Qb=Qb)
        if self.mode == "neural":
            self.worst = max(self.worst, 0.5 - min(np.abs(v - 0.5).min() for v in L.values()))

    def step(self, inputs=None):
        """One clock tick. Returns nothing; read display with outputs()."""
        V = self._eval(self.groups, inputs)
        D = V[self.d_w].astype(float)
        if self.dff == "ideal":
            self.Q = V[self.d_w].copy()
        else:
            self._settle(0.0, D)        # clock low: master follows D, slave holds
            self._settle(1.0, D)        # clock high: master holds, slave copies it -> Q
            self.Q = self.L["Q"].copy() if self.mode == "neural" else np.round(self.L["Q"]).astype(np.int64)

    def outputs(self):
        """Evaluate only the gates that drive the outputs (they depend on the state alone)."""
        V = self._eval(self.out_groups)
        return {n: V[np.array(ws)].copy() for n, ws in self.net.outputs.items()}

    def regs(self):
        q = np.round(self.Q).astype(int)
        return {g["name"]: int(v) for g, v in zip(self.net.regs, q)}
