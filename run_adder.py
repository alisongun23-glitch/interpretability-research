"""Tiling trained logic gates into a 4-bit adder, and comparing it with one network
trained on 4-bit addition from scratch.

    python run_adder.py               # about 1-2 minutes, writes results_adder/

PART 1  Train XOR, AND and OR gate networks (2 -> 4 -> 1, tanh), many seeds each,
        at two safety margins.
PART 2  Tile them into a ripple-carry adder (the circuit in every CPU):
            full adder:  s = XOR(XOR(a,b), cin)
                         cout = OR(AND(a,b), AND(XOR(a,b), cin))
        Each gate position gets a different randomly chosen trained network, so the
        adder mixes gates built by different internal recipes.
        Two wiring modes:  SNAP (round each gate output to 0/1, like a real chip)
                           SOFT (pass the raw probability to the next gate)
PART 3  Train ordinary networks (8 -> 64 -> 64 -> 5, tanh) on 4-bit addition, with
        20% of the 256 problems held out.
PART 4  Look inside them for the adder's parts:
        a) probes   - can each layer linearly read out the tiled adder's intermediate
                      signals (a_i XOR b_i, a_i AND b_i, the carries)? vs untrained nets
        b) ablation - knock out the units most tied to carry k: do exactly the output
                      bits that depend on carry k break? vs random units
        c) carry insertion (a transplant of a tiled gate's output) - take the carry a
                      tiled gate computes for a DIFFERENT problem and write it into the
                      network along the probe direction. Does the network's answer change
                      exactly the way the tiled adder predicts? vs a random direction
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
from itertools import product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xorlab.model import train_batch, X as X4

OUT = "results_adder"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
T0 = time.time()
TRUTH = {"XOR": [0, 1, 1, 0], "AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1]}
SEEDS = list(range(20))
MARGINS = [2.0, 6.0]
TCFG = {"optimizer": "adam", "lr": 0.05, "betas": [0.9, 0.999], "max_steps": 20000,
        "log_every": 1000, "init_std": 1.0}
summary = {}

# ============ PART 1: gate networks ============
gates = {}
for m in MARGINS:
    for g, y in TRUTH.items():
        runs, *_ = train_batch(SEEDS, "tanh", 4, dict(TCFG, margin=m), y=y)
        ok = [r["params"] for r in runs if r["converged"]]
        gates[(g, m)] = ok
        summary[f"gate_{g}_margin{m:g}_converged"] = f"{len(ok)}/{len(runs)}"
print("gates trained", {k: len(v) for k, v in gates.items()}, f"{time.time() - T0:.1f}s")


def gate_fn(p):
    def f(a, b):                                   # a, b arrays in [0, 1]
        h = np.tanh(np.stack([a, b], -1) @ p["W1"] + p["b1"])
        return 1 / (1 + np.exp(-(h @ p["w2"] + p["b2"])))
    return f


# ============ PART 2: tiled adder ============
A_ALL, B_ALL = np.array(list(product(range(16), range(16)))).T
BITS = lambda v: np.stack([(v >> i) & 1 for i in range(4)], 1).astype(float)   # LSB first
TRUE_SUM = A_ALL + B_ALL


def tiled_adder(a_bits, b_bits, G, snap, carry_override=None):
    """G: dict position -> gate function. carry_override=(k, values) forces carry INTO bit k."""
    r = (lambda v: np.round(v)) if snap else (lambda v: v)
    c = np.zeros(len(a_bits)); out, inter = [], {}
    for i in range(4):
        if carry_override is not None and carry_override[0] == i:
            c = carry_override[1].astype(float)
        x1 = r(G[("XOR1", i)](a_bits[:, i], b_bits[:, i]))
        s = r(G[("XOR2", i)](x1, c))
        g1 = r(G[("AND1", i)](a_bits[:, i], b_bits[:, i]))
        g2 = r(G[("AND2", i)](x1, c))
        c = r(G[("OR", i)](g1, g2))
        inter[f"xor{i}"], inter[f"and{i}"], inter[f"carry{i + 1}"] = x1, g1, c
        out.append(s)
    out.append(c)
    return np.stack(out, 1), inter


def random_adder_gates(m, rng):
    G = {}
    for i in range(4):
        for slot, g in [("XOR1", "XOR"), ("XOR2", "XOR"), ("AND1", "AND"), ("AND2", "AND"), ("OR", "OR")]:
            pool = gates[(g, m)]
            G[(slot, i)] = gate_fn(pool[rng.integers(len(pool))])
    return G


def to_int(bits):
    return (np.round(bits) * (2 ** np.arange(5))).sum(1).astype(int)


rng = np.random.default_rng(0)
Ab, Bb = BITS(A_ALL), BITS(B_ALL)
tile_res = {}
for m in MARGINS:
    for snap in (True, False):
        accs, devs = [], []
        for trial in range(20):                    # 20 different random gate assemblies
            G = random_adder_gates(m, rng)
            out, _ = tiled_adder(Ab, Bb, G, snap)
            accs.append(np.mean(to_int(out) == TRUE_SUM))
            exact = np.stack([(TRUE_SUM >> i) & 1 for i in range(5)], 1)
            devs.append(np.mean(np.abs(out - exact), 0))
        key = f"margin{m:g}_{'snap' if snap else 'soft'}"
        tile_res[key] = {"mean_acc": float(np.mean(accs)), "min_acc": float(np.min(accs)),
                         "mean_abs_dev_per_bit": np.mean(devs, 0).tolist()}
summary["tiled_adder"] = tile_res
print("tiled adder", {k: round(v["mean_acc"], 3) for k, v in tile_res.items()})

# reference intermediates from one snapped, high-margin tiled adder (exact by construction)
G_ref = random_adder_gates(MARGINS[-1], np.random.default_rng(1))
_, INTER = tiled_adder(Ab, Bb, G_ref, True)
INTER = {k: v.astype(int) for k, v in INTER.items()}

# ============ PART 3: monolithic networks ============
Xin = np.concatenate([Ab, Bb], 1)                              # 256 x 8
Yout = np.stack([(TRUE_SUM >> i) & 1 for i in range(5)], 1).astype(float)
perm = np.random.default_rng(7).permutation(256)
TR, TE = perm[:205], perm[205:]


class MLP:
    def __init__(self, sizes, seed):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def forward(self, x, edit=None):
        """Returns list of hidden layers and output logits. edit(layer_index, h) -> h."""
        hs = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(z)
                if edit is not None:
                    h = edit(i, h)
                hs.append(h)
            else:
                return hs, z

    def step(self, x, y, lr=0.003):
        hs, z = self.forward(x)
        d = (1 / (1 + np.exp(-z)) - y) / len(x)
        acts = [x] + hs
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d; gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        self.t += 1
        for j, (p, g) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            p -= lr * (self.m[j] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.t)) + 1e-8)


def acc_sum(net, idx, edit=None):
    _, z = net.forward(Xin[idx], edit)
    return float(np.mean(to_int((z > 0).astype(float)) == TRUE_SUM[idx]))


MONO_SEEDS = [0, 1, 2, 3, 4]
WIDTHS = [16, 64]
NETS, UNTR, mono = {}, {}, []
for W in WIDTHS:
    NETS[W], UNTR[W] = [], []
    for sd in MONO_SEEDS:
        UNTR[W].append(MLP([8, W, W, 5], 1000 + sd))
        net = MLP([8, W, W, 5], sd)
        for _ in range(8000):
            net.step(Xin[TR], Yout[TR])
        NETS[W].append(net)
        mono.append({"width": W, "seed": sd, "train_acc": acc_sum(net, TR), "test_acc": acc_sum(net, TE)})
summary["monolithic"] = mono
print("monolithic", [(m["width"], m["train_acc"], round(m["test_acc"], 2)) for m in mono], f"{time.time() - T0:.1f}s")


# ============ PART 4a: probes ============
def probe(H, y, tr=TR, te=TE):
    Hb = np.hstack([H, np.ones((len(H), 1))])
    w = np.linalg.lstsq(Hb[tr], 2 * y[tr] - 1, rcond=None)[0]
    return float(np.mean((Hb[te] @ w > 0) == (y[te] == 1))), w


SIGNALS = [f"xor{i}" for i in range(4)] + [f"and{i}" for i in range(4)] + [f"carry{i}" for i in range(1, 5)]
probe_tab = {"raw_input": {s: probe(Xin, INTER[s])[0] for s in SIGNALS}}
for W in WIDTHS:
    for kind, group in [("trained", NETS[W]), ("untrained", UNTR[W])]:
        probe_tab[f"{kind}_w{W}"] = {s: np.mean([[probe(n.forward(Xin)[0][L], INTER[s])[0] for L in range(2)]
                                                  for n in group], 0).tolist() for s in SIGNALS}
summary["probes"] = probe_tab

# ============ PART 4b: ablation of carry-correlated units ============
def bit_acc(net, edit=None):
    _, z = net.forward(Xin, edit)
    return np.mean((z > 0) == (Yout == 1), 0)


abl = {}
r_ab = np.random.default_rng(3)
for W in WIDTHS:
    K = max(2, W // 8)
    for s in ["carry1", "carry2", "carry3"]:
        rows_c, rows_r = [], []
        for net in NETS[W]:
            L = 1
            H = net.forward(Xin)[0][L]; Hm = H.mean(0)
            c = np.nan_to_num([abs(np.corrcoef(H[:, j], INTER[s])[0, 1]) for j in range(W)])
            U = np.argsort(-c)[:K]
            R = r_ab.choice(W, K, replace=False)

            def mk(units):
                def e(i, h):
                    if i != L:
                        return h
                    h = h.copy(); h[:, units] = Hm[units]; return h
                return e
            rows_c.append(bit_acc(net, mk(U))); rows_r.append(bit_acc(net, mk(R)))
        abl[f"w{W}_{s}"] = {"k_units": K, "carry_units": np.mean(rows_c, 0).tolist(),
                            "random_units": np.mean(rows_r, 0).tolist()}
summary["ablation_bit_accuracy"] = abl

# ============ PART 4c: carry insertion ============
# For each problem, a SOURCE problem with the same high bits (positions >= k) but the
# opposite carry into bit k. The tiled adder predicts the answer with that carry forced.
def sources(k):
    hi = (A_ALL >> k, B_ALL >> k)
    src = np.full(256, -1)
    for i in range(256):
        cand = np.where((hi[0] == hi[0][i]) & (hi[1] == hi[1][i]) & (INTER[f"carry{k}"] != INTER[f"carry{k}"][i]))[0]
        if len(cand):
            src[i] = cand[0]
    return src


ins = {}
r_in = np.random.default_rng(5)
DOSES = [1.5, 4.0, 10.0]
for W in WIDTHS:
    for k in [1, 2, 3]:
        s = f"carry{k}"
        src = sources(k); valid = src >= 0
        out_cf, _ = tiled_adder(Ab, Bb, G_ref, True, carry_override=(k, 1 - INTER[s]))
        target = to_int(out_cf)
        res = {f"patch_L{L + 1}_{d}": [] for L in range(2) for d in ("probe", "random")}
        res.update({f"push{x:g}_probe": [] for x in DOSES})
        res["unchanged_after_patch_L2_probe"] = []
        for net in NETS[W]:
            Hs = net.forward(Xin)[0]
            for L in range(2):
                _, w = probe(Hs[L], INTER[s], tr=np.arange(256), te=np.arange(256))
                d = w[:-1] / np.linalg.norm(w[:-1])
                dr = r_in.normal(size=d.shape); dr /= np.linalg.norm(dr)
                for name, u in [("probe", d), ("random", dr)]:
                    def e(i, h, u=u, L=L):
                        if i != L:
                            return h
                        return h + ((Hs[L][src] - h) @ u)[:, None] * u[None, :]   # copy the source's coordinate along u
                    _, z = net.forward(Xin, e)
                    got = to_int((z > 0).astype(float))
                    res[f"patch_L{L + 1}_{name}"].append(float(np.mean(got[valid] == target[valid])))
                    if name == "probe" and L == 1:
                        res["unchanged_after_patch_L2_probe"].append(float(np.mean(got[valid] == TRUE_SUM[valid])))
            L = 1
            _, w = probe(Hs[L], INTER[s], tr=np.arange(256), te=np.arange(256))
            dv, b0 = w[:-1], w[-1]
            for x in DOSES:
                def e(i, h, x=x):
                    if i != L:
                        return h
                    goal = np.where(INTER[s] == 0, 1.0, -1.0) * x
                    return h + ((goal - (h @ dv + b0)) / (dv @ dv))[:, None] * dv[None, :]
                _, z = net.forward(Xin, e)
                got = to_int((z > 0).astype(float))
                res[f"push{x:g}_probe"].append(float(np.mean(got[valid] == target[valid])))
        ins[f"w{W}_{s}"] = {kk: float(np.mean(v)) for kk, v in res.items()}
summary["carry_insertion"] = ins


# ============ PART 4d: does the net collapse each problem to its total? ============
# eta^2 = share of a layer's variation explained by the total a+b alone (1.0 = problems
# with the same total look identical inside). Distance ratio = distance between two
# problems with the same total / distance between two random problems (0 = full collapse).
# (Swapping a whole layer with a same-total problem's activity is NOT a valid test: the
# network then simply answers that other problem, which has the same answer.)
def eta2(H, groups):
    tot = np.sum((H - H.mean(0)) ** 2)
    within = sum(np.sum((H[groups == g] - H[groups == g].mean(0)) ** 2) for g in np.unique(groups))
    return float(1 - within / tot)


pairs_same = [(i, j) for i in range(256) for j in range(i + 1, 256) if TRUE_SUM[i] == TRUE_SUM[j]]
ii, jj = np.array(pairs_same).T
rr = np.random.default_rng(9); ri, rj = rr.integers(0, 256, 20000), rr.integers(0, 256, 20000)
keep = ri != rj; ri, rj = ri[keep], rj[keep]


def dist_ratio(H):
    return float(np.mean(np.linalg.norm(H[ii] - H[jj], axis=1)) / np.mean(np.linalg.norm(H[ri] - H[rj], axis=1)))


collapse = {"input_eta2": eta2(Xin, TRUE_SUM), "input_distance_ratio": dist_ratio(Xin)}
for W in WIDTHS:
    for kind, group in [("trained", NETS[W]), ("untrained", UNTR[W])]:
        collapse[f"{kind}_w{W}_eta2"] = np.mean([[eta2(n.forward(Xin)[0][L], TRUE_SUM) for L in range(2)] for n in group], 0).tolist()
        collapse[f"{kind}_w{W}_distance_ratio"] = np.mean([[dist_ratio(n.forward(Xin)[0][L]) for L in range(2)] for n in group], 0).tolist()
summary["collapse_to_total"] = collapse
print("collapse", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in collapse.items()})
summary["runtime_seconds"] = time.time() - T0
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

# ============ figures ============
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3.6))
keys = list(tile_res)
ax.bar(range(len(keys)), [tile_res[k]["mean_acc"] for k in keys],
       yerr=[[tile_res[k]["mean_acc"] - tile_res[k]["min_acc"] for k in keys], [0] * len(keys)], capsize=4)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("_", "\n") for k in keys])
ax.set_ylabel("share of 256 sums correct"); ax.set_title("1. Tiled adder from trained gates (20 random assemblies)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_tiled_adder.png"), dpi=130); plt.close(fig)

fig, axes = plt.subplots(len(WIDTHS), 2, figsize=(13, 3.6 * len(WIDTHS)), sharey=True)
for row, W in enumerate(WIDTHS):
    for L in range(2):
        ax = axes[row, L]; x = np.arange(len(SIGNALS)); w = 0.27
        ax.bar(x - w, [probe_tab["raw_input"][s] for s in SIGNALS], w, label="raw input bits", color="lightgray")
        ax.bar(x, [probe_tab[f"untrained_w{W}"][s][L] for s in SIGNALS], w, label="untrained net")
        ax.bar(x + w, [probe_tab[f"trained_w{W}"][s][L] for s in SIGNALS], w, label="trained net")
        ax.set_xticks(x); ax.set_xticklabels(SIGNALS, rotation=45, fontsize=7); ax.axhline(0.5, ls=":", c="gray")
        ax.set_title(f"2. Probes, width {W}, hidden layer {L + 1}")
axes[0, 0].set_ylabel("held-out probe accuracy"); axes[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_probes.png"), dpi=130); plt.close(fig)

fig, axes = plt.subplots(len(WIDTHS), 3, figsize=(13, 3.2 * len(WIDTHS)), sharey=True)
for row, W in enumerate(WIDTHS):
    for col, s in enumerate(["carry1", "carry2", "carry3"]):
        ax = axes[row, col]; A_ = abl[f"w{W}_{s}"]; x = np.arange(5)
        ax.bar(x - 0.2, A_["carry_units"], 0.4, label=f"top {A_['k_units']} {s} units")
        ax.bar(x + 0.2, A_["random_units"], 0.4, label=f"{A_['k_units']} random units")
        ax.set_xticks(x); ax.set_xticklabels([f"bit {i}" for i in range(5)])
        ax.axvspan(int(s[-1]) - 0.5, 4.5, color="orange", alpha=0.08)
        ax.set_title(f"3. Width {W}: ablate {s} units (shaded = depends on it)", fontsize=8); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_ablation.png"), dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, len(WIDTHS), figsize=(12, 3.8), sharey=True)
cols = ["patch_L1_probe", "patch_L1_random", "patch_L2_probe", "patch_L2_random"] + [f"push{x:g}_probe" for x in DOSES]
for ax, W in zip(axes, WIDTHS):
    x = np.arange(3); w = 0.8 / len(cols)
    for j, c in enumerate(cols):
        ax.bar(x + j * w, [ins[f"w{W}_carry{k}"][c] for k in [1, 2, 3]], w, label=c)
    ax.set_xticks(x + 0.4); ax.set_xticklabels([f"carry{k}" for k in [1, 2, 3]]); ax.set_ylim(0, 1)
    ax.set_title(f"4. Width {W}: insert the opposite carry")
axes[0].set_ylabel("answer matches tiled-adder prediction"); axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "4_carry_insertion.png"), dpi=130); plt.close(fig)
for k, v in ins.items():
    print(k, {kk: round(x, 2) for kk, x in v.items()})
for k, v in abl.items():
    print("ablation", k, [round(x, 2) for x in v["carry_units"]], "random", [round(x, 2) for x in v["random_units"]])
for k, v in probe_tab.items():
    if k != "raw_input":
        print(k, {s: [round(x, 2) for x in val] for s, val in v.items() if s.startswith("xor")})
print(f"done {time.time() - T0:.1f}s")
