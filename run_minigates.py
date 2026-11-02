"""How small can a neural logic gate be and still work inside a big machine?

    python run_minigates.py           # about 2 minutes, writes results_minigates/

PART A  Architecture search. Every gate (NOT AND OR NAND NOR XOR XNOR) is trained with 100
        seeds on six architectures, from one sigmoid neuron up to the 2 -> 4 -> 1 network
        used in run_adder.py. Smallest architecture that trains = candidate.
PART B  "Works on its own" is not enough. Inside a machine a gate sees inputs that are
        themselves outputs of gates (0.03, not 0). Static-discipline test: every input
        within eps of a rail must give an output within eps of the correct rail. If that
        holds, signals cannot drift through any depth of gates or around any feedback loop.
        Swept over the training margin for the candidate architecture of each gate.
        Prespecified rule: the library uses the smallest margin at which >= 90% of
        converged seeds of EVERY gate pass at the interface level eps = 0.05.
PART C  Weight precision: fewest bits per weight that keep each library gate passing.
PART D  Stress tests: 400-layer random circuits and a NAND latch, pure analog
        (no rounding anywhere), library gates vs margin-2 gates (the run_adder.py margin).
Output: results_minigates/library.json (the gate pools used by the snake machine),
        summary.json, figures/.
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatesnake import gates as G

OUT = "results_minigates"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
T0 = time.time()
SEEDS = list(range(100))
ARCH_MARGIN = 4.0
MARGINS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
EPS_I = 0.05            # interface: logic 0 is any value <= 0.05, logic 1 any value >= 0.95
PASS_FRAC = 0.9
POOL = 8                # trained networks kept per gate; each gate instance in a machine draws one
summary = {"eps_interface": EPS_I}


def arch_list(g):
    return ["neuron"] if G.N_IN[g] == 1 else list(G.ARCHS)


# ============ PART A: architecture search ============
arch_tab = {}
for g in G.TRUTH:
    arch_tab[g] = {}
    for a in arch_list(g):
        runs = G.train(g, a, ARCH_MARGIN, SEEDS)
        ok = [r for r in runs if r["converged"]]
        arch_tab[g][a] = {"params": G.n_params(G.N_IN[g], a), "neurons": G.n_neurons(a),
                          "converged": len(ok) / len(runs),
                          "median_steps": float(np.median([r["steps"] for r in ok])) if ok else None}
minimal = {g: min((a for a in arch_tab[g] if arch_tab[g][a]["converged"] > 0),
                  key=lambda a: arch_tab[g][a]["params"]) for g in G.TRUTH}
summary["arch_search"] = arch_tab
summary["minimal_arch"] = minimal
print("minimal archs", minimal, f"{time.time() - T0:.0f}s")

# ============ PART B: margin sweep, static discipline ============
sweep, nets = {}, {}
for g in G.TRUTH:
    sweep[g] = {}
    for m in MARGINS:
        runs = G.train(g, minimal[g], m, SEEDS, max_steps=40000)
        ok = [r for r in runs if r["converged"]]
        passing = [r for r in ok if G.restores(r["params"], g, EPS_I)]
        rng_ = np.array([G.restore_range(r["params"], g) for r in ok[:30]])
        nets[(g, m)] = (ok, passing)
        sweep[g][m] = {"converged": len(ok) / len(runs), "pass_at_eps": len(passing) / max(len(ok), 1),
                       "median_eps_lo": float(np.nanmedian(rng_[:, 0])) if np.isfinite(rng_).any() else None,
                       "median_eps_hi": float(np.nanmedian(rng_[:, 1])) if np.isfinite(rng_).any() else None,
                       "median_max_weight": float(np.median([max(np.abs(v).max() for v in r["params"].values() if v.size)
                                                             for r in ok])) if ok else None}
summary["margin_sweep"] = {g: {str(m): v for m, v in d.items()} for g, d in sweep.items()}
LIB_M = next(m for m in MARGINS if all(sweep[g][m]["pass_at_eps"] >= PASS_FRAC and sweep[g][m]["converged"] > 0
                                     for g in G.TRUTH))
summary["library_margin"] = LIB_M
print("library margin", LIB_M, {g: sweep[g][LIB_M]["pass_at_eps"] for g in G.TRUTH}, f"{time.time() - T0:.0f}s")

# ============ PART C: weight precision ============
lib = {}
bits_tab = {}
for g in G.TRUTH:
    pool = nets[(g, LIB_M)][1][:POOL]
    lib[g] = pool
    need = []
    for r in pool:
        b_ok = None
        for b in range(2, 13):
            q, _ = G.quantize(r["params"], b)
            if G.restores(q, g, EPS_I):
                b_ok = b; break
        need.append(b_ok)
    bits_tab[g] = need
summary["bits_needed"] = bits_tab
print("bits", {g: (min(v), max(v)) for g, v in bits_tab.items()})


# ============ PART D: stress tests, fully analog ============
TYPES = list(G.TRUTH)


def pools_for(m):
    """Up to POOL converged nets per gate at margin m (passing or not: we want to see failures)."""
    return {g: [r["params"] for r in nets[(g, m)][0][:POOL]] for g in TYPES}


def run_random_circuit(pools, seed, width=128, depth=400, in_noise=0.0):
    r = np.random.default_rng(seed)
    exact = r.integers(0, 2, width).astype(float)
    soft = np.clip(np.abs(exact - r.uniform(0, in_noise, width)), 0, 1)
    wrong, dev = [], []
    for _ in range(depth):
        t = r.integers(0, len(TYPES), width); i = r.integers(0, width, width); j = r.integers(0, width, width)
        k = r.integers(0, POOL, width)
        new_e = np.empty(width); new_s = np.empty(width)
        for ti, g in enumerate(TYPES):
            sel = t == ti
            if not sel.any():
                continue
            tt = np.array(G.TRUTH[g])
            if G.N_IN[g] == 1:
                new_e[sel] = tt[exact[i[sel]].astype(int)]
                xs = soft[i[sel]][:, None]
            else:
                new_e[sel] = tt[(2 * exact[i[sel]] + exact[j[sel]]).astype(int)]
                xs = np.stack([soft[i[sel]], soft[j[sel]]], 1)
            pl = pools[g]
            out = np.empty(sel.sum())
            for kk in range(len(pl)):
                s2 = k[sel] % len(pl) == kk
                if s2.any():
                    out[s2] = G.forward(pl[kk], xs[s2])
            new_s[sel] = out
        exact, soft = new_e, new_s
        wrong.append(np.mean(np.round(soft) != exact)); dev.append(np.max(np.abs(soft - exact)))
    return np.array(wrong), np.array(dev)


STRESS_M = sorted(set([2.0, 3.0, LIB_M]))
stress = {}
for m in STRESS_M:
    pl = pools_for(m)
    for noise in (0.0, EPS_I):
        W, D = zip(*[run_random_circuit(pl, s, in_noise=noise) for s in range(10)])
        W, D = np.array(W), np.array(D)
        stress[(m, noise)] = (W.mean(0), np.median(D, 0))
summary["stress_random_circuits"] = {f"margin{m:g}_noise{n:g}": {
    "wrong_frac_at_layer": {str(L): float(w[L - 1]) for L in (1, 10, 100, 400)},
    "median_max_dev_at_layer": {str(L): float(d[L - 1]) for L in (1, 10, 100, 400)}}
    for (m, n), (w, d) in stress.items()}


def latch_hold(nand, start, steps=2000):
    """SR latch: Q = NAND(S', Qb), Qb = NAND(R', Q) with S' = R' = 1 (hold), analog, synchronous."""
    q, qb = start
    for _ in range(steps):
        q, qb = (G.forward(nand, np.array([1.0, qb])), G.forward(nand, np.array([1.0, q])))
    return float(q), float(qb)


latch = {}
for m in STRESS_M:
    nand = nets[("NAND", m)][0][0]["params"]
    latch[f"margin{m:g}"] = {"from_1_0": latch_hold(nand, (1.0, 0.0)), "from_0.7_0.3": latch_hold(nand, (0.7, 0.3)),
                             "from_0.3_0.7": latch_hold(nand, (0.3, 0.7))}
summary["latch_hold"] = latch
print("stress + latch done", f"{time.time() - T0:.0f}s")

# ============ library.json for the snake machine ============
def to_list(p):
    return {k: np.asarray(v).tolist() for k, v in p.items()}


library = {"margin": LIB_M, "eps_interface": EPS_I, "activation": "sigmoid",
           "gates": {g: {"arch": minimal[g], "hidden": G.ARCHS[minimal[g]][0], "skip": G.ARCHS[minimal[g]][1],
                         "cost": G.gate_cost(g, minimal[g]),
                         "pool": [to_list(r["params"]) for r in lib[g]],
                         "pool_seeds": [r["seed"] for r in lib[g]]} for g in G.TRUTH}}
# reference library: the earlier experiments' gate size (2 -> 4 -> 1) at their margin, for cost comparison only
library["baseline"] = {g: {"arch": "neuron" if G.N_IN[g] == 1 else "mlp-4",
                           "cost": G.gate_cost(g, "neuron" if G.N_IN[g] == 1 else "mlp-4")} for g in G.TRUTH}
json.dump(library, open(os.path.join(OUT, "library.json"), "w"), indent=1)
summary["library_cost"] = {g: library["gates"][g]["cost"] for g in G.TRUTH}
summary["runtime_s"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ============ figures ============
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2, "lines.solid_capstyle": "round"})

# 1: convergence by architecture
archs = list(G.ARCHS)
M_ = np.full((len(G.TRUTH), len(archs)), np.nan)
for i, g in enumerate(G.TRUTH):
    for j, a in enumerate(archs):
        if a in arch_tab[g]:
            M_[i, j] = arch_tab[g][a]["converged"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.grid(False)
im = ax.imshow(M_, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#86b6ef", "#1c5cab"]),
               vmin=0, vmax=1, aspect="auto")
for i in range(M_.shape[0]):
    for j in range(M_.shape[1]):
        if np.isfinite(M_[i, j]):
            v = M_[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
            if archs[j] == minimal[list(G.TRUTH)[i]]:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=INK, lw=1.5))
        else:
            ax.text(j, i, "–", ha="center", va="center", color=MUTED)
ax.set_xticks(range(len(archs)))
ax.set_xticklabels([f"{a}\n{G.n_params(2, a)} params" for a in archs], fontsize=8)
ax.set_yticks(range(len(G.TRUTH))); ax.set_yticklabels(list(G.TRUTH))
ax.set_title(f"Share of 100 seeds that learn the truth table (margin {ARCH_MARGIN:g}); box = smallest that works",
             fontsize=9, loc="left")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_arch_search.png"), dpi=150); plt.close(fig)

# 2: static discipline vs margin
fig, ax = plt.subplots(figsize=(6.4, 3.4))
groups = [("NOT (1 input)", ["NOT"], C1),
          ("every 2-input gate (AND OR NAND NOR XOR XNOR)", ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"], C2)]
for name, gs, c in groups:
    ys = [np.mean([sweep[g][m]["pass_at_eps"] for g in gs]) for m in MARGINS]
    ax.plot(MARGINS, ys, color=c, marker="o", ms=5, mec="#fcfcfb", mew=1.5, label=name)
ax.axvline(LIB_M, color=MUTED, lw=1); ax.text(LIB_M + 0.1, 0.05, f"library\nmargin {LIB_M:g}", color=INK2, fontsize=8)
ax.set_xlabel("training margin (|logit| required on the truth table)")
ax.set_ylabel(f"share passing static discipline\n(inputs within {EPS_I} of rail -> output within {EPS_I})")
ax.set_ylim(-0.03, 1.05); ax.legend(frameon=False, fontsize=8, loc="center right")
ax.set_title("Truth table right is not enough: small margins let errors grow gate by gate", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_margin_vs_discipline.png"), dpi=150); plt.close(fig)

# 3: transfer curves of an inverter
fig, ax = plt.subplots(figsize=(4.6, 4.0))
xs = np.linspace(0, 1, 401)
ax.fill_between([0, 1], 0, EPS_I, color=GRID, alpha=0.6, lw=0); ax.fill_between([0, 1], 1 - EPS_I, 1, color=GRID, alpha=0.6, lw=0)
ax.plot(xs, xs, color=BASE, lw=1)
for m, c in [(2.0, C2), (LIB_M, C1)]:
    p = nets[("NOT", m)][0][0]["params"]
    ax.plot(xs, G.forward(p, xs[:, None]), color=c, label=f"NOT gate, margin {m:g}")
ax.set_xlabel("input"); ax.set_ylabel("output"); ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Inverter transfer curve; shaded = within {EPS_I} of a rail", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_transfer_curve.png"), dpi=150); plt.close(fig)

# 4: random deep circuits
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
cols = {2.0: C2, 3.0: "#eda100", LIB_M: C1}
for m in STRESS_M:
    w, d = stress[(m, EPS_I)]
    L = np.arange(1, len(w) + 1)
    axs[0].plot(L, w, color=cols.get(m, C3), label=f"margin {m:g}")
    axs[1].plot(L, d, color=cols.get(m, C3), label=f"margin {m:g}")
axs[0].set_ylabel("share of signals wrong"); axs[1].set_ylabel("worst distance from the exact 0/1")
axs[1].axhline(0.5, color=MUTED, lw=1)
for a in axs:
    a.set_xlabel("layer of gates"); a.set_xscale("log")
axs[0].legend(frameon=False, fontsize=8)
axs[0].text(30, 0.015, "margins 3 and 4: no wrong signals", fontsize=8, color=INK2)
axs[0].set_title(f"400-layer random circuits, 128 wires, input noise {EPS_I}", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "4_deep_circuits.png"), dpi=150); plt.close(fig)
print("done", f"{time.time() - T0:.0f}s")
