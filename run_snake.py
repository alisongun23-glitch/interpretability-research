"""Snake built from the smallest working neural logic gates. How big is it, does it work,
and what does it cost?

    python run_minigates.py     # first: trains the gate library (results_minigates/library.json)
    python run_snake.py         # about 5 minutes, writes results_snake/ and snake_ui/net16.json

PART 1  Size. Build the machine for boards 4x4 ... 64x64 and count gates, flip-flops,
        neurons, weights and multiply-adds per tick for three gate libraries:
          minimal   - the gates from run_minigates.py (1 neuron; XOR/XNOR 2 neurons)
          NAND-only - the whole machine lowered to one gate, the 1-neuron NAND
          baseline  - same netlist with the 2 -> 4 -> 1 gates of run_adder.py (cost only)
PART 2  Correctness. Run the machine in five ways against a plain-Python snake, tick by tick,
        driven by a bot that eats, crashes and resets: Boolean gates, Boolean gates with
        NAND flip-flops, neural gates, neural gates with neural NAND flip-flops (the fully
        analog machine, nothing rounded anywhere), and the NAND-only analog machine.
PART 3  Where the cost goes, per part of the machine.
PART 4  Does it still work with smaller gates? Same fully analog machine with gate pools
        trained at margins 1.5 ... 4, and with noise added to every neuron.
PART 5  Speed, and export of the 16x16 machine for the browser player (snake_ui/).
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small arrays; must precede numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatesnake import gates as G
from gatesnake.design import build_snake, reg_part
from gatesnake.hdl import lower_to_nand, GATES, DFF_NANDS
from gatesnake.ref import SnakeRef
from gatesnake.sim import Machine, load_library, make_library
from gatesnake.bot import bot_inputs, cycle_inputs

OUT = "results_snake"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs("snake_ui", exist_ok=True)
T0 = time.time()
LIB = json.load(open("results_minigates/library.json"))
LIB_M = LIB["margin"]
COST = {"minimal": {t: LIB["gates"][t]["cost"] for t in GATES},
        "NAND-only": {t: LIB["gates"]["NAND"]["cost"] for t in GATES},
        "baseline": {t: LIB["baseline"][t]["cost"] for t in GATES}}
LATCH_ITERS = 6
summary = {"library_margin": LIB_M}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


# ============ PART 1: size ============
SIZES = [4, 8, 16, 32, 64]
builds, size_tab = {}, {}
for N in SIZES:
    c, info = build_snake(N)
    net = c.finalize()
    nn = lower_to_nand(c).finalize()
    builds[N] = (c, net, nn, info)
    row = {"board": f"{N}x{N}", "counter_bits": info["W"]}
    for lib_name, net_ in (("minimal", net), ("NAND-only", nn), ("baseline", net)):
        s = net_.stats(COST[lib_name])
        dff_macs = len(net_.regs) * DFF_NANDS * COST[lib_name]["NAND"]["macs"]
        s["macs_per_tick_simulated"] = s["macs_comb"] + 2 * LATCH_ITERS * dff_macs
        row[lib_name] = s
    size_tab[N] = row
    s = row["minimal"]
    log(f"N={N}: {s['total_gates']:,} gates ({s['comb_gates']:,} logic + {s['dff_gates']:,} in {s['registers']:,} flip-flops), "
        f"{s['neurons']:,} neurons, {s['params']:,} weights, depth {s['logic_depth']} gates / {s['neuron_depth_comb']} neurons")
summary["size"] = {str(N): v for N, v in size_tab.items()}
summary["unique_trained_weights"] = {
    "minimal": sum(len(LIB["gates"][t]["pool"]) * LIB["gates"][t]["cost"]["params"] for t in GATES),
    "NAND-only": len(LIB["gates"]["NAND"]["pool"]) * LIB["gates"]["NAND"]["cost"]["params"]}


# ============ PART 2: correctness ============
def run_vs_ref(net, N, T, seed=0, driver=None, **mk):
    m = Machine(net, seed=seed, **mk)
    ref = SnakeRef(N); r = np.random.default_rng(100 + seed)
    stats = {"ticks": 0, "eats": 0, "games": 1, "max_len": 0, "first_error": None}
    t0 = time.time()
    for i in range(T):
        inp = (driver or bot_inputs)(ref, r)
        L0 = ref.len
        m.step(inp); ref.step(**inp)
        stats["ticks"] = i + 1
        stats["eats"] += int(ref.len > L0); stats["games"] += inp["rst"]; stats["max_len"] = max(stats["max_len"], ref.len)
        if m.regs() != ref.regs():
            stats["first_error"] = i + 1
            break
    stats["ms_per_tick"] = (time.time() - t0) / stats["ticks"] * 1e3
    stats["worst_signal_dev"] = float(m.worst)
    return stats


lib_min = load_library()
lib_nand = load_library(nand_only=True)
MODES = [("Boolean gates, ideal flip-flops", {"mode": "logic", "dff": "ideal"}, False),
         ("Boolean gates, NAND flip-flops", {"mode": "logic", "dff": "gates"}, False),
         ("neural gates, ideal flip-flops", {"mode": "neural", "dff": "ideal", "library": lib_min}, False),
         ("neural gates, neural NAND flip-flops (fully analog)", {"mode": "neural", "dff": "gates", "library": lib_min}, False),
         ("NAND-only, fully analog", {"mode": "neural", "dff": "gates", "library": lib_nand}, True)]
VERIFY = {8: 4000, 16: 3000, 32: 400}
verify = {}
for N, T in VERIFY.items():
    c, net, nn, info = builds[N]
    for name, mk, use_nand in MODES:
        if N == 32 and mk["mode"] == "logic":
            continue
        runs = [run_vs_ref(nn if use_nand else net, N, T, seed=s, latch_iters=LATCH_ITERS, **mk) for s in range(3 if N < 32 else 1)]
        verify[f"{N}x{N} | {name}"] = {
            "ticks": sum(r["ticks"] for r in runs), "errors": sum(r["first_error"] is not None for r in runs),
            "first_errors": [r["first_error"] for r in runs], "eats": sum(r["eats"] for r in runs),
            "games": sum(r["games"] for r in runs), "max_len": max(r["max_len"] for r in runs),
            "ms_per_tick": float(np.mean([r["ms_per_tick"] for r in runs])),
            "worst_signal_dev": max(r["worst_signal_dev"] for r in runs)}
        v = verify[f"{N}x{N} | {name}"]
        log(f"  {N}x{N} {name}: {v['ticks']} ticks, {v['errors']} runs with errors, {v['eats']} eats, {v['games']} games, "
            f"max len {v['max_len']}, {v['ms_per_tick']:.2f} ms/tick, worst signal {v['worst_signal_dev']:.3f}")
# fill the board: a player that follows a loop through every cell grows the snake to N*N - 1,
# so every counter runs through its whole range
for N, T in ((8, 1700), (16, 18300)):
    g = run_vs_ref(builds[N][1], N, T, seed=7, driver=cycle_inputs, mode="neural", dff="gates", library=lib_min,
                   latch_iters=LATCH_ITERS)
    verify[f"{N}x{N} | fully analog, board-filling game"] = g
    log(f"  {N}x{N} board-filling game", g)
summary["verify"] = verify

# plain Python reference speed
ref = SnakeRef(16); r = np.random.default_rng(0); t0 = time.time()
for i in range(3000):
    ref.step(**bot_inputs(ref, r))
summary["python_reference_ms_per_tick_16"] = (time.time() - t0) / 3000 * 1e3

# ============ PART 3: breakdown ============
c, net, nn, info = builds[16]
bd = net.breakdown(reg_part)
summary["breakdown_16"] = bd
summary["gates_by_type_16"] = net.counts()

# ============ PART 4: smaller gates, noise ============
SURV_T = 1500
margins = [1.5, 2.0, 2.5, 3.0, 3.25, 3.5, LIB_M]
surv = {}
for m in margins:
    pools = {}
    for g in G.TRUTH:
        runs = G.train(g, LIB["gates"][g]["arch"], m, list(range(40)), max_steps=40000)
        pools[g] = [r["params"] for r in runs if r["converged"]][:8]
    lib_m = make_library(pools, margin=m)
    res = [run_vs_ref(net, 16, SURV_T, seed=s, mode="neural", dff="gates", library=lib_m, latch_iters=LATCH_ITERS)
           for s in range(3)]
    surv[m] = {"first_errors": [r["first_error"] for r in res], "ticks_ok": [r["ticks"] - (r["first_error"] is not None) for r in res],
               "worst_signal_dev": max(r["worst_signal_dev"] for r in res)}
    log(f"  margin {m}: first errors {surv[m]['first_errors']}, worst signal {surv[m]['worst_signal_dev']:.3f}")
    if m == 2.0:
        pools_m2 = pools
    if m == 3.0:
        pools_m3 = pools
    # the self-collision check is a 256-input OR tree fed with zeros: follow a zero up the tree
    orp, andp = pools["OR"][0], pools["AND"][0]
    z = float(G.forward(andp, np.array([0.0, 1.0])))            # a typical analog 0 leaving an AND gate
    lv = [z]
    for _ in range(8):
        z = float(G.forward(orp, np.array([z, z]))); lv.append(z)
    surv[m]["or_tree_zero_by_level"] = lv
summary["margin_survival_16"] = {str(k): v for k, v in surv.items()}

noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
noise = {}
for sd in noise_levels:
    res = [run_vs_ref(net, 16, SURV_T, seed=s, mode="neural", dff="gates", library=lib_min, latch_iters=LATCH_ITERS, noise=sd)
           for s in range(3)]
    noise[sd] = {"first_errors": [r["first_error"] for r in res], "ticks_ok": [r["ticks"] - (r["first_error"] is not None) for r in res]}
    log(f"  noise {sd}: first errors {noise[sd]['first_errors']}")
summary["noise_survival_16"] = {str(k): v for k, v in noise.items()}


# ============ PART 5: export for the browser ============
def export(N, net, path, trace_T=2000):
    lib = lib_min
    m = Machine(net, mode="neural", library=lib, dff="gates", latch_iters=LATCH_ITERS, seed=0)
    TYPES = list(GATES)
    gates = net.gates
    pools = {}
    for tag, pl in (("m%g" % LIB_M, {t: LIB["gates"][t]["pool"] for t in GATES}), ("m3", pools_m3), ("m2", pools_m2)):
        L = make_library(pl)
        pools[tag] = {t: [{"U": p["U"].tolist(), "c": p["c"].tolist(), "w": p["w"].tolist(), "v": p["v"].tolist(), "bo": p["bo"]}
                          for p in L[t]] for t in GATES}
    # a test trace: bot inputs and a hash of the register bits after every tick
    ref = SnakeRef(N); r = np.random.default_rng(5); ins, hashes = [], []
    for i in range(trace_T):
        inp = bot_inputs(ref, r); ref.step(**inp)
        ins.append([inp[k] for k in ("up", "down", "left", "right", "rst")])
        h = 0
        rg = ref.regs()
        for g in net.regs:
            h = (h * 31 + rg[g["name"]]) & 0xFFFFFFFF
        hashes.append(h)
    s = size_tab[N]
    data = {
        "N": N, "W": builds[N][3]["W"], "n_wires": net.n_wires, "types": TYPES, "latch_iters": LATCH_ITERS,
        "inputs": net.inputs, "outputs": net.outputs,
        "regs": {"q": [g["q"] for g in net.regs], "d": [g["d"] for g in net.regs], "init": [g["init"] for g in net.regs]},
        "gates": {"out": [w for w, _, _ in gates], "a": [a[0] for _, _, a in gates], "b": [a[-1] for _, _, a in gates],
                  "type": [TYPES.index(t) for _, t, _ in gates], "pick": [m.choice[w] for w, _, _ in gates]},
        "out_cone": sorted(int(i) for i in np.flatnonzero(np.isin([w for w, _, _ in gates],
                                                                  np.concatenate([g.out for g in m.out_groups])))),
        "dff_pick": [int(x) for x in np.random.default_rng(1).integers(0, 8, len(net.regs) * DFF_NANDS)],
        "pools": pools, "default_pool": "m%g" % LIB_M,
        "trace": {"inputs": ins, "hashes": hashes},
        "stats": {"total_gates": s["minimal"]["total_gates"], "comb_gates": s["minimal"]["comb_gates"],
                  "registers": s["minimal"]["registers"], "neurons": s["minimal"]["neurons"],
                  "params": s["minimal"]["params"], "macs_per_tick": s["minimal"]["macs_per_tick_simulated"],
                  "logic_depth": s["minimal"]["logic_depth"], "neuron_depth": s["minimal"]["neuron_depth_comb"],
                  "gates_by_type": net.counts()},
    }
    json.dump(data, open(path, "w"), separators=(",", ":"))
    return os.path.getsize(path)


for N in (8, 16):
    sz = export(N, builds[N][1], f"snake_ui/net{N}.json")
    log(f"exported snake_ui/net{N}.json ({sz / 1e6:.2f} MB)")
summary["runtime_s"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

# ============ figures ============
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2, "lines.solid_capstyle": "round"})

# 1: size vs board
fig, ax = plt.subplots(figsize=(6.2, 3.6))
cells = np.array([N * N for N in SIZES])
for lib_name, col in (("baseline", C3), ("NAND-only", C2), ("minimal", C1)):
    ys = np.array([size_tab[N][lib_name]["params"] for N in SIZES])
    ax.plot(cells, ys, color=col, marker="o", ms=5, mec="#fcfcfb", mew=1.5,
            label={"minimal": "minimal gates (1-2 neurons)", "NAND-only": "NAND neurons only",
                   "baseline": "2-4-1 gates of run_adder.py"}[lib_name])
    ax.text(cells[-1] * 1.15, ys[-1], f"{ys[-1] / 1e6:.1f}M", fontsize=8, color=INK2,
            va={"minimal": "top", "NAND-only": "bottom"}.get(lib_name, "center"))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(cells); ax.set_xticklabels([f"{N}x{N}" for N in SIZES])
ax.set_xlim(cells[0] / 1.5, cells[-1] * 2.2)
ax.set_xlabel("board"); ax.set_ylabel("weights in the whole machine")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Snake machine size: weights grow with the number of cells", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_size_vs_board.png"), dpi=150); plt.close(fig)

# 2: breakdown at 16x16
parts = sorted(bd, key=lambda k: -(bd[k]["comb"] + bd[k]["dff"]))
fig, ax = plt.subplots(figsize=(6.2, 3.0))
y = np.arange(len(parts))[::-1]
comb = np.array([bd[p]["comb"] for p in parts]); dff = np.array([bd[p]["dff"] for p in parts])
ax.barh(y, comb, height=0.5, color=C1, label="logic gates")
ax.barh(y, dff, left=comb, height=0.5, color=C2, label="memory (8 NAND per flip-flop)")
for yi, cc, dd in zip(y, comb, dff):
    ax.text(cc + dd + 300, yi, f"{cc + dd:,}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(parts); ax.set_xlabel("gates (16x16 board)")
ax.grid(axis="y", visible=False); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_xlim(0, (comb + dff).max() * 1.15)
ax.set_title("Where the 16x16 machine's gates go", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_breakdown_16.png"), dpi=150); plt.close(fig)

# 3: smaller gates and noise
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
xs = margins
ys = [np.mean(surv[m]["ticks_ok"]) for m in xs]
axs[0].plot(xs, ys, color=C1, marker="o", ms=5, mec="#fcfcfb", mew=1.5)
axs[0].set_xlabel("training margin of every gate"); axs[0].set_ylabel(f"ticks before the first wrong bit\n(mean of 3 runs, stop at {SURV_T})")
axs[0].set_title("Fully analog 16x16 machine vs gate margin", fontsize=9, loc="left")
ys = [np.mean(noise[s]["ticks_ok"]) for s in noise_levels]
axs[1].plot(noise_levels, ys, color=C2, marker="o", ms=5, mec="#fcfcfb", mew=1.5)
axs[1].set_xlabel("noise added to every neuron output (std)")
axs[1].set_title(f"Library gates (margin {LIB_M:g}) under noise", fontsize=9, loc="left")
for a in axs:
    a.set_ylim(-50, SURV_T * 1.08)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_margin_and_noise.png"), dpi=150); plt.close(fig)
log("done")
