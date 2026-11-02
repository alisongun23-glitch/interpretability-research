"""An 8-bit computer built from the smallest working neural logic gates.

    python run_minigates.py     # first: trains the gate library (results_minigates/library.json)
    python run_cpu.py           # about 3 minutes, writes results_cpu/ and cpu_ui/cpu.json

PART 1  Size: gates, flip-flops, neurons, weights, depth and multiply-adds per clock tick,
        per unit (memory, ALU, control, bus, registers, program port), for the minimal gate
        library, NAND neurons only, and the 2 -> 4 -> 1 gates of run_adder.py (cost only).
PART 2  Correctness: every register bit on every tick against gatecpu.ref.CpuRef, in five
        ways (Boolean gates, Boolean NAND flip-flops, neural gates, fully analog neural,
        NAND-only analog), on the four demo programs loaded through the program port and on
        random-byte programs with random resets and reloads.
PART 3  Smaller gates (margins 2-4) and noise on every neuron: ticks to the first wrong bit.
PART 4  Speed, and export of the netlist, gate pools, demos and a self-test trace for cpu_ui/.
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small arrays; must precede numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatesnake import gates as G
from gatesnake.hdl import lower_to_nand, GATES, DFF_NANDS
from gatesnake.sim import Machine, load_library, make_library
from gatecpu.design import build_cpu, reg_part, LINES, OPS, NWORDS
from gatecpu.ref import CpuRef, port_inputs, load_ticks, run_isa
from gatecpu.asm import assemble, DEMOS

OUT = "results_cpu"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs("cpu_ui", exist_ok=True)
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
circ, info = build_cpu()
net = circ.finalize()
nand_net = lower_to_nand(circ).finalize()
size = {}
for lib_name, nt in (("minimal", net), ("NAND-only", nand_net), ("baseline", net)):
    s = nt.stats(COST[lib_name])
    s["macs_per_tick_simulated"] = s["macs_comb"] + 2 * LATCH_ITERS * len(nt.regs) * DFF_NANDS * COST[lib_name]["NAND"]["macs"]
    size[lib_name] = s
bd = net.breakdown(reg_part)
summary.update(size=size, breakdown=bd, gates_by_type=net.counts(),
               unique_trained_weights=sum(len(LIB["gates"][t]["pool"]) * LIB["gates"][t]["cost"]["params"] for t in GATES))
s = size["minimal"]
log(f"CPU: {s['total_gates']:,} gates ({s['comb_gates']:,} logic + {s['dff_gates']:,} in {s['registers']} flip-flops), "
    f"{s['neurons']:,} neurons, {s['params']:,} weights, depth {s['logic_depth']} gates / {s['neuron_depth_comb']} neurons")
log("breakdown", bd)


# ============ PART 2: correctness ============
def ref_step(ref, inp):
    return ref.step(inp["rst"], inp["prog"], inp["prog_we"],
                    sum(inp[f"prog_addr{j}"] << j for j in range(5)), sum(inp[f"prog_data{j}"] << j for j in range(8)))


def demo_inputs(program, run_ticks):
    return [port_inputs(**t) for t in load_ticks(program)] + [port_inputs() for _ in range(run_ticks)]


def fuzz_inputs(seed, ticks):
    """Random-byte programs; now and then a reset, now and then a fresh random program."""
    r = np.random.default_rng(seed)
    seq = demo_inputs([int(x) for x in r.integers(0, 256, NWORDS)], 0)
    while len(seq) < ticks:
        u = r.random()
        if u < 0.003:
            seq.append(port_inputs(rst=1))
        elif u < 0.004:
            seq += demo_inputs([int(x) for x in r.integers(0, 256, NWORDS)], 0)
        else:
            seq.append(port_inputs())
    return seq[:ticks]


def run_vs_ref(nt, seq, seed=0, **mk):
    m = Machine(nt, seed=seed, latch_iters=LATCH_ITERS, **mk)
    ref = CpuRef()
    st = {"ticks": 0, "first_error": None, "outputs": [], "instructions": 0}
    t0 = time.time()
    for i, inp in enumerate(seq):
        L = ref_step(ref, inp)
        if L["OUT_in"]:
            st["outputs"].append(ref.bus)
        if L["IR_in"]:
            st["instructions"] += 1
        m.step(inp)
        st["ticks"] = i + 1
        if m.regs() != ref.regs():
            st["first_error"] = i + 1
            break
    st["ms_per_tick"] = (time.time() - t0) / st["ticks"] * 1e3
    st["worst_signal_dev"] = float(m.worst)
    st["halted"] = bool(ref.halt)
    return st


lib_min = load_library()
lib_nand = load_library(nand_only=True)
MODES = [("Boolean gates, ideal flip-flops", net, {"mode": "logic", "dff": "ideal"}),
         ("Boolean gates, NAND flip-flops", net, {"mode": "logic", "dff": "gates"}),
         ("neural gates, ideal flip-flops", net, {"mode": "neural", "dff": "ideal", "library": lib_min}),
         ("neural gates, neural NAND flip-flops (fully analog)", net, {"mode": "neural", "dff": "gates", "library": lib_min}),
         ("NAND-only, fully analog", nand_net, {"mode": "neural", "dff": "gates", "library": lib_nand})]
WORKLOADS = {name: demo_inputs(assemble(src), 800) for name, (src, _) in DEMOS.items()}
WORKLOADS.update({f"random programs {s}": fuzz_inputs(s, 4000) for s in range(3)})
verify = {}
for mname, nt, mk in MODES:
    for wname, seq in WORKLOADS.items():
        st = run_vs_ref(nt, seq, **mk)
        if wname in DEMOS:
            st["expected_outputs"] = DEMOS[wname][1]
            st["outputs_correct"] = st["outputs"] == DEMOS[wname][1]
        verify[f"{mname} | {wname}"] = st
    rows = [v for k, v in verify.items() if k.startswith(mname)]
    log(f"  {mname}: {sum(v['ticks'] for v in rows)} ticks, {sum(v['first_error'] is not None for v in rows)} runs with errors, "
        f"demo outputs right: {all(v.get('outputs_correct', True) for v in rows)}, "
        f"{np.mean([v['ms_per_tick'] for v in rows]):.2f} ms/tick, worst signal {max(v['worst_signal_dev'] for v in rows):.3f}")
summary["verify"] = verify
summary["demo_isa"] = {name: dict(zip(("outputs", "halted", "instructions"), run_isa(assemble(src)))) for name, (src, _) in DEMOS.items()}

# plain Python speed
seq = WORKLOADS["random programs 0"]; ref = CpuRef(); t0 = time.time()
for inp in seq:
    ref_step(ref, inp)
summary["python_reference_ms_per_tick"] = (time.time() - t0) / len(seq) * 1e3
analog = [v for k, v in verify.items() if k.startswith("neural gates, neural NAND")]
summary["ticks_per_instruction"] = sum(v["ticks"] for v in analog) / max(1, sum(v["instructions"] for v in analog))

# ============ PART 3: smaller gates, noise ============
SURV_T = 3000
survive_seq = fuzz_inputs(10, SURV_T)
margins = [2.0, 2.5, 3.0, 3.25, 3.5, LIB_M]
pools_by_m, surv = {}, {}
for m in margins:
    if m == LIB_M:
        pools = {g: LIB["gates"][g]["pool"] for g in GATES}
    else:
        pools = {g: [r["params"] for r in G.train(g, LIB["gates"][g]["arch"], m, list(range(40)), max_steps=40000)
                     if r["converged"]][:8] for g in G.TRUTH}
    pools_by_m[m] = pools
    res = [run_vs_ref(net, survive_seq, seed=s, mode="neural", dff="gates", library=make_library(pools)) for s in range(3)]
    surv[m] = {"first_errors": [r["first_error"] for r in res], "worst_signal_dev": max(r["worst_signal_dev"] for r in res),
               "ticks_ok": [r["ticks"] - (r["first_error"] is not None) for r in res]}
    log(f"  margin {m}: first errors {surv[m]['first_errors']}, worst signal {surv[m]['worst_signal_dev']:.3f}")
summary["margin_survival"] = {str(k): v for k, v in surv.items()}
noise_levels = [0.0, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
noise = {}
for sd in noise_levels:
    res = [run_vs_ref(net, survive_seq, seed=s, mode="neural", dff="gates", library=lib_min, noise=sd) for s in range(3)]
    noise[sd] = {"first_errors": [r["first_error"] for r in res], "ticks_ok": [r["ticks"] - (r["first_error"] is not None) for r in res]}
    log(f"  noise {sd}: first errors {noise[sd]['first_errors']}")
summary["noise_survival"] = {str(k): v for k, v in noise.items()}

# ============ PART 4: export for the browser ============
m0 = Machine(net, mode="neural", library=lib_min, dff="gates", latch_iters=LATCH_ITERS, seed=0)
pools_js = {}
for tag, m in (("m4", LIB_M), ("m3", 3.0), ("m2", 2.0)):
    L = make_library(pools_by_m[m])
    pools_js[tag] = {t: [{"U": p["U"].tolist(), "c": p["c"].tolist(), "w": p["w"].tolist(), "v": p["v"].tolist(), "bo": p["bo"]}
                         for p in L[t]] for t in GATES}
trace_seq = [inp for name in DEMOS for inp in demo_inputs(assemble(DEMOS[name][0]), 600)]
ref = CpuRef(); hashes = []
for inp in trace_seq:
    ref_step(ref, inp)
    h = 0; rg = ref.regs()
    for g in net.regs:
        h = (h * 31 + rg[g["name"]]) & 0xFFFFFFFF
    hashes.append(h)
in_names = list(net.inputs)
data = {
    "n_wires": net.n_wires, "types": list(GATES), "latch_iters": LATCH_ITERS, "ops": OPS, "lines": LINES,
    "inputs": net.inputs, "outputs": net.outputs,
    "regs": {"q": [g["q"] for g in net.regs], "d": [g["d"] for g in net.regs], "init": [g["init"] for g in net.regs],
             "name": [g["name"] for g in net.regs]},
    "gates": {"out": [w for w, _, _ in net.gates], "a": [a[0] for _, _, a in net.gates], "b": [a[-1] for _, _, a in net.gates],
              "type": [list(GATES).index(t) for _, t, _ in net.gates], "pick": [m0.choice[w] for w, _, _ in net.gates]},
    "out_cone": sorted(int(i) for i in np.flatnonzero(np.isin([w for w, _, _ in net.gates],
                                                              np.concatenate([g.out for g in m0.out_groups])))),
    "dff_pick": [int(x) for x in np.random.default_rng(1).integers(0, 8, len(net.regs) * DFF_NANDS)],
    "pools": pools_js, "default_pool": "m4",
    "demos": {k: {"source": v[0], "expected": v[1]} for k, v in DEMOS.items()},
    "trace": {"input_names": in_names, "inputs": [[inp[k] for k in in_names] for inp in trace_seq], "hashes": hashes},
    "stats": {"total_gates": s["total_gates"], "comb_gates": s["comb_gates"], "registers": s["registers"],
              "neurons": s["neurons"], "params": s["params"], "macs_per_tick": s["macs_per_tick_simulated"],
              "logic_depth": s["logic_depth"], "neuron_depth": s["neuron_depth_comb"], "breakdown": bd},
}
json.dump(data, open("cpu_ui/cpu.json", "w"), separators=(",", ":"))
log(f"exported cpu_ui/cpu.json ({os.path.getsize('cpu_ui/cpu.json') / 1e6:.2f} MB, trace {len(trace_seq)} ticks)")
summary["runtime_s"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

# ============ figures ============
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C1, C2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2, "lines.solid_capstyle": "round"})
parts = sorted(bd, key=lambda k: -(bd[k]["comb"] + bd[k]["dff"]))
fig, ax = plt.subplots(figsize=(6.2, 2.9))
y = np.arange(len(parts))[::-1]
comb = np.array([bd[p]["comb"] for p in parts]); dff = np.array([bd[p]["dff"] for p in parts])
ax.barh(y, comb, height=0.5, color=C1, label="logic gates")
ax.barh(y, dff, left=comb, height=0.5, color=C2, label="memory (8 NAND per flip-flop)")
for yi, cc, dd in zip(y, comb, dff):
    ax.text(cc + dd + 40, yi, f"{cc + dd:,}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(parts); ax.set_xlabel("gates")
ax.grid(axis="y", visible=False); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_xlim(0, (comb + dff).max() * 1.15)
ax.set_title("Where the computer's gates go", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_breakdown.png"), dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
axs[0].plot(margins, [np.mean(surv[m]["ticks_ok"]) for m in margins], color=C1, marker="o", ms=5, mec="#fcfcfb", mew=1.5)
axs[0].set_xlabel("training margin of every gate")
axs[0].set_ylabel(f"ticks before the first wrong bit\n(mean of 3 runs, stop at {SURV_T})")
axs[0].set_title("Fully analog computer vs gate margin", fontsize=9, loc="left")
axs[1].plot(noise_levels, [np.mean(noise[s]["ticks_ok"]) for s in noise_levels], color=C2, marker="o", ms=5, mec="#fcfcfb", mew=1.5)
axs[1].set_xlabel("noise added to every neuron output (std)")
axs[1].set_title(f"Library gates (margin {LIB_M:g}) under noise", fontsize=9, loc="left")
for a_ in axs:
    a_.set_ylim(-80, SURV_T * 1.08)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_margin_and_noise.png"), dpi=150); plt.close(fig)
log("done")
