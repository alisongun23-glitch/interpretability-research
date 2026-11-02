"""A small GPT with the neural-gate computer as one of its attention heads.

    python run_cpugpt.py             # about 15 minutes; writes results_cpugpt/ and cpugpt_ui/model.json
    python run_cpugpt.py --analyse   # reuse the trained models in results_cpugpt/models (about 2 minutes)

PART 1  The computer's answers: the calculator program (gatecpu/calc.py) runs on the gate CPU for
        every (op, x, y), 262,144 calls, bit-sliced; checked against the instruction-level reference.
PART 2  Train three conditions, 3 seeds each, same size and steps:
          baseline        4 ordinary heads in layer 2
          CPU head        3 ordinary heads + the CPU head; pointing learns from the answer's
                          log-likelihood only (more samples, no call cost: the most
                          favourable settings found in pilot runs)
          CPU head + check  same head, plus +1 reward when the computer's result equals the answer,
                          and a small cost per call (0.01), so calling has to be worth it
          ... pricier calls   the same with a call cost of 0.05
PART 3  Evaluate on held-out operand pairs: accuracy by operation and template (Wilson 95% CI),
        pointing accuracy, gate on math vs non-math, ablations (head off, wrong operation).
PART 4  Live check: 500 test problems whose CPU calls run on the fully analog neural computer
        (neural gates, NAND flip-flops) instead of the table.
PART 5  Figures, summary.json, model export for cpugpt_ui/.
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatecpu.design import build_cpu
from gatecpu.batch import build_table
from gatecpu.calc import PROGRAM, ADDR, CAP, run_calc, call_ticks
from gatecpu.ref import port_inputs
from gatesnake.sim import Machine, load_library
from cpugpt import data as D
from cpugpt.model import GPT, losses

OUT = "results_cpugpt"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
os.makedirs("cpugpt_ui", exist_ok=True)
T0 = time.time()
SEEDS = [0, 1, 2]
STEPS = 6000
EVAL_EVERY = 250
CONDITIONS = {
    "baseline": dict(cpu=False),
    "CPU head": dict(cpu=True, shaping=0.0, K=16, entropy=0.02, gate_bias=2.0, call_cost=0.0),
    "CPU head + check": dict(cpu=True, shaping=1.0, K=8, entropy=0.01, gate_bias=2.0, call_cost=0.01),
    "CPU head + check, pricier calls": dict(cpu=True, shaping=1.0, K=8, entropy=0.01, gate_bias=2.0, call_cost=0.05),
}
summary = {"steps": STEPS, "seeds": SEEDS, "conditions": CONDITIONS}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


# ============ PART 1: the computer's answers ============
net = build_cpu()[0].finalize()
t0 = time.time()
table, halt_tick = build_table(net)
mism = sum(int(table[op, x, y] != run_calc(x, y, op)[0]) for op in range(4) for x in range(256) for y in range(256))
np.savez_compressed(os.path.join(OUT, "cpu_table.npz"), out=table, halt_tick=halt_tick)
summary["table"] = {"calls": 4 * 256 * 256, "seconds_on_gates": round(time.time() - t0, 1), "mismatches_vs_reference": mism}
log(f"table: 262,144 calls on the gate CPU, {mism} mismatches")
CPU = dict(numval=D.NUMVAL, opval=D.OPVAL, table=table)


def tb(b):
    return {k: torch.as_tensor(v) for k, v in b.items()}


EVAL_SETS = {split: tb(D.batch(np.random.default_rng(12345 if split == "test" else 54321), 7000, split,
                                templates=list(range(len(D.TEMPLATES))))) for split in ("test", "train")}


def evaluate(m, b, mode=None, op_map=None, results=None):
    m.eval()
    with torch.no_grad():
        o = m(b["tokens"], b["last"], mode=mode or ("gated" if m.cpuhead is not None else "argmax"),
              op_map=op_map, results=results)
    m.train()
    pred = o["logits"][:, 0].argmax(-1)
    ok = (pred == b["target"]).numpy()
    kind = b["kind"].numpy(); tid = b["tid"].numpy()
    res = {"pred": pred.numpy(), "ok": ok}
    res["by_kind"] = {nm: [int(ok[kind == k].sum()), int((kind == k).sum())] for k, nm in ((1, "add"), (2, "sub"), (3, "mul"), (0, "non-math"))}
    res["by_template"] = {D.TEMPLATES[t][0]: [int(ok[tid == t].sum()), int((tid == t).sum())] for t in range(len(D.TEMPLATES))}
    if m.cpuhead is not None:
        mathm = kind > 0
        x, y, op = o["x"][:, 0].numpy(), o["y"][:, 0].numpy(), o["op"][:, 0].numpy()
        a, bb = b["a"].numpy(), b["b"].numpy()
        res["point"] = {"A": float((x == a)[mathm].mean()), "B": float((y == bb)[mathm].mean()),
                        "OP": float((op == kind)[mathm].mean()), "all": float(((x == a) & (y == bb) & (op == kind))[mathm].mean())}
        g = o["gate"].numpy()
        res["gate"] = {"math_mean": float(g[mathm].mean()), "nonmath_mean": float(g[~mathm].mean()),
                       "math_calls": float((g[mathm] >= 0.5).mean()), "nonmath_calls": float((g[~mathm] >= 0.5).mean())}
        res["calls"] = {"x": x, "y": y, "op": op, "gate": g}
        res["scores"] = o["scores"].softmax(-1).nan_to_num(0).numpy()
    return res


# ============ PART 2: training (one process per run) ============
def train_run(args):
    cname, seed = args
    torch.set_num_threads(1)
    cfg = CONDITIONS[cname]
    torch.manual_seed(seed); r = np.random.default_rng(1000 + seed)
    m = GPT(len(D.VOCAB), D.T_MAX, cpu=CPU if cfg["cpu"] else None)
    if cfg["cpu"]:
        with torch.no_grad():
            m.cpuhead.gate.bias.fill_(cfg["gate_bias"])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / STEPS))))
    curve = []
    t1 = time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            e = evaluate(m, EVAL_SETS["test"])
            bk = e["by_kind"]
            row = {"step": step, **{k: v[0] / v[1] for k, v in bk.items()}}
            if cfg["cpu"]:
                row.update(point=e["point"]["all"], gate_math=e["gate"]["math_mean"], gate_non=e["gate"]["nonmath_mean"])
            curve.append(row)
        if step == STEPS:
            break
        b = tb(D.batch(r, 256))
        if cfg["cpu"]:
            loss, info = losses(m, b, K=cfg["K"], shaping=cfg["shaping"], entropy=cfg["entropy"], call_cost=cfg["call_cost"])
        else:
            loss, info = losses(m, b, K=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    path = os.path.join(OUT, "models", f"{cname.replace(' ', '_').replace('+', 'plus')}_seed{seed}.pt")
    torch.save(m.state_dict(), path)
    return cname, seed, path, curve, time.time() - t1


if __name__ == "__main__":
    import multiprocessing as mp
    import sys
    jobs = [(c, s) for c in CONDITIONS for s in SEEDS]
    if "--analyse" in sys.argv:
        old = json.load(open(os.path.join(OUT, "summary.json")))["runs"]
        done = [(c, s, os.path.join(OUT, "models", f"{c.replace(' ', '_').replace('+', 'plus')}_seed{s}.pt"),
                 old[f"{c} | seed {s}"]["curve"], float("nan")) for c, s in jobs]
    else:
        with mp.get_context("fork").Pool(min(len(jobs), max(1, (os.cpu_count() or 2) - 1))) as pool:
            done = pool.map(train_run, jobs)
    torch.set_num_threads(4)
    runs = {}
    for cname, seed, path, curve, secs in done:
        m = GPT(len(D.VOCAB), D.T_MAX, cpu=CPU if CONDITIONS[cname]["cpu"] else None)
        m.load_state_dict(torch.load(path)); runs[(cname, seed)] = (m, curve)
        last = curve[-1]
        log(f"{cname} seed {seed}: held-out add {last['add']:.3f} sub {last['sub']:.3f} mul {last['mul']:.3f} "
            f"non-math {last['non-math']:.3f}" + (f" point {last['point']:.3f} gate math {last['gate_math']:.2f} non {last['gate_non']:.2f}"
                                                   if CONDITIONS[cname]["cpu"] else "") + f" ({secs:.0f}s)")

# ============ PART 3: evaluation ============
OPMAP_WRONG = torch.tensor([0, 2, 3, 1])           # add -> sub -> mul -> add
evals = {}
for (cname, seed), (m, curve) in runs.items():
    e_test = evaluate(m, EVAL_SETS["test"]); e_train = evaluate(m, EVAL_SETS["train"])
    row = {"curve": curve,
           "test": {k: {"correct": v[0], "n": v[1], "acc": v[0] / v[1], "ci95": wilson(*v)} for k, v in e_test["by_kind"].items()},
           "train": {k: v[0] / v[1] for k, v in e_train["by_kind"].items()},
           "test_by_template": {k: v[0] / v[1] for k, v in e_test["by_template"].items()}}
    if m.cpuhead is not None:
        row["point"] = e_test["point"]; row["gate"] = e_test["gate"]
        off = evaluate(m, EVAL_SETS["test"], mode="off"); wrong = evaluate(m, EVAL_SETS["test"], op_map=OPMAP_WRONG)
        row["ablation_head_off"] = {k: v[0] / v[1] for k, v in off["by_kind"].items()}
        row["ablation_wrong_op"] = {k: v[0] / v[1] for k, v in wrong["by_kind"].items()}
        c = e_test["calls"]; called = c["gate"] >= 0.5
        tid = EVAL_SETS["test"]["tid"].numpy(); a_ = EVAL_SETS["test"]["a"].numpy(); b_ = EVAL_SETS["test"]["b"].numpy()
        row["pointers_by_template"] = {D.TEMPLATES[t][0]: {
            "x=a, y=b": float(((c["x"] == a_) & (c["y"] == b_))[tid == t].mean()),
            "swapped": float(((c["x"] == b_) & (c["y"] == a_) & (a_ != b_))[tid == t].mean()),
            "calls": float(called[tid == t].mean())} for t in range(len(D.TEMPLATES))}
        row["mean_cpu_ticks_per_call"] = float(np.mean(halt_tick[c["op"][called], c["x"][called], c["y"][called]])) if called.any() else None
    evals[f"{cname} | seed {seed}"] = row
summary["runs"] = evals

# ============ PART 4: live analog check ============
def page_score(k):
    """Model shown on the page: most accurate on held-out math, then fewest calls on non-math."""
    ev = evals[f"{k[0]} | seed {k[1]}"]
    return (round(np.mean([ev["test"][o]["acc"] for o in ("add", "sub", "mul")]), 3), -ev["gate"]["nonmath_calls"], ev["point"]["all"])


best = max((k for k in runs if CONDITIONS[k[0]]["cpu"]), key=page_score)
mbest = runs[best][0]
live_b = tb(D.batch(np.random.default_rng(777), 500, "test", math_frac=1.0))
e = evaluate(mbest, live_b)
lib = load_library()
cpu = Machine(net, mode="neural", library=lib, dff="gates")
halt_i = [i for i, g in enumerate(net.regs) if g["name"] == "halt"][0]
out_i = [i for i, g in enumerate(net.regs) if g["name"].startswith("out[")]
for i, byte in enumerate(PROGRAM):
    cpu.step(port_inputs(prog=1, we=1, addr=i, data=byte))
live = np.zeros(500, np.int64); ticks = np.zeros(500, np.int64); t1 = time.time()
for i in range(500):
    x, y, op = int(e["calls"]["x"][i]), int(e["calls"]["y"][i]), int(e["calls"]["op"][i])
    for t in call_ticks(x, y, op):
        cpu.step(port_inputs(**t))
    for t in range(CAP):
        cpu.step(port_inputs())
        if cpu.Q[halt_i] > 0.5:
            break
    ticks[i] = t + 1
    live[i] = sum(int(cpu.Q[j] > 0.5) << k for k, j in enumerate(out_i))
tab_res = table[e["calls"]["op"], e["calls"]["x"], e["calls"]["y"]]
e_live = evaluate(mbest, live_b, results=torch.as_tensor(live)[:, None])
summary["live_analog_check"] = {
    "model": f"{best[0]} | seed {best[1]}", "problems": 500, "calls_run_on_analog_cpu": 500,
    "results_differing_from_table": int((live != tab_res).sum()),
    "answers_differing": int((e_live["pred"] != e["pred"]).sum()),
    "accuracy_with_live_cpu": float(e_live["ok"].mean()), "seconds": round(time.time() - t1, 1),
    "mean_ticks": float(ticks.mean()), "worst_signal_dev": float(cpu.worst)}
log("live analog check", summary["live_analog_check"])

# ============ PART 5: export, figures ============
def export(m):
    return {k: np.round(v.detach().numpy().astype(np.float64), 5).tolist() for k, v in m.state_dict().items()
            if not k.startswith("cpuhead.") or k.split(".")[1] in ("q", "k", "read", "gate")}


ui = {"vocab": D.VOCAB, "t_max": D.T_MAX, "numval": D.NUMVAL.tolist(), "opval": D.OPVAL.tolist(),
      "templates": [{"name": t[0], "words": t[1], "kind": t[2] if isinstance(t[2], int) else 0} for t in D.TEMPLATES],
      "program": PROGRAM, "addr": ADDR, "cap": CAP, "d": 96, "heads": 4,
      "models": {"cpu": {"label": f"{best[0]}, seed {best[1]}", "weights": export(mbest)},
                 "baseline": {"label": "baseline, seed 0", "weights": export(runs[("baseline", 0)][0])}},
      "results": {k: {kk: v["test"][kk]["acc"] for kk in v["test"]} for k, v in evals.items()}}
json.dump(ui, open("cpugpt_ui/model.json", "w"), separators=(",", ":"))
log(f"exported cpugpt_ui/model.json ({os.path.getsize('cpugpt_ui/model.json') / 1e6:.1f} MB)")
summary["runtime_s"] = round(time.time() - T0, 1)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL = {"baseline": "#898781", "CPU head": "#eb6834", "CPU head + check": "#2a78d6", "CPU head + check, pricier calls": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2, "lines.solid_capstyle": "round"})
fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
for cname in CONDITIONS:
    curves = [runs[(cname, s)][1] for s in SEEDS]
    steps = [r["step"] for r in curves[0]]
    mathacc = np.mean([[np.mean([r["add"], r["sub"], r["mul"]]) for r in c] for c in curves], 0)
    axs[0].plot(steps, mathacc, color=COL[cname], label=cname)
    if CONDITIONS[cname]["cpu"]:
        axs[1].plot(steps, np.mean([[r["point"] for r in c] for c in curves], 0), color=COL[cname], label=cname)
axs[0].set_ylabel("held-out math accuracy (3-seed mean)"); axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
axs[1].set_ylabel("pointers on the right A, B and operator")
for a_ in axs:
    a_.set_xlabel("training step"); a_.set_ylim(-0.03, 1.05)
axs[0].set_title("Answers on operand pairs never seen in training", fontsize=9, loc="left")
axs[1].set_title("Learning to use the computer", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_learning.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 3.4))
kinds = ["add", "sub", "mul", "non-math"]; w = 0.2
for i, cname in enumerate(CONDITIONS):
    accs = np.array([[evals[f"{cname} | seed {s}"]["test"][k]["acc"] for k in kinds] for s in SEEDS])
    xs = np.arange(len(kinds)) + (i - 1.5) * w
    ax.bar(xs, accs.mean(0), width=w * 0.9, color=COL[cname], label=cname)
    for s in range(len(SEEDS)):
        ax.plot(xs, accs[s], "o", ms=3, color=INK, alpha=0.6)
ax.set_xticks(np.arange(len(kinds))); ax.set_xticklabels(["a + b", "a - b", "a x b", "non-math"])
ax.set_ylabel("held-out accuracy"); ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.28))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_accuracy.png"), dpi=150); plt.close(fig)
log("done")
