"""Calibration on REAL programs, at the level of the compiler's own circuit (the IR), where every node has a name,
known inputs and a known function.

    cd <qwenputer clone>/experimental/riscv && uv run python <this file> OUT_DIR [MAX_POSITIONS]

Builds the RV32 circuit with Qwenputer's compiler, runs the riscv-tests and small C programs on the reference
machine, evaluates the circuit on their traces (ir.evaluate: the residual stream after every node, exact in
float64), and scores three discovery methods against the graph:
  M2  mean-ablation of each input dim (results_storyroom inside()): is the input found?
  M1  correlation ranking (xorlab): rank earlier dims by |corr| with the node, take the top k (k = true fan-in).
  M3  gate labels (xorlab bool_label / clean_carry label_k): with the true inputs GIVEN and all of them 0/1,
      pick the Boolean function that best matches the observed activation; is it the node's true function?
Writes OUT_DIR/calib_real.json and prints a summary.
"""
import collections, itertools, json, os, sys, time
import numpy as np
from chipformer.q35 import ir
from chipformer.q35.ir import FFN, Head
from chipformer.dsl.graph import ONE
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.programs import ALL, program

OUT = sys.argv[1]; MAXPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
c = core.build()
print("circuit:", c.stats(), "dims", len(c.dims), "nodes", len(c.nodes), f"[{time.time()-t0:.0f}s]", flush=True)
ffn = [(i, n) for i, n in enumerate(c.nodes) if isinstance(n, FFN)]
print("FFN nodes", len(ffn), "head nodes", len(c.nodes) - len(ffn))

# ---- traces of real programs ----
DATA = os.path.join("tests", "data", "rv")
elfs = sorted(json.load(open(os.path.join(DATA, "qemu.json"))))
skip = {"c-sieve", "c-sort", "c-crc32p", "c-gmpmul", "c-fib", "c-sudoku", "c-qsort", "c-sieve4", "mpy-s1", "mpy-s5"}
progs = [(n, program(n)) for n in ALL] + [(n, elf.program(os.path.join(DATA, f"{n}.elf"))) for n in elfs if n not in skip]
mats, used = [], []
npos = 0
for name, prog in progs:
    m = ref.Machine.from_program(prog)
    toks = m.run(10**6)
    if npos + len(toks) > MAXPOS and mats: continue
    core.snapshot(c, ref.Machine.from_program(prog))
    mat = ir.evaluate(c, toks, check=False)
    pred = core.predictions(c, mat)
    wrong = sum(pred[i] != toks[i + 1] for i in range(len(toks) - 1))
    mats.append(mat); used.append((name, len(toks), int(wrong))); npos += len(toks)
    print(f"  {name:16s} {len(toks):6d} tokens, wrong predictions {wrong}  [{time.time()-t0:.0f}s]", flush=True)
M = np.concatenate(mats); N = M.shape[0]
print("positions", N, "programs", len(used), flush=True)
level = np.array([d.level for d in c.dims])
varies = (M.max(0) - M.min(0)) > 0
Mc = M - M.mean(0); Ms = Mc / (np.linalg.norm(Mc, axis=0) + 1e-300)

def ev(e, m): return sum((m[:, d] * float(w) for d, w in e.terms.items()), np.zeros(m.shape[0]))
def node_act(n, m):
    return sum((float(co) * ev(a, m) * np.maximum(ev(b, m), 0) for co, a, b in n.terms), np.zeros(m.shape[0]))

# activations of every FFN node first, so that the correlation of every dim with every node is one matmul
acts = {}
for i, n in ffn:
    ins = sorted({d for co, a, b in n.terms for d in list(a.terms) + list(b.terms) if d != ONE})
    if any(varies[d] for d in ins): acts[i] = node_act(n, M).astype(np.float32)
A = np.stack([acts[i] for i in acts], 1) if acts else np.zeros((N, 0), np.float32)
Ac = A - A.mean(0); As = Ac / (np.linalg.norm(Ac, axis=0) + 1e-30)
CORR = np.abs(Ms.astype(np.float32).T @ As)                                   # (dims, nodes with inputs)
col = {i: j for j, i in enumerate(acts)}
print("correlation matrix", CORR.shape, f"[{time.time()-t0:.0f}s]", flush=True)
rows = []
by_fan = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0, 0, 0])   # n, active, m2, m1, m3 tested, m3 right, m3 full-table
for i, n in ffn:
    ins = sorted({d for co, a, b in n.terms for d in list(a.terms) + list(b.terms) if d != ONE})
    truth = [d for d in ins if varies[d]]
    k = len(truth)
    if k == 0: continue
    act = acts[i].astype(np.float64)
    active = bool((act != 0).any())
    fb = by_fan[min(k, 8)]; fb[0] += 1
    rec = {"node": n.name, "level": int(n.level), "fanin": k, "active": active}
    if active:
        fb[1] += 1
        # M2: mean-ablation of each true input
        found = 0
        for d in truth:
            m2 = M.copy() if False else None
            cols = {d: M[:, d].mean()}
            # recompute the node with dim d replaced by its mean: build the needed columns only
            mm = {dd: M[:, dd] for co, a, b in n.terms for dd in list(a.terms) + list(b.terms)}
            mm[d] = np.full(N, cols[d])
            a2 = sum((float(co) * sum((mm[dd] * float(w) for dd, w in a.terms.items()), 0.0) *
                      np.maximum(sum((mm[dd] * float(w) for dd, w in b.terms.items()), 0.0), 0) for co, a, b in n.terms), np.zeros(N))
            if np.abs(a2 - act).max() > 1e-9 * (np.abs(act).max() + 1e-300): found += 1
        rec["m2"] = found / k; fb[2] += rec["m2"]
        # M1: correlation ranking among dims produced earlier
        cand = np.nonzero((level < n.level) & varies)[0]
        corr = CORR[cand, col[i]]
        top = cand[np.argsort(-corr)[:k]]
        rec["m1_at_k"] = len(set(top.tolist()) & set(truth)) / k; fb[3] += rec["m1_at_k"]
        # M3: gate label with inputs given, when every input is a bit
        X = M[:, truth]
        if k <= 4 and np.all((X == 0) | (X == 1)):
            fb[4] += 1
            combos = list(itertools.product((0, 1), repeat=k))
            # the true table: evaluate the node on every combo (ONE = 1, inputs as given, other dims 0)
            mt = np.zeros((len(combos), len(c.dims))); mt[:, ONE] = 1.0
            for j, d in enumerate(truth): mt[:, d] = [cb[j] for cb in combos]
            true_tab = tuple(int(v != 0) for v in node_act(n, mt))
            # observed: majority activation per observed combo; unobserved combos are unknown
            seen = {}
            y = (act != 0).astype(int)
            for cb, yy in zip(map(tuple, X.astype(int)), y): seen.setdefault(cb, []).append(yy)
            obs = {cb: int(np.mean(v) > 0.5) for cb, v in seen.items()}
            rec["combos_seen"] = len(obs) / len(combos)
            # xorlab bool_label: best-matching Boolean function by correlation on the observed rows (ties: first)
            seen_idx = [j for j, cb in enumerate(combos) if cb in obs]; ys = np.array([obs[combos[j]] for j in seen_idx], float)
            nf = 1 << len(combos)                                            # every Boolean function of k inputs, as rows of a table
            Fall = ((np.arange(nf)[:, None] >> np.arange(len(combos))[None, :]) & 1).astype(float)
            Fo = Fall[:, seen_idx]
            if ys.std() == 0:
                sc = np.where((Fo == ys).all(1), 1.0, -1.0)
            else:
                Fc = Fo - Fo.mean(1, keepdims=True); yc = ys - ys.mean()
                sc = (Fc @ yc) / (np.linalg.norm(Fc, axis=1) * np.linalg.norm(yc) + 1e-300)
                sc[Fo.std(1) == 0] = -1.0
            best = tuple(int(v) for v in Fall[int(np.argmax(sc))])
            rec["m3_right"] = best == true_tab; fb[5] += int(rec["m3_right"])
            rec["m3_right_on_seen"] = all(best[combos.index(cb)] == true_tab[combos.index(cb)] for cb in obs)
            fb[6] += int(len(obs) == len(combos))
            rec["true_kind"] = "AND" if true_tab == tuple(int(all(cb)) for cb in combos) else ("OR" if true_tab == tuple(int(any(cb)) for cb in combos) else "other")
    rows.append(rec)

n_all = len(rows); n_act = sum(r["active"] for r in rows)
m3 = [r for r in rows if "m3_right" in r]
summary = {"positions": N, "programs": used, "ffn_nodes_scored": n_all, "active": n_act,
           "m2_recall_active": float(np.mean([r["m2"] for r in rows if r["active"]])) if n_act else 0,
           "m2_recall_all": float(sum(r.get("m2", 0) for r in rows) / n_all),
           "m1_recall_at_k_active": float(np.mean([r["m1_at_k"] for r in rows if r["active"]])) if n_act else 0,
           "m3_tested": len(m3), "m3_right": int(sum(r["m3_right"] for r in m3)),
           "m3_right_on_seen_combos": int(sum(r["m3_right_on_seen"] for r in m3)),
           "m3_full_table_seen": int(sum(r["combos_seen"] == 1.0 for r in m3)),
           "m3_right_when_full_table": int(sum(r["m3_right"] for r in m3 if r["combos_seen"] == 1.0)),
           "m3_by_kind": {kd: [sum(1 for r in m3 if r["true_kind"] == kd), sum(r["m3_right"] for r in m3 if r["true_kind"] == kd)] for kd in ("AND", "OR", "other")},
           "by_fanin": {k: {"n": v[0], "active": v[1], "m2": v[2] / max(v[1], 1), "m1_at_k": v[3] / max(v[1], 1), "m3_tested": v[4], "m3_right": v[5], "m3_full": v[6]} for k, v in sorted(by_fan.items())},
           "by_level": {}}
for lv in sorted({r["level"] for r in rows}):
    rr = [r for r in rows if r["level"] == lv]; ra = [r for r in rr if r["active"]]
    summary["by_level"][lv] = {"n": len(rr), "active": len(ra), "m2": float(np.mean([r["m2"] for r in ra])) if ra else 0, "m1_at_k": float(np.mean([r["m1_at_k"] for r in ra])) if ra else 0}
json.dump({"summary": summary, "nodes": rows}, open(os.path.join(OUT, "calib_real.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)[:4000])
print(f"done [{time.time()-t0:.0f}s]")
