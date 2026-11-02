"""Self-checks for the whole project. Run from the project root:  python tests/check_all.py
Every check prints PASS/FAIL. Nothing here changes results."""
import sys, os, importlib.util
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xorlab import model as M, analysis as A

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")

# ---------- 1. gradient of the XOR training loop (model.train_batch) ----------
def loss_of(p, act, Yt):
    L = M.logits(p, act); return np.mean(np.logaddexp(0, L) - Yt * L)

def analytic_grad(p, act, Yt):
    Z = M.X @ p["W1"] + p["b1"]; H = M.act_fn(act)(Z); L = H @ p["w2"] + p["b2"]
    dL = (1 / (1 + np.exp(-L)) - Yt) / 4.0
    dZ = dL[:, None] * p["w2"][None, :] * M.act_grad(act, Z, H)
    return {"w2": H.T @ dL, "b2": dL.sum(), "W1": M.X.T @ dZ, "b1": dZ.sum(0)}

for act in ["tanh", "relu"]:
    p = M.init_params(3, 4, 1.0); g = analytic_grad(p, act, M.Y); worst = 0
    for k in p:
        it = np.nditer(p[k], flags=["multi_index"], op_flags=["readwrite"]) if p[k].ndim else [None]
        idx_list = [tuple(i) for i in np.ndindex(p[k].shape)] if p[k].ndim else [()]
        for idx in idx_list:
            orig = p[k][idx] if idx else p[k].item()
            eps = 1e-6
            def setv(v):
                if idx: p[k][idx] = v
                else: p[k] = np.array(v)
            setv(orig + eps); lp = loss_of(p, act, M.Y); setv(orig - eps); lm = loss_of(p, act, M.Y); setv(orig)
            num = (lp - lm) / (2 * eps); an = g[k][idx] if idx else g[k]
            worst = max(worst, abs(num - an))
    check(f"XOR gradient ({act}) matches finite differences", worst < 1e-6, f"max err {worst:.1e}")

# ---------- 2. batched training == single-network training ----------
cfg = {"optimizer": "adam", "lr": 0.05, "betas": [0.9, 0.999], "max_steps": 300, "margin": 2.0, "log_every": 50, "init_std": 1.0}
runs_batch, *_ = M.train_batch([0, 1, 2], "tanh", 4, cfg)
runs_single, *_ = M.train_batch([1], "tanh", 4, cfg)
same = all(np.allclose(runs_batch[1]["params"][k], runs_single[0]["params"][k]) for k in runs_single[0]["params"])
check("batched training gives the same weights as training seed 1 alone", same)
check("frozen networks keep their weights after converging",
      runs_batch[1]["converge_step"] >= 0 and same)

# ---------- 3. ablation and circuit logic ----------
p = runs_batch[0]["params"]
H = M.hidden_acts(p, "tanh")
lg_full = A.forward(p, "tanh")
lg_abl = A.forward(p, "tanh", ablate=[0], mode="mean")
expected = lg_full - p["w2"][0] * (H[:, 0] - H[:, 0].mean())
check("mean ablation = subtract the unit's input-dependent contribution", np.allclose(lg_abl, expected))
check("ablating every unit gives a constant output", np.ptp(A.forward(p, "tanh", ablate=[0, 1, 2, 3])) < 1e-12)
mins = A.minimal_circuits(p, "tanh")
for c in mins:
    others = [i for i in range(4) if i not in c["units"]]
    check(f"minimal circuit {c['units']} is sufficient", np.all(A.correct(A.forward(p, "tanh", ablate=others))))
    for i in c["units"]:
        sub = [u for u in c["units"] if u != i]; others2 = [u for u in range(4) if u not in sub]
        check(f"  subset {sub} is NOT sufficient", not np.all(A.correct(A.forward(p, "tanh", ablate=others2))))

# ---------- 4. transplant arithmetic ----------
rec = runs_batch[0]["params"]; don = runs_batch[2]["params"]
Cr = mins[0]["units"]
t = A.transplant(rec, "tanh", Cr, A.contribution(don, "tanh", Cr))
check("native transplant reproduces the recipient's own logits", np.allclose(t["native"], lg_full))
check("ablated transplant equals mean-ablating the recipient circuit",
      np.allclose(t["ablated"], A.forward(rec, "tanh", ablate=Cr, mode="mean")))
c_d = A.contribution(don, "tanh", Cr); c_r = A.contribution(rec, "tanh", Cr)
mm = A.moment_match(c_d, c_r)
check("moment matching preserves recipient mean and std", abs(mm.mean() - c_r.mean()) < 1e-12 and abs(mm.std() - c_r.std()) < 1e-12)
ref = A.reference_contribution("tanh", 8.0)
# a contribution is added to the recipient's bias and other units, so test its SHAPE: XOR=1 inputs above XOR=0 inputs
check("hand-built tanh reference has XOR shape", ref[1] > ref[0] + 1 and ref[2] > ref[3] + 1 and abs(ref[0] - ref[3]) < 1e-9, f"{np.round(ref, 2)}")
ref_r = A.reference_contribution("relu", 8.0)
check("hand-built relu reference has XOR shape", ref_r[1] > ref_r[0] + 1 and ref_r[2] > ref_r[3] + 1 and abs(ref_r[0] - ref_r[3]) < 1e-9, f"{np.round(ref_r, 2)}")

# ---------- 5. Boolean labels ----------
check("AND pattern labelled AND", A.bool_label([0, 0, 0, 1])[0] == "AND")
check("negated AND pattern is NOT labelled AND", A.bool_label([1, 1, 1, 0])[0] == "NAND")
check("constant vector labelled constant", A.bool_label([0.5, 0.5, 0.5, 0.5])[0] == "constant")

# ---------- 6. CKA invariances ----------
rng = np.random.default_rng(0); Aq = rng.normal(size=(4, 6)); Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
check("CKA invariant to rotation and scale", abs(A.linear_cka(Aq, 3 * Aq @ Q) - 1) < 1e-12)
check("CKA invariant to neuron permutation", abs(A.linear_cka(Aq, Aq[:, rng.permutation(6)]) - 1) < 1e-12)

# ---------- 7. adder scripts: load their helper functions without running them ----------
import ast
def load_funcs(path, _unused=None):
    """Execute only the imports, constants, classes and functions of a script (no training)."""
    tree = ast.parse(open(path).read()); keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name != "probe":   # probe() defaults need training data
            keep.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in ("NB", "rng", "OUT", "T0") for t in node.targets):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[]); ns = {}
    exec(compile(mod, path, "exec"), ns); return ns

a16 = load_funcs("run_adder16.py", "# ---------------- data ----------------")
a, b = np.array([0b0111111111111111, 5, 0]), np.array([1, 3, 0])
check("carries(): 0111..1 + 1 carries into bits 1..15", list(a16["carries"](a, b)[0]) == [1] * 15 + [0])
check("chain_len(): that problem has chain length 15", a16["chain_len"](a, b)[0] == 15)
check("chain_len(): 5 + 3 (0101+0011) has chain 3", a16["chain_len"](a, b)[1] == 3)
check("chain_len(): 0 + 0 has chain 0", a16["chain_len"](a, b)[2] == 0)
# MLP gradient check (16-bit script's class)
MLP = a16["MLP"]; net = MLP([4, 5, 5, 3], 0); x = rng.normal(size=(7, 4)); y = (rng.random((7, 3)) > 0.5).astype(float)
def mlp_loss(net):
    _, z = net.forward(x); return np.mean(np.logaddexp(0, z) - y * z) * 3   # step() divides by len(x) only, per-output summed
import copy
n2 = copy.deepcopy(net); hs, z = n2.forward(x); d = (1 / (1 + np.exp(-z)) - y) / len(x)
acts = [x] + hs; gW = [None] * 3
for i in reversed(range(3)):
    gW[i] = acts[i].T @ d
    if i: d = (d @ n2.W[i].T) * (1 - acts[i] ** 2)
worst = 0
for i in range(3):
    for idx in [(0, 0), (1, 2)]:
        o = net.W[i][idx]; eps = 1e-6
        net.W[i][idx] = o + eps; lp = mlp_loss(net); net.W[i][idx] = o - eps; lm = mlp_loss(net); net.W[i][idx] = o
        worst = max(worst, abs((lp - lm) / (2 * eps) - gW[i][idx]))
check("adder MLP gradient matches finite differences", worst < 1e-6, f"max err {worst:.1e}")
# opposite_carry_source really flips the carry and keeps high bits
rr = np.random.default_rng(1); ka, kb = a16["make"](200, rr)
for k in [4, 8, 12]:
    a2, b2, c = a16["opposite_carry_source"](ka, kb, k, rr)
    c_new = ((a2 % 2 ** k) + (b2 % 2 ** k)) >= 2 ** k
    check(f"opposite_carry_source(k={k}) flips the carry and keeps high bits",
          np.all(c_new != c) and np.all(a2 >> k == ka >> k) and np.all(b2 >> k == kb >> k))
# built_distance builds the carry travel distance it claims
for L in [4, 8, 14]:
    la, lb = a16["built_distance"](300, L, np.random.default_rng(L))
    td = a16["travel_dist"](la, lb)
    check(f"built_distance(L={L}) gives travel distance of at least {L}", np.all(td >= L), f"min {td.min()}, max {td.max()}")

# ---------- 8. 4-bit tiled adder logic with PERFECT gates ----------
a4 = load_funcs("run_adder.py")
from itertools import product
A_ALL, B_ALL = np.array(list(product(range(16), range(16)))).T
BITS = lambda v: np.stack([(v >> i) & 1 for i in range(4)], 1).astype(float)
perfect = {"XOR": lambda a, b: (a != b).astype(float), "AND": lambda a, b: a * b, "OR": lambda a, b: np.maximum(a, b)}
G = {}
for i in range(4):
    for slot, g in [("XOR1", "XOR"), ("XOR2", "XOR"), ("AND1", "AND"), ("AND2", "AND"), ("OR", "OR")]:
        G[(slot, i)] = perfect[g]
out, inter = a4["tiled_adder"](BITS(A_ALL), BITS(B_ALL), G, True)
check("tiled_adder with perfect gates adds correctly on all 256 problems", np.all(a4["to_int"](out) == A_ALL + B_ALL))
# forcing the carry into bit k should change the sum by exactly 2^k
for k in [1, 2, 3]:
    c_true = inter[f"carry{k}"]
    out_cf, _ = a4["tiled_adder"](BITS(A_ALL), BITS(B_ALL), G, True, carry_override=(k, 1 - c_true))
    check(f"forcing opposite carry into bit {k} shifts the sum by 2^{k}",
          np.all(a4["to_int"](out_cf) - (A_ALL + B_ALL) == np.where(c_true == 1, -1, 1) * 2 ** k))

# ---------- 9. gate networks (gatesnake.gates) ----------
from gatesnake import gates as GT
for g, arch in [("NAND", "neuron"), ("XOR", "skip-1"), ("NOT", "neuron")]:
    rs = GT.train(g, arch, 4.0, [0, 1, 2])
    good = all(r["converged"] and np.array_equal(np.round(GT.forward(r["params"], GT.truth_inputs(g))), GT.TRUTH[g]) for r in rs)
    check(f"{g} {arch} trains to its truth table at margin 4", good)
check("XOR cannot be one sigmoid neuron (no seed converges)", not any(r["converged"] for r in GT.train("XOR", "neuron", 1.0, [0, 1], max_steps=3000)))
nand_hand = {"Wh": np.zeros((2, 0)), "bh": np.zeros(0), "wx": np.array([-12., -12.]), "wh": np.zeros(0), "bo": np.array(18.)}
check("hand-built NAND sigmoid(18 - 12a - 12b) restores at eps 0.05", GT.restores(nand_hand, "NAND", 0.05))
nand_weak = {k: v / 4 for k, v in nand_hand.items()}
check("the same NAND with weights / 4 does not restore", not GT.restores(nand_weak, "NAND", 0.05))
check("parameter count: neuron 3, skip-1 7, mlp-4 17", (GT.n_params(2, "neuron"), GT.n_params(2, "skip-1"), GT.n_params(2, "mlp-4")) == (3, 7, 17))

# ---------- 10. circuit builder and the snake machine ----------
from gatesnake.hdl import Circuit, lower_to_nand
from gatesnake.design import build_snake
from gatesnake.ref import SnakeRef
from gatesnake.sim import Machine, make_library
from gatesnake.bot import bot_inputs, cycle_dir
c = Circuit(); xs = [c.input(f"x{i}") for i in range(4)]
s_inc, carry = c.incdec(xs, c.input("en"), c.input("dec"))
dec_out = c.decoder(xs)
q = c.reg("q"); c.set_d(q, q); c.output("s", s_inc + [carry] + dec_out)
nt = c.finalize()
good = True
for v in range(16):
    for en in (0, 1):
        for dc in (0, 1):
            vals = {nt.inputs[f"x{i}"]: (v >> i) & 1 for i in range(4)}; vals.update({nt.inputs["en"]: en, nt.inputs["dec"]: dc, q: 0})
            o = nt.eval_logic(vals); ws = nt.outputs["s"]
            got = sum(o[ws[i]] << i for i in range(4)); want = (v + en) % 16 if not dc else (v - en) % 16
            wall = int((v == 15 and en and not dc) or (v == 0 and en and dc))
            good &= got == want and o[ws[4]] == wall and [o[w] for w in ws[5:]] == [int(i == v) for i in range(16)]
check("incdec (all 16 values, +-1, carry = wall) and 4->16 decoder are exact", good)
c4, _ = build_snake(4); n4 = c4.finalize(); l4 = lower_to_nand(c4).finalize()
check("NAND-only lowering uses only NAND (and NAND-as-NOT) gates", set(l4.counts()) <= {"NAND", "NOT"})
runs = {}
for name, nt_ in (("netlist", n4), ("NAND-only", l4)):
    m = Machine(nt_, "logic", dff="gates"); ref = SnakeRef(4); r = np.random.default_rng(3); same = True
    for _ in range(600):
        inp = bot_inputs(ref, r); m.step(inp); ref.step(**inp); same &= m.regs() == ref.regs()
    check(f"4x4 snake ({name}, Boolean gates, NAND flip-flops) matches the Python snake for 600 ticks", same)
# neural machine with hand-built restoring gates (no training needed)
def neuron(w0, w1, b): return {"Wh": np.zeros((2, 0)), "bh": np.zeros(0), "wx": np.array([w0, w1]), "wh": np.zeros(0), "bo": np.array(b)}
k = 12.
hand = {"AND": [neuron(k, k, -1.5 * k)], "OR": [neuron(k, k, -0.5 * k)], "NAND": [neuron(-k, -k, 1.5 * k)],
        "NOR": [neuron(-k, -k, 0.5 * k)],
        "NOT": [{"Wh": np.zeros((1, 0)), "bh": np.zeros(0), "wx": np.array([-k]), "wh": np.zeros(0), "bo": np.array(k / 2)}],
        "XOR": [{"Wh": np.array([[k], [k]]), "bh": np.array([-1.5 * k]), "wx": np.array([k, k]), "wh": np.array([-2 * k]), "bo": np.array(-0.5 * k)}]}
hand["XNOR"] = [{kk: -v if kk in ("wx", "wh", "bo") else v for kk, v in hand["XOR"][0].items()}]
m = Machine(n4, "neural", library=make_library(hand), dff="gates"); ref = SnakeRef(4); r = np.random.default_rng(4); same = True
for _ in range(400):
    inp = bot_inputs(ref, r); m.step(inp); ref.step(**inp); same &= m.regs() == ref.regs()
check("4x4 snake with hand-built neuron gates, fully analog, matches for 400 ticks", same, f"worst signal {m.worst:.3f}")
for Nc in (4, 8, 16):
    seen, x, y = set(), 0, 0
    for _ in range(Nc * Nc):
        seen.add((x, y)); dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][cycle_dir(Nc, x, y)]; x, y = x + dx, y + dy
    check(f"board-filling loop visits all {Nc * Nc} cells and returns to the start", len(seen) == Nc * Nc and (x, y) == (0, 0))

# ---------- 11. the gate computer (gatecpu) ----------
from gatecpu.design import build_cpu, OPS as CPU_OPS
from gatecpu.ref import CpuRef, port_inputs, load_ticks, run_isa
from gatecpu.asm import assemble, AsmError, DEMOS
for name, (src, want) in DEMOS.items():
    outs, halted, _ = run_isa(assemble(src))
    check(f"demo '{name}' prints {want[:4]}{'...' if len(want) > 4 else ''} and halts (instruction-level run)", outs == want and halted)
p = assemble("start: OUT\nHLT\nJMP start\n.byte 0x1F")
check("assembler: OUT = STA 31, HLT = jump to itself, labels, hex data", p[:4] == [(4 << 5) | 31, (5 << 5) | 1, (5 << 5) | 0, 31])
try:
    assemble("LDA 40"); bad = False
except AsmError:
    bad = True
check("assembler rejects an address outside memory", bad)
cpu_c, _ = build_cpu(); cpu = cpu_c.finalize()
q_of = {g["name"]: g["q"] for g in cpu.regs}; d_of = {g["name"]: g["d"] for g in cpu.regs}
def alu_gates(op, av, bv):
    """One T4 tick of an ALU instruction, evaluated gate by gate: returns (new A, C, Z)."""
    vals = {w: 0 for w in cpu.inputs.values()}
    vals.update({g["q"]: 0 for g in cpu.regs})
    for i in range(8):
        vals[q_of[f"a[{i}]"]] = (av >> i) & 1; vals[q_of[f"b[{i}]"]] = (bv >> i) & 1
        vals[q_of[f"ir[{i}]"]] = ((CPU_OPS.index(op) << 5) >> i) & 1
    vals[q_of["t[2]"]] = 1                                     # step T4
    o = cpu.eval_logic(vals)
    return sum(o[d_of[f"a[{i}]"]] << i for i in range(8)), o[d_of["fc"]], o[d_of["fz"]]
rr = np.random.default_rng(5)
pairs = [(0, 0), (255, 1), (255, 255), (1, 1), (128, 128), (0, 255)] + [tuple(int(x) for x in rr.integers(0, 256, 2)) for _ in range(150)]
good = True
for av, bv in pairs:
    good &= alu_gates("ADD", av, bv) == ((av + bv) & 255, int(av + bv > 255), int((av + bv) & 255 == 0))
    good &= alu_gates("SUB", av, bv) == ((av - bv) & 255, int(av >= bv), int(av == bv))
    nv = ~(av & bv) & 255
    good &= alu_gates("NAND", av, bv) == (nv, 0, int(nv == 0))
check(f"gate ALU: ADD, SUB, NAND and flags right on {len(pairs)} operand pairs", good)
def cpu_run(nt, seq, **mk):
    m = Machine(nt, **mk); ref = CpuRef(); same = True; outs = []
    for inp in seq:
        L = ref.step(inp["rst"], inp["prog"], inp["prog_we"], sum(inp[f"prog_addr{j}"] << j for j in range(5)),
                     sum(inp[f"prog_data{j}"] << j for j in range(8)))
        if L["OUT_in"]: outs.append(ref.bus)
        m.step(inp); same &= m.regs() == ref.regs()
    return same, outs, m
rr = np.random.default_rng(6); prog_bytes = [int(x) for x in rr.integers(0, 256, 32)]
seq = [port_inputs(**t) for t in load_ticks(prog_bytes)]
same, _, m = cpu_run(cpu, seq, mode="logic", dff="gates")
check("program port loads all 32 RAM bytes", same and all(sum(m.regs()[f"ram{w}[{i}]"] << i for i in range(8)) == prog_bytes[w] for w in range(32)))
seq += [port_inputs(rst=int(rr.random() < 0.01)) for _ in range(500)]
check("gate CPU (Boolean gates, NAND flip-flops) matches the Python CPU for 500 ticks of a random program", cpu_run(cpu, seq, mode="logic", dff="gates")[0])
src, want = DEMOS["Multiply"]
seq = [port_inputs(**t) for t in load_ticks(assemble(src))] + [port_inputs() for _ in range(250)]
same, outs, m = cpu_run(cpu, seq, mode="neural", dff="gates", library=make_library(hand))
check("gate CPU with hand-built neuron gates, fully analog, runs Multiply and prints 42", same and outs == want, f"worst signal {m.worst:.3f}")

# ---------- 12. GPT with a CPU head (gatecpu.calc, gatecpu.batch, cpugpt) ----------
from gatecpu.calc import PROGRAM as CALC, ADDR as CADDR, CAP as CCAP, run_calc, call_ticks, expected
from gatecpu.batch import BitMachine, port as bport
check("calculator program fits in RAM below the output address", len([b for b in CALC]) == 32 and CALC[31] == 0)
ref = CpuRef()
for i, byte in enumerate(CALC):
    ref.step(prog=1, prog_we=1, prog_addr=i, prog_data=byte)
good = True
for x, y, op in [(200, 55, 1), (77, 200, 2), (13, 15, 3), (5, 17, 3), (5, 18, 3), (9, 3, 0), (1, 1, 7)]:
    for t in call_ticks(x, y, op):
        ref.step(prog=1, prog_we=1, prog_addr=t["addr"], prog_data=t["data"])
    got = (0, None)
    for k in range(CCAP):
        L = ref.step()
        if L["OUT_in"]:
            got = (ref.bus, k + 1)
    good &= got == run_calc(x, y, op) and (got[1] is None or got[0] == expected(x, y, op))
check("calculator: tick-level CPU = instruction-level run (+, -, x, timeout at y = 18, no-op)", good)
calls = [(200, 55, 1), (77, 200, 2), (13, 15, 3), (5, 18, 3), (9, 3, 0), (255, 255, 1), (0, 0, 3), (100, 2, 3)]
bm = BitMachine(cpu, len(calls))
for i, byte in enumerate(CALC):
    bm.step(bport(i, byte))
for k in ("x", "y", "op"):
    bm.step(bport(CADDR[k], np.array([c[("x", "y", "op").index(k)] for c in calls])))
bm.step(bport(CADDR["prod"], 0))
for _ in range(CCAP):
    bm.step({})
check("bit-sliced gate CPU gives the calculator's answers for 8 calls at once",
      list(bm.bus("out", 8)) == [run_calc(*c)[0] for c in calls])
rr = np.random.default_rng(8); progs = rr.integers(0, 256, (70, 32))
bm = BitMachine(cpu, 70); mm = [Machine(cpu, "logic") for _ in range(2)]
same = True
for i in range(32):
    bm.step(bport(i, progs[:, i]))
    for j, mj in enumerate(mm):
        mj.step(port_inputs(prog=1, we=1, addr=i, data=int(progs[j, i])))
for _ in range(150):
    bm.step({})
    for j, mj in enumerate(mm):
        mj.step(port_inputs())
        same &= all(bm.bit(n)[j] == v for n, v in mj.regs().items())
check("bit-sliced runner equals the one-machine simulator on random programs for 150 ticks", same)

import torch
from cpugpt import data as CD
# the held-out split (fixed 1 Nov 2026: the old rule held out whole addition RESULT values, a + b == 2 mod 5, never pairs)
_kinds = (CD.ADD, CD.SUB, CD.MUL, "nocall")
_grid = [(a, b) for a in range(64) for b in range(64)]
_rate = {k: float(np.mean([CD.is_test(k, a, b) for a, b in _grid])) for k in _kinds}
_sums = {s: [sum(1 for a in range(s + 1) if a <= s - a and not CD.is_test(CD.ADD, a, s - a)),
             sum(1 for a in range(s + 1) if a <= s - a and CD.is_test(CD.ADD, a, s - a))] for s in range(64)}
_no_test = [s for s, (tr, te) in _sums.items() if te == 0]
_by_res = [float(np.mean([CD.is_test(CD.ADD, a, b) for a, b in _grid if a + b < 64 and (a + b) % 5 == m])) for m in range(5)]
check("held-out split: symmetric in (a, b), 15-25 % of pairs per kind; addition below 64: every sum has training pairs, every sum "
      "with more than ten pairs has held-out pairs too, and 10-30 % held out in every residue class of the sum (no result-value split)",
      all(CD.is_test(k, a, b) == CD.is_test(k, b, a) for k in _kinds for a, b in _grid) and all(0.15 <= r <= 0.25 for r in _rate.values())
      and all(tr > 0 for tr, te in _sums.values()) and all(s // 2 + 1 <= 10 for s in _no_test) and all(0.1 <= r <= 0.3 for r in _by_res),
      f"held out per kind {dict((str(k), round(r, 3)) for k, r in _rate.items())}, sums with no held-out pair {_no_test}")
from cpugpt.model import GPT, losses as cg_losses
good = True
for name, words, kind in CD.TEMPLATES:
    ids = CD.encode(words, 12, 3, 99)
    text = " ".join(CD.VOCAB[i] for i in ids[1:])
    good &= CD.tokenize(text) == ids
check("tokenizer round-trips every template", good)
tiny_table = np.zeros((4, 256, 256), np.uint8)
for opc in (1, 2, 3):
    xs, ys = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    tiny_table[opc] = [(xs + ys) & 255, (xs - ys) & 255, (xs * ys) & 255][opc - 1]
torch.manual_seed(0)
mg = GPT(len(CD.VOCAB), CD.T_MAX, cpu=dict(numval=CD.NUMVAL, opval=CD.OPVAL, table=tiny_table))
tbt = lambda b_: {k: torch.as_tensor(v) for k, v in b_.items()}
bt = tbt(CD.batch(np.random.default_rng(0), 64, templates=[0]))
loss, _ = cg_losses(mg, bt, K=4, shaping=1.0)
loss.backward()
check("the computer's table is a frozen buffer (no gradient), pointing weights get a gradient",
      not mg.cpuhead.table.requires_grad and mg.cpuhead.q.weight.grad is not None and float(mg.cpuhead.q.weight.grad.abs().sum()) > 0)
def p_plus(m_):
    with torch.no_grad():
        s = m_(bt["tokens"], bt["last"], mode="always")["scores"].softmax(-1)
    return float(s[:, 2, 2].mean())                       # slot OP on the "+" token ("<bos> A + B =")
before = p_plus(mg)
opt = torch.optim.Adam(mg.parameters(), lr=3e-3)
for _ in range(40):
    b_ = tbt(CD.batch(np.random.default_rng(_ + 1), 128, templates=[0]))
    l_, _i = cg_losses(mg, b_, K=8, shaping=1.0); opt.zero_grad(); l_.backward(); opt.step()
check("REINFORCE with the result check moves the OP pointer onto '+'", p_plus(mg) > before + 0.2, f"{before:.2f} -> {p_plus(mg):.2f}")

# ---------- 13. story GPT with a CPU head (cpugpt.text, cpugpt.lm) ----------
from cpugpt import text as TX
from cpugpt.lm import StoryGPT, lm_losses
toy = ['Once upon a time, Tom had a red ball. He said, "I like it!" Then he played.',
       "Lily went to the park. She saw a big dog. The dog was happy."] * 3
tv = TX.Vocab(toy, n_words=200)
s_ = 'tom had a red ball. he said, "i like it!" 12 + 5 = 17.'
check("tokenizer + decoder round-trip story text and math", tv.decode(tv.encode(TX.words(s_))) == s_, repr(tv.decode(tv.encode(TX.words(s_)))))
check("number v is token NUM0 + v, and the CPU tables read it", tv.stoi["200"] == TX.NUM0 + 200 and tv.numval[tv.stoi["200"]] == 200
      and tv.opval[tv.stoi["times"]] == 3)
stream_, apos, ainfo = TX.build_stream(tv, toy, np.random.default_rng(0), "train", per_story=2.0)
check("every inserted problem's answer follows its question", len(apos) > 0 and
      all(stream_[p + 1] == TX.NUM0 + inf["target"] and not CD.is_test(inf["kind"], inf["a"], inf["b"]) for p, inf in zip(apos, ainfo)))
torch.manual_seed(0)
sg = StoryGPT(len(tv), 16, d=32, n_layers=2, n_heads=4, d_mlp=64, cpu=dict(numval=tv.numval, opval=tv.opval, table=tiny_table), num0=TX.NUM0)
a_, b_ = TX.windows(stream_, np.random.default_rng(1), 4, 16)
a_, b_ = torch.as_tensor(a_), torch.as_tensor(b_)
loss, _ = lm_losses(sg, a_, b_, K=2, P=8); loss.backward()
check("story GPT: computer table frozen, pointers and gate get gradients",
      not sg.table.requires_grad and float(sg.q.weight.grad.abs().sum()) > 0 and float(sg.gate.weight.grad.abs().sum()) > 0)
with torch.no_grad():
    l1 = sg(a_, mode="always")["logits"]; a2 = a_.clone(); a2[:, 10:] = 7; l2 = sg(a2, mode="always")["logits"]
check("story GPT is causal, CPU head included: changing later tokens leaves earlier predictions alone",
      torch.allclose(l1[:, :10], l2[:, :10], atol=1e-5))
wv = TX.Vocab(toy + [" ".join(["he", "she", "his", "her", "him"] + TX.BOYS + TX.GIRLS + TX.THINGS + TX.FOOD + [w for bx in TX.BOXES for w in bx]) + ". " +
                     " ".join(w for t in TX.WORD_PROBLEMS + TX.NEW_WORDINGS for w in t[1].split() if len(w) > 1)], n_words=400)
good, rw = True, np.random.default_rng(3)
for tpl in (TX.WORD_PROBLEMS, TX.NEW_WORDINGS):
    for tid, (_, _, kind, opword) in enumerate(tpl):
        for split in ("train", "test"):
            ids, off, inf = TX.word_snippet(wv, rw, split, tid=tid, templates=tpl)
            good &= (TX.UNK not in ids and ids[off + 1] == TX.NUM0 + inf["target"] and inf["target"] == CD.answer(kind, inf["a"], inf["b"])
                     and ids[inf["at"][0]] == TX.NUM0 + inf["a"] and ids[inf["at"][1]] == TX.NUM0 + inf["b"]
                     and wv.opval[ids[inf["at"][2]]] == kind and CD.is_test(kind, inf["a"], inf["b"]) == (split == "test"))
check("word problems: answer after the question, operands and operation word where 'at' says, operation word in the frozen table, split kept", good)
stream_, apos, ainfo = TX.build_stream(wv, toy * 4, np.random.default_rng(0), "train", per_story=2.0, word_per_story=1.5)
check("training stream with word problems: every answer follows its question, train operand pairs only",
      sum(i["word"] for i in ainfo) > 0 and sum(not i["word"] for i in ainfo) > 0 and
      all(stream_[p + 1] == TX.NUM0 + inf["target"] and not CD.is_test(inf["kind"], inf["a"], inf["b"]) for p, inf in zip(apos, ainfo)))
torch.manual_seed(0)
se = StoryGPT(len(tv), 16, d=32, n_layers=2, n_heads=4, d_mlp=64, cpu=dict(numval=tv.numval, opval=tv.opval, table=tiny_table), num0=TX.NUM0,
              read_emb=True, read_copy=True)
loss, _ = lm_losses(se, a_, b_, K=2, P=8); loss.backward()
with torch.no_grad():
    o1 = se(a_, mode="always"); o0 = se(a_, mode="off")
    xr = o1["logits"] - o0["logits"]
    s1 = se(a_, mode="always"); a3 = a_.clone(); a3[:, 10:] = 7; s2 = se(a3, mode="always")
check("read-back with the result's token embedding and the copy bonus: both gains get a gradient, a call changes the prediction, still causal",
      all(g.grad is not None and float(g.grad.abs()) > 0 for g in (se.read_gain, se.copy_gain)) and float(xr.abs().max()) > 0
      and torch.allclose(s1["logits"][:, :10], s2["logits"][:, :10], atol=1e-5))
sd_ = se.state_dict(); sd_["opval"] = torch.zeros_like(sd_["opval"]); before_ = tv.opval.copy()
se.load_state_dict(sd_)
check("loading a story-GPT checkpoint leaves the vocabulary's frozen tables alone (buffers are copies)", (tv.opval == before_).all())

# ---------- 14. story GPT that calls the computer in writing (cpugpt.text tool stream, generator, cpugpt.tool) ----------
from cpugpt.tool import Calculator, decode, call_right, OPMAP_WRONG
tmpl_words = " ".join(w for fam in (TX.WORD_PROBLEMS, TX.NEW_WORDINGS, TX.TWO_STEP, TX.NOCALL) for t in fam for w in t[1].split() if len(w) > 1)
tsent = toy + [" ".join(["he", "she", "his", "her", "him"] + TX.BOYS + TX.GIRLS + TX.THINGS + TX.FOOD + [w for bx in TX.BOXES for w in bx]) + ". " + tmpl_words
               + " . " + " ".join(TX.GENERATOR_WORDS)]
tw, tw0 = TX.Vocab(tsent, n_words=1000, tool_tokens=True), TX.Vocab(tsent, n_words=1000)
check("tool tokens go at the end of the vocabulary: every other id unchanged, bus tables 0 for them",
      tw.itos[:len(tw0)] == tw0.itos and tw.itos[-2:] == ["<calc>", "</calc>"] and tw.CALC == len(tw0) and TX.NUM0 == 4
      and tw.numval[tw.CALC] == 0 and tw.opval[tw.END] == 0)
def strip_calls(ids):
    out, inside = [], False
    for t in ids:
        if t == tw.CALC: inside = True
        elif t == tw.END: inside = False
        elif not inside: out.append(t)
    return out
good, rng14 = True, 14
makers = [lambda r, s, c: TX.snippet(tw, r, s, call=c), lambda r, s, c: TX.word_snippet(tw, r, s, call=c),
          lambda r, s, c: TX.two_step_snippet(tw, r, s, call=c), lambda r, s, c: TX.nocall_snippet(tw, r, s)]
makers += [lambda r, s, c, h=h: TX.gen_word(tw, r, s, held=h, call=c) for h in (None, "shape", "verb")]
makers += [lambda r, s, c, h=h: TX.gen_two(tw, r, s, held=h, call=c) for h in (None, "shape", "verb")]
makers += [lambda r, s, c, h=h: TX.gen_nocall(tw, r, s, held=h) for h in (None, "shape", "verb")]
for mk in makers:
    for split in ("train", "test"):
        for k in range(12):
            (ids, off, inf), (ids0, off0, inf0) = mk(np.random.default_rng(rng14 + k), split, True), mk(np.random.default_rng(rng14 + k), split, False)
            env = inf["env"]
            good &= TX.UNK not in ids and len(env) == len(ids) and (inf["target"] < 0 or (ids[off + 1] == TX.NUM0 + inf["target"] and ids0[off0 + 1] == TX.NUM0 + inf0["target"]))
            good &= strip_calls(ids) == ids0 and inf0["target"] == inf["target"] and sum(env) == 2 * len(inf["calls"])
            good &= all(ids[j] == TX.NUM0 + c[3] and ids[j + 1] == tw.END and env[j] and env[j + 1] for j, c in
                        zip([j for j in range(len(ids)) if env[j] and ids[j] != tw.END], inf["calls"]))
            good &= (not inf["calls"]) or (ids[inf["cue"] + 1] == tw.CALC and ids0[inf0["cue"]] == ids[inf["cue"]])
            for a_, op_, b_, r_ in inf["calls"]:
                good &= r_ == CD.answer(op_, a_, b_) and 0 <= r_ <= 255 and CD.is_test(op_, a_, b_) == (split == "test")
check("call blocks (fixed lists and generator): machine-written tokens flagged (result, </calc>), answer after the block, same problem without blocks, split kept", good)
good = True
for tid in range(len(TX.TWO_STEP)):
    for split in ("train", "test"):
        ids, off, inf = TX.two_step_snippet(tw, np.random.default_rng(tid), split, tid=tid)
        c1, c2 = inf["calls"]
        good &= c2[0] == c1[3] == inf["r1"] and inf["kind2"] != CD.MUL and 0 <= inf["target"] <= 255 and tw.CALC in ids and ids.count(tw.CALC) == 2
check("two-step problems: the second call starts from the first result, results stay 0-255, never x second", good)
good = True
for tid in range(len(TX.NOCALL)):
    ids, off, inf = TX.nocall_snippet(tw, np.random.default_rng(tid), "train", tid=tid)
    good &= tw.CALC not in ids and inf["target"] in (inf["a"], inf["b"]) and not inf["calls"] and ids[off + 1] == TX.NUM0 + inf["target"]
check("no-call sentences: no call block, the answer restates a number already in the sentence", good)
# the compositional generator: held-out frames and verbs
held_frames = {f["text"] for f in TX._all_frames() if f["held"]}
good, seen_shape, seen_verb, types = True, set(), set(), set()
check("generator: held-out verbs are never trained verbs; the word list covers every frame; frames hold out some of every kind",
      not set(TX.HELD_OUT_VERBS) & set(TX.TRAINED_VERBS) and len(held_frames) > 20 and "every" in TX.HELD_OUT_VERBS
      and all(any(f["held"] for f in fr) and any(not f["held"] for f in fr) for fr in (TX.STARTS, TX.ADD_CHANGES, TX.SUB_CHANGES, TX.MUL_FRAMES, TX.ASKS_ALL)))
for i in range(200):
    for fn in (TX.gen_word, TX.gen_two, TX.gen_nocall):
        for h in (None, "shape", "verb"):
            ids, off, inf = fn(tw, np.random.default_rng(300 + i), "train", held=h)
            n_held = sum(t in held_frames for t in inf["shape"].split(" | ")) + sum(w in TX.HELD_OUT_VERBS for w in inf["verb"].split())
            good &= n_held == (0 if h is None else 1) and (inf["held"] == "") == (h is None)
            if h == "shape": seen_shape.add(inf["held"])
            if h == "verb": seen_verb.add(inf["held"])
            if inf["calls"]:                                  # every operand is in the text before the call block
                pre = ids[:inf["cue"] + 1]
                good &= all(TX.NUM0 + c[0] in pre or j > 0 for j, c in enumerate(inf["calls"])) and all(TX.NUM0 + c[2] in pre for c in inf["calls"])
                good &= all(t not in (tw.CALC, tw.END) for t in pre)
            if fn is TX.gen_two:
                good &= inf["calls"][1][0] == inf["calls"][0][3] == inf["r1"] and inf["kind2"] != CD.MUL and ids.count(tw.CALC) == 2
            if fn is TX.gen_nocall:
                types.add(inf["nocall_type"])
                good &= tw.CALC not in ids and not inf["calls"] and (inf["target"] >= 0) == (inf["nocall_type"] != "numberless" and TX.NUM0 + inf["target"] in ids) \
                        and (inf["target"] < 0 or ids[off + 1] == TX.NUM0 + inf["target"])
check("generator: trained mode uses no held-out frame or verb, held modes exactly one; operands in the text before the call; two-step chains; "
      "no-call sentences restate a number or have none (target -1)",
      good and len(seen_shape) > 25 and seen_verb == set(TX.HELD_OUT_VERBS) and types == {"fixed", "setup", "numberless"},
      f"{len(seen_shape)} held-out frames, verbs {sorted(seen_verb)}, no-call types {sorted(types)}")
good = True
for i in range(100):
    ids, off, inf = TX.gen_nocall(tw, np.random.default_rng(i), "test", restate_only=True)
    good &= inf["target"] >= 0 and ids[off + 1] == TX.NUM0 + inf["target"]
check("generator: restate_only no-call sentences always restate a number", good)
S14, E14, P14, I14 = TX.build_tool_stream(tw, toy * 20, np.random.default_rng(0), "train")
S14b, E14b, P14b, I14b = TX.build_tool_stream(tw, toy * 20, np.random.default_rng(0), "train", calls=False)
check("tool stream: calls=False is the same stream with the call blocks removed; flags count the machine's tokens; answers follow; all families",
      strip_calls(S14.tolist()) == S14b.tolist() and not E14b.any() and E14.sum() == 2 * sum(len(i["calls"]) for i in I14) > 0
      and all(i["target"] < 0 or S14[p + 1] == TX.NUM0 + i["target"] for p, i in zip(P14, I14)) and all(np.bincount([i["family"] for i in I14], minlength=4) > 0)
      and all(not CD.is_test(c[1], c[0], c[2]) for i in I14 for c in i["calls"]) and TX.UNK not in S14)
S14f = TX.build_tool_stream(tw, toy * 20, np.random.default_rng(0), "train", varied=False)[0]
check("tool stream with varied=False uses the fixed lists (a different stream, same families)", S14f.tolist() != S14.tolist())
a14, b14, e14 = TX.windows(S14, np.random.default_rng(1), 16, 24, env=E14)
check("windows with flags: every flagged target is a result after '=' or a </calc> after a result", e14.any() and
      all((tw.itos[a14[i, j]] == "=" and TX.NUM0 <= b14[i, j] < TX.NUM0 + 256) or (b14[i, j] == tw.END and TX.NUM0 <= a14[i, j] < TX.NUM0 + 256)
          for i in range(16) for j in range(24) if e14[i, j]))
torch.manual_seed(0)
sm = StoryGPT(len(tw), 24, d=32, n_layers=2, n_heads=4, d_mlp=64, cpu=None, num0=TX.NUM0)
A14, B14, M14 = torch.as_tensor(a14), torch.as_tensor(b14), torch.as_tensor(e14)
with torch.no_grad():
    l1 = lm_losses(sm, A14, B14, ignore=M14)[0]; Bx = B14.clone(); Bx[M14] = 7; l2 = lm_losses(sm, A14, Bx, ignore=M14)[0]
    By = B14.clone(); By[~M14 & (B14 != 0)] = 7; l3 = lm_losses(sm, A14, By, ignore=M14)[0]
    l4 = lm_losses(sm, A14, B14)[0]; l5 = lm_losses(sm, A14, B14, ignore=torch.zeros_like(M14))[0]
check("masked loss: changing what the machine wrote leaves the loss unchanged, changing a model token changes it, no mask = old loss",
      torch.allclose(l1, l2) and not torch.allclose(l1, l3) and torch.allclose(l4, l5))
# the decoder, driven by a scripted model: the next token is a function of the row so far
class Scripted:
    def __init__(self, script): self.script = script
    def eval(self): pass
    def train(self): pass
    def __call__(self, rows, **kw):
        lg = torch.full((rows.shape[0], rows.shape[1], 1, len(tw)), -10.0)
        for j in range(rows.shape[0]):
            last = int((rows[j] != 0).nonzero().max())
            for tok, score in self.script(rows[j, :last + 1].tolist()):
                lg[j, last, 0, tok] = score
        return {"logits": lg}
N_, PLUS, EQ, W = lambda v: TX.NUM0 + v, tw.stoi["+"], tw.stoi["="], tw.stoi["apples"]
def script(p):
    tag, last = p[0] - TX.NUM0, p[-1]              # the first token names the test row
    if tag == 1:                                   # a well-formed call, then copy the result
        return {tw.stoi["has"]: [(tw.CALC, 9)], tw.CALC: [(N_(12), 9)], N_(12): [(PLUS, 9)], PLUS: [(N_(5), 9)], N_(5): [(EQ, 9)], tw.END: [(p[-2], 9)]}.get(last, [(W, 9)])
    if tag == 2:                                   # a malformed call (a word for the operator): the model writes its own result
        return {tw.stoi["has"]: [(tw.CALC, 9)], tw.CALC: [(N_(12), 9)], N_(12): [(W, 9)], W: [(N_(5), 9)], N_(5): [(EQ, 9)], EQ: [(N_(99), 9)], tw.END: [(N_(99), 9)]}.get(last, [(W, 9)])
    if tag == 3:                                   # "=" in the prompt is not a call; the answer comes straight away
        return [(N_(17), 9)]
    if tag == 4:                                   # wants <calc> inside a call and again after two calls; both must be refused
        n_calls = p.count(tw.CALC)
        if last == tw.stoi["has"] or last == tw.END and n_calls < 2: return [(tw.CALC, 9)]
        if last == tw.END: return [(tw.CALC, 9), (N_(40), 5)]
        return {tw.CALC: [(tw.CALC, 9), (N_(12), 5)], N_(12): [(PLUS, 9)], PLUS: [(N_(5), 9)], N_(5): [(EQ, 9)]}.get(last, [(W, 9)])
    if tag == 5:                                   # a call that never reaches "=": closed after 5 tokens
        return [(tw.CALC, 9)] if last == tw.stoi["has"] else [(N_(3), 9)] if last == tw.END else [(W, 9)]
    if tag == 6:                                   # loves </calc>: must never be sampled, even at temperature 0.8
        return [(tw.END, 9), (W, 2), (N_(8), 1)]
    if tag == 7:                                   # <calc> ahead of "apples" by 1.5: a margin decides whether the call opens
        return {tw.stoi["has"]: [(tw.CALC, 9), (W, 7.5)], tw.CALC: [(N_(12), 9)], N_(12): [(PLUS, 9)], PLUS: [(N_(5), 9)], N_(5): [(EQ, 9)],
                tw.END: [(p[-2], 9)], W: [(N_(1), 9)]}.get(last, [(W, 9)])
    return [(W, 9)]
TAB = np.load("results_cpugpt/cpu_table.npz")["out"] if os.path.exists("results_cpugpt/cpu_table.npz") else tiny_table
calc14 = Calculator(TAB, tw)
def rows_for(tags, pre):
    rows = torch.zeros(len(tags), 24, dtype=torch.long); ptr = torch.zeros(len(tags), dtype=torch.long)
    for i, (tag, n) in enumerate(zip(tags, pre)):
        seq = [N_(tag)] + [W] * n + ([N_(12), PLUS, N_(5), EQ] if tag == 3 else [tw.stoi["has"]])
        rows[i, :len(seq)] = torch.as_tensor(seq); ptr[i] = len(seq) - 1
    return rows, ptr
R_, P_ = rows_for([1, 2, 3, 4, 5, 1], [1, 3, 2, 1, 1, 6])
res = decode(Scripted(script), R_, P_, calc14, n_new=14, max_calls=2)
d = lambda i: [tw.itos[t] for t in res[i]["written"]]
check("decoder: a well-formed call gets the machine's answer and </calc>, the copy is the answer, rows of different lengths are independent",
      d(0) == ["<calc>", "12", "+", "5", "=", "17", "</calc>", "17"] and res[0]["answer"] == 17 and res[0]["calls"][0]["by"] == "machine"
      and call_right(res[0]["calls"][0], 12, CD.ADD, 5) and call_right(res[0]["calls"][0], 5, CD.ADD, 12) and not call_right(res[0]["calls"][0], 12, CD.SUB, 5)
      and d(5) == d(0) and res[5]["answer"] == 17)
check("decoder: a malformed call is closed after the model's own result and counted; '=' in the prompt does not call",
      d(1) == ["<calc>", "12", "apples", "5", "=", "99", "</calc>", "99"] and res[1]["malformed"] == 1 and res[1]["calls"][0]["by"] == "model"
      and res[1]["calls"][0]["r"] == 99 and res[1]["answer"] == 99 and d(2) == ["17"] and res[2]["calls"] == [] and res[2]["answer"] == 17)
check("decoder: no <calc> inside an open call or after two calls; a call without '=' is closed after 5 tokens",
      d(3) == ["<calc>", "12", "+", "5", "=", "17", "</calc>"] * 2 + ["40"] and res[3]["answer"] == 40 and len(res[3]["calls"]) == 2
      and d(4) == ["<calc>"] + ["apples"] * 5 + ["</calc>", "3"] and res[4]["malformed"] == 1 and res[4]["answer"] == 3)
R6, P6 = rows_for([6], [2])
res6 = decode(Scripted(script), R6, P6, calc14, n_new=10, temperature=0.8, stop_at_number=False)
check("decoder: </calc> is never sampled (temperature 0.8), and without stopping at numbers it writes n_new tokens",
      tw.END not in res6[0]["written"] and res6[0]["n_model"] == 10 and res6[0]["answer"] is None)
res_off = decode(Scripted(script), R_[:1], P_[:1], Calculator(TAB, tw, "off"), n_new=14)
check("decoder with the machine off: the model fills in its own result after '='",
      [tw.itos[t] for t in res_off[0]["written"]][:7] == ["<calc>", "12", "+", "5", "=", "apples", "</calc>"] and res_off[0]["calls"][0]["by"] == "model")
R7, P7 = rows_for([7, 7, 7], [1, 1, 1])
rm = [decode(Scripted(script), R7[i:i + 1], P7[i:i + 1], calc14, n_new=8, calc_margin=mg)[0] for i, mg in enumerate((0.0, 2.0, -3.0))]
check("decoder margin: <calc> 1.5 ahead of the best word opens the call at margin 0 and -3 but not at margin 2 (the word is written instead)",
      [tw.itos[t] for t in rm[0]["written"]][:5] == ["<calc>", "12", "+", "5", "="] and rm[0]["answer"] == 17 and rm[2]["answer"] == 17
      and [tw.itos[t] for t in rm[1]["written"]] == ["apples", "1"] and rm[1]["calls"] == [] and rm[1]["answer"] == 1)
cw = Calculator(TAB, tw, "wrong")
check("calculator: table answers, the wrong-operation ablation swaps every opcode, overflow is counted, off answers nothing",
      calc14(12, CD.ADD, 5) == 17 and calc14(13, CD.MUL, 15) == 195 and cw(12, CD.ADD, 5) == 7 and list(OPMAP_WRONG) == [0, 2, 3, 1]
      and calc14(200, CD.ADD, 100) == 44 and calc14.overflow == 1 and Calculator(TAB, tw, "off")(1, CD.ADD, 1) is None
      and calc14.parse([N_(12), tw.stoi["more"], N_(5)]) is None and calc14.parse([N_(12), PLUS, N_(5)]) == (12, CD.ADD, 5))


# ---------- 15. the computer that can be wrong, and the model's retry (cpugpt.check, cpugpt.tool, cpugpt.text) ----------
from cpugpt.check import FlakyMachine
tc = TX.Vocab(tsent, n_words=1000, tool_tokens=True, extra=TX.CHECK_TOKENS)
check("the doubt token goes after the tool tokens: every other id unchanged; a vocabulary without it has none",
      tc.itos[:len(tw)] == tw.itos and tc.itos[-1] == "<unsure>" and tc.UNSURE == len(tw) and tw.UNSURE is None and tc.CALC == tw.CALC)
def blocks_of(ids):
    starts = [i for i, t in enumerate(ids) if t == tc.CALC] + [len(ids)]
    return [ids[s:e] for s, e in zip(starts, starts[1:])]
MINUS = tc.stoi["-"]
check("a machine that is never wrong writes one block, the same as trace(): the result and </calc> flagged",
      FlakyMachine(TAB, tc, p_err=0.0, seed=0).blocks(tc, 12, CD.ADD, 5, 17) == tuple(TX.trace(tc, 12, CD.ADD, 5, 17)))
always = FlakyMachine(TAB, tc, p_err=1.0, signal=True, hit=1.0, false_alarm=0.0, max_tries=3, seed=1, errs=("bit", "op"))
good, kinds = True, set()
for _ in range(60):
    ids, env = always.blocks(tc, 40, CD.SUB, 13, 27)
    bl = blocks_of(ids)
    good &= len(bl) == 3 and all(b[:5] == [tc.CALC, N_(40), MINUS, N_(13), EQ] and b[-1] == tc.END for b in bl)
    good &= all(len(b) == 8 and b[5] != N_(27) and b[6] == tc.UNSURE for b in bl[:2]) and bl[2][5:] == [N_(27), tc.END]
    good &= all(b[5] - TX.NUM0 in {27 ^ (1 << k) for k in range(8)} | {(40 + 13) & 255, (40 * 13) & 255} for b in bl[:2])
    good &= env == ([False] * 5 + [True] * 3) * 2 + [False] * 5 + [True] * 2
    kinds |= {"op" if b[5] - TX.NUM0 in (53, 8) else "bit" for b in bl[:2]}
check("a machine that is always wrong: two wrong, doubted answers (one bit flipped or another operation's answer), each followed by the "
      "same call again, the third right and not doubted; only the machine's tokens flagged", good and kinds == {"op", "bit"})
doubt = lambda policy: FlakyMachine(TAB, tc, p_err=0.0, signal=True, hit=1.0, false_alarm=1.0, max_tries=3, seed=2, policy=policy)
check("policy: the text retries after a doubted answer under the signal policy, only after a wrong one under the oracle policy",
      len(blocks_of(doubt("signal").blocks(tc, 12, CD.ADD, 5, 17)[0])) == 3 and len(blocks_of(doubt("oracle").blocks(tc, 12, CD.ADD, 5, 17)[0])) == 1
      and doubt("oracle").blocks(tc, 12, CD.ADD, 5, 17)[0][5:] == [N_(17), tc.UNSURE, tc.END])
fm15 = FlakyMachine(TAB, tc, p_err=0.25, signal=True, hit=0.8, false_alarm=0.1, seed=3)
trip = np.random.default_rng(3).integers([0, 1, 0], [128, 4, 128], (4000, 3))
ds = [fm15.draw(int(a_), int(o_), int(b_)) for a_, o_, b_ in trip]
wrong15 = [d for d in ds if d["wrong"]]
check("draw(): about a quarter of answers wrong and never the right one, of four kinds (a zero only when the answer is not 0), a flipped bit differs "
      "by a power of two, doubt mostly after wrong answers",
      0.21 < len(wrong15) / len(ds) < 0.29 and all(d["r"] != d["right"] for d in wrong15) and all(d["r"] == d["right"] for d in ds if not d["wrong"])
      and all(d["r"] ^ d["right"] == 1 << d["bit"] for d in wrong15 if d["err"] == "bit") and {d["err"] for d in wrong15} == {"op", "bit", "random", "zero"}
      and all(d["r"] == 0 for d in wrong15 if d["err"] == "zero") and all(d["right"] != 0 for d in wrong15 if d["err"] == "zero")
      and 0.7 < np.mean([d["unsure"] for d in wrong15]) < 0.9 and 0.05 < np.mean([d["unsure"] for d in ds if not d["wrong"]]) < 0.15)
class Scripted15(Scripted):
    def __call__(self, rows, **kw):
        lg = torch.full((rows.shape[0], rows.shape[1], 1, len(tc)), -10.0)
        for j in range(rows.shape[0]):
            last = int((rows[j] != 0).nonzero().max())
            for tok, score in self.script(rows[j, :last + 1].tolist()):
                lg[j, last, 0, tok] = score
        return {"logits": lg}
def script15(p):                                   # calls 12 + 5; after a block, calls again if it was doubted, else copies the result
    last = p[-1]
    if last == tc.END:
        return [(tc.CALC, 9)] if p[-2] == tc.UNSURE else [(p[-2], 9)]
    return {tw.stoi["has"]: [(tc.CALC, 9)], tc.CALC: [(N_(12), 9)], N_(12): [(PLUS, 9)], PLUS: [(N_(5), 9)], N_(5): [(EQ, 9)]}.get(last, [(W, 9)])
R15 = torch.zeros(40, 30, dtype=torch.long); R15[:, 0] = N_(1); R15[:, 1] = W; R15[:, 2] = tw.stoi["has"]; P15 = torch.full((40,), 2)
fm = FlakyMachine(TAB, tc, p_err=0.6, signal=True, hit=1.0, false_alarm=0.0, seed=4)
res15 = decode(Scripted15(script15), R15, P15, fm, n_new=16, max_calls=3)
good, n_retries = True, 0
for r_ in res15:
    c = r_["calls"]; w_ = [tc.itos[t] for t in r_["written"]]
    good &= 1 <= len(c) <= 3 and all(x["ok"] and x["by"] == "machine" and (x["a"], x["op"], x["b"]) == (12, CD.ADD, 5) for x in c)
    good &= all(x["wrong"] and x["unsure"] and x["r"] != 17 for x in c[:-1]) and all(x["wrong"] == (x["r"] != 17) and x["unsure"] == x["wrong"] for x in c)
    good &= all((x["err"] == "bit") == (x["bit"] is not None) and (x["err"] != "bit" or x["r"] ^ 17 == 1 << x["bit"]) for x in c)
    good &= (r_["answer"] == 17 and not c[-1]["wrong"]) or (r_["answer"] is None and len(c) == 3 and c[-1]["wrong"])
    good &= w_.count("<unsure>") == sum(x["wrong"] for x in c) and w_.count("<calc>") == len(c) and fm.n_calls > 0
    n_retries += len(c) - 1
check("decoder with a flaky machine: a doubted wrong answer is followed by the same call again, the final right answer is copied, "
      "three wrong answers leave no answer; every call records wrong, unsure, err and the flipped bit; <unsure> written only after wrong answers",
      good and n_retries > 20 and fm.n_calls == sum(len(r_["calls"]) for r_ in res15) and fm.n_wrong == sum(x["wrong"] for r_ in res15 for x in r_["calls"]))
res_plain = decode(Scripted15(script15), R15[:3], P15[:3], Calculator(TAB, tc), n_new=16, max_calls=3)
check("the plain calculator through the same decoder: one call, never wrong or doubted, no <unsure> written",
      all(len(r_["calls"]) == 1 and not r_["calls"][0]["wrong"] and not r_["calls"][0]["unsure"] and r_["calls"][0]["err"] is None and r_["answer"] == 17
          and tc.UNSURE not in r_["written"] for r_ in res_plain))
S15, E15, P15s, I15 = TX.build_tool_stream(tc, toy * 20, np.random.default_rng(0), "train", calls=FlakyMachine(TAB, tc, p_err=0.5, signal=True, seed=5))
S15b = TX.build_tool_stream(tc, toy * 20, np.random.default_rng(0), "train", calls=False)[0]
def strip15(ids):
    out, inside = [], False
    for t in ids:
        if t == tc.CALC: inside = True
        elif t == tc.END: inside = False
        elif not inside: out.append(t)
    return out
n_blocks, n_calls15 = int((S15 == tc.CALC).sum()), sum(len(i["calls"]) for i in I15)
check("tool stream with a flaky machine: the same problems as without blocks; more blocks than calls (retries); the flags count exactly the results, "
      "doubts and </calc>; every answer follows its last block",
      strip15(S15.tolist()) == S15b.tolist() and n_blocks > 1.3 * n_calls15 and E15.sum() == 2 * n_blocks + int((S15 == tc.UNSURE).sum())
      and all((TX.NUM0 <= S15[j] < TX.NUM0 + 256) or S15[j] in (tc.UNSURE, tc.END) for j in np.nonzero(E15)[0])
      and all(i["target"] < 0 or S15[p + 1] == TX.NUM0 + i["target"] for p, i in zip(P15s, I15))
      and all(S15[p] == tc.END and S15[p - 1] in (TX.NUM0 + i["calls"][-1][3], tc.UNSURE) for p, i in zip(P15s, I15) if i["calls"]))


# ---------- 16. the capacity experiment: token roles, weighted loss, number range, probes (cpugpt.text, cpugpt.lm, cpugpt.probes) ----------
from cpugpt import probes as PRB
R16, R16b = TX.roles(S14, E14, P14, I14, tw.CALC), TX.roles(S14b, E14b, P14b, I14b)
check("tool stream records where every problem starts and how long it is; the answer and the machine's tokens lie inside",
      all(i["start"] <= p < i["start"] + i["len"] and i["len"] == len(i["env"]) and E14[i["start"]:i["start"] + i["len"]].tolist() == i["env"] for p, i in zip(P14, I14)))
story_only = TX.build_stream(tw, toy * 20, np.random.default_rng(0), "train", per_story=0.0)[0]
n_calls16 = sum(len(i["calls"]) for i in I14)
check("roles: machine tokens are role 3, answers role 2 (restated numbers 5), five call-syntax tokens per call (role 4), and the role-0 tokens are exactly the stories",
      ((R16 == 3) == E14).all() and all(R16[p + 1] == (5 if i["family"] == 3 else 2) for p, i in zip(P14, I14) if i["target"] >= 0)
      and int((R16 == 4).sum()) == 5 * n_calls16 > 0 and all(S14[j] == tw.CALC or TX.NUM0 <= S14[j] < TX.NUM0 + 256 or tw.itos[S14[j]] in "+-*=" for j in np.nonzero(R16 == 4)[0])
      and S14[R16 == 0].tolist() == story_only.tolist() and S14b[R16b == 0].tolist() == story_only.tolist()
      and not ((R16b == 3) | (R16b == 4)).any() and (R16b == 2).sum() == (R16 == 2).sum() and (R16b == 5).sum() == (R16 == 5).sum() and (R16 == 1).sum() > 0)
a16, b16, e16, r16 = TX.windows(S14, np.random.default_rng(1), 16, 24, E14, R16)
ab16, bb16, eb16, rb16 = TX.windows(S14b, np.random.default_rng(1), 16, 24, E14b, R16b)
check("windows passes extra arrays aligned with the targets: before an answer target stands </calc> in the tool stream and a word in the plain one",
      ((r16 == 3) == e16).all() and (r16 == 2).any() and all(a16[i, j] == tw.END for i, j in zip(*np.nonzero(r16 == 2)))
      and (rb16 == 2).any() and all(ab16[i, j] not in (tw.END, tw.CALC) for i, j in zip(*np.nonzero(rb16 == 2)))
      and len(TX.windows(S14, np.random.default_rng(1), 4, 24)) == 2 and len(TX.windows(S14, np.random.default_rng(1), 4, 24, env=E14)) == 3)
with torch.no_grad():
    W1 = torch.ones_like(B14, dtype=torch.float); Wk = W1.clone(); Wk[torch.as_tensor(r16 == 2)] = 8.0       # the answers (same windows as a14: same seed)
    lw1 = lm_losses(sm, A14, B14, ignore=M14, weight=W1)[0]; lw3 = lm_losses(sm, A14, B14, ignore=M14, weight=3 * W1)[0]
    W0 = W1.clone(); W0[M14] = 0; lw0 = lm_losses(sm, A14, B14, weight=W0)[0]; l1_16 = lm_losses(sm, A14, B14, ignore=M14)[0]
    lwk = lm_losses(sm, A14, B14, ignore=M14, weight=Wk)[0]
    lp16 = sm(A14)["logits"][:, :, 0].log_softmax(-1).gather(2, B14[..., None])[..., 0]
    mw = (~M14 & (B14 != 0)).float() * Wk; hand = -(lp16 * mw).sum() / mw.sum()
check("weighted loss: all-ones equals the plain loss, a uniform weight changes nothing, zero weight equals ignore, k on a token equals the hand-computed weighted mean",
      torch.allclose(lw1, l1_16) and torch.allclose(lw3, l1_16) and torch.allclose(lw0, l1_16) and torch.allclose(lwk, hand) and not torch.allclose(lwk, l1_16))
check("plain loss helper agrees with lm_losses", abs(PRB.nll(sm, A14, B14) - float(lm_losses(sm, A14, B14)[0])) < 1e-5)
old_range = CD.MAX_RESULT, CD.OPS
try:
    CD.MAX_RESULT, CD.OPS = 32, (CD.ADD,)
    good = True; seen = []
    mk16 = [lambda r, s: TX.snippet(tw, r, s), lambda r, s: TX.word_snippet(tw, r, s), lambda r, s: TX.two_step_snippet(tw, r, s),
            lambda r, s: TX.gen_word(tw, r, s), lambda r, s: TX.gen_two(tw, r, s), lambda r, s: TX.gen_nocall(tw, r, s, restate_only=True)]
    for mk in mk16:
        for k in range(60):
            ids, off, inf = mk(np.random.default_rng(k), "train")
            good &= all(c[1] == CD.ADD for c in inf["calls"]) and 0 <= inf["target"] < 32 and inf["a"] < 32 and inf["b"] < 32
            good &= all(not CD.is_test(c[1], c[0], c[2]) for c in inf["calls"]) and all(not (TX.NUM0 + 32 <= t < TX.NUM0 + 256) for t in ids)
            seen.append(inf["target"])
    S16, E16, P16, I16 = TX.build_tool_stream(tw, toy * 10, np.random.default_rng(3), "train")
    good &= all(TX.NUM0 + 32 > t for t in S16[(S16 >= TX.NUM0) & (S16 < TX.NUM0 + 256)]) and len(I16) > 0
finally:
    CD.MAX_RESULT, CD.OPS = old_range
big = [TX.gen_word(tw, np.random.default_rng(k), "train")[2]["target"] for k in range(60)]
check("MAX_RESULT and OPS bound every maker (addition only, everything below 32, the held-out rule kept) and are restored afterwards",
      good and max(seen) < 32 and len(set(seen)) > 10 and CD.MAX_RESULT == 256 and max(big) >= 32)
c16 = PRB.cache(sm, A14)
with torch.no_grad():
    lg16 = sm(A14)["logits"][:, :, 0]; back = sm.lnf(c16["resid"][-1]) @ sm.emb.weight.T + sm.out_bias
check("cache: residual stream per layer and after the last block, heads (B, T, H, dh), neurons (B, T, d_mlp), logits; the final residual reproduces the logits; hooks removed",
      len(c16["resid"]) == 3 and all(r.shape == (16, 24, 32) for r in c16["resid"]) and len(c16["heads"]) == 2 and all(h.shape == (16, 24, 4, 8) for h in c16["heads"])
      and all(n.shape == (16, 24, 64) for n in c16["neurons"]) and torch.allclose(c16["logits"], lg16) and torch.allclose(back, lg16, atol=1e-5)
      and not any(sm.blocks[l].ln1._forward_pre_hooks or sm.blocks[l].attn.o._forward_pre_hooks or sm.blocks[l].mlp[1]._forward_hooks for l in range(2)) and not sm.lnf._forward_pre_hooks)
zeros = {"head": [torch.zeros(4, 8), torch.zeros(4, 8)], "neuron": [torch.zeros(64), torch.zeros(64)]}
with torch.no_grad():
    with PRB.ablated(sm, "head", 0, 0, zeros), PRB.ablated(sm, "head", 0, 1, zeros), PRB.ablated(sm, "head", 0, 2, zeros), PRB.ablated(sm, "head", 0, 3, zeros):
        lg_abl = sm(A14)["logits"][:, :, 0]
    x16 = sm.emb(A14) + sm.pos(torch.arange(24))[None]
    x16 = x16 + sm.blocks[0].attn.o.bias; x16 = x16 + sm.blocks[0].mlp(sm.blocks[0].ln2(x16))
    last = sm.blocks[-1]; x16 = x16 + last.attn(last.ln1(x16))[0]; xk = x16[:, :, None, :]; xk = xk + last.mlp(last.ln2(xk))
    lg_hand = (sm.lnf(xk) @ sm.emb.weight.T + sm.out_bias)[:, :, 0]
    lg_after = sm(A14)["logits"][:, :, 0]
    means16 = PRB.unit_means(c16)
    with PRB.ablated(sm, "neuron", 1, 5, means16):
        lg_n = sm(A14)["logits"][:, :, 0]
check("ablation: zeroing all four heads of block 0 equals the hand computation (only the output bias remains); a mean-ablated neuron changes the logits; the model is restored afterwards",
      torch.allclose(lg_abl, lg_hand, atol=1e-5) and not torch.allclose(lg_abl, lg16) and torch.allclose(lg_after, lg16) and not torch.allclose(lg_n, lg16))
import copy as _copy
sd16 = _copy.deepcopy(sm); sd16.blocks[0].mlp[0].bias.data[7] = -100.0
cd16 = PRB.cache(sd16, A14)
with torch.no_grad():
    with PRB.ablated(sd16, "neuron", 0, 7, zeros):
        lg_dead = sd16(A14)["logits"][:, :, 0]
    lg_dead0 = sd16(A14)["logits"][:, :, 0]
check("dead units: a neuron pushed below zero everywhere is reported dead and costs nothing when ablated",
      abs(PRB.dead_units(cd16)[0] - PRB.dead_units(c16)[0] - 1 / 64) < 1e-9 and float(cd16["neurons"][0][..., 7].max()) == 0.0 and torch.allclose(lg_dead, lg_dead0))
rp = np.random.default_rng(16); F16 = rp.standard_normal((400, 10)); w16 = rp.standard_normal(10)
y16 = np.clip(np.round(128 + 40 * (F16 @ w16) / np.sqrt(10)), 0, 255)
pr16 = PRB.linear_probe(F16, y16, rp.random(400) < 0.7)
pr_noise = PRB.linear_probe(F16, rp.permutation(y16), rp.random(400) < 0.7)
check("linear probe: reads a linear signal out (R2 > 0.9, the top bit right > 90%) and nothing out of shuffled labels",
      pr16["r2"] > 0.9 and pr16["bits"][7] > 0.9 and pr_noise["r2"] < 0.2 and pr16["n_train"] + pr16["n_test"] == 400)
torch.manual_seed(16)
check("effective rank: white noise in 8 dimensions uses nearly 8, a rank-one matrix exactly 1",
      PRB.effective_rank(torch.randn(4000, 8)) > 7.9 and abs(PRB.effective_rank(torch.arange(1, 21.)[:, None] @ torch.ones(1, 8)) - 1) < 1e-6)
probe_words = ("was were has have happy sad tired hungry scared excited sleepy proud angry glad each a one they the kids boys girls friends mom dad "
               "dog cat said lost wanted put down gave it to played with ran then had it was a nice day the sun was out mom was in the kitchen "
               "the birds sang it was time to play the wind blew a dog ran by the sky was blue and . " + " ".join(s for pl, s in TX.CONTAINERS + TX.GROUPS + TX.BOXES))
tp = TX.Vocab(tsent + [probe_words], n_words=1200, tool_tokens=True)
pairs16 = {p: PRB.minimal_pairs(tp, np.random.default_rng(5), p, 200) for p in PRB.PARADIGMS}
rec16 = PRB.recall_pairs(tp, np.random.default_rng(6), 200, gap=2)
check("minimal pairs: every paradigm gives at least 150 distinct in-vocabulary pairs that differ in exactly one token; recall pairs end at the name with the same prefix",
      all(len(v) >= 150 and len({tuple(g) for g, b in v}) >= 100 and all(len(g) == len(b) and sum(x != y for x, y in zip(g, b)) == 1 and TX.UNK not in g + b for g, b in v) for v in pairs16.values())
      and len(rec16) >= 150 and all(g[:-1] == b[:-1] and g[-1] != b[-1] and tp.itos[g[-1]] in TX.BOYS + TX.GIRLS and tp.itos[b[-1]] in TX.BOYS + TX.GIRLS for g, b in rec16))
class Const16:
    def __init__(self, v, fav=None): self.v, self.fav = v, fav
    def eval(self): pass
    def train(self): pass
    def __call__(self, inp):
        lg = torch.zeros(*inp.shape, 1, self.v)
        if self.fav is not None: lg[..., self.fav] = 5.0
        return {"logits": lg}
hand_pairs = [([TX.EOS, 300, 301], [TX.EOS, 300, 302]), ([TX.EOS, 303, 301, 304], [TX.EOS, 303, 302, 304])]
check("pair score: a model with no preference scores 0.5 (ties count half), one that favours the good token scores 1.0",
      PRB.pair_score(Const16(len(tp)), pairs16["was_were"])["score"] == 0.5 and PRB.pair_score(Const16(len(tp), fav=301), hand_pairs)["score"] == 1.0
      and PRB.pair_score(Const16(len(tp), fav=302), hand_pairs)["score"] == 0.0)
ind16 = PRB.induction_test(sm, tw, n=20, run=5)
check("induction test runs on the small model and reports both copies' losses and the copy accuracy",
      set(ind16) >= {"first_copy_loss", "second_copy_loss", "induction", "copy_accuracy"} and 0 <= ind16["copy_accuracy"] <= 1 and abs(ind16["induction"] - (ind16["first_copy_loss"] - ind16["second_copy_loss"])) < 1e-6)
br16 = PRB.loss_by_role(sm, torch.as_tensor(a16), torch.as_tensor(b16), torch.as_tensor(r16))
bf16 = PRB.loss_by_frequency(tw, tw) if False else PRB.loss_by_frequency(sm, torch.as_tensor(a16), torch.as_tensor(b16), tw)
check("loss by role and by frequency cover every counted target once",
      sum(v["n"] for v in br16.values()) == int((b16 != 0).sum()) and sum(v["n"] for v in bf16.values()) == int((b16 != 0).sum()) and br16["story"]["n"] > 0 and br16["machine"]["n"] > 0)

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
