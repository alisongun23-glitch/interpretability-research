"""Checks for transformer_gates.py. Run from the project root:  python tests/test_circuits.py
Every check prints PASS/FAIL. Nothing here changes results. Needs no trained model."""
import sys, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transformer_gates as TG
import run_transformer as RT

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")

r = np.random.default_rng(0)

# ---------- layouts ----------
a, b = r.integers(0, 2 ** 10, 200), r.integers(0, 2 ** 10, 200)
S = RT.bits(a + b, 11)
F = TG.Flat.tokens(a, b)
check("flat layout = run_transformer.encode lsb", np.array_equal(F, RT.encode(a, b, "lsb")))
check("flat: token after q_k is s_k, a_k / b_k at their positions",
      all(np.array_equal(F[:, TG.Flat.q(k) + 1], S[:, k]) for k in range(11))
      and all(np.array_equal(F[:, TG.Flat.pos_a(k)], RT.bits(a, 10)[:, k]) and np.array_equal(F[:, TG.Flat.pos_b(k)], RT.bits(b, 10)[:, k]) for k in range(10)))
check("flat: q_k holds s_(k-1) (and '=' at k = 0)",
      np.all(F[:, TG.Flat.q(0)] == RT.EQ) and all(np.array_equal(F[:, TG.Flat.q(k)], S[:, k - 1]) for k in range(1, 11)))
for n in (10, 16):
    a2, b2 = r.integers(0, 2 ** n, 200), r.integers(0, 2 ** n, 200)
    I = TG.Inter.tokens(a2, b2, n); S2 = RT.bits(a2 + b2, n + 1)
    good = I.shape[1] == 3 * (n + 1) and np.all(I[:, 3 * n] == 0) and np.all(I[:, 3 * n + 1] == 0)
    good &= all(np.array_equal(I[:, TG.Inter.q(k) + 1], S2[:, k]) for k in range(n + 1))
    q = TG.Inter.q(5)                                            # fixed offsets from the query of s_5
    good &= np.array_equal(I[:, q], RT.bits(b2, n + 1)[:, 5]) and np.array_equal(I[:, q - 1], RT.bits(a2, n + 1)[:, 5])
    good &= np.array_equal(I[:, q - 2], S2[:, 4]) and np.array_equal(I[:, q - 3], RT.bits(b2, n + 1)[:, 4]) and np.array_equal(I[:, q - 4], RT.bits(a2, n + 1)[:, 4])
    check(f"interleaved layout ({n} bits): triples, a_n = b_n = 0, inputs of s_k at offsets 0..4", good)

# ---------- the carry identity ----------
rows = [(x, y, c) for x in (0, 1) for y in (0, 1) for c in (0, 1)]
maj = [int(x + y + c >= 2) for x, y, c in rows]
thr = [int(x + y - (x ^ y ^ c) >= 1) for x, y, c in rows]
check("carry = MAJ(a, b, c) = [a + b - s >= 1] for all 8 (a, b, c)", maj == thr)
check("all 8 (a, b, s) combos are reachable", len({(x, y, x ^ y ^ c) for x, y, c in rows}) == 8)
check("CARRY_MASK = [a + b - s >= 1] over (a, b, s)", TG.CARRY_MASK.tolist() == [float(x + y - s >= 1) for x in (0, 1) for y in (0, 1) for s in (0, 1)])
L = TG.labels(a, b)
pred = np.stack([(L["a"][:, k - 1] + L["b"][:, k - 1] - L["s"][:, k - 1] >= 1).astype(int) for k in range(1, 11)], 1)
check("carry_into(k) = [a_(k-1) + b_(k-1) - s_(k-1) >= 1] on 200 random sums", np.array_equal(pred, L["c"][:, 1:]))

# ---------- gate adder ----------
A = r.integers(0, 2, (3000, 64)).astype(float); B_ = r.integers(0, 2, (3000, 64)).astype(float)
Sg, Cg = TG.gate_adder(A, B_)
ref = TG.exact_ripple(A, B_)
check("gate adder of library neurons is exact on 3,000 random 64-bit sums", np.array_equal(np.round(Sg), ref), f"worst signal {np.abs(Sg - ref).max():.3f}")
a3, b3 = r.integers(0, 2 ** 10, 500), r.integers(0, 2 ** 10, 500)
S3, C3 = TG.gate_adder(TG.bitsf(a3, 10), TG.bitsf(b3, 10))
check("gate adder carries = carry_into, sums = a + b (10 bits)",
      np.array_equal(np.round(C3).astype(int), TG.carry_bits(a3, b3, 10)) and np.array_equal(TG.bits_to_int(S3), a3 + b3))
good = True
for k in (1, 4, 10):
    c = RT.carry_into(a3, b3, k)
    So, _ = TG.gate_adder(TG.bitsf(a3, 10), TG.bitsf(b3, 10), carry_override=(k, 1 - c))
    good &= np.array_equal(TG.bits_to_int(So) - (a3 + b3), np.where(c == 1, -1, 1) * 2 ** k)
check("carry_override flips the carry into bit k: sum changes by exactly 2^k (k = 1, 4, 10)", good)

# ---------- C2 model ----------
w = TG.rel_gradient_check()
check("RelPosGPT gradients match finite differences", w < 1e-5, f"worst rel err {w:.1e}")
m = TG.RelPosGPT(4, 16, 2, 1, 32, seed=3)
m.P["0.rel"] = r.normal(0, 1, m.P["0.rel"].shape).astype(np.float32)
tok = r.integers(0, 2, (5, 30))
lg = m.forward(tok)
check("RelPosGPT is causal (changing a later token leaves earlier logits alone)",
      np.allclose(lg[:, :20], m.forward(np.concatenate([tok[:, :20], 1 - tok[:, 20:]], 1))[:, :20]))
check("RelPosGPT has no W_pos and a (heads, 64) bias per layer", "W_pos" not in m.P and m.P["0.rel"].shape == (2, 64))

# ---------- hooks: identity patch and MLP replacement ----------
for fam in ("B", "C2"):
    mdl = TG.make_model(fam, 5)
    for k_ in mdl.P:                                            # a non-trivial model
        mdl.P[k_] = (mdl.P[k_] + r.normal(0, 0.2, mdl.P[k_].shape)).astype(np.float32)
    lay = TG.LAYOUT[fam]
    a4, b4 = r.integers(0, 2 ** 10, 64), r.integers(0, 2 ** 10, 64)
    tk = lay.tokens(a4, b4); q = lay.q(6)
    base = mdl.forward(tk[:, :q + 1])
    x_src, _ = TG.mid_at(mdl, tk, q)
    U = np.linalg.qr(r.normal(size=(64, 4)))[0]; P = U @ U.T
    same = mdl.forward(tk[:, :q + 1], hooks=TG.patch_q(q, lambda xq: xq + (x_src - xq) @ P))
    check(f"{fam}: interchange with source = base leaves the logits unchanged", np.allclose(same, base, atol=1e-5))
    keep = {}
    def grab(x):
        keep["x"] = x.copy(); return x
    def write(x):
        x = x.copy(); x[:, q] = keep["x"][:, q] + (x[:, q] - keep["x"][:, q]); return x   # the real MLP output
    rep = mdl.forward(tk[:, :q + 1], hooks={TG.MID: grab, TG.POST: write})
    check(f"{fam}: MLP-replacement hook with the real MLP output leaves the logits unchanged", np.allclose(rep, base, atol=1e-5))
    pred_full = TG.generate(mdl, lay, a4, b4)
    pred_hooked = TG.generate(mdl, lay, a4, b4, hooks_fn=lambda k, rows: TG.patch_q(lay.q(k), lambda xq: xq))
    check(f"{fam}: generation with an identity hook = without", np.array_equal(pred_full, pred_hooked))

# ---------- probes, statistics ----------
d = r.normal(size=20); d /= np.linalg.norm(d)
X = r.normal(size=(6000, 20)); y = (X @ d + 0.1 * r.normal(size=6000) > 0).astype(int)
pr = TG.Probe(X[:4000], y[:4000])
check("probe recovers a known direction", abs(pr.direction() @ d) > 0.98 and pr.acc(X[4000:], y[4000:]) > 0.95,
      f"cos {abs(pr.direction() @ d):.3f}, acc {pr.acc(X[4000:], y[4000:]):.3f}")
lo, hi = TG.wilson(50, 100)
check("Wilson 95% interval for 50/100 = (0.4038, 0.5962)", abs(lo - 0.40383) < 1e-4 and abs(hi - 0.59617) < 1e-4, f"({lo:.5f}, {hi:.5f})")
idx = r.integers(0, 8, 4000)
act = ((idx >> 2) & (idx >> 1) & 1).astype(float)[:, None] * 2.5 + 0.3       # AND of the first two inputs
R2, masks = TG.bool_fit(act, idx)
f = int(R2[:, 0].argmax())
and_mask = TG.mask_of(lambda a_, b_, s_: a_ & b_)
check("best Boolean fit finds AND(a, b) (or its complement, same R^2) with R^2 = 1",
      abs(R2[f, 0] - 1) < 1e-9 and (np.array_equal(masks[f], and_mask) or np.array_equal(masks[f], 1 - and_mask)))
z = np.concatenate([r.normal(-2, 0.3, 500), r.normal(3, 0.3, 700)])
lo2, hi2, t2 = TG.two_means(z)
check("1-D 2-means finds the two levels and the midpoint", abs(lo2 + 2) < 0.1 and abs(hi2 - 3) < 0.1 and abs(t2 - 0.5) < 0.1, f"{lo2:.2f} {hi2:.2f} {t2:.2f}")
tab = r.random(8)
g = TG.trilinear(tab)
corners = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], float)
check("trilinear gate equals its table at the 8 corners", np.allclose(g(corners[:, 0], corners[:, 1], corners[:, 2]), tab))
okr, worst = TG.restores3(TG.trilinear(TG.CARRY_MASK), TG.CARRY_MASK, 0.05)
check("trilinear interpolation of the exact carry table does NOT restore at 0.05 (gain 2 at (0,1,1))", not okr and worst > 0.09, f"worst {worst:.3f}")
check("bimodality coefficient: two clusters > 0.555 > one Gaussian", TG.bimodality(z) > 0.555 > TG.bimodality(r.normal(size=2000)))
n_u, _ = TG.greedy_readout(np.hstack([y[:4000, None] * 1.0, r.normal(size=(4000, 10))]), y[:4000].astype(float),
                           np.hstack([y[4000:, None] * 1.0, r.normal(size=(2000, 10))]), y[4000:])
check("greedy readout needs 1 unit when one unit is the label", n_u == 1)
p = TG.train_neuron(corners, TG.CARRY_MASK, margin=4.0)
L_ = corners @ p["wx"] + p["bo"]
check("train_neuron reaches margin 4 on the 3-input carry table", np.all(L_ * (2 * TG.CARRY_MASK - 1) >= 4))

# ---------- pair builder ----------
a0, b0, a1, b1 = TG.opposite_pairs(3000, 6, np.random.default_rng(1))
check("opposite_pairs: same bits >= k, opposite carry into k",
      np.all(a0 >> 6 == a1 >> 6) and np.all(b0 >> 6 == b1 >> 6) and np.all(RT.carry_into(a0, b0, 6) != RT.carry_into(a1, b1, 6)))

# ---------- clean_carry.py: labeller, readers, MLP hooks, arm R (before any real model) ----------
import clean_carry as CC
r = np.random.default_rng(5)
N = 4000
bits3 = r.integers(0, 2, (N, CC.NB + 1, 3))
combo = 4 * bits3[..., 0] + 2 * bits3[..., 1] + bits3[..., 2]
count = bits3.sum(2)
Hp = np.zeros((N, CC.NB + 1, 4), np.float32)
Hp[:, 2, 0] = 1.5 * (count[:, 2] >= 1) + 0.01 * r.normal(size=N)              # unit 0: OR3 at k = 2 ...
Hp[:, 7, 0] = 2.0 * (count[:, 7] >= 3) + 0.01 * r.normal(size=N)              # ... and AND3 at k = 7
for k in CC.KS:
    Hp[:, k, 1] = 1.2 * (count[:, k] >= 2) + 0.01 * r.normal(size=N)          # unit 1: majority at every k
    Hp[:, k, 2] = count[:, k]                                                   # unit 2: a pure count ramp
Hp = np.maximum(Hp, 0)
codes = CC.label_all(Hp, combo, count, "A")
v = CC.verdicts(codes, "A")
check("planted OR3 at k=2 and AND3 at k=7 -> packed", v[0] == "packed", str(codes[0]))
check("planted majority at every k -> one job, labelled as the count threshold t = 2",
      v[1] == "one" and all(CC.is_thr(c) and CC.thr_t(c) == 2 and CC.thr_sign(c) == 1 for c in codes[1]), str(codes[1]))
check("planted pure count ramp -> a threshold (count-ramp) label, one job",
      v[2] == "one" and all(CC.is_thr(c) for c in codes[2]), str(codes[2]))
check("silent unit -> dead", v[3] == "dead")
shuf = r.choice([1001 + 0, 1010, 1020, 1030, 23, 150], size=(300, 9)).astype(int)
d_, _, _ = CC.delta(shuf, 200, np.random.default_rng(0))
check("delta ~ 0 when labels are shuffled at random", abs(d_) < 0.03, f"{d_:.4f}")
same = np.repeat(r.choice([1010, 1020, 1030], size=(300, 1)), 9, axis=1)
d2_, _, _ = CC.delta(same, 200, np.random.default_rng(0))
check("delta > 0.3 when every unit keeps one label across positions", d2_ > 0.3, f"{d2_:.3f}")
m = CC.make(8, 0)
vk = r.normal(size=64); vk -= vk.mean(); vk /= np.linalg.norm(vk)
m.P["0.W_in"][:, 0] = (vk / m.P["0.ln2.w"]).astype(np.float32)
Rc = CC.read_cos(m, {k: vk for k in CC.KS})
check("planted reader: |read cosine| = 1 for the unit whose W_in is v_k, small for random units",
      np.allclose(np.abs(Rc[0]), 1, atol=1e-5) and np.abs(Rc[1:]).max() < 0.6, f"{np.abs(Rc[1:]).max():.2f}")
tok = TG.Flat.tokens(*TG.problems(64, CC.NB, r))
q = TG.Flat.q(4)
lg0 = m.forward(tok[:, :q + 1])
lg1 = m.forward(tok[:, :q + 1], hooks=CC.mlp_replace(m, q, lambda h: h))
check("mlp_replace with the identity leaves the logits unchanged", np.allclose(lg0, lg1, atol=1e-4))
h0, _ = CC.mlp_at(m, tok, q)
lg2 = m.forward(tok[:, :q + 1], hooks=CC.mlp_replace(m, q, lambda h: h0.copy()))
check("holding every unit at its own value leaves the logits unchanged", np.allclose(lg0, lg2, atol=1e-4))
lg3 = m.forward(tok[:, :q + 1], hooks=CC.mlp_replace(m, q, lambda h: np.zeros_like(h)))
check("mlp_replace actually acts (zeroed units change the logits)", not np.allclose(lg0[:, q], lg3[:, q], atol=1e-6))
check("arm R tables: 11 distinct balanced functions of all three inputs",
      len({tuple(t) for t in CC.TAB}) == CC.NB + 1 and all(t.sum() == 4 for t in CC.TAB)
      and all(CC._depends_on_all(CC.fval(t)) for t in CC.TAB))
a_, b_ = TG.problems(500, CC.NB, r)
t_ = CC.FlatTable.targets(a_, b_); prev = np.zeros(500, int); okr = True
for k in range(CC.NB + 1):
    want = CC.TAB[k][4 * ((a_ >> k) & 1) + 2 * ((b_ >> k) & 1) + prev]
    okr &= np.array_equal(t_[:, k], want); prev = want
tk = CC.FlatTable.tokens(a_, b_)
check("FlatTable: targets follow t_k = T_k(a_k, b_k, t_(k-1)) and sit after '='",
      okr and np.array_equal(tk[:, RT.POS_EQ + 1:], t_) and np.array_equal(tk[:, :RT.POS_EQ + 1], RT.encode(a_, b_, "lsb")[:, :RT.POS_EQ + 1]))
check("width-256 arm-A model starts from the same initialisation as TG.make_model('B')",
      all(np.array_equal(CC.make(256, 3).P[k], TG.make_model("B", 3).P[k]) for k in TG.make_model("B", 3).P))

# ---------- clean_carry_long.py: gate labels, schedule, learning rate ----------
import clean_carry_long as CL
Hg = np.zeros((N, CC.NB + 1, 6), np.float32)
for k in CC.KS:
    Hg[:, k, 0] = 1.5 * (count[:, k] >= 1)                                     # silent-off OR3: a gate
    Hg[:, k, 1] = 0.5 + 1.5 * (count[:, k] >= 1)                               # same step, but 0.5 when off: not a gate
    Hg[:, k, 2] = 1.2 * (count[:, k] >= 2)                                     # silent-off majority
    Hg[:, k, 3] = 0.9 * (count[:, k] >= 3)                                     # silent-off AND3
    Hg[:, k, 4] = 1.1 * (count[:, k] < 2)                                      # silent where count >= 2: a below-threshold gate
    Hg[:, k, 5] = 0.7 * (count[:, k] >= 2)
g = CL.gate_label_all(Hg, combo)
check("gate label: a silent-off OR3 is a gate (threshold t = 1) at every k", all(c == 1010 for c in g[0]), str(g[0]))
check("gate label: the same step with an off level of 0.5 is NOT a gate", all(c == 0 for c in g[1]), str(g[1]))
check("gate label: majority, AND3 and a below-threshold unit are gates with the right codes",
      all(c == 1020 for c in g[2]) and all(c == 1030 for c in g[3]) and all(c == 1021 for c in g[4]))
Q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(6, 6)))
gr = CL.gate_label_all(np.einsum("nku,uv->nkv", Hg.astype(float), Q), combo)
check("gate label: a random rotation of silent gates leaves (almost) no gates", (gr > 0).mean() < 0.2 * (g > 0).mean(),
      f"{(gr > 0).mean():.2f} vs {(g > 0).mean():.2f}")
check("lr_at(step, 2500) is the first study's learning-rate schedule",
      all(abs(CL.lr_at(st, 2500) - 1e-3 * min(1, (st + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * st / 2500)))) < 1e-15
          for st in (0, 50, 99, 100, 1000, 2499)))
sl = CL.schedule_long(20000)
check("schedule_long: 0, the 25-step grid to 1000, the 250-step grid to 5000, the 1000-step grid to S, and S",
      sl == sorted(set(sl)) and {0, 25, 1000, 1250, 5000, 6000, 19000, 20000} <= set(sl) and 1025 not in sl and 5250 not in sl
      and max(sl) == 20000)
check("arm L starts from clean_carry.make (same initialisation as arm A)",
      all(np.array_equal(CL.CC.make(16, 2).P[k], CC.make(16, 2).P[k]) for k in CC.make(16, 2).P))
rng_u = np.random.default_rng(3); au, bu = TG.problems(4000, TG.NB, rng_u); Lu = TG.labels(au, bu)
cu = Lu["a"][:, 5] + Lu["b"][:, 5] + Lu["c"][:, 5]
xu = np.stack([1.0 * (cu >= 1), 1.0 * (cu >= 2) + 0.8 * Lu["s"][:, 4]], 1)      # OR3; majority plus the token s_4
su, _ = CL._bit_sets(Lu, 5)
r_loc = CL._cv_table_r2(xu, *CL._groups(su["local"]), np.random.default_rng(0))
r_tok = CL._cv_table_r2(xu, *CL._groups(su["token"]), np.random.default_rng(0))
check("unit functions: a function of a_k, b_k, c_k has table R2 ~ 1 on those inputs; one that also uses the token s_(k-1) only with it",
      r_loc[0] > 0.98 and r_loc[1] < 0.9 and r_tok[1] > 0.98, f"local {r_loc.round(2)} token {r_tok.round(2)}")
mt = CC.make(16, 0); tk = TG.Flat.tokens(au[:200], bu[:200]); q5 = TG.Flat.q(5)
_, acts0, _ = mt.forward(tk, cache=True)
hk, kp = CL.token_patch_hooks(mt, [q5], "flip", "all"); _, acts1, _ = mt.forward(tk, hooks=hk, cache=True)
dE = mt.P["W_E"][1 - tk[:, q5]].astype(float) - mt.P["W_E"][tk[:, q5]].astype(float)
d = acts1[TG.MID].astype(float) - acts0[TG.MID].astype(float)
check("token path: the hooks read the token at q off resid_pre and flip exactly its embedding there, nowhere else",
      np.array_equal(kp["tok"][q5], tk[:, q5]) and np.allclose(d[:, q5], dE, atol=1e-5) and np.allclose(np.delete(d, q5, 1), 0))
hk, kp = CL.token_patch_hooks(mt, [q5], "remove", "mlp"); _, acts2, _ = mt.forward(tk, hooks=hk, cache=True)
dP = acts2[TG.POST].astype(float) - acts0[TG.POST].astype(float)
check("token path, scope mlp: resid_mid is untouched, resid_post changes at q only",
      np.allclose(acts2[TG.MID], acts0[TG.MID]) and np.abs(dP[:, q5]).max() > 1e-6 and np.allclose(np.delete(dP, q5, 1), 0))
hk, kp = CL.token_pre_hooks(mt, q5, "flip"); _, acts3, _ = mt.forward(tk, hooks=hk, cache=True)
d3 = acts3[TG.HOOKS["resid_pre"]].astype(float) - acts0[TG.HOOKS["resid_pre"]].astype(float)
dM = acts3[TG.MID].astype(float) - acts0[TG.MID].astype(float)
check("token path at resid_pre: the flip lands before attention at q only, and later positions' resid_mid changes too",
      np.array_equal(kp["tok"], tk[:, q5]) and np.allclose(d3[:, q5], dE, atol=1e-5) and np.allclose(np.delete(d3, q5, 1), 0)
      and np.abs(dM[:, q5 + 1:]).max() > 1e-6 and np.allclose(dM[:, :q5], 0))

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
