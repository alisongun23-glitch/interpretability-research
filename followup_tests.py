"""Follow-up tests for "The Carry Is Part of a Count, Not a Gate" (EXPLORATORY: run after the paper, not pre-registered).

    python followup_tests.py              # all 8 models, ~10 min
    python followup_tests.py B:1 C2:0     # some models
    -> results_followup/<fam>-<seed>.json, then  python followup_tests.py summary  -> results_followup/SUMMARY.md

Reuses transformer_gates.py (same models, same PC1-4 flip directions, same probe directions, same data seed).
Changes nothing in transformer_gates.py or results_transformer_gates/.

T1  What PC1 encodes. OLS of the PC1 coordinate at q_k on c_k, a_k, b_k, a_(k-1), b_(k-1), s_(k-1), with a
    per-k intercept. If PC1 were "the carry", only c_k would matter. Also the cosine of PC1 with the query
    token's embedding difference W_E[1] - W_E[0].
T2  Context-conditioned transplant. Set the PC1 (or PC1-4) coordinates at q_k to the training mean for
    (k, a_k, b_k, carry), with carry = the exact gate carry or its flip. The paper's version used means
    for (k, carry) only. Teacher-forced at every k, plus full generation with the gate carry written at every q_k.
T3  Steering instead of swapping. Add m x (class gap along v) along v = probe direction, PC1, or a random
    direction (same shift size as PC1) to carry-0 rows; share whose output flips.
T4  Carry or answer? Interchange PC1 from sources that ALSO differ in a_k, b_k. A carry variable predicts
    a_k(base) xor b_k(base) xor c(source); an answer direction predicts the source's own s_k.
T5  Token confound (B). Interchange PC1 on pairs that also share s_(k-1), the token at q_k.
T6  Gate margin (the paper's Model -> gate table) for PC1, for PC1 with each (a_k, b_k) group's mean removed
    (inputs only, no carry labels), and for the top component of PC1-4 after the same removal.
T7  MLP units at q_k (k = 1..9, as P4b): eta^2 of (a_k, b_k, c_k) and of the count a_k + b_k + c_k,
    pooled and within k; best Boolean fit; share of count thresholds.
T8  Later bits. Teacher-force s_k flipped (what a PC1 patch emits) and let the model continue: does the next
    bit follow the local rule [a_k + b_k - s_k' >= 1] or the original carry? Attention from q_(k+1).
T9  What MLP units depend on (k = 1..9): median eta^2 of the current digits (a_k, b_k, c_k), of those plus
    the bit position k, and of those plus k and the previous digits (a_(k-1), b_(k-1)).
"""
import json, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import transformer_gates as TG
import run_transformer as RT

NB, MID = TG.NB, TG.MID
OUT = "results_followup"
VARS = ["c_k", "a_k", "b_k", "a_(k-1)", "b_(k-1)", "s_(k-1)"]


def mid_q(model, lay, a, b, bs=2000):
    X, O = [], []
    qs = [lay.q(k) for k in range(NB + 1)]
    for i in range(0, len(a), bs):
        lg, acts, _ = model.forward(lay.tokens(a[i:i + bs], b[i:i + bs]), cache=True)
        X.append(acts[MID][:, qs].astype(float)); O.append(lg[:, qs, :2].argmax(2))
    return np.concatenate(X), np.concatenate(O)


def feats(L, k):
    return np.stack([L["c"][:, k], L["a"][:, k], L["b"][:, k], L["a"][:, k - 1], L["b"][:, k - 1], L["s"][:, k - 1]], 1).astype(float)


def t1_composition(model, U, X, L, fam):
    ks = range(1, NB + 1)
    z = np.concatenate([X[:, k] @ U[k][:, 0] for k in ks])
    F = np.concatenate([feats(L, k) for k in ks])
    D = np.concatenate([np.eye(NB)[np.full(len(X), k - 1)] for k in ks])           # per-k intercepts
    def fit(M):
        w, *_ = np.linalg.lstsq(M, z, rcond=None); r = z - M @ w
        return w, 1 - r.var() / z.var()
    w_full, r2_full = fit(np.hstack([F, D]))
    _, r2_c = fit(np.hstack([F[:, :1], D]))
    _, r2_k = fit(D)
    coef = dict(zip(VARS, w_full[:6].tolist()))
    WE = model.P["W_E"].astype(float); tokdir = WE[1] - WE[0]; tokdir /= np.linalg.norm(tokdir)
    cos = [float(abs(U[k][:, 0] @ tokdir)) for k in ks]
    return {"coef": coef, "coef_over_c": {v: coef[v] / coef["c_k"] for v in VARS},
            "R2_full": float(r2_full), "R2_c_only": float(r2_c), "R2_k_only": float(r2_k),
            "query_token": "s_(k-1)" if fam == "B" else "b_k",
            "abs_cos_pc1_token_embedding": {"mean": float(np.mean(cos)), "max": float(np.max(cos))}}


def ctx_means(Xtr, Ltr):
    """M[k][a_k, b_k, c] = mean resid_mid at q_k; Mc[k][c] = mean for the carry alone (the paper's)."""
    M, Mc = {}, {}
    for k in range(1, NB + 1):
        a, b, c = Ltr["a"][:, k], Ltr["b"][:, k], Ltr["c"][:, k]
        M[k] = np.zeros((2, 2, 2, Xtr.shape[2]))
        for ak in (0, 1):
            for bk in (0, 1):
                for cc in (0, 1):
                    sel = (a == ak) & (b == bk) & (c == cc)          # empty only for k = NB, where a_k = b_k = 0 always
                    M[k][ak, bk, cc] = Xtr[sel, k].mean(0) if sel.any() else np.nan
        Mc[k] = np.stack([Xtr[c == 0, k].mean(0), Xtr[c == 1, k].mean(0)])
    return M, Mc


def t2_transplant(model, lay, U, M, Mc, a, b, L):
    tok = lay.tokens(a, b); res = {}
    for basis in ("pc1", "pc1-4"):
        for kind in ("context_gate", "context_flipped", "carry_only_gate"):
            ok = []
            for k in range(1, NB + 1):
                q = lay.q(k); V = U[k][:, :1] if basis == "pc1" else U[k]; P = V @ V.T
                c = L["c"][:, k]; cw = 1 - c if kind == "context_flipped" else c
                tgt = M[k][L["a"][:, k], L["b"][:, k], cw] if kind.startswith("context") else Mc[k][cw]
                out = model.forward(tok[:, :q + 1], hooks=TG.patch_q(q, lambda xq: xq + (tgt - xq) @ P))[:, q, :2].argmax(1)
                want = tok[:, q + 1] if kind != "context_flipped" else 1 - tok[:, q + 1]
                ok.append(out == want)
            res[f"{basis}/{kind}"] = TG.acc_ci(np.concatenate(ok))
    # full generation: gate carry written (context means) at every q_k, k >= 1
    for basis in ("pc1", "pc1-4"):
        def hooks_fn(k, rows, basis=basis):
            if k == 0:
                return None
            V = U[k][:, :1] if basis == "pc1" else U[k]; P = V @ V.T
            cg = RT.carry_into(a[rows], b[rows], k)
            tgt = M[k][L["a"][rows, k], L["b"][rows, k], cg]
            return TG.patch_q(lay.q(k), lambda xq: xq + (tgt - xq) @ P)
        res[f"generation/{basis}/context_gate"] = TG.acc_ci(TG.generate(model, lay, a, b, NB, hooks_fn=hooks_fn) == a + b)
    return res


def t3_steering(model, lay, U, pdir, Mc, a, b, L, mults=(1, 2, 4, 8)):
    tok = lay.tokens(a, b); rr = np.random.default_rng(3); res = {}
    d = U[1].shape[0]
    bases = {k: model.forward(tok[:, :lay.q(k) + 1])[:, lay.q(k), :2].argmax(1) for k in range(1, NB + 1)}
    for name in ("probe", "pc1", "random"):
        for m in mults:
            flips = []
            for k in range(1, NB + 1):
                q = lay.q(k); c0 = L["c"][:, k] == 0
                u = U[k][:, 0]; gap_u = (Mc[k][1] - Mc[k][0]) @ u
                if name == "probe":
                    v = pdir[k]; shift = (Mc[k][1] - Mc[k][0]) @ v
                elif name == "pc1":
                    v, shift = u, gap_u
                else:
                    v = rr.normal(size=d); v /= np.linalg.norm(v); shift = abs(gap_u)
                base = bases[k]
                out = model.forward(tok[:, :q + 1], hooks=TG.patch_q(q, lambda xq: xq + m * shift * v[None]))[:, q, :2].argmax(1)
                valid = c0 & (base == tok[:, q + 1])
                flips.append((out != base)[valid])
            res[f"{name}/x{m}"] = TG.acc_ci(np.concatenate(flips))
    res["gap_along_probe_over_gap_along_pc1"] = float(np.mean([abs((Mc[k][1] - Mc[k][0]) @ pdir[k]) /
                                                                abs((Mc[k][1] - Mc[k][0]) @ U[k][:, 0]) for k in range(1, NB + 1)]))
    return res


def pairs(n, k, rng, mode):
    """Base/source with opposite carry into k. mode 'same': bits >= k shared (the paper's pairs).
    'cross': bits > k shared, bit k random in the source. 'token': as 'same', plus the same s_(k-1)."""
    a0, b0 = TG.problems(n, NB, rng); lo = 2 ** k
    c0 = RT.carry_into(a0, b0, k); s0 = (((a0 % lo) + (b0 % lo)) >> (k - 1)) & 1
    hi_a, hi_b = a0 - a0 % lo, b0 - b0 % lo
    if mode == "cross":
        hi_a = a0 - a0 % (2 * lo) + rng.integers(0, 2, n) * lo; hi_b = b0 - b0 % (2 * lo) + rng.integers(0, 2, n) * lo
    x = np.zeros(n, np.int64); y = np.zeros(n, np.int64); todo = np.arange(n); tries = 0
    while len(todo) and tries < 500:                 # 'token' at k = 1 with s_0 = 1 has no partner: dropped below
        xx, yy = rng.integers(0, lo, len(todo)), rng.integers(0, lo, len(todo))
        ok = ((xx + yy) >= lo).astype(int) != c0[todo]
        if mode == "token":
            ok &= (((xx + yy) >> (k - 1)) & 1) == s0[todo]
        x[todo[ok]] = xx[ok]; y[todo[ok]] = yy[ok]; todo = todo[~ok]; tries += 1
    keep = np.ones(n, bool); keep[todo] = False
    return a0[keep], b0[keep], (hi_a + x)[keep], (hi_b + y)[keep]


def interchange(model, lay, U, k, a0, b0, a1, b1):
    q = lay.q(k); tok0 = lay.tokens(a0, b0)
    x1, _ = TG.mid_at(model, lay.tokens(a1, b1), q); _, base = TG.mid_at(model, tok0, q)
    P = U[k][:, :1] @ U[k][:, :1].T
    out = model.forward(tok0[:, :q + 1], hooks=TG.patch_q(q, lambda xq: xq + (x1 - xq) @ P))[:, q, :2].argmax(1)
    return out, base == tok0[:, q + 1], tok0[:, q + 1]


def t4_t5(model, lay, U, fam, seed, n=1000):
    rng = np.random.default_rng(800 + seed); res = {}
    for mode in (["same", "cross", "token"] if fam == "B" else ["same", "cross"]):
        carry, answer, neither, succ = [], [], [], []
        for k in range(1, NB + 1):
            a0, b0, a1, b1 = pairs(n, k, rng, mode)
            out, valid, s0 = interchange(model, lay, U, k, a0, b0, a1, b1)
            p_carry = 1 - s0                                         # a_k(base) xor b_k(base) xor flipped carry
            p_answer = ((a1 + b1) >> k) & 1                          # the source's own s_k
            if mode == "cross":
                disc = valid & (p_carry != p_answer)
                carry.append((out == p_carry)[disc]); answer.append((out == p_answer)[disc])
            else:
                succ.append((out == p_carry)[valid])
        if mode == "cross":
            res["cross/follows_carry"] = TG.acc_ci(np.concatenate(carry))
            res["cross/follows_source_answer"] = TG.acc_ci(np.concatenate(answer))
        else:
            res[f"{mode}/pc1_success"] = TG.acc_ci(np.concatenate(succ))
    return res


def t6_margins(U, Xte, Lte, fam):
    """Gate margin (the paper's 6b table on (a_(k-1), b_(k-1), s_(k-1))) for PC1 as is, for PC1 with the
    current digits' contribution removed (per-k mean for each (a_k, b_k) subtracted; uses inputs only, no
    carry labels), and for the top PC of PC1-4 after the same removal."""
    ks = range(1, NB + 1)
    combos = {k: 4 * Lte["a"][:, k - 1] + 2 * Lte["b"][:, k - 1] + Lte["s"][:, k - 1] for k in ks}
    g = {k: 2 * Lte["a"][:, k] + Lte["b"][:, k] for k in ks}
    def resid(Z, k):
        Z = np.array(Z, float); out = Z.copy()
        for v in range(4):
            m = g[k] == v
            if m.any():
                out[m] -= Z[m].mean(0)
        return out + Z.mean(0)
    raw = {k: Xte[:, k] @ U[k][:, 0] for k in ks}
    res1 = {k: resid(raw[k], k) for k in ks}
    Z4 = {k: resid(Xte[:, k] @ U[k], k) for k in ks}
    allz = np.concatenate(list(Z4.values())); _, _, Vt = np.linalg.svd(allz - allz.mean(0), full_matrices=False)
    res4 = {k: Z4[k] @ Vt[0] for k in ks}
    out = {}
    for name, z in (("pc1", raw), ("pc1_minus_current_digits", res1), ("pc1-4_minus_current_digits", res4)):
        t = TG.gate_table(z, combos, fam)
        out[name] = {"gate_margin": t["gate_margin"], "table": t["table"], "snapped_is_carry": bool(t["snapped"] == TG.CARRY_MASK.astype(int).tolist())}
    return out


def t7_units(model, lay, a, b, L):
    """MLP units at q_k: how much of each unit's activity a function of (a_k, b_k, c_k) can explain
    (eta^2 over the 8 combos), how much the count n = a_k + b_k + c_k alone explains, pooled over k and
    within each k, and the best single Boolean function (as P4b)."""
    ks = range(1, NB)                                                # as the paper's P4b: the top bit has a_k = b_k = 0
    qs = [lay.q(k) for k in ks]; H = []
    for i in range(0, len(a), 2000):
        _, acts, _ = model.forward(lay.tokens(a[i:i + 2000], b[i:i + 2000]), cache=True)
        H.append(acts["blocks.0.mlp.hook_post"][:, qs].astype(float))
    H = np.concatenate(H)                                             # (N, 10, units)
    combo = np.stack([4 * L["a"][:, k] + 2 * L["b"][:, k] + L["c"][:, k] for k in ks], 1)
    count = np.stack([L["a"][:, k] + L["b"][:, k] + L["c"][:, k] for k in ks], 1)
    def eta2(X, grp, m):
        sst = ((X - X.mean(0)) ** 2).sum(0); ssb = 0
        for v in range(m):
            sel = grp == v
            if sel.any():
                ssb = ssb + sel.sum() * (X[sel].mean(0) - X.mean(0)) ** 2
        return ssb / np.maximum(sst, 1e-12)
    Xp = H.reshape(-1, H.shape[2]); cp = combo.reshape(-1); np_ = count.reshape(-1)
    active = ((Xp > 0).mean(0) >= 0.01) & (Xp.std(0) > 1e-6)
    Xa = Xp[:, active]
    e_combo, e_count = eta2(Xa, cp, 8), eta2(Xa, np_, 4)
    e_combo_k = np.mean([eta2(H[:, j][:, active], combo[:, j], 8) for j in range(len(ks))], 0)
    R2, masks = TG.bool_fit(Xa, cp, 3)
    best = R2.max(0); f = R2.argmax(0)
    thr = {tuple(TG.mask_of(lambda x, y, z, t=t: int(x + y + z >= t))) for t in (1, 2, 3)}
    thr |= {tuple(1 - np.array(m)) for m in thr}
    is_thr = np.array([tuple(masks[i]) in thr for i in f])
    clean = e_combo >= 0.8
    return {"n_active": int(active.sum()),
            "units_eta2_combo_ge_0.8_pooled": int(clean.sum()),
            "units_eta2_combo_ge_0.8_within_k_only": int(((e_combo_k >= 0.8) & ~clean).sum()),
            "share_of_those_symmetric (count explains >= 95%)": float(np.mean(e_count[clean] >= 0.95 * e_combo[clean])) if clean.any() else None,
            "units_best_boolean_R2_ge_0.8": int((best >= 0.8).sum()),
            "share_of_those_count_thresholds": float(np.mean(is_thr[best >= 0.8])) if (best >= 0.8).any() else None,
            "median_eta2_combo": float(np.median(e_combo)), "median_eta2_count": float(np.median(e_count)),
            "median_eta2_combo_within_k": float(np.median(e_combo_k))}


def t8_continuation(model, lay, a, b, L):
    """Teacher-force the sum with s_k flipped (what a PC1 patch emits), then let the model continue.
    In a 1-layer model only that token carries the flip forward. Does the next bit follow the local rule
    [a_k + b_k - s_k' >= 1] (the flipped carry), or the original carry? And where does q_(k+1) attend?"""
    tok0 = lay.tokens(a, b); res = {"next_follows_local_rule": [], "next_follows_original_carry": [], "rest_matches_flipped_adder": []}
    att = {"a_k": [], "b_k": [], "a_(k-1)": [], "b_(k-1)": [], "s_(k-1)": []}
    for k in range(1, NB):
        tok = tok0.copy(); ps = lay.pos_s(k); tok[:, ps] = 1 - tok[:, ps]
        cflip = 1 - L["c"][:, k]
        c_next_flip = ((L["a"][:, k] + L["b"][:, k] + cflip) >= 2).astype(int)
        s_next_local = L["a"][:, k + 1] ^ L["b"][:, k + 1] ^ c_next_flip
        s_next_orig = L["s"][:, k + 1]
        q = lay.q(k + 1)
        lg, acts, _ = model.forward(tok[:, :q + 1], cache=True)
        nxt = lg[:, q, :2].argmax(1); diff = s_next_local != s_next_orig
        res["next_follows_local_rule"].append((nxt == s_next_local)[diff]); res["next_follows_original_carry"].append((nxt == s_next_orig)[diff])
        A = acts["blocks.0.attn.hook_pattern"][:, :, q, :].mean(0)          # (heads, positions)
        for name, pos in (("a_k", lay.pos_a(k)), ("b_k", lay.pos_b(k)), ("a_(k-1)", lay.pos_a(k - 1)), ("b_(k-1)", lay.pos_b(k - 1)), ("s_(k-1)", lay.pos_s(k - 1))):
            att[name].append(float(A[:, pos].max()))
        # full continuation from the flipped prefix vs the adder with the carry flipped at k
        c = c_next_flip.copy(); want = []
        for j in range(k + 1, NB + 1):
            want.append(L["a"][:, j] ^ L["b"][:, j] ^ c); c = ((L["a"][:, j] + L["b"][:, j] + c) >= 2).astype(int)
        for j in range(k + 1, NB + 1):
            qj = lay.q(j); tok[:, qj + 1] = model.forward(tok[:, :qj + 1])[:, qj, :2].argmax(1)
        got = np.stack([tok[:, lay.q(j) + 1] for j in range(k + 1, NB + 1)], 1)
        res["rest_matches_flipped_adder"].append(np.all(got == np.stack(want, 1), 1))
    out = {kk: TG.acc_ci(np.concatenate(v)) for kk, v in res.items()}
    out["max_head_attention_from_q_(k+1)"] = {kk: float(np.mean(v)) for kk, v in att.items()}
    return out


def t9_unit_dependence(model, lay, a, b, L):
    """Median eta^2 over active MLP units at q_k (k = 1..9) for groupings of increasing size."""
    ks = list(range(1, NB)); qs = [lay.q(k) for k in ks]; H = []
    for i in range(0, len(a), 2000):
        _, acts, _ = model.forward(lay.tokens(a[i:i + 2000], b[i:i + 2000]), cache=True)
        H.append(acts["blocks.0.mlp.hook_post"][:, qs].astype(float))
    X = np.concatenate(H).transpose(1, 0, 2).reshape(-1, H[0].shape[2])           # rows: k-major
    kk = np.repeat(np.array(ks), len(a))
    cur = np.concatenate([4 * L["a"][:, k] + 2 * L["b"][:, k] + L["c"][:, k] for k in ks])
    prev = np.concatenate([2 * L["a"][:, k - 1] + L["b"][:, k - 1] for k in ks])
    active = ((X > 0).mean(0) >= 0.01) & (X.std(0) > 1e-6)
    X = X[:, active]; sst = ((X - X.mean(0)) ** 2).sum(0)
    def eta2(grp):
        ssb = 0
        for v in np.unique(grp):
            sel = grp == v
            ssb = ssb + sel.sum() * (X[sel].mean(0) - X.mean(0)) ** 2
        return ssb / np.maximum(sst, 1e-12)
    out = {"n_active": int(active.sum())}
    for name, grp in (("current_digits", cur), ("current_digits+k", kk * 8 + cur), ("current+previous_digits+k", (kk * 8 + cur) * 4 + prev)):
        e = eta2(grp)
        out[name] = {"median_eta2": float(np.median(e)), "units_ge_0.8": int((e >= 0.8).sum()), "n_groups": int(len(np.unique(grp)))}
    return out


def run(fam, seed):
    t0 = time.time(); lay = TG.LAYOUT[fam]
    model, _ = TG.load(fam, seed)
    rng0 = np.random.default_rng(7)                                  # the paper's data split
    a_tr, b_tr = TG.problems(8000, NB, rng0); a_te, b_te = TG.problems(4000, NB, rng0)
    Ltr, Lte = TG.labels(a_tr, b_tr), TG.labels(a_te, b_te)
    Xtr, _ = mid_q(model, lay, a_tr, b_tr); Xte, _ = mid_q(model, lay, a_te, b_te)
    U = TG.flip_dirs(model, lay, fam, seed)                          # same directions as the paper
    pdir = TG.probe_dirs_c(fam, {"q": {"resid_mid": Xtr}}, Ltr)
    M, Mc = ctx_means(Xtr, Ltr)
    path = os.path.join(OUT, f"{fam}-{seed}.json")
    R = json.load(open(path)) if os.path.exists(path) else {"model": f"{fam}:{seed}"}
    steps = [("T1_pc1_composition", lambda: t1_composition(model, U, Xte, Lte, fam)),
             ("T2_context_transplant", lambda: t2_transplant(model, lay, U, M, Mc, a_te, b_te, Lte)),
             ("T3_steering", lambda: t3_steering(model, lay, U, pdir, Mc, a_te[:2000], b_te[:2000], {v: L[:2000] for v, L in Lte.items()})),
             ("T4_T5_interchange", lambda: t4_t5(model, lay, U, fam, seed)),
             ("T6_margins", lambda: t6_margins(U, Xte, Lte, fam)),
             ("T7_mlp_units", lambda: t7_units(model, lay, a_te, b_te, Lte)),
             ("T8_continuation", lambda: t8_continuation(model, lay, a_te, b_te, Lte)),
             ("T9_unit_dependence", lambda: t9_unit_dependence(model, lay, a_te, b_te, Lte))]
    for key, fn in steps:                            # resumable: each test is saved as soon as it is done
        if key not in R:
            R[key] = fn(); R["seconds_" + key] = round(time.time() - t0, 1)
            json.dump(R, open(path, "w"), indent=1, default=float)
        print(fam, seed, key, f"{time.time() - t0:.0f}s", flush=True)


def summary():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(OUT, f))))
    pct = lambda d: f"{100 * d['acc']:.1f}"
    L = ["# Follow-up tests (exploratory)", "", "Generated by `followup_tests.py summary`. Percentages over all bits k = 1..10 unless a section says otherwise.", ""]
    L += ["## T1: what PC1 encodes (OLS coefficients relative to c_k, test split)", "",
          "| model | " + " | ".join(VARS[1:]) + " | R2 full | R2 c only | R2 k only | abs cos(PC1, token) |", "|" + "---|" * 10]
    for r in rows:
        t = r["T1_pc1_composition"]
        L.append(f"| {r['model']} | " + " | ".join(f"{t['coef_over_c'][v]:+.2f}" for v in VARS[1:]) +
                 f" | {t['R2_full']:.2f} | {t['R2_c_only']:.2f} | {t['R2_k_only']:.2f} | {t['abs_cos_pc1_token_embedding']['mean']:.2f} |")
    keys = ["pc1/carry_only_gate", "pc1/context_gate", "pc1/context_flipped", "pc1-4/carry_only_gate", "pc1-4/context_gate",
            "pc1-4/context_flipped", "generation/pc1/context_gate", "generation/pc1-4/context_gate"]
    L += ["", "## T2: transplant with means for (k, carry) vs (k, a_k, b_k, carry)", "",
          "Teacher-forced: % of bits whose output equals the intended bit (gate carry: the true s_k; flipped: its flip). "
          "Generation: % exact 10-bit sums.", "", "| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for r in rows:
        L.append(f"| {r['model']} | " + " | ".join(pct(r["T2_context_transplant"][k]) for k in keys) + " |")
    keys = [f"{d}/x{m}" for d in ("probe", "pc1", "random") for m in (1, 2, 4, 8)]
    L += ["", "## T3: steering carry-0 rows by m x the class gap (% of outputs that flip)", "",
          "| model | " + " | ".join(keys) + " | gap along probe / gap along PC1 |", "|" + "---|" * (len(keys) + 2)]
    for r in rows:
        t = r["T3_steering"]
        L.append(f"| {r['model']} | " + " | ".join(pct(t[k]) for k in keys) + f" | {t['gap_along_probe_over_gap_along_pc1']:.2f} |")
    L += ["", "## T4/T5: PC1 interchange with different pairs", "",
          "| model | paper's pairs: success | same s_(k-1): success | cross pairs: follows carry | cross pairs: follows source's answer |", "|---|---|---|---|---|"]
    for r in rows:
        t = r["T4_T5_interchange"]
        L.append(f"| {r['model']} | {pct(t['same/pc1_success'])} | {pct(t['token/pc1_success']) if 'token/pc1_success' in t else '-'} | "
                 f"{pct(t['cross/follows_carry'])} | {pct(t['cross/follows_source_answer'])} |")
    L += ["", "## T6: gate margin (min over the 8 (a_(k-1), b_(k-1), s_(k-1)) entries of |mean - 0.5| / sd)", "",
          "| model | PC1 | PC1, current digits removed | top PC of PC1-4, current digits removed | snapped = carry table |", "|---|---|---|---|---|"]
    for r in rows:
        t = r["T6_margins"]
        L.append(f"| {r['model']} | {t['pc1']['gate_margin']:.2f} | {t['pc1_minus_current_digits']['gate_margin']:.2f} | "
                 f"{t['pc1-4_minus_current_digits']['gate_margin']:.2f} | {all(v['snapped_is_carry'] for v in t.values())} |")
    L += ["", "## T7: MLP units at q_k, k = 1..9", "",
          "| model | active | eta2(a,b,c) >= 0.8 | of those, count explains >= 95% | best Boolean R2 >= 0.8 | of those, count thresholds | median eta2 (a,b,c) | median eta2 count | median eta2 (a,b,c) within k |",
          "|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: "-" if v is None else f"{v:.2f}"
    for r in rows:
        t = r["T7_mlp_units"]
        L.append(f"| {r['model']} | {t['n_active']} | {t['units_eta2_combo_ge_0.8_pooled']} | {f2(t['share_of_those_symmetric (count explains >= 95%)'])} | "
                 f"{t['units_best_boolean_R2_ge_0.8']} | {f2(t['share_of_those_count_thresholds'])} | {t['median_eta2_combo']:.2f} | "
                 f"{t['median_eta2_count']:.2f} | {t['median_eta2_combo_within_k']:.2f} |")
    L += ["", "## T8: later bits after s_k is flipped (teacher-forced), where the local rule and the original carry disagree", "",
          "| model | next bit follows local rule | follows original carry | whole rest = adder with carry flipped | max-head attention q_(k+1) -> a_k / b_k / a_(k-1) / b_(k-1) / s_(k-1) |",
          "|---|---|---|---|---|"]
    for r in rows:
        t = r["T8_continuation"]; at = t["max_head_attention_from_q_(k+1)"]
        L.append(f"| {r['model']} | {pct(t['next_follows_local_rule'])} | {pct(t['next_follows_original_carry'])} | {pct(t['rest_matches_flipped_adder'])} | "
                 + " / ".join(f"{at[x]:.2f}" for x in ("a_k", "b_k", "a_(k-1)", "b_(k-1)", "s_(k-1)")) + " |")
    L += ["", "## T9: what MLP units at q_k depend on (k = 1..9): median eta2 over active units (units >= 0.8)", "",
          "| model | active | (a_k, b_k, c_k) | + bit position k | + k + (a_(k-1), b_(k-1)) |", "|---|---|---|---|---|"]
    for r in rows:
        t = r["T9_unit_dependence"]
        L.append(f"| {r['model']} | {t['n_active']} | " + " | ".join(
            f"{t[x]['median_eta2']:.2f} ({t[x]['units_ge_0.8']})" for x in ("current_digits", "current_digits+k", "current+previous_digits+k")) + " |")
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    if args == ["summary"]:
        summary()
    else:
        todo = [a.split(":") for a in args] or [(f, s) for f in TG.SEEDS for s in TG.SEEDS[f]]
        for f, s in todo:
            run(f, int(s))
