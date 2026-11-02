"""Cammarata's clean-neuron test x the carry direction ("squeeze, watch, snap, read").

    python clean_carry.py prereg-check          # locks results_clean_carry/PREREG.md (hash + time); must pass first
    python clean_carry.py stage0                # saved B:0-B:4: H1, D1, D5, V4, V5 -> analysis/stage0.json
    python clean_carry.py train                 # arms A and R, step-stamped checkpoints, several processes at once
    python clean_carry.py analyse               # every model: labels (+ carry direction for arm A) per checkpoint
    python clean_carry.py report                # V1-V5, H1-H4, D1-D5 by code -> results.json, VERDICTS.md, figures/
                                                # (REPORT.md is then written from results.json)
    python tests/test_circuits.py               # includes the labeller / reader / hook checks used here

Arm A: the B adder recipe of transformer_gates.py at MLP widths 256, 32, 16, 8, 4 (seeds 0-4), plus width 256
seeds 5-9. Arm R (positive control): same model and format, but each answer token is a different random
balanced Boolean function of (a_k, b_k, previous token), so per-position jobs are unrelated by construction.
Every definition and threshold is in results_clean_carry/PREREG.md. Reuses transformer_gates.py and
followup_tests.py; changes neither, nor any earlier result.
"""
import argparse, hashlib, json, os, subprocess, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np

from tinygpt import TinyGPT
import run_transformer as RT
import transformer_gates as TG
import followup_tests as FU

OUT = "results_clean_carry"
MODELS, AN, FIG = (os.path.join(OUT, d) for d in ("models", "analysis", "figures"))
PREREG, LOCK = os.path.join(OUT, "PREREG.md"), os.path.join(OUT, "prereg_lock.json")
NB, STEPS, BATCH = TG.NB, 2500, 128
KS = list(range(1, NB))                              # k = 1..9 (as P4b: the top bit has a_k = b_k = 0)
WIDTHS = {"A": [256, 32, 16, 8, 4], "R": [256, 32, 16, 8]}
SEEDS, EXTRA = list(range(5)), list(range(5, 10))
DENSE = (256, 16)
D1_EXTRA = [("A", 256, s) for s in EXTRA]


def runs():
    out = [(arm, w, s) for arm in ("A", "R") for w in WIDTHS[arm] for s in SEEDS]
    return out + [("A", 256, s) for s in EXTRA]


def schedule(arm, w, s):
    if w in DENSE and s in SEEDS:
        return [0] + list(range(25, 1001, 25)) + list(range(1250, STEPS + 1, 250))
    return [0] + list(range(250, STEPS + 1, 250))


# ======================= pre-registration lock =======================
def prereg_hash():
    if not os.path.exists(PREREG):
        raise SystemExit(f"{PREREG} is missing: write the pre-registration first")
    return hashlib.sha256(open(PREREG, "rb").read()).hexdigest()


def prereg_check(lock=False):
    h = prereg_hash()
    if not os.path.exists(LOCK):
        if not lock:
            raise SystemExit("PREREG not locked yet: run  python clean_carry.py prereg-check")
        json.dump({"sha256": h, "locked": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(LOCK, "w"), indent=1)
        print("locked PREREG.md", h)
        return h
    L = json.load(open(LOCK))
    if L["sha256"] != h:
        raise SystemExit("PREREG.md changed after it was locked: refusing to run")
    print("PREREG.md unchanged since", L["locked"], h)
    return h


# ======================= arm R: random per-position tables =======================
def _depends_on_all(f):
    for j in range(3):
        if all(((f >> c) & 1) == ((f >> (c ^ (1 << j))) & 1) for c in range(8)):
            return False
    return True


def _tables():
    cands = [f for f in range(256) if bin(f).count("1") == 4 and _depends_on_all(f)]
    picks = np.random.default_rng(4242).choice(cands, NB + 1, replace=False)
    return np.array([[(int(f) >> c) & 1 for c in range(8)] for f in picks], int)   # TAB[k, 4a + 2b + t_prev]


TAB = _tables()


class FlatTable(TG.Flat):
    """Arm R: a0..a9 + b0..b9 = t0..t10 with t_k = T_k(a_k, b_k, t_(k-1)), t_(-1) = 0."""
    name = "R"

    @staticmethod
    def targets(a, b):
        A, B_ = RT.bits(a, NB + 1), RT.bits(b, NB + 1)
        t = np.zeros((len(a), NB + 1), int); prev = np.zeros(len(a), int)
        for k in range(NB + 1):
            t[:, k] = TAB[k][4 * A[:, k] + 2 * B_[:, k] + prev]; prev = t[:, k]
        return t

    @staticmethod
    def tokens(a, b, n=NB):
        tok = RT.encode(a, b, "lsb").copy()
        tok[:, RT.POS_EQ + 1:] = FlatTable.targets(a, b)
        return tok


LAY = {"A": TG.Flat, "R": FlatTable}


# ======================= models and training =======================
def make(w, seed):
    return TinyGPT(d_vocab=4, n_ctx=RT.T, d_model=64, n_heads=4, n_layers=1, d_mlp=w, seed=seed)


def name_of(arm, w, s):
    return f"{arm}-w{w}-s{s}"


def ckpt(arm, w, s, step):
    return os.path.join(MODELS, name_of(arm, w, s), f"step{step:04d}.npz")


def load_ckpt(arm, w, s, step):
    m = make(w, s); z = np.load(ckpt(arm, w, s, step), allow_pickle=True)
    m.load({k: z[k] for k in m.P})
    return m


def train_width(arm, w, s):
    """TG.train's recipe and data stream (default_rng([seed, 1000])) with a d_mlp argument and
    parameter-only checkpoints at every step of schedule(). Width 256, arm A reproduces TG.train."""
    final = ckpt(arm, w, s, STEPS)
    if os.path.exists(final):
        return
    os.makedirs(os.path.dirname(final), exist_ok=True)
    lay, model, sched = LAY[arm], make(w, s), set(schedule(arm, w, s))
    r = np.random.default_rng([s, 1000])
    mask = np.zeros((BATCH, RT.T), bool); mask[:, [lay.q(k) for k in range(NB + 1)]] = True
    log, t0 = [], time.time()

    def save(step):
        np.savez(ckpt(arm, w, s, step), step=step, log=np.array(log if log else [(0, np.nan)]), **model.P)
    save(0)
    for step in range(STEPS):
        a, b = TG.problems(BATCH, NB, r)
        lr = 1e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / STEPS)))
        loss, Gr = model.loss_and_grads(lay.tokens(a, b), mask)
        model.adam_step(Gr, lr, wd=0.01)
        if step % 50 == 0:
            log.append((step, loss))
            print(f"{name_of(arm, w, s)} step {step} loss {loss:.5f}  {time.time() - t0:.0f}s", flush=True)
        if step + 1 in sched:
            save(step + 1)


def pool(jobs, sub):
    """Run `python clean_carry.py <sub> --only arm:w:s` for every job, min(7, cores - 1) at once."""
    cores = os.cpu_count() or 2
    n_par = max(1, min(7, cores - 1))
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    print(f"{cores} cores -> {n_par} processes; {len(jobs)} jobs", flush=True)
    todo, running = list(jobs), []
    while todo or running:
        while todo and len(running) < n_par:
            arm, w, s = todo.pop(0)
            logf = open(os.path.join(OUT, "logs", f"{sub}-{name_of(arm, w, s)}.log"), "a")
            running.append(((arm, w, s), subprocess.Popen([sys.executable, __file__, sub, "--only", f"{arm}:{w}:{s}"],
                                                         stdout=logf, stderr=subprocess.STDOUT)))
        time.sleep(3)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                print(f"{sub} {name_of(*item[0])} finished (exit {item[1].returncode})", flush=True)


# ======================= data, activations, hooks =======================
_DATA = {}


def data():
    if not _DATA:
        rng0 = np.random.default_rng(7)
        a_tr, b_tr = TG.problems(8000, NB, rng0); a_te, b_te = TG.problems(4000, NB, rng0)
        _DATA.update(a_tr=a_tr, b_tr=b_tr, a_te=a_te, b_te=b_te, Ltr=TG.labels(a_tr, b_tr), Lte=TG.labels(a_te, b_te))
    return _DATA


def hidden(model, lay, a, b, bs=2000):
    """Teacher-forced: hook_post (N, 11, U) and resid_mid (N, 11, 64) at q_0..q_10, and the argmax outputs (N, 11)."""
    qs = [lay.q(k) for k in range(NB + 1)]; H, X, O = [], [], []
    for i in range(0, len(a), bs):
        lg, acts, _ = model.forward(lay.tokens(a[i:i + bs], b[i:i + bs]), cache=True)
        H.append(acts["blocks.0.mlp.hook_post"][:, qs].astype(np.float32))
        X.append(acts[TG.MID][:, qs].astype(float)); O.append(lg[:, qs, :2].argmax(2))
    return np.concatenate(H), np.concatenate(X), np.concatenate(O)


def inputs(arm, a, b, L):
    """Per k = 0..10: combo index 4a + 2b + x (x = carry for A, previous answer token for R), and the count (A)."""
    A, B_ = L["a"], L["b"]
    x = L["c"] if arm == "A" else np.concatenate([np.zeros((len(a), 1), int), FlatTable.targets(a, b)[:, :-1]], 1)
    return 4 * A + 2 * B_ + x, A + B_ + x


def _ln2(model, xm):
    P = model.P
    xc = xm - xm.mean(-1, keepdims=True)
    return xc / np.sqrt((xc ** 2).mean(-1, keepdims=True) + 1e-5) * P["0.ln2.w"] + P["0.ln2.b"]


def mlp_replace(model, q, fn, mid_fn=None):
    """Hooks that recompute the MLP at position q with fn applied to its post-ReLU activations
    (fn: (rows, U) -> (rows, U)); mid_fn optionally patches resid_mid at q first. No edit to tinygpt.py:
    a grab on resid_mid and a write on resid_post (the pattern of TG.step_module_transplant)."""
    keep = {}
    P = model.P

    def grab(x):
        if mid_fn is not None:
            x = x.copy(); x[:, q] = mid_fn(x[:, q].astype(float))
        keep["x"] = x
        return x

    def write(x):
        xm = keep["x"][:, q].astype(float)
        h = np.maximum(_ln2(model, xm) @ P["0.W_in"] + P["0.b_in"], 0)
        x = x.copy(); x[:, q] = xm + fn(h) @ P["0.W_out"] + P["0.b_out"]
        return x
    return {TG.MID: grab, TG.POST: write}


# ======================= the labeller =======================
def fval(mask):
    return int(sum(int(m) << c for c, m in enumerate(mask)))


THR_CODE = {}
for _t in (1, 2, 3):
    _m = TG.mask_of(lambda x, y, z, t=_t: int(x + y + z >= _t)).astype(int)
    THR_CODE[fval(_m)] = 1000 + 10 * _t                  # unit high when count >= t
    THR_CODE[fval(1 - _m)] = 1000 + 10 * _t + 1          # unit high when count < t
DICTATORS = set()
for _j in range(3):
    _m = TG.mask_of(lambda x, y, z, j=_j: (x, y, z)[j]).astype(int)
    DICTATORS |= {fval(_m), fval(1 - _m)}


def is_thr(code):
    return code >= 1000


def thr_t(code):
    return (code - 1000) // 10


def thr_sign(code):
    return +1 if code % 10 == 0 else -1


def label_k(Hk, combo, count, arm, min_active=0.05):
    """Labels of every unit at one position: -1 inactive, 0 unclear, 1..254 Boolean f (oriented: the unit is
    high where f = 1), 1000 + 10 t (+1) count threshold (or below-threshold)."""
    U = Hk.shape[1]; codes = np.full(U, -1, int)
    act = ((Hk > 0).mean(0) >= min_active) & (Hk.std(0) > 1e-6)
    if not act.any():
        return codes
    Ha = np.asarray(Hk[:, act], float)
    R2, masks = TG.bool_fit(Ha, combo, 3)
    best, fi = R2.max(0), R2.argmax(0)
    out = np.zeros(Ha.shape[1], int)
    for j in np.where(best >= 0.8)[0]:
        m = masks[fi[j]].astype(int)
        on = m[combo] == 1
        if Ha[on, j].mean() < Ha[~on, j].mean():
            m = 1 - m
        f = fval(m)
        out[j] = THR_CODE.get(f, f)
    if arm == "A":                                        # count ramps: graded ReLU units are not penalised for being analog
        for j in np.where(best < 0.8)[0]:
            x = Ha[:, j]; mu = x.mean(); sst = ((x - mu) ** 2).sum()
            means = np.array([x[count == n].mean() if (count == n).any() else np.nan for n in range(4)])
            if np.isnan(means).any() or sst <= 0:
                continue
            ssb = sum((count == n).sum() * (means[n] - mu) ** 2 for n in range(4))
            d = np.diff(means); tol = 1e-6 * (np.ptp(means) + 1e-12)
            if ssb / sst >= 0.9 and ((d >= -tol).all() or (d <= tol).all()):
                rising = means[-1] > means[0]
                theta = 1 + int(np.argmax(d if rising else -d))
                out[j] = 1000 + 10 * theta + (0 if rising else 1)
    codes[act] = out
    return codes


def label_all(H, combo, count, arm):
    """(U, 9) label codes for k = 1..9."""
    return np.stack([label_k(H[:, k], combo[:, k], count[:, k], arm) for k in KS], 1)


def verdicts(codes, arm, drop_dictators=False):
    """Per unit: 'dead', 'unclear', 'one', 'two', 'packed' (PREREG definitions)."""
    out = []
    for row in codes:
        labs = row[row >= 0]
        if len(labs) == 0:
            out.append("dead"); continue
        if drop_dictators:
            labs = np.array([0 if (0 < c < 1000 and c in DICTATORS) else c for c in labs])
        if (labs == 0).any():
            out.append("unclear"); continue
        s = sorted(set(labs.tolist()))
        if len(s) == 1:
            out.append("one")
        elif (arm == "A" and len(s) == 2 and all(is_thr(c) for c in s) and thr_sign(s[0]) == thr_sign(s[1])
              and abs(thr_t(s[0]) - thr_t(s[1])) == 1):
            out.append("two")
        else:
            out.append("packed")
    return np.array(out)


def _same_rate(C, lab):
    num = den = 0
    for i in range(C.shape[1]):
        for j in range(i + 1, C.shape[1]):
            both = lab[:, i] & lab[:, j]
            den += int(both.sum()); num += int((C[both, i] == C[both, j]).sum())
    return num / den if den else float("nan")


def delta(codes, perms, rng):
    """Within-unit same-label rate over pairs of labelled positions, minus its permutation mean (labels shuffled
    across units within each position). Returns (delta, observed, null mean)."""
    lab = codes > 0
    obs = _same_rate(codes, lab)
    if not np.isfinite(obs):
        return float("nan"), obs, float("nan")
    null = []
    for _ in range(perms):
        C = codes.copy()
        for k in range(C.shape[1]):
            idx = np.where(lab[:, k])[0]
            C[idx, k] = rng.permutation(C[idx, k])
        null.append(_same_rate(C, lab))
    return float(obs - np.mean(null)), float(obs), float(np.mean(null))


def summarize(codes, arm, perms, seed=0):
    v = verdicts(codes, arm); vd = verdicts(codes, arm, drop_dictators=True)
    alive = v != "dead"; n = max(int(alive.sum()), 1)
    n_act = (codes >= 0).sum(1)
    multi = alive & (v != "unclear") & (n_act >= 2)
    d, obs, null = delta(codes, perms, np.random.default_rng(seed))
    return {"units_alive": int(alive.sum()),
            "one": int((v == "one").sum()), "two": int((v == "two").sum()), "packed": int((v == "packed").sum()),
            "unclear": int((v == "unclear").sum()),
            "cleanliness": float(((v == "one") | (v == "two")).sum() / n),
            "cleanliness_no_dictators": float(((vd == "one") | (vd == "two")).sum() / n),
            "packed_share_multi": float((v[multi] == "packed").mean()) if multi.any() else None,
            "n_multi": int(multi.sum()),
            "positions_per_unit": float(n_act[alive].mean()) if alive.any() else 0.0,
            "delta": d, "same_rate": obs, "same_rate_null": null,
            "clean_threshold_pairs": int((codes >= 1000).sum()),
            "shared_thresholds": sorted({int(thr_t(row[row >= 0][0])) for row, vv in zip(codes, v)
                                         if vv == "one" and (row >= 0).sum() >= 2 and is_thr(row[row >= 0][0])})}


# ======================= the carry direction =======================
def directions(model, seed, n_pairs):
    U = TG.flip_dirs(model, TG.Flat, "B", seed, n_pairs=n_pairs)
    return U, {k: U[k][:, 0] for k in KS}


def read_cos(model, V):
    """(U, 9): cosine of each unit's effective read weight ln2.w * W_in[:, j] with v_k, mean components removed."""
    P = model.P
    W = (P["0.ln2.w"][:, None] * P["0.W_in"]).astype(float); W = W - W.mean(0, keepdims=True)
    W = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    out = []
    for k in KS:
        v = V[k] - V[k].mean(); v = v / max(np.linalg.norm(v), 1e-12)
        out.append(W.T @ v)
    return np.stack(out, 1)


def mlp_at(model, tok, q):
    lg, acts, _ = model.forward(tok[:, :q + 1], cache=True)
    return acts["blocks.0.mlp.hook_post"][:, q].astype(float), lg[:, q, :2].argmax(1)


def interchange(model, V, rng, n_pairs, clamp=None):
    """Rank-1 interchange along v_k at resid_mid q_k, k = 1..9, pooled. clamp: {k: unit indices} held at their
    unpatched values while the MLP reads the patched resid_mid."""
    ok = []
    for k in KS:
        a0, b0, a1, b1 = TG.opposite_pairs(n_pairs, k, rng)
        q, tok0 = TG.Flat.q(k), TG.Flat.tokens(a0, b0)
        x1, _ = TG.mid_at(model, TG.Flat.tokens(a1, b1), q)
        h0, base = mlp_at(model, tok0, q)
        valid = base == tok0[:, q + 1]
        target = ((a0 >> k) & 1) ^ ((b0 >> k) & 1) ^ (1 - RT.carry_into(a0, b0, k))
        v = V[k][:, None]; P = v @ v.T
        patch = lambda xq: xq + (x1 - xq) @ P
        if clamp is None:
            hooks = TG.patch_q(q, patch)
        else:
            u = np.asarray(clamp.get(k, []), int)

            def fn(h, u=u):
                h = h.copy(); h[:, u] = h0[:, u]; return h
            hooks = mlp_replace(model, q, fn, mid_fn=patch)
        out = model.forward(tok0[:, :q + 1], hooks=hooks)[:, q, :2].argmax(1)
        ok.append((out == target)[valid])
    ok = np.concatenate(ok)
    return float(ok.mean()) if len(ok) else float("nan")


def clamp_sanity(model, n=1000, seed=0):
    """V5: every unit held at its own unpatched value, no patch: outputs must agree with the plain run."""
    r = np.random.default_rng(seed); agree = []
    for k in KS:
        a, b = TG.problems(n, NB, r); tok, q = TG.Flat.tokens(a, b), TG.Flat.q(k)
        h0, base = mlp_at(model, tok, q)
        out = model.forward(tok[:, :q + 1], hooks=mlp_replace(model, q, lambda h: h0.copy()))[:, q, :2].argmax(1)
        agree.append(out == base)
    return float(np.concatenate(agree).mean())


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([pos, neg]); r = rank(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def readers(codes, R, model, V, seed, n_pairs=1000, with_clamp=True):
    """D1: AUC of |r| (threshold pairs vs other active pairs), sign agreement, clamp mediation."""
    act = codes >= 0; thr = codes >= 1000
    out = {"clean_threshold_pairs": int(thr.sum()),
           "auc_abs_r": auc(np.abs(R[thr]), np.abs(R[act & ~thr])),
           "median_abs_r_threshold": float(np.median(np.abs(R[thr]))) if thr.any() else None,
           "median_abs_r_other": float(np.median(np.abs(R[act & ~thr]))) if (act & ~thr).any() else None}
    if thr.any():
        sg = np.array([thr_sign(c) for c in codes[thr]])
        out["sign_agree"] = int((np.sign(R[thr]) == sg).sum()); out["sign_n"] = int(thr.sum())
    if with_clamp and thr.any():
        plain = interchange(model, V, np.random.default_rng(700 + seed), n_pairs)
        cl = {k: np.where(codes[:, i] >= 1000)[0] for i, k in enumerate(KS)}
        clamped = interchange(model, V, np.random.default_rng(700 + seed), n_pairs, clamp=cl)
        rr = np.random.default_rng(800 + seed); rnd = []
        for _ in range(5):
            cr = {}
            for i, k in enumerate(KS):
                pool_ = np.where((codes[:, i] >= 0) & (codes[:, i] < 1000))[0]
                cr[k] = rr.choice(pool_, min(len(cl[k]), len(pool_)), replace=False) if len(pool_) else np.array([], int)
            rnd.append(interchange(model, V, np.random.default_rng(700 + seed), n_pairs, clamp=cr))
        out.update(interchange_plain=plain, interchange_clean_held=clamped, interchange_random_held=float(np.mean(rnd)),
                   drop_clean=plain - clamped, drop_random=plain - float(np.mean(rnd)))
    return out


RAND_BAND = None


def basis_stats(v):
    v = v / np.linalg.norm(v)
    return float((v ** 2).max()), float(1 / (v ** 4).sum())


def rand_band():
    global RAND_BAND
    if RAND_BAND is None:
        g = np.random.default_rng(31).normal(size=(1000, 64))
        st = np.array([basis_stats(x) for x in g])
        RAND_BAND = {"max_sq": np.percentile(st[:, 0], [2.5, 97.5]).tolist(), "pr": np.percentile(st[:, 1], [2.5, 97.5]).tolist()}
    return RAND_BAND


def d5(V):
    b = rand_band(); rows = []
    for k in KS:
        m, pr = basis_stats(V[k])
        rows.append({"k": k, "max_sq": m, "pr": pr,
                     "inside": bool(b["max_sq"][0] <= m <= b["max_sq"][1] and b["pr"][0] <= pr <= b["pr"][1])})
    return rows


def debiased_margin(U, X, L):
    Ufull = {k: U[k] for k in range(1, NB + 1)}
    return FU.t6_margins(Ufull, X, L, "B")["pc1_minus_current_digits"]["gate_margin"]


def rotation_null(H, combo, count, arm, draws=10, seed=0):
    """V4: cleanliness after mixing the units by a random orthogonal matrix."""
    U = H.shape[2]; rng = np.random.default_rng(1100 + seed); vals = []
    for _ in range(draws):
        Q, _ = np.linalg.qr(rng.normal(size=(U, U)))
        Hr = np.einsum("nku,uv->nkv", H.astype(float), Q)
        codes = np.stack([label_k(Hr[:, k], combo[:, k], count[:, k], arm) for k in KS], 1)
        vv = verdicts(codes, arm); alive = vv != "dead"
        vals.append(float(((vv == "one") | (vv == "two")).sum() / max(alive.sum(), 1)))
    return float(np.mean(vals))


# ======================= accuracy and snaps (H4) =======================
def exact_acc(model, n=2000):
    D = data(); a, b = D["a_te"][:n], D["b_te"][:n]
    return float(np.mean(TG.generate(model, TG.Flat, a, b) == a + b))


def bit_acc_R(O, a, b):
    t = FlatTable.targets(a, b)
    return float(np.mean([np.mean(O[:, k] == t[:, k]) for k in KS]))


def snaps(model, Htr, count_tr, n=2000):
    """H4 edits at q_k (k = 1..9) during greedy generation of n test problems."""
    D = data(); a, b, Lte = D["a_te"][:n], D["b_te"][:n], D["Lte"]
    count_te = Lte["a"][:n] + Lte["b"][:n] + Lte["c"][:n]
    U = Htr.shape[2]; P = {}
    for k in KS:
        Hk = Htr[:, k].astype(float)
        p75, tau = np.full(U, -np.inf), np.full(U, np.inf)
        for u in range(U):
            nz = Hk[:, u][Hk[:, u] > 0]
            if len(nz) == 0:
                continue
            p75[u] = np.percentile(nz, 75)
            mass = nz[nz <= p75[u]].sum()
            srt = np.sort(nz)[::-1]; cs = np.cumsum(srt)
            tau[u] = srt[min(int(np.searchsorted(cs, mass)), len(srt) - 1)]
        cmean = np.stack([Hk[count_tr[:, k] == c].mean(0) if (count_tr[:, k] == c).any() else np.zeros(U) for c in range(4)])
        lo, hi, th = np.zeros(U), np.zeros(U), np.zeros(U)
        for u in range(U):
            if Hk[:, u].std() > 1e-9:
                lo[u], hi[u], th[u] = TG.two_means(Hk[:, u])
            else:
                lo[u] = hi[u] = th[u] = Hk[:, u].mean()
        P[k] = (p75, tau, cmean, lo, hi, th)

    def run(edit):
        def hooks_fn(k, rows):
            if k not in P:
                return None
            return mlp_replace(model, TG.Flat.q(k), lambda h, k=k, rows=rows: edit(h, k, rows))
        return float(np.mean(TG.generate(model, TG.Flat, a, b, hooks_fn=hooks_fn) == a + b))

    def weak(h, k, rows):
        p75 = P[k][0]; h = h.copy(); h[(h > 0) & (h <= p75)] = 0; return h

    def strong(h, k, rows):
        tau = P[k][1]; h = h.copy(); h[h >= tau] = 0; return h

    def cnt(h, k, rows):
        return P[k][2][count_te[rows, k]]

    def two(h, k, rows):
        lo, hi, th = P[k][3:]
        return np.where(h > th, hi, lo)
    base = run(lambda h, k, rows: h)
    out = {"base": base, "weak_removed": run(weak), "strong_removed": run(strong), "count_snap": run(cnt),
           "two_level_snap": run(two)}
    out["weak_cost_points"] = 100 * (base - out["weak_removed"]); out["strong_cost_points"] = 100 * (base - out["strong_removed"])
    return out


# ======================= per-model analysis =======================
def analyse_codes(model, arm, a, b, L):
    H, X, O = hidden(model, LAY[arm], a, b)
    combo, count = inputs(arm, a, b, L)
    return H, X, O, combo, count, label_all(H, combo, count, arm)


def analyse_model(arm, w, s):
    path = os.path.join(AN, name_of(arm, w, s) + ".json")
    if os.path.exists(path):
        return
    D = data(); a, b, L = D["a_te"], D["b_te"], D["Lte"]
    steps = schedule(arm, w, s); dense = len(steps) > 12
    res = {"arm": arm, "width": w, "seed": s, "checkpoints": []}
    T0 = time.time()
    for step in steps:
        model = load_ckpt(arm, w, s, step)
        final = step == STEPS
        H, X, O, combo, count, codes = analyse_codes(model, arm, a, b, L)
        row = {"step": step, "labels": summarize(codes, arm, 1000 if final else 200, seed=s)}
        if arm == "A":
            row["exact_acc"] = exact_acc(model)
        else:
            row["bit_acc"] = bit_acc_R(O, a, b)
        if dense:
            row["codes"] = codes.tolist()
        if arm == "A" and (dense or step in (0, STEPS)):
            U, V = directions(model, s, 2000 if final else 500)
            row["interchange"] = interchange(model, V, np.random.default_rng(600 + s), 1000 if final else 300)
            R = read_cos(model, V)
            if dense:
                row["read_cos"] = np.round(R, 4).tolist()
            row["readers"] = readers(codes, R, model, V, s, with_clamp=final and w == 256)
            if final:
                row["d5"] = d5(V)
                row["debiased_margin"] = debiased_margin(U, X, L)
                if row["exact_acc"] >= 0.99:
                    Htr, _, _ = hidden(model, TG.Flat, D["a_tr"], D["b_tr"])
                    _, count_tr = inputs("A", D["a_tr"], D["b_tr"], D["Ltr"])
                    row["snaps"] = snaps(model, Htr, count_tr)
                if w == 256:
                    row["clamp_sanity"] = clamp_sanity(model, seed=s)
        res["checkpoints"].append(row)
        print(f"{name_of(arm, w, s)} step {step}: clean {row['labels']['cleanliness']:.2f} "
              f"packed {row['labels']['packed_share_multi']} acc {row.get('exact_acc', row.get('bit_acc')):.3f}"
              + (f" ic {row['interchange']:.3f}" if "interchange" in row else "") + f"  [{time.time() - T0:.0f}s]", flush=True)
    os.makedirs(AN, exist_ok=True)
    json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def stage0():
    """Saved B:0-B:4 (no training): labels, H1, D1 (with clamp mediation), D5, debiased margin, V4, V5."""
    path = os.path.join(AN, "stage0.json")
    D = data(); a, b, L = D["a_te"], D["b_te"], D["Lte"]
    out = {}
    for s in range(5):
        T0 = time.time()
        model, _ = TG.load("B", s)
        H, X, O, combo, count, codes = analyse_codes(model, "A", a, b, L)
        M = {"labels": summarize(codes, "A", 1000, seed=s), "exact_acc": exact_acc(model)}
        U, V = directions(model, s, 2000)
        R = read_cos(model, V)
        M["interchange"] = interchange(model, V, np.random.default_rng(600 + s), 1000)
        M["readers"] = readers(codes, R, model, V, s)
        M["d5"] = d5(V)
        M["debiased_margin"] = debiased_margin(U, X, L)
        M["rotation_cleanliness"] = rotation_null(H[:, :, :], combo, count, "A", seed=s)
        M["clamp_sanity"] = clamp_sanity(model, seed=s)
        M["codes"] = codes.tolist(); M["read_cos"] = np.round(R, 4).tolist()
        out[f"B:{s}"] = M
        print(f"B:{s} clean {M['labels']['cleanliness']:.3f} packed {M['labels']['packed_share_multi']} delta {M['labels']['delta']:.3f} "
              f"rot {M['rotation_cleanliness']:.3f} ic {M['interchange']:.3f} auc {M['readers']['auc_abs_r']} "
              f"drops {M['readers'].get('drop_clean')} / {M['readers'].get('drop_random')}  [{time.time() - T0:.0f}s]", flush=True)
    os.makedirs(AN, exist_ok=True)
    json.dump(out, open(path, "w"))


# ======================= verdicts =======================
def load_an(arm, w, s):
    p = os.path.join(AN, name_of(arm, w, s) + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def final_row(r):
    return r["checkpoints"][-1]


def learns(arm, row):
    return row["exact_acc"] >= 0.99 if arm == "A" else row["bit_acc"] >= 0.95


def judge():
    S0 = json.load(open(os.path.join(AN, "stage0.json")))
    A = {(w, s): load_an("A", w, s) for w in WIDTHS["A"] for s in SEEDS}
    A.update({(256, s): load_an("A", 256, s) for s in EXTRA})
    Rm = {(w, s): load_an("R", w, s) for w in WIDTHS["R"] for s in SEEDS}
    V, H, Dv, X = {}, {}, {}, {}

    # ---- V1 reproducibility
    rows = {}
    for s in (0, 2, 3, 4):
        saved, _ = TG.load("B", s); new = load_ckpt("A", 256, s, STEPS)
        diff = max(float(np.abs(saved.P[k].astype(float) - new.P[k].astype(float)).max()) for k in saved.P)
        fa, s0 = final_row(A[(256, s)]), S0[f"B:{s}"]
        loose = (abs(fa["exact_acc"] - s0["exact_acc"]) <= 0.005 and
                 abs(fa["labels"]["clean_threshold_pairs"] - s0["labels"]["clean_threshold_pairs"]) <= 0.1 * max(s0["labels"]["clean_threshold_pairs"], 1)
                 and abs(fa["interchange"] - s0["interchange"]) <= 0.02)
        rows[s] = {"max_param_diff": diff, "pass": diff <= 1e-3 or loose}
    V["V1"] = {"pass": all(r["pass"] for r in rows.values()), "seeds": rows}

    # ---- V2 sensitivity (arm R)
    rr = [(w, s, final_row(Rm[(w, s)])) for w in WIDTHS["R"] if w <= 16 for s in SEEDS if Rm[(w, s)]]
    lr = [(w, s, f["labels"]["packed_share_multi"]) for w, s, f in rr if learns("R", f)]
    hit = [p for _, _, p in lr if p is not None and p >= 0.30]
    V["V2"] = {"pass": bool(lr) and len(hit) >= len(lr) / 2, "learning_models": len(lr),
               "packed_ge_0.30": len(hit), "values": [[w, s, p] for w, s, p in lr]}

    # ---- V3 specificity (arm 0 = step 0, width 256 seeds 0-4)
    z = [A[(256, s)]["checkpoints"][0] for s in SEEDS]
    cl0 = float(np.mean([r["labels"]["cleanliness_no_dictators"] for r in z]))
    aucs0 = [r["readers"]["auc_abs_r"] for r in z if r["readers"]["clean_threshold_pairs"] >= 20 and r["readers"]["auc_abs_r"] is not None]
    V["V3"] = {"pass": cl0 <= 0.10 and all(x <= 0.6 for x in aucs0), "mean_cleanliness_no_dictators_init": cl0, "init_aucs": aucs0}

    # ---- V4 privileged basis (saved models)
    c = float(np.mean([S0[f"B:{s}"]["labels"]["cleanliness"] for s in range(5)]))
    cr = float(np.mean([S0[f"B:{s}"]["rotation_cleanliness"] for s in range(5)]))
    V["V4"] = {"pass": cr <= 0.5 * c, "cleanliness": c, "rotated": cr}

    # ---- V5 clamp sanity
    vals = [S0[f"B:{s}"]["clamp_sanity"] for s in range(5)] + [final_row(A[(256, s)])["clamp_sanity"] for s in EXTRA]
    V["V5"] = {"pass": min(vals) >= 0.999, "min_agreement": min(vals)}

    gate = lambda *g: all(V[x]["pass"] for x in g)
    nov = lambda gs: {"verdict": "no verdict (gate failed: " + ", ".join(x for x in gs if not V[x]["pass"]) + ")"}

    # ---- H1
    b3 = S0["B:3"]["labels"]
    h1 = "pass" if b3["cleanliness"] >= 0.70 else ("fail" if (b3["packed_share_multi"] or 0) >= 0.30 and b3["delta"] <= 0.05 else "inconclusive")
    H["H1"] = {"verdict": h1 if gate("V3", "V4") else nov(["V3", "V4"])["verdict"],
               "per_model": {m: {k: S0[m]["labels"][k] for k in ("cleanliness", "packed_share_multi", "delta", "positions_per_unit", "n_multi")} for m in S0}}

    # ---- H2 (development)
    def h2_seed(r):
        cps = r["checkpoints"]; steps = [c["step"] for c in cps]
        C = np.array([c["codes"] for c in cps])                      # (T, U, 9)
        fin = C[-1]; v = verdicts(fin, "A")
        ones = [u for u in range(fin.shape[0]) if v[u] == "one"]
        if len(ones) < 5:
            return None
        same, match, nulls = [], [], []
        for u in ones:
            final_lab = int(fin[u][fin[u] >= 0][0])
            labelled = [(t, C[t, u][C[t, u] > 0]) for t in range(len(cps)) if (C[t, u] > 0).any()]
            if not labelled:
                continue
            t0, labs = labelled[0]
            vals_, cnts = np.unique(labs, return_counts=True); first = int(vals_[cnts.argmax()])
            same.append(first == final_lab)
            for i in np.where((fin[u] > 0) & ~(C[t0, u] > 0))[0]:
                tj = next((t for t in range(t0, len(cps)) if C[t, u, i] > 0), None)
                if tj is None:
                    continue
                match.append(int(C[tj, u, i]) == first)
                col = C[tj, :, i]; labd = col[col > 0]
                nulls.append(float(np.mean(labd == first)) if len(labd) else 0.0)
        return {"one_job_units": len(ones), "first_eq_final": float(np.mean(same)) if same else None,
                "joining_positions": len(match),
                "excess": float(np.mean(match) - np.mean(nulls)) if match else None}
    h2 = {w: {s: h2_seed(A[(w, s)]) for s in SEEDS} for w in DENSE}
    ok_w, bad = True, False
    for w in DENSE:
        good = [x for x in h2[w].values() if x and x["first_eq_final"] is not None and x["excess"] is not None]
        npass = sum(x["first_eq_final"] >= 0.8 and x["excess"] >= 0.2 for x in good)
        nfail = sum(x["first_eq_final"] < 0.6 or x["excess"] < 0.05 for x in good)
        ok_w &= npass >= 4; bad |= nfail >= 2
    h2v = "fail" if bad else ("pass" if ok_w else "inconclusive")
    H["H2"] = {"verdict": h2v if gate("V1", "V3", "V4") else nov(["V1", "V3", "V4"])["verdict"],
               "per_width": {str(w): {str(s): h2[w][s] for s in SEEDS} for w in DENSE}}

    # ---- H3 (squeeze)
    per_w = {}
    for w in WIDTHS["A"]:
        fr = [final_row(A[(w, s)]) for s in SEEDS]
        lw = [f for f in fr if learns("A", f)]
        per_w[w] = {"learning_seeds": len(lw), "learns": len(lw) >= 3,
                    "cleanliness": float(np.mean([f["labels"]["cleanliness"] for f in lw])) if lw else None,
                    "delta": float(np.nanmean([f["labels"]["delta"] for f in lw])) if lw else None,
                    "positions_per_unit": float(np.mean([f["labels"]["positions_per_unit"] for f in lw])) if lw else None,
                    "packed_share_multi": [f["labels"]["packed_share_multi"] for f in lw],
                    "packed_and_low_delta_seeds": sum(1 for f in lw if (f["labels"]["packed_share_multi"] or 0) >= 0.30 and f["labels"]["delta"] < 0.05),
                    "shared_thresholds": sorted({t for f in lw for t in f["labels"]["shared_thresholds"]})}
    lw = [w for w in WIDTHS["A"] if per_w[w]["learns"]]
    rho = spearman(lw, [per_w[w]["positions_per_unit"] for w in lw]) if len(lw) >= 3 else None
    passh3 = (bool(lw) and all(per_w[w]["cleanliness"] >= 0.70 and per_w[w]["delta"] >= 0.20 for w in lw)
              and (rho is None or rho <= -0.8) and set(per_w[min(lw)]["shared_thresholds"]) >= {1, 2, 3})
    failh3 = V["V2"]["pass"] and any(per_w[w]["packed_and_low_delta_seeds"] >= 3 for w in lw if w <= 16)
    H["H3"] = {"verdict": ("fail" if failh3 else "pass" if passh3 else "inconclusive") if gate("V1", "V3", "V4") else nov(["V1", "V3", "V4"])["verdict"],
               "rho_width_positions": rho, "per_width": {str(w): v for w, v in per_w.items()},
               "note": "the fail clause needs V2" if not V["V2"]["pass"] else ""}

    # ---- H4 (snaps)
    sn = [final_row(A[k])["snaps"] for k in A if A[k] and "snaps" in final_row(A[k])]
    p4 = sum(x["weak_cost_points"] <= 2 and x["strong_cost_points"] >= 50 and x["count_snap"] >= 0.99 for x in sn)
    f4 = sum(x["count_snap"] < 0.95 or x["weak_cost_points"] > 5 for x in sn)
    H["H4"] = {"verdict": "pass" if sn and p4 >= 0.8 * len(sn) else ("fail" if sn and f4 >= 0.5 * len(sn) else "inconclusive"),
               "models": len(sn), "meet_pass": p4, "meet_fail": f4,
               "two_level": {"digital": sum(x["two_level_snap"] >= 0.99 for x in sn),
                             "analog": sum(x["two_level_snap"] < 0.90 and x["count_snap"] >= 0.99 for x in sn)}}

    # ---- D1
    d1m = {m: S0[m]["readers"] for m in S0}
    d1m.update({f"A-w256-s{s}": final_row(A[(256, s)])["readers"] for s in EXTRA})
    el = {m: r for m, r in d1m.items() if r["clean_threshold_pairs"] >= 20 and r["auc_abs_r"] is not None}
    au = [r["auc_abs_r"] for r in el.values()]
    d1a = "pass" if el and np.mean([x >= 0.70 for x in au]) >= 0.8 else ("fail" if el and np.mean([x < 0.60 for x in au]) >= 0.5 else "inconclusive")
    sa = sum(r.get("sign_agree", 0) for r in el.values()); sn_ = sum(r.get("sign_n", 0) for r in el.values())
    share = sa / sn_ if sn_ else None
    d1b = "pass" if share is not None and share >= 0.9 else ("fail" if share is not None and share < 0.75 else "inconclusive")
    dc = [(r["drop_clean"], r["drop_random"]) for r in el.values() if "drop_clean" in r]
    d1c = ("pass" if dc and np.mean([a_ >= 0.5 and b_ <= 0.1 for a_, b_ in dc]) >= 0.8
           else "fail" if dc and np.mean([a_ < 0.2 for a_, _ in dc]) >= 0.5 else "inconclusive")
    Dv["D1a"] = {"verdict": d1a if gate("V3") else nov(["V3"])["verdict"], "eligible": list(el), "auc": dict(zip(el, au))}
    Dv["D1b"] = {"verdict": d1b, "sign_share": share, "n": sn_}
    Dv["D1c"] = {"verdict": d1c if gate("V5") else nov(["V5"])["verdict"], "drops": {m: [r.get("drop_clean"), r.get("drop_random")] for m, r in el.items()}}
    Dv["D1_all_models"] = d1m

    # ---- D2 (order of emergence)
    d2 = {}
    for w in DENSE:
        d2[w] = {}
        for s in SEEDS:
            cps = A[(w, s)]["checkpoints"]
            fin_n = cps[-1]["labels"]["clean_threshold_pairs"]
            t_dir = next((c["step"] for c in cps if c.get("interchange", 0) >= 0.8), None)
            t_cl = next((c["step"] for c in cps if c["labels"]["clean_threshold_pairs"] >= 0.5 * fin_n), None) if fin_n >= 20 else None
            # per-unit onset vs read-cosine onset
            C = np.array([c["codes"] for c in cps]); Rc = np.array([c["read_cos"] for c in cps]); st = [c["step"] for c in cps]
            fin = C[-1]; on_l, on_r = [], []
            for u, i in zip(*np.where(fin >= 1000)):
                tl = next((t for t in range(len(cps)) if C[t, u, i] == fin[u, i]), None)
                tr_ = next((t for t in range(len(cps)) if abs(Rc[t, u, i]) > 0.2), None)
                if tl is not None and tr_ is not None:
                    on_l.append(st[tl]); on_r.append(st[tr_])
            d2[w][s] = {"t_dir": t_dir, "t_clean": t_cl, "final_clean_pairs": fin_n,
                        "unit_onset_rho": spearman(on_l, on_r), "unit_pairs": len(on_l)}
    def order(w):
        u = [x for x in d2[w].values() if x["t_dir"] is not None and x["t_clean"] is not None]
        if len(u) < 3:
            return None
        return sum(x["t_dir"] <= x["t_clean"] for x in u), sum(x["t_dir"] > x["t_clean"] for x in u), len(u)
    o = {w: order(w) for w in DENSE}
    if any(v is None for v in o.values()):
        d2v = "no verdict (fewer than 3 usable seeds at a width)"
    elif all(v[0] >= 4 for v in o.values()):
        d2v = "pass (writer before readers)"
    elif all(v[1] >= 4 for v in o.values()):
        d2v = "fail (readers first)"
    else:
        d2v = "mixed"
    Dv["D2"] = {"verdict": d2v if gate("V1") else nov(["V1"])["verdict"], "per_width": {str(w): {str(s): d2[w][s] for s in SEEDS} for w in DENSE},
                "counts_dir_first_clean_first_n": {str(w): o[w] for w in DENSE}}

    # ---- D3 (squeeze)
    pts = [(w, s, final_row(A[(w, s)])) for w in WIDTHS["A"] for s in SEEDS]
    pts = [(w, s, f) for w, s, f in pts if learns("A", f)]
    low = {w: sum(1 for ww, _, f in pts if ww == w and f["interchange"] < 0.8) for w in WIDTHS["A"]}
    rho3 = spearman([f["labels"]["cleanliness"] for _, _, f in pts], [f["interchange"] for _, _, f in pts])
    Dv["D3a"] = {"verdict": ("pass" if all(v < 2 for v in low.values()) else "fail") if gate("V1") else nov(["V1"])["verdict"],
                 "seeds_below_0.8_per_width": {str(w): v for w, v in low.items()},
                 "interchange": {f"{w}:{s}": f["interchange"] for w, s, f in pts}}
    Dv["D3b"] = {"verdict": (("independent" if rho3 is not None and abs(rho3) < 0.3 else "go together" if rho3 is not None and rho3 >= 0.3
                              else "trade off" if rho3 is not None else "no verdict") if gate("V1") else nov(["V1"])["verdict"]), "rho": rho3, "n": len(pts)}

    # ---- D4 (seeds)
    s4 = [final_row(A[(256, s)]) for s in SEEDS + EXTRA]
    s4 = [f for f in s4 if learns("A", f)]
    rho4 = spearman([f["labels"]["cleanliness"] for f in s4], [f["debiased_margin"] for f in s4])
    d4 = "go together" if rho4 is not None and rho4 >= 0.6 else ("independent" if rho4 is not None and abs(rho4) <= 0.2 else "inconclusive")
    Dv["D4"] = {"verdict": d4 if gate("V1") else nov(["V1"])["verdict"], "rho": rho4, "n": len(s4),
                "points": [[f["labels"]["cleanliness"], f["debiased_margin"]] for f in s4]}

    # ---- D5 (basis)
    rows5 = [x for m in S0 for x in S0[m]["d5"]] + [x for _, _, f in pts for x in f["d5"]]
    inside = float(np.mean([x["inside"] for x in rows5]))
    Dv["D5"] = {"verdict": "pass" if inside >= 0.9 else ("fail" if inside < 0.5 else "inconclusive"), "share_inside": inside,
                "n": len(rows5), "random_band": rand_band()}

    # ---- exploratory: arm R squeeze
    X["arm_R"] = {f"{w}:{s}": {"bit_acc": final_row(Rm[(w, s)])["bit_acc"], **{k: final_row(Rm[(w, s)])["labels"][k]
                  for k in ("cleanliness", "packed_share_multi", "delta", "positions_per_unit")}} for w in WIDTHS["R"] for s in SEEDS if Rm[(w, s)]}
    X["arm_A_final"] = {f"{w}:{s}": {"exact_acc": f["exact_acc"], "interchange": f["interchange"],
                                     **{k: f["labels"][k] for k in ("cleanliness", "packed_share_multi", "delta", "positions_per_unit", "clean_threshold_pairs")}}
                        for (w, s), r in A.items() if r for f in [final_row(r)]}
    return {"validity": V, "cammarata": H, "direction": Dv, "exploratory": X}


# ======================= report =======================
def figures(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for col, w in enumerate(DENSE):
        raw = max(load_an("A", w, s)["checkpoints"][-1]["labels"]["clean_threshold_pairs"] for s in SEEDS) < 20
        for s in SEEDS:
            r = load_an("A", w, s)
            st = [c["step"] for c in r["checkpoints"]]
            n = 1 if raw else max(r["checkpoints"][-1]["labels"]["clean_threshold_pairs"], 1)
            axs[0, col].plot(st, [c.get("interchange", np.nan) for c in r["checkpoints"]], color=f"C{s}", label=f"seed {s}")
            axs[1, col].plot(st, [c["labels"]["clean_threshold_pairs"] / n for c in r["checkpoints"]], color=f"C{s}")
        axs[0, col].set_title(f"width {w}: carry-direction interchange"); axs[0, col].legend(fontsize=7)
        axs[1, col].set_title(f"width {w}: clean threshold pairs" + (" (raw count; never learned)" if raw else " / final count"))
        for ax in axs[:, col]:
            ax.set_xscale("symlog", linthresh=50); ax.set_xlabel("step")
            if ax is axs[0, col] or not raw:
                ax.axhline(0.8 if ax is axs[0, col] else 0.5, ls="--", color="grey")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_emergence.png"), dpi=130); plt.close(fig)

    S0 = json.load(open(os.path.join(AN, "stage0.json")))
    fig, axs = plt.subplots(1, 5, figsize=(18, 3.4), sharey=True)
    for i, m in enumerate(S0):
        C, Rc = np.array(S0[m]["codes"]), np.abs(np.array(S0[m]["read_cos"]))
        axs[i].hist([Rc[C >= 1000], Rc[(C >= 0) & (C < 1000)]], bins=30, density=True, label=["threshold", "other active"])
        axs[i].set_title(f"{m}: |read cosine|"); axs[i].set_xlabel("|r|")
    axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_readers.png"), dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for arm, mk in (("A", "o-"), ("R", "s--")):
        ws = WIDTHS[arm]
        for j, key in enumerate(("cleanliness", "packed_share_multi")):
            ys = [np.nanmean([final_row(load_an(arm, w, s))["labels"][key] if final_row(load_an(arm, w, s))["labels"][key] is not None else np.nan for s in SEEDS]) for w in ws]
            axs[j].plot(ws, ys, mk, label=f"arm {arm}")
    ws = WIDTHS["A"]
    axs[2].plot(ws, [np.mean([final_row(load_an("A", w, s))["interchange"] for s in SEEDS]) for w in ws], "o-")
    for ax, t in zip(axs, ("cleanliness", "packed share (multi-position units)", "arm A carry-direction interchange")):
        ax.set_xscale("log", base=2); ax.set_title(t); ax.set_xlabel("MLP width"); ax.legend() if ax is not axs[2] else None
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_squeeze.png"), dpi=130); plt.close(fig)

    pts = np.array(R["direction"]["D4"]["points"])
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1])
    ax.set_xlabel("cleanliness"); ax.set_ylabel("debiased PC1 margin"); ax.set_title(f"D4, width 256 (rho {R['direction']['D4']['rho']})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_seeds.png"), dpi=130); plt.close(fig)


def report():
    h = prereg_check()
    R = judge(); R["prereg_sha256"] = h; R["prereg_locked"] = json.load(open(LOCK))["locked"]
    R["settings"] = {"steps": STEPS, "widths": WIDTHS, "seeds": SEEDS, "extra_seeds_w256": EXTRA, "tables_R": TAB.tolist()}
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    figures(R)
    L = ["# Cammarata's clean-neuron test × the carry direction: verdicts", "",
         f"Generated by `python clean_carry.py report` from `results.json` (PREREG sha256 `{h[:16]}…`, locked {R['prereg_locked']}).", "",
         "## Validity gates", "", "| gate | pass | detail |", "|---|---|---|"]
    for k, v in R["validity"].items():
        L.append(f"| {k} | {'yes' if v['pass'] else 'NO'} | " + ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("pass", "seeds", "values")) + " |")
    L += ["", "## Verdicts", "", "| # | verdict |", "|---|---|"]
    for grp in ("cammarata", "direction"):
        for k, v in R[grp].items():
            if isinstance(v, dict) and "verdict" in v:
                L.append(f"| {k} | {v['verdict']} |")
    open(os.path.join(OUT, "VERDICTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ======================= exploratory (after the verdicts; never judged) =======================
def explore():
    """EXPLORATORY, added after seeing V4 fail and the narrow arm-A models not learn:
    1. a basis-sensitive variant of cleanliness: share of active (unit, k) pairs whose activity is a sharp step
       (best Boolean R^2 >= 0.8; count ramps NOT accepted), real units vs the same 10 random rotations as V4;
    2. teacher-forced per-bit accuracy of the narrow arm-A models (exact sums were 0-13%)."""
    D = data(); a, b, L = D["a_te"], D["b_te"], D["Lte"]
    out = {"note": "exploratory, defined after the verdicts; not pre-registered"}

    def step_share(H, combo):
        num = den = 0
        for k in KS:
            Hk = H[:, k].astype(float)
            act = ((Hk > 0).mean(0) >= 0.05) & (Hk.std(0) > 1e-6)
            if not act.any():
                continue
            R2, _ = TG.bool_fit(Hk[:, act], combo[:, k], 3)
            num += int((R2.max(0) >= 0.8).sum()); den += int(act.sum())
        return num / max(den, 1)
    rows = {}
    for s in range(5):
        model, _ = TG.load("B", s)
        H, _, _ = hidden(model, TG.Flat, a, b)
        combo, _ = inputs("A", a, b, L)
        rng = np.random.default_rng(1100 + s); rot = []
        for _ in range(10):
            Q, _ = np.linalg.qr(rng.normal(size=(H.shape[2], H.shape[2])))
            rot.append(step_share(np.einsum("nku,uv->nkv", H.astype(float), Q), combo))
        rows[f"B:{s}"] = {"step_share_real": step_share(H, combo), "step_share_rotated": float(np.mean(rot))}
        print("step share", f"B:{s}", rows[f"B:{s}"], flush=True)
    out["step_labels_vs_rotation"] = rows
    nar = {}
    for w in (32, 16, 8, 4):
        for s in SEEDS:
            model = load_ckpt("A", w, s, STEPS)
            _, _, O = hidden(model, TG.Flat, a, b)
            nar[f"{w}:{s}"] = float(np.mean([np.mean(O[:, k] == L["s"][:, k]) for k in range(NB + 1)]))
        print("narrow", w, [round(nar[f'{w}:{s}'], 3) for s in SEEDS], flush=True)
    out["narrow_arm_A_teacher_forced_bit_acc"] = nar
    json.dump(out, open(os.path.join(OUT, "exploratory.json"), "w"), indent=1)


# ======================= main =======================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg-check", "stage0", "train", "analyse", "report", "explore"])
    ap.add_argument("--only", default=None)
    A_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if A_.cmd == "prereg-check":
        prereg_check(lock=True)
    elif A_.cmd == "stage0":
        prereg_check(); stage0()
    elif A_.cmd == "train":
        prereg_check()
        if A_.only:
            arm, w, s = A_.only.split(":"); train_width(arm, int(w), int(s))
        else:
            todo = sorted([r for r in runs() if not os.path.exists(ckpt(*r, STEPS))], key=lambda r: -r[1])
            pool(todo, "train")
    elif A_.cmd == "analyse":
        prereg_check()
        if A_.only:
            arm, w, s = A_.only.split(":"); analyse_model(arm, int(w), int(s))
        else:
            todo = [r for r in runs() if not os.path.exists(os.path.join(AN, name_of(*r) + ".json"))]
            todo.sort(key=lambda r: (-(len(schedule(*r)) * (3 if r[0] == "A" else 1) * (2 if r[1] == 256 else 1))))
            pool(todo, "analyse")
    elif A_.cmd == "explore":
        prereg_check(); explore()
    else:
        report()
