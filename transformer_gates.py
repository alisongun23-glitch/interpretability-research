"""Is a logic gate inside a transformer the same as a gate neuron?

    python transformer_gates.py train                # trains the missing models, several processes at once
    python transformer_gates.py train --only B:2     # one model (family:seed)
    python transformer_gates.py analyse              # steps 2-6 from saved models -> results.json, figures/
    python tests/test_circuits.py                    # checks for the pieces below

A 1-layer transformer adding binary numbers lowest bit first has to compute the carry into bit k.
Over the previous digit (a, b, s) that carry is one threshold function, c = [a + b - s >= 1], the
kind of function one gate neuron of results_minigates/library.json computes. Where is it inside the
transformer, does it look like a gate neuron, and can the two be swapped?

MODELS  B   TinyGPT, 1 layer, learned positions, a0..a9 + b0..b9 = s0..s10. Seed 1 is
            results_transformer/lsb-L1.npz; seeds 0, 2, 3, 4 are trained here with its recipe.
        C2  RelPosGPT (below): no position embedding, a learned per-head bias on the relative
            offset, triples a0 b0 s0 a1 b1 s1 ... a_n b_n s_n (a_n = b_n = 0). Every input of
            s_k then sits at a fixed offset from the query, so the model can run on longer sums.

Pre-registered predictions: results_transformer_gates/PREREG.md. Report: REPORT.md.
"""
import argparse, json, os, subprocess, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np

from tinygpt import TinyGPT
import run_transformer as RT                        # main-guarded: importing trains nothing
from gatesnake import gates as G

OUT = "results_transformer_gates"
MODELS_DIR = os.path.join(OUT, "models")
NB = 10                                             # training length (bits per operand)
SEEDS = {"B": [0, 1, 2, 3, 4], "C2": [0, 1, 2]}
STEPS = {"B": 2500, "C2": 4000}
BATCH = 128
LIB = json.load(open("results_minigates/library.json"))


# ======================= task layouts =======================
class Flat:
    """B: a0..a9 + b0..b9 = s0..s10 (run_transformer's lsb format, fixed at 10 bits)."""
    name = "B"

    @staticmethod
    def tokens(a, b, n=NB):
        assert n == NB, "B has learned positions for 10 bits only"
        return RT.encode(a, b, "lsb")

    @staticmethod
    def q(k, n=NB): return RT.POS_EQ + k            # predicts s_k; its token is s_(k-1), '=' for k = 0
    @staticmethod
    def pos_a(k, n=NB): return k
    @staticmethod
    def pos_b(k, n=NB): return NB + 1 + k
    @staticmethod
    def pos_s(k, n=NB): return RT.POS_EQ + 1 + k


class Inter:
    """C2: triples a_k b_k s_k for k = 0..n, with a_n = b_n = 0; s_k is predicted at 3k + 1."""
    name = "C2"

    @staticmethod
    def tokens(a, b, n=NB):
        A, B_, S = RT.bits(a, n + 1), RT.bits(b, n + 1), RT.bits(a + b, n + 1)
        return np.stack([A, B_, S], 2).reshape(len(a), 3 * (n + 1))

    @staticmethod
    def q(k, n=NB): return 3 * k + 1                 # token there is b_k
    @staticmethod
    def pos_a(k, n=NB): return 3 * k
    @staticmethod
    def pos_b(k, n=NB): return 3 * k + 1
    @staticmethod
    def pos_s(k, n=NB): return 3 * k + 2


LAYOUT = {"B": Flat, "C2": Inter}


def problems(n_prob, n, rng):
    return rng.integers(0, 2 ** n, n_prob), rng.integers(0, 2 ** n, n_prob)


def carry_bits(a, b, n):
    """(problems, n + 1): carry INTO bit k, k = 0..n."""
    return np.stack([RT.carry_into(a, b, k) for k in range(n + 1)], 1)


# ======================= the C2 model =======================
class RelPosGPT(TinyGPT):
    """TinyGPT without W_pos. Head h adds rel[h, min(i - j, R - 1)] to the score of query i on key j,
    so a head can only prefer keys by their distance. Forward and backward are TinyGPT's with that
    bias added (same hook names); gradients checked by rel_gradient_check()."""
    R = 64

    def __init__(self, d_vocab, d_model, n_heads, n_layers, d_mlp, seed=0, dtype=np.float32):
        super().__init__(d_vocab, 1, d_model, n_heads, n_layers, d_mlp, seed=seed, dtype=dtype)
        del self.P["W_pos"]
        for l in range(n_layers):
            self.P[f"{l}.rel"] = np.zeros((n_heads, self.R), dtype)
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}

    def _offsets(self, T):
        return np.clip(np.arange(T)[:, None] - np.arange(T)[None], 0, self.R - 1)

    def forward(self, tokens, hooks=None, cache=False):
        P, c = self.P, self.cfg
        hooks = hooks or {}
        B, T = tokens.shape
        H, dh = c["n_heads"], c["d_head"]
        acts, tape = {}, {"tokens": tokens}
        x = P["W_E"][tokens]
        mask = np.triu(np.ones((T, T), bool), 1)
        off = self._offsets(T)
        for l in range(c["n_layers"]):
            name = f"blocks.{l}.hook_resid_pre"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            ln1, ln1_tape = self._ln(x, P[f"{l}.ln1.w"], P[f"{l}.ln1.b"])
            ln1h = ln1[:, None]
            q = ln1h @ P[f"{l}.W_Q"][None] + P[f"{l}.b_Q"][None, :, None, :]
            k = ln1h @ P[f"{l}.W_K"][None] + P[f"{l}.b_K"][None, :, None, :]
            v = ln1h @ P[f"{l}.W_V"][None] + P[f"{l}.b_V"][None, :, None, :]
            scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh) + P[f"{l}.rel"][:, off][None]
            scores = np.where(mask[None, None], -1e9, scores)
            scores = scores - scores.max(-1, keepdims=True)
            A = np.exp(scores); A = A / A.sum(-1, keepdims=True)
            name = f"blocks.{l}.attn.hook_pattern"
            if name in hooks: A = hooks[name](A)
            acts[name] = A
            z = (A @ v).transpose(0, 2, 1, 3)
            name = f"blocks.{l}.attn.hook_z"
            if name in hooks: z = hooks[name](z)
            acts[name] = z
            x = x + z.reshape(B, T, H * dh) @ P[f"{l}.W_O"].reshape(H * dh, -1) + P[f"{l}.b_O"]
            name = f"blocks.{l}.hook_resid_mid"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            ln2, ln2_tape = self._ln(x, P[f"{l}.ln2.w"], P[f"{l}.ln2.b"])
            pre = ln2 @ P[f"{l}.W_in"] + P[f"{l}.b_in"]
            h = np.maximum(pre, 0)
            acts[f"blocks.{l}.mlp.hook_post"] = h
            x = x + h @ P[f"{l}.W_out"] + P[f"{l}.b_out"]
            name = f"blocks.{l}.hook_resid_post"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            tape[l] = dict(ln1=ln1, ln1_tape=ln1_tape, q=q, k=k, v=v, A=A, z=z, ln2=ln2,
                           ln2_tape=ln2_tape, pre=pre, h=h)
        lnf, lnf_tape = self._ln(x, P["ln_f.w"], P["ln_f.b"])
        logits = lnf @ P["W_U"] + P["b_U"]
        acts["ln_final.hook_normalized"] = lnf
        tape["lnf"] = lnf; tape["lnf_tape"] = lnf_tape
        if cache:
            return logits, acts, tape
        return logits

    def loss_and_grads(self, tokens, loss_mask):
        P, c = self.P, self.cfg
        H, dh = c["n_heads"], c["d_head"]
        logits, acts, tape = self.forward(tokens, cache=True)
        B, T, V = logits.shape
        targets = tokens[:, 1:]
        lg = logits[:, :-1]
        m = loss_mask[:, :-1].astype(logits.dtype)
        n = m.sum()
        lg = lg - lg.max(-1, keepdims=True)
        p = np.exp(lg); p /= p.sum(-1, keepdims=True)
        loss = -(np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None], targets] + 1e-12) * m).sum() / n
        dlogits = np.zeros_like(logits)
        dp = p.copy(); dp[np.arange(B)[:, None], np.arange(T - 1)[None], targets] -= 1
        dlogits[:, :-1] = dp * m[..., None] / n
        Gr = {}
        Gr["W_U"] = tape["lnf"].reshape(-1, c["d_model"]).T @ dlogits.reshape(-1, V)
        Gr["b_U"] = dlogits.reshape(-1, V).sum(0)
        dx, Gr["ln_f.w"], Gr["ln_f.b"] = self._ln_back(dlogits @ P["W_U"].T, tape["lnf_tape"])
        off = self._offsets(T); low = np.tril(np.ones((T, T), bool))
        for l in reversed(range(c["n_layers"])):
            tp = tape[l]
            dh_ = dx @ P[f"{l}.W_out"].T
            Gr[f"{l}.W_out"] = tp["h"].reshape(-1, c["d_mlp"]).T @ dx.reshape(-1, c["d_model"])
            Gr[f"{l}.b_out"] = dx.reshape(-1, c["d_model"]).sum(0)
            dpre = dh_ * (tp["pre"] > 0)
            Gr[f"{l}.W_in"] = tp["ln2"].reshape(-1, c["d_model"]).T @ dpre.reshape(-1, c["d_mlp"])
            Gr[f"{l}.b_in"] = dpre.reshape(-1, c["d_mlp"]).sum(0)
            dx2, Gr[f"{l}.ln2.w"], Gr[f"{l}.ln2.b"] = self._ln_back(dpre @ P[f"{l}.W_in"].T, tp["ln2_tape"])
            dx = dx + dx2
            Bq, Tq = dx.shape[:2]
            dz = (dx @ P[f"{l}.W_O"].reshape(H * dh, -1).T).reshape(Bq, Tq, H, dh)
            Gr[f"{l}.W_O"] = (tp["z"].reshape(-1, H * dh).T @ dx.reshape(-1, c["d_model"])).reshape(H, dh, -1)
            Gr[f"{l}.b_O"] = dx.reshape(-1, c["d_model"]).sum(0)
            dzh = dz.transpose(0, 2, 1, 3)
            dA = dzh @ tp["v"].transpose(0, 1, 3, 2)
            dv = tp["A"].transpose(0, 1, 3, 2) @ dzh
            A = tp["A"]
            dS = A * (dA - (dA * A).sum(-1, keepdims=True))         # gradient of the scores (bias included)
            S_sum = dS.sum(0)                                        # (H, T, T)
            Gr[f"{l}.rel"] = np.stack([np.bincount(off[low], weights=S_sum[h][low], minlength=self.R)
                                       for h in range(H)]).astype(P[f"{l}.rel"].dtype)
            dscores = dS / np.sqrt(dh)
            dq = dscores @ tp["k"]
            dk = dscores.transpose(0, 1, 3, 2) @ tp["q"]
            ln1T = tp["ln1"].transpose(0, 2, 1)[:, None]
            Gr[f"{l}.W_Q"] = (ln1T @ dq).sum(0); Gr[f"{l}.b_Q"] = dq.sum((0, 2))
            Gr[f"{l}.W_K"] = (ln1T @ dk).sum(0); Gr[f"{l}.b_K"] = dk.sum((0, 2))
            Gr[f"{l}.W_V"] = (ln1T @ dv).sum(0); Gr[f"{l}.b_V"] = dv.sum((0, 2))
            dln1 = ((dq @ P[f"{l}.W_Q"].transpose(0, 2, 1)[None]).sum(1) + (dk @ P[f"{l}.W_K"].transpose(0, 2, 1)[None]).sum(1)
                    + (dv @ P[f"{l}.W_V"].transpose(0, 2, 1)[None]).sum(1))
            dx1, Gr[f"{l}.ln1.w"], Gr[f"{l}.ln1.b"] = self._ln_back(dln1, tp["ln1_tape"])
            dx = dx + dx1
        Gr["W_E"] = np.zeros_like(P["W_E"])
        np.add.at(Gr["W_E"], tokens.ravel(), dx.reshape(-1, c["d_model"]))
        return float(loss), Gr


def rel_gradient_check(seed=0):
    """Central finite differences on a tiny float64 RelPosGPT; returns the worst relative error."""
    m = RelPosGPT(d_vocab=5, d_model=8, n_heads=2, n_layers=2, d_mlp=12, seed=seed, dtype=np.float64)
    r = np.random.default_rng(seed)
    for k in m.P:
        m.P[k] = m.P[k] + r.normal(0, 0.3, m.P[k].shape)
    tokens = r.integers(0, 5, (3, 9)); mask = r.random((3, 9)) < 0.6; mask[:, -1] = False
    _, Gr = m.loss_and_grads(tokens, mask)
    worst = 0.0
    for k in m.P:
        for _ in range(3):
            idx = tuple(r.integers(0, s) for s in m.P[k].shape)
            if k.endswith(".rel"):
                idx = (idx[0], int(r.integers(0, 9)))             # offsets that occur in a length-9 sequence
            old = m.P[k][idx]; eps = 1e-5
            m.P[k][idx] = old + eps; lp, _ = m.loss_and_grads(tokens, mask)
            m.P[k][idx] = old - eps; lm, _ = m.loss_and_grads(tokens, mask)
            m.P[k][idx] = old
            num = (lp - lm) / (2 * eps); ana = Gr[k][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


# ======================= models: make, train, load =======================
def make_model(fam, seed):
    if fam == "B":
        return TinyGPT(d_vocab=4, n_ctx=RT.T, d_model=64, n_heads=4, n_layers=1, d_mlp=256, seed=seed)
    return RelPosGPT(d_vocab=4, d_model=64, n_heads=4, n_layers=1, d_mlp=256, seed=seed)


def ckpt_path(fam, seed):
    if fam == "B" and seed == 1:
        return os.path.join("results_transformer", "lsb-L1.npz")      # read only
    return os.path.join(MODELS_DIR, f"{fam}-seed{seed}.npz")


def trained_steps(fam, seed):
    p = ckpt_path(fam, seed)
    return int(np.load(p)["step"]) if os.path.exists(p) else 0


def train(fam, seed):
    """run_transformer.train's recipe: batches of 128 random 10-bit problems, AdamW (wd 0.01), lr 1e-3
    with 100 warm-up steps and cosine decay to 0.1x, loss on the sum tokens only. Resumable."""
    path, steps, lay = ckpt_path(fam, seed), STEPS[fam], LAYOUT[fam]
    assert not (fam == "B" and seed == 1), "B seed 1 is run_transformer's lsb-L1, not retrained"
    model = make_model(fam, seed); log = []; step0 = 0
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        model.load({k: z[k] for k in model.P}); step0 = int(z["step"]); log = [tuple(x) for x in z["log"]]
        model.m = {k: z["m_" + k] for k in model.P}; model.v = {k: z["v_" + k] for k in model.P}; model.t = step0
    r = np.random.default_rng([seed, 1000 + step0])
    T = lay.tokens(np.zeros(1, int), np.zeros(1, int)).shape[1]
    mask = np.zeros((BATCH, T), bool); mask[:, [lay.q(k) for k in range(NB + 1)]] = True
    t0 = time.time()
    for step in range(step0, steps):
        a, b = problems(BATCH, NB, r)
        lr = 1e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        loss, Gr = model.loss_and_grads(lay.tokens(a, b), mask)
        model.adam_step(Gr, lr, wd=0.01)
        if step % 50 == 0:
            log.append((step, loss))
            print(f"{fam}:{seed} step {step} loss {loss:.5f}  {time.time() - t0:.0f}s", flush=True)
        if step % 250 == 249 or step == steps - 1:
            np.savez(path, step=step + 1, log=np.array(log), **model.P,
                     **{"m_" + k: v for k, v in model.m.items()}, **{"v_" + k: v for k, v in model.v.items()})
    return model


def load(fam, seed):
    z = np.load(ckpt_path(fam, seed), allow_pickle=True)
    model = make_model(fam, seed)
    model.load({k: z[k] for k in model.P})
    return model, [tuple(x) for x in z["log"]]


def train_all(only=None):
    """Train every missing model, min(7, cores - 1) processes at once, one BLAS thread each."""
    todo = [(f, s) for f in SEEDS for s in SEEDS[f] if not (f == "B" and s == 1) and trained_steps(f, s) < STEPS[f]]
    if only:
        f, s = only.split(":"); train(f, int(s)); return
    cores = os.cpu_count() or 2
    n_par = max(1, min(7, cores - 1))
    print(f"{cores} cores -> {n_par} training processes at once; to train: {todo}", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < n_par:
            f, s = todo.pop(0)
            logf = open(os.path.join(MODELS_DIR, f"{f}-seed{s}.log"), "a")
            running.append((f, s, subprocess.Popen([sys.executable, __file__, "train", "--only", f"{f}:{s}"],
                                                   stdout=logf, stderr=subprocess.STDOUT)))
        time.sleep(5)
        for item in list(running):
            if item[2].poll() is not None:
                running.remove(item)
                print(f"{item[0]}:{item[1]} finished (exit {item[2].returncode}), steps {trained_steps(item[0], item[1])}", flush=True)


# ======================= decoding and hooks =======================
def generate(model, lay, a, b, n=NB, hooks_fn=None, bs=2000):
    """Greedy decoding of s_0..s_n with the operands given. hooks_fn(k, rows) -> hooks for the
    forward pass that predicts s_k on problems `rows` (a slice), or None. Returns predicted sums."""
    out = []
    for i in range(0, len(a), bs):
        rows = slice(i, min(i + bs, len(a)))
        tok = lay.tokens(a[rows], b[rows], n).copy()
        qs = [lay.q(k, n) for k in range(n + 1)]
        tok[:, [q + 1 for q in qs]] = 0
        for k, q in enumerate(qs):
            hooks = hooks_fn(k, rows) if hooks_fn else None
            lg = model.forward(tok[:, :q + 1], hooks=hooks)
            tok[:, q + 1] = lg[:, q, :2].argmax(1)
        s = tok[:, [q + 1 for q in qs]]
        out.append((s * (2 ** np.arange(n + 1))).sum(1))
    return np.concatenate(out)


def wilson(k, n, z=1.96):
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def acc_ci(ok):
    ok = np.asarray(ok, bool)
    lo, hi = wilson(int(ok.sum()), len(ok))
    return {"acc": float(ok.mean()), "n": int(len(ok)), "ci95": [lo, hi]}


# ======================= gate neurons and the gate adder =======================
def lib_gate(t, i=0):
    """Library gate neuron number i of type t as a function of analog inputs."""
    pool = LIB["gates"][t]["pool"]
    p = {k: np.array(v, float) for k, v in pool[i % len(pool)].items()}
    return lambda *xs: G.forward(p, np.stack(np.broadcast_arrays(*xs), -1))


def gate_adder(A, B_, carry_override=None, carry_gate=None):
    """Ripple adder of library gate neurons, analog, nothing snapped.
    A, B_: (problems, n) bits as floats, lowest first. Bit i uses pool member i of each gate type.
    s_i = XOR(XOR(a_i, b_i), c_i);  c_(i+1) = OR(AND(a_i, b_i), AND(XOR(a_i, b_i), c_i)).
    carry_override=(k, values) forces the carry INTO bit k (k = n forces the top sum bit).
    carry_gate(a, b, s) -> next carry replaces the AND/OR carry logic (6b: the transformer's gate).
    Returns (sums (problems, n + 1), carries (problems, n + 1)), carry k = carry INTO bit k."""
    A = np.asarray(A, float); B_ = np.asarray(B_, float)
    n = A.shape[1]
    c = np.zeros(len(A)); S, C = [], []
    for i in range(n):
        if carry_override is not None and carry_override[0] == i:
            c = np.asarray(carry_override[1], float)
        C.append(c)
        XOR, AND, OR = lib_gate("XOR", i), lib_gate("AND", i), lib_gate("OR", i)
        x = XOR(A[:, i], B_[:, i]); s = XOR(x, c); S.append(s)
        c = carry_gate(A[:, i], B_[:, i], s) if carry_gate else OR(AND(A[:, i], B_[:, i]), AND(x, c))
    if carry_override is not None and carry_override[0] == n:
        c = np.asarray(carry_override[1], float)
    C.append(c); S.append(c)
    return np.stack(S, 1), np.stack(C, 1)


def bits_to_int(bits):
    return (np.round(bits).astype(np.int64) * (2 ** np.arange(bits.shape[1]))).sum(1)


def bitsf(v, n):
    return RT.bits(v, n).astype(float)


def train_neuron(X, y, margin=4.0, seed=0, max_steps=20000, lr=0.05, b1=0.9, b2=0.999):
    """gatesnake.gates.train for one sigmoid neuron on any truth table (X rows, y targets):
    full-batch Adam, stop at the first step where every row is right with |logit| >= margin."""
    r = np.random.default_rng(seed)
    p = {"Wh": np.zeros((X.shape[1], 0)), "bh": np.zeros(0), "wx": r.normal(0, 1.0, X.shape[1]),
         "wh": np.zeros(0), "bo": np.zeros(())}
    th = np.concatenate([p["wx"], [0.0]]); m = np.zeros_like(th); v = np.zeros_like(th)
    X1 = np.hstack([X, np.ones((len(X), 1))]); sgn = 2 * y - 1
    for t in range(1, max_steps + 1):
        L = X1 @ th
        if np.all(L * sgn >= margin):
            break
        g = X1.T @ (sig(L) - y) / len(X)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        th = th - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    p["wx"], p["bo"] = th[:-1], np.array(th[-1])
    return p


# ======================= analysis tools =======================
def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


HOOKS = {"resid_pre": "blocks.0.hook_resid_pre", "attn_z": "blocks.0.attn.hook_z",
         "resid_mid": "blocks.0.hook_resid_mid", "mlp_post": "blocks.0.mlp.hook_post",
         "resid_post": "blocks.0.hook_resid_post"}
MID, POST = HOOKS["resid_mid"], HOOKS["resid_post"]


def collect(model, lay, a, b, n=NB, bs=1000):
    """Teacher-forced activations, float32, at the query of every sum bit ("q": (N, n + 1, d), slot k = q_k)
    and at the operand / sum tokens ("a", "b", "s": (N, n, d), slot j = digit j); "attn": (N, n + 1, H, T)
    = the attention rows of the queries."""
    roles = {"q": [lay.q(k, n) for k in range(n + 1)], "a": [lay.pos_a(j, n) for j in range(n)],
             "b": [lay.pos_b(j, n) for j in range(n)], "s": [lay.pos_s(j, n) for j in range(n)]}
    out = {r: {h: [] for h in HOOKS} for r in roles}; att = []
    for i in range(0, len(a), bs):
        _, acts, _ = model.forward(lay.tokens(a[i:i + bs], b[i:i + bs], n), cache=True)
        for h, name in HOOKS.items():
            x = acts[name]
            x = x.reshape(x.shape[0], x.shape[1], -1)
            for r, pos in roles.items():
                out[r][h].append(x[:, pos].astype(np.float32))
        att.append(acts["blocks.0.attn.hook_pattern"][:, :, roles["q"], :].transpose(0, 2, 1, 3).astype(np.float32))
    res = {r: {h: np.concatenate(v) for h, v in d.items()} for r, d in out.items()}
    res["attn"] = np.concatenate(att)
    return res


def labels(a, b, n=NB):
    """Bits as (N, n + 1): a, b, s (sum), c (carry INTO bit k)."""
    return {"a": RT.bits(a, n + 1), "b": RT.bits(b, n + 1), "s": RT.bits(a + b, n + 1), "c": carry_bits(a, b, n)}


class Probe:
    """L2-regularised logistic regression fitted by IRLS on standardised features. Near-constant features
    (sd < 1e-4, e.g. dead ReLU units) get weight 0: standardising them would blow them up and IRLS diverges."""

    def __init__(self, X, y, l2=1.0, iters=40):
        X = np.asarray(X, float); y = np.asarray(y, float)
        sd = X.std(0)
        self.mu = X.mean(0); self.sd = np.where(sd < 1e-4, np.inf, sd)
        Z = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        reg = np.full(Z.shape[1], l2); reg[-1] = 0.0
        w = np.zeros(Z.shape[1])

        def loss(w):
            z = Z @ w
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(reg * w * w)
        for _ in range(iters):
            p = sig(Z @ w)
            Hs = (Z * (p * (1 - p) + 1e-9)[:, None]).T @ Z + np.diag(reg)
            step = np.linalg.solve(Hs, Z.T @ (p - y) + reg * w)
            t, f0 = 1.0, loss(w)
            while t > 1e-4 and loss(w - t * step) > f0:      # step-halving keeps Newton from diverging
                t /= 2
            w = w - t * step
            if np.abs(t * step).max() < 1e-5:
                break
        self.w = w

    def logit(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.w[:-1] + self.w[-1]

    def predict(self, X):
        return (self.logit(X) > 0).astype(int)

    def acc(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def direction(self):
        d = self.w[:-1] / self.sd
        return d / np.linalg.norm(d)


def stack(Xr, Y, slots, ks):
    """Rows for the k in ks: activations Xr[:, slots(k)] and labels Y(k), stacked over k."""
    return np.concatenate([Xr[:, slots(k)] for k in ks]), np.concatenate([Y(k) for k in ks])


def subsample(X, y, cap, seed):
    if len(X) <= cap:
        return X, y
    i = np.random.default_rng(seed).choice(len(X), cap, replace=False)
    return X[i], y[i]


def probe_acc(Xtr, ytr, Xte, yte, cap_tr=30000, cap_te=20000, seed=0):
    """Test accuracy of a probe; None if the training labels are constant."""
    Xtr, ytr = subsample(Xtr, ytr, cap_tr, seed); Xte, yte = subsample(Xte, yte, cap_te, seed + 1)
    if ytr.min() == ytr.max():
        return None
    return Probe(Xtr, ytr).acc(Xte, yte)


def bool_fit(act, idx, n_in=3):
    """act (N, U), idx (N,) the input combo (first input most significant). For each of the
    2^(2^n_in) - 2 non-constant Boolean f: R^2 of act ~ alpha + beta * f(combo).
    Returns R2 (n_f, U) and masks (n_f, 2^n_in), masks[f, c] = f(combo c)."""
    m = 2 ** n_in
    masks = np.array([[(f >> c) & 1 for c in range(m)] for f in range(1, 2 ** m - 1)], float)
    act = np.asarray(act, float)
    cnt = np.bincount(idx, minlength=m).astype(float)
    sums = np.stack([act[idx == c].sum(0) for c in range(m)])
    N = len(act); tot = act.sum(0); mean = tot / N
    sst = ((act - mean) ** 2).sum(0)
    n1 = np.maximum(masks @ cnt, 1e-9); n0 = np.maximum(N - masks @ cnt, 1e-9)
    s1 = masks @ sums; s0 = tot - s1
    ssb = n1[:, None] * (s1 / n1[:, None] - mean) ** 2 + n0[:, None] * (s0 / n0[:, None] - mean) ** 2
    return ssb / np.maximum(sst, 1e-12), masks


def mask_of(fn, n_in=3):
    """Truth table of fn(bits...) over combos, first input most significant."""
    return np.array([fn(*[(c >> (n_in - 1 - j)) & 1 for j in range(n_in)]) for c in range(2 ** n_in)], float)


CARRY_MASK = mask_of(lambda a, b, s: int(a + b - s >= 1))
COUNT_THRESH = [mask_of(lambda a, b, c, t=t: int(a + b + c >= t)) for t in (1, 2, 3)]


def bimodality(x):
    """Sarle's bimodality coefficient (> 0.555 suggests two modes)."""
    x = np.asarray(x, float); n = len(x)
    sd = x.std()
    if sd < 1e-12 or n < 4:
        return float("nan")
    z = (x - x.mean()) / sd
    g = (z ** 3).mean(); k = (z ** 4).mean() - 3
    return float((g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def switch_stats(act, idx, mask):
    """Level gap / within-level std, for the split given by mask over the combos."""
    on = mask[idx] == 1
    a1, a0 = act[on], act[~on]
    sse = ((a1 - a1.mean()) ** 2).sum() + ((a0 - a0.mean()) ** 2).sum()
    sd = np.sqrt(sse / max(len(act) - 2, 1))
    return float(abs(a1.mean() - a0.mean()) / max(sd, 1e-12))


def greedy_readout(Xtr, ytr, Xte, yte, target=0.99, cap=40):
    """Fewest units for a least-squares readout (y in {0,1}, threshold 0.5) reaching `target` test
    accuracy: greedy forward selection on train SSE. Returns (units needed or None, accuracy curve)."""
    Xtr = np.hstack([np.ones((len(Xtr), 1)), np.asarray(Xtr, float)])
    Xte = np.hstack([np.ones((len(Xte), 1)), np.asarray(Xte, float)])
    Gm = Xtr.T @ Xtr; Xy = Xtr.T @ ytr; yy = float(ytr @ ytr)
    live = [j for j in range(1, Xtr.shape[1]) if Xtr[:, j].std() > 1e-9]
    sel, curve = [0], []
    for _ in range(cap):
        best, bsse = None, np.inf
        for j in live:
            if j in sel:
                continue
            S = sel + [j]
            beta = np.linalg.solve(Gm[np.ix_(S, S)] + 1e-8 * np.eye(len(S)), Xy[S])
            sse = yy - beta @ Xy[S]
            if sse < bsse:
                best, bsse = j, sse
        if best is None:
            break
        sel.append(best)
        beta = np.linalg.solve(Gm[np.ix_(sel, sel)] + 1e-8 * np.eye(len(sel)), Xy[sel])
        acc = float(np.mean((Xte[:, sel] @ beta > 0.5) == (yte == 1)))
        curve.append(acc)
        if acc >= target:
            return len(sel) - 1, curve
    return None, curve


def two_means(z, iters=100):
    """1-D 2-means, no labels: (low level, high level, threshold = midpoint)."""
    z = np.asarray(z, float)
    lo, hi = z.min(), z.max()
    for _ in range(iters):
        t = (lo + hi) / 2
        nlo, nhi = z[z <= t].mean(), z[z > t].mean()
        if nlo == lo and nhi == hi:
            break
        lo, hi = nlo, nhi
    return float(lo), float(hi), float((lo + hi) / 2)


def trilinear(table):
    """Analog gate from an 8-entry table over (a, b, s) (index 4a + 2b + s): multilinear interpolation."""
    T = np.asarray(table, float)

    def g(a, b, s):
        a, b, s = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(s, float))
        out = np.zeros(a.shape)
        for c in range(8):
            ca, cb, cs = (c >> 2) & 1, (c >> 1) & 1, c & 1
            out = out + T[c] * (a if ca else 1 - a) * (b if cb else 1 - b) * (s if cs else 1 - s)
        return out
    return g


def restores3(gate, table01, eps, pts=9):
    """gatesnake.gates.restores for a 3-input analog gate: every input within eps of its rail gives an
    output within eps of table01[combo]. Returns (passes, worst deviation)."""
    g = np.linspace(0, eps, pts); worst = 0.0
    for c in range(8):
        axes = [g if ((c >> (2 - j)) & 1) == 0 else 1 - g for j in range(3)]
        X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        worst = max(worst, float(np.abs(gate(X[:, 0], X[:, 1], X[:, 2]) - table01[c]).max()))
    return worst <= eps, worst


def exact_ripple(A, B_):
    """Exact sum bits (problems, n + 1) of bit arrays, lowest first."""
    A = A.astype(int); B_ = B_.astype(int); c = np.zeros(len(A), int); out = []
    for i in range(A.shape[1]):
        out.append(A[:, i] ^ B_[:, i] ^ c); c = (A[:, i] & B_[:, i]) | (c & (A[:, i] ^ B_[:, i]))
    out.append(c)
    return np.stack(out, 1)


def mid_at(model, tok, q, hooks=None):
    """resid_mid at position q (teacher-forced prefix up to q)."""
    keep = {}

    def grab(x):
        keep["x"] = x[:, q].copy(); return x
    h = dict(hooks or {})
    if MID in h:
        f = h[MID]; h[MID] = lambda x: grab(f(x))
    else:
        h[MID] = grab
    lg = model.forward(tok[:, :q + 1], hooks=h)
    return keep["x"].astype(float), lg[:, q, :2].argmax(1)


def patch_q(q, fn):
    """Hook on resid_mid that replaces x[:, q] by fn(x[:, q])."""
    def hk(x):
        x = x.copy(); x[:, q] = fn(x[:, q].astype(float)); return x
    return {MID: hk}


def opposite_pairs(n_pairs, k, rng, n=NB):
    """Base and source problems that agree on every bit >= k and have the opposite carry into bit k
    (run_transformer.analyse's pair builder, vectorised)."""
    a0, b0 = problems(n_pairs, n, rng)
    lo = 2 ** k; c0 = RT.carry_into(a0, b0, k)
    x = np.zeros(n_pairs, np.int64); y = np.zeros(n_pairs, np.int64); todo = np.arange(n_pairs)
    while len(todo):
        xx, yy = rng.integers(0, lo, len(todo)), rng.integers(0, lo, len(todo))
        ok = ((xx + yy) >= lo).astype(int) != c0[todo]
        x[todo[ok]] = xx[ok]; y[todo[ok]] = yy[ok]; todo = todo[~ok]
    return a0, b0, a0 - a0 % lo + x, b0 - b0 % lo + y


# ======================= the analysis, step by step =======================
def flip_dirs(model, lay, fam, seed, n_pairs=2000):
    """Label-free carry-flip directions: top-4 PCs of x_src - x_base (resid_mid at q_k), PC1 signed so
    base c = 0 -> source c = 1 is positive. B: one set per k. C2: one set pooled over k (used at any k).
    Returns {k: (d, 4)} for k = 1..NB, and the raw differences (for the 6b table)."""
    rng = np.random.default_rng(500 + seed)
    D = {}
    for k in range(1, NB + 1):
        a0, b0, a1, b1 = opposite_pairs(n_pairs, k, rng)
        q = lay.q(k)
        x0, _ = mid_at(model, lay.tokens(a0, b0), q); x1, _ = mid_at(model, lay.tokens(a1, b1), q)
        D[k] = (x1 - x0, RT.carry_into(a0, b0, k))

    def pcs(d, c0):
        _, _, Vt = np.linalg.svd(d - d.mean(0), full_matrices=False)
        U = Vt[:4].T.copy()
        if (d[c0 == 0] @ U[:, 0]).mean() < 0:
            U[:, 0] *= -1
        return U
    if fam == "B":
        return {k: pcs(*D[k]) for k in D}
    U = pcs(np.concatenate([D[k][0] for k in D]), np.concatenate([D[k][1] for k in D]))
    return {k: U for k in D}


def kk(k):
    """Index into the per-k direction sets (C2's pooled set is stored under every k <= NB)."""
    return min(k, NB)


def step_accuracy(model, lay, fam, n_test=10000):
    out = {}
    for n in ([NB] if fam == "B" else [NB, 12, 16, 20]):
        a, b = problems(n_test, n, np.random.default_rng(20000 + n))
        out[str(n)] = acc_ci(generate(model, lay, a, b, n) == a + b)
    return out


def step_locate(fam, model, rand, tr, te, Ltr, Lte):
    """Probes (trained and random-init), shuffled labels, attention."""
    ks = range(1, NB + 1)
    res = {}
    for kind, (TR, TE) in (("trained", (tr, te)), ("random_init", (rand["tr"], rand["te"]))):
        r = {"c_at_q": {}, "inputs_at_q": {}, "c_at_input_pos": {}}
        for h in HOOKS:                                     # carry at the query: per k and pooled
            per_k = [probe_acc(TR["q"][h][:, k], Ltr["c"][:, k], TE["q"][h][:, k], Lte["c"][:, k]) for k in ks]
            Xa, ya = stack(TR["q"][h], lambda k: Ltr["c"][:, k], lambda k: k, ks)
            Xb, yb = stack(TE["q"][h], lambda k: Lte["c"][:, k], lambda k: k, ks)
            r["c_at_q"][h] = {"pooled": probe_acc(Xa, ya, Xb, yb), "per_k": per_k}
        targets = {"a_prev": ("a", -1, ks), "b_prev": ("b", -1, ks), "s_prev": ("s", -1, ks),
                   "a_k": ("a", 0, range(1, NB)), "b_k": ("b", 0, range(1, NB)), "s_k": ("s", 0, ks)}
        for tname, (lab, d, kr) in targets.items():           # what else is readable at the query
            r["inputs_at_q"][tname] = {}
            for h in HOOKS:
                Xa, ya = stack(TR["q"][h], lambda k: Ltr[lab][:, k + d], lambda k: k, kr)
                Xb, yb = stack(TE["q"][h], lambda k: Lte[lab][:, k + d], lambda k: k, kr)
                r["inputs_at_q"][tname][h] = probe_acc(Xa, ya, Xb, yb)
        for role in (("a", "b", "s") if fam == "C2" else ("a", "b")):   # carry at the previous digit's tokens
            r["c_at_input_pos"][role] = {}
            for h in HOOKS:
                Xa, ya = stack(TR[role][h], lambda k: Ltr["c"][:, k], lambda k: k - 1, ks)
                Xb, yb = stack(TE[role][h], lambda k: Lte["c"][:, k], lambda k: k - 1, ks)
                r["c_at_input_pos"][role][h] = probe_acc(Xa, ya, Xb, yb)
        res[kind] = r
    Xa, ya = stack(tr["q"]["resid_mid"], lambda k: Ltr["c"][:, k], lambda k: k, ks)
    Xb, yb = stack(te["q"]["resid_mid"], lambda k: Lte["c"][:, k], lambda k: k, ks)
    res["shuffled_c_mid"] = probe_acc(Xa, np.random.default_rng(3).permutation(ya), Xb, yb)
    res["majority_c"] = float(max(yb.mean(), 1 - yb.mean()))
    first = [h for h in HOOKS if (res["trained"]["c_at_q"][h]["pooled"] or 0) >= 0.99]
    res["first_hook_c_readable_99"] = first[0] if first else None
    # attention from q_k to the inputs of s_k (k = 1..NB; a_k, b_k only below the top bit)
    lay = LAYOUT[fam]; A = te["attn"]
    att = {}
    for name, pos, kr in (("a_k", lay.pos_a, range(1, NB)), ("b_k", lay.pos_b, range(1, NB)),
                          ("a_prev", lambda k: lay.pos_a(k - 1), ks), ("b_prev", lambda k: lay.pos_b(k - 1), ks),
                          ("s_prev", lambda k: lay.pos_s(k - 1), ks)):
        att[name] = [float(np.mean([A[:, k, h, pos(k)].mean() for k in kr])) for h in range(A.shape[2])]
    res["attention"] = att
    res["P1_max_head"] = {nm: max(v) for nm, v in att.items()}
    return res


def gate_like_reference():
    """Gate-likeness stats of library gate neurons and of a 3-input carry neuron trained like them
    (margin 4), with inputs drawn within 0.05 of their rails."""
    rng = np.random.default_rng(11)
    out = {}

    def stats(fn, n_in, name):
        combos = np.repeat(np.arange(2 ** n_in), 1000)
        bits = np.stack([(combos >> (n_in - 1 - j)) & 1 for j in range(n_in)], 1).astype(float)
        u = rng.uniform(0, 0.05, bits.shape)
        x = bits * (1 - u) + (1 - bits) * u
        act = fn(x)[:, None]
        R2, masks = bool_fit(act, combos, n_in)
        f = int(R2[:, 0].argmax())
        out[name] = {"best_R2": float(R2[f, 0]), "gap_over_sd": switch_stats(act[:, 0], combos, masks[f]),
                     "bimodality": bimodality(act[:, 0]), "levels_by_combo": [float(act[combos == c, 0].mean()) for c in range(2 ** n_in)]}
    for t in ("AND", "OR", "NAND", "NOR", "XOR"):
        p = {k: np.array(v, float) for k, v in LIB["gates"][t]["pool"][0].items()}
        stats(lambda x, p=p: G.forward(p, x), 2, t)
    X = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)], float)
    pc = train_neuron(X, CARRY_MASK, margin=4.0, seed=0)
    stats(lambda x: G.forward(pc, x), 3, "carry neuron (margin 4)")
    out["carry neuron (margin 4)"]["weights"] = [float(w) for w in pc["wx"]] + [float(pc["bo"])]
    out["carry neuron (margin 4)"]["restores_0.05"] = restores3(lambda a, b, s: G.forward(pc, np.stack([a, b, s], -1)),
                                                                 CARRY_MASK, 0.05)[0]
    return out


def step_gate_likeness(tr, te, Ltr, Lte):
    """MLP units at q_k (pooled k): Boolean fit, clean switch, fewest units for a 99% readout."""
    ks = range(1, NB + 1)
    act = np.concatenate([te["q"]["mlp_post"][:, k] for k in ks]).astype(float)
    a_p = np.concatenate([Lte["a"][:, k - 1] for k in ks]); b_p = np.concatenate([Lte["b"][:, k - 1] for k in ks])
    s_p = np.concatenate([Lte["s"][:, k - 1] for k in ks])
    idx_c = 4 * a_p + 2 * b_p + s_p
    k9 = range(1, NB)                                       # a_k, b_k exist below the top bit
    act9 = np.concatenate([te["q"]["mlp_post"][:, k] for k in k9]).astype(float)
    idx_s = np.concatenate([4 * Lte["a"][:, k] + 2 * Lte["b"][:, k] + Lte["c"][:, k] for k in k9])
    active = ((act > 0).mean(0) >= 0.01) & (act.std(0) > 1e-6)
    units = np.flatnonzero(active)
    R2c, masks = bool_fit(act[:, units], idx_c)
    R2s, _ = bool_fit(act9[:, units], idx_s)
    carry_f = int(np.flatnonzero((masks == CARRY_MASK).all(1))[0])
    best_c, best_s = R2c.max(0), R2s.max(0)
    fs = R2s.argmax(0)
    thr = [any((masks[f] == m).all() or (masks[f] == 1 - m).all() for m in COUNT_THRESH) for f in fs]
    good = best_s >= 0.8
    top = np.argsort(-best_c)[:6]
    gallery = [{"unit": int(units[u]), "best_R2_carry_inputs": float(best_c[u]), "R2_carry_table": float(R2c[carry_f, u]),
                "best_f_carry_inputs": masks[R2c[:, u].argmax()].astype(int).tolist(),
                "gap_over_sd": switch_stats(act[:, units[u]], idx_c, masks[R2c[:, u].argmax()]),
                "bimodality": bimodality(act[:, units[u]]),
                "mean_by_combo": [float(act[idx_c == c, units[u]].mean()) for c in range(8)]} for u in top]
    # fewest units for a 99% readout (train: greedy; test: accuracy)
    act_tr = np.concatenate([tr["q"]["mlp_post"][:, k] for k in ks]).astype(float)
    c_tr = np.concatenate([Ltr["c"][:, k] for k in ks]).astype(float); c_te = np.concatenate([Lte["c"][:, k] for k in ks])
    s_tr = np.concatenate([Ltr["s"][:, k] for k in ks]).astype(float); s_te = np.concatenate([Lte["s"][:, k] for k in ks])
    Xtr, ctr = subsample(act_tr, c_tr, 20000, 5); Xte, cte = subsample(act, c_te, 20000, 6)
    _, str_ = subsample(act_tr, s_tr, 20000, 5); _, ste = subsample(act, s_te, 20000, 6)
    n_c, curve_c = greedy_readout(Xtr, ctr, Xte, cte)
    n_s, curve_s = greedy_readout(Xtr, str_, Xte, ste)
    return {"n_units": int(act.shape[1]), "n_active": int(len(units)),
            "R2_carry_table": {"max": float(R2c[carry_f].max()), "n_ge_0.9": int((R2c[carry_f] >= 0.9).sum())},
            "best_R2_carry_inputs": {"median": float(np.median(best_c)), "n_ge_0.8": int((best_c >= 0.8).sum())},
            "best_R2_sum_inputs": {"median": float(np.median(best_s)), "n_ge_0.8": int(good.sum()),
                                   "share_count_threshold_among_ge_0.8": float(np.mean(np.array(thr)[good])) if good.any() else None},
            "units_for_99pct": {"carry": n_c, "sum": n_s, "carry_curve": curve_c, "sum_curve": curve_s},
            "gallery": gallery}


def step_interchange(model, lay, fam, seed, U, probe_dir, rng, n_pairs=1000, n_random=10):
    """Patch x + P(x_src - x) at resid_mid q_k. Success = the output becomes the gate adder's s_k with
    the carry into k flipped, among pairs whose unpatched output is right."""
    rr = np.random.default_rng(900 + seed)
    d = U[1].shape[0]
    rand1 = [np.linalg.qr(rr.normal(size=(d, 1)))[0] for _ in range(n_random)]
    rand4 = [np.linalg.qr(rr.normal(size=(d, 4)))[0] for _ in range(n_random)]
    tallies = {name: [] for name in ["pc1", "pc1-4", "probe"] + [f"random1_{i}" for i in range(n_random)]
               + [f"random4_{i}" for i in range(n_random)]}
    gen_ok, per_k = [], {}
    for k in range(1, NB + 1):
        a0, b0, a1, b1 = opposite_pairs(n_pairs, k, rng)
        tok0, q = lay.tokens(a0, b0), lay.q(k)
        x1, _ = mid_at(model, lay.tokens(a1, b1), q)
        _, base = mid_at(model, tok0, q)
        c0 = RT.carry_into(a0, b0, k)
        S, _ = gate_adder(bitsf(a0, NB), bitsf(b0, NB), carry_override=(k, 1 - c0))
        target = np.round(S[:, k]).astype(int)
        valid = base == tok0[:, q + 1]
        subs = {"pc1": U[k][:, :1], "pc1-4": U[k], "probe": probe_dir[k][:, None]}
        subs.update({f"random1_{i}": v for i, v in enumerate(rand1)}); subs.update({f"random4_{i}": v for i, v in enumerate(rand4)})
        for name, V in subs.items():
            P = V @ V.T
            out = model.forward(tok0[:, :q + 1], hooks=patch_q(q, lambda xq: xq + (x1 - xq) @ P))[:, q, :2].argmax(1)
            tallies[name].append((out == target)[valid])
            if name == "pc1":
                per_k[k] = float(np.mean((out == target)[valid])) if valid.any() else None
        # full generation with the PC1 patch at q_k vs the gate adder with carry k flipped
        P = U[k][:, :1] @ U[k][:, :1].T
        pred = generate(model, lay, a0, b0, NB, hooks_fn=lambda kk_, rows: patch_q(q, lambda xq, rows=rows: xq + (x1[rows] - xq) @ P)
                        if kk_ == k else None)
        gen_ok.append((pred == bits_to_int(S))[valid])
    res = {name: acc_ci(np.concatenate(v)) for name, v in tallies.items()}
    out = {"pc1": res["pc1"], "pc1-4": res["pc1-4"], "probe_direction": res["probe"],
           "random1_mean": float(np.mean([res[f"random1_{i}"]["acc"] for i in range(n_random)])),
           "random1_max": float(np.max([res[f"random1_{i}"]["acc"] for i in range(n_random)])),
           "random4_mean": float(np.mean([res[f"random4_{i}"]["acc"] for i in range(n_random)])),
           "random4_max": float(np.max([res[f"random4_{i}"]["acc"] for i in range(n_random)])),
           "pc1_per_k": per_k, "pc1_full_generation_vs_gate_adder": acc_ci(np.concatenate(gen_ok))}
    return out


def class_means(model, lay, fam, tr, Ltr):
    """Mean resid_mid at q_k for carry 0 and 1, and overall (train split). C2: pooled over k."""
    mu = {}
    for k in range(1, NB + 1):
        X = tr["q"]["resid_mid"][:, k].astype(float); c = Ltr["c"][:, k]
        mu[k] = (X[c == 0].mean(0), X[c == 1].mean(0), X.mean(0))
    if fam == "C2":
        X = np.concatenate([tr["q"]["resid_mid"][:, k] for k in range(1, NB + 1)]).astype(float)
        c = np.concatenate([Ltr["c"][:, k] for k in range(1, NB + 1)])
        pooled = (X[c == 0].mean(0), X[c == 1].mean(0), X.mean(0))
        mu = {k: pooled for k in mu}
    return mu


def step_carry_transplant(model, lay, fam, U, mu, n_test=4000):
    """6a: at every q_k (k >= 1) remove the model's carry subspace (PC1-4) and write the class mean of
    the gate adder's carry. Controls: own carry removed (subspace projected out, as pre-registered; and,
    extra, mean-ablated: the overall mean written instead), flipped gate carry written."""
    out = {}
    for n in ([NB] if fam == "B" else [NB, 12, 16, 20]):
        a, b = problems(n_test, n, np.random.default_rng(30000 + n))
        _, C = gate_adder(bitsf(a, n), bitsf(b, n)); cg = np.round(C).astype(int)
        row = {"none": acc_ci(generate(model, lay, a, b, n) == a + b)}
        for mode in ("removed", "removed_mean_ablated", "gate", "flipped"):
            def hooks_fn(k, rows, mode=mode):
                if k == 0:
                    return None
                V = U[kk(k)]; P = V @ V.T; m0, m1, mbar = mu[kk(k)]
                c = cg[rows, k][:, None]
                if mode == "removed":
                    w = np.zeros((1, len(mbar)))
                elif mode == "removed_mean_ablated":
                    w = mbar[None]
                else:
                    w = np.where((c == 1) ^ (mode == "flipped"), m1, m0)
                return patch_q(lay.q(k, n), lambda xq: xq - xq @ P + w @ P)
            row[mode] = acc_ci(generate(model, lay, a, b, n, hooks_fn=hooks_fn) == a + b)
        out[str(n)] = row
    return out


def pc1_diagnostic(model, lay, U, mu, te, Lte):
    """Exploratory (not pre-registered), why 6a can fail: teacher-forced at q_k, k = 1..NB.
    - set: PC1 coordinate replaced by the class mean of the TRUE carry -> share of right outputs
    - shift: the class difference added along PC1 to carry-0 rows -> share that flip
    - spread: within-carry-class sd of the PC1 coordinate / gap between the class means"""
    tok = lay.tokens(*[(Lte["a"][:, :NB] * 2 ** np.arange(NB)).sum(1), (Lte["b"][:, :NB] * 2 ** np.arange(NB)).sum(1)])
    set_ok, shift_ok, spread = [], [], []
    for k in range(1, NB + 1):
        q = lay.q(k); v = U[kk(k)][:, 0]; m0, m1, _ = mu[kk(k)]
        c = Lte["c"][:, k]; tgt = tok[:, q + 1]
        x = te["q"]["resid_mid"][:, k].astype(float) @ v
        spread.append(np.sqrt((x[c == 0].var() + x[c == 1].var()) / 2) / abs(x[c == 1].mean() - x[c == 0].mean()))
        want = np.where(c[:, None] == 1, m1 @ v, m0 @ v)
        out = model.forward(tok[:, :q + 1], hooks=patch_q(q, lambda xq: xq + (want - xq @ v[:, None]) * v[None]))[:, q, :2].argmax(1)
        set_ok.append(out == tgt)
        out = model.forward(tok[:, :q + 1], hooks=patch_q(q, lambda xq: xq + (c[:, None] == 0) * ((m1 - m0) @ v) * v[None]))[:, q, :2].argmax(1)
        shift_ok.append((out != tgt)[c == 0])
    return {"set_pc1_to_true_class_mean": acc_ci(np.concatenate(set_ok)),
            "shift_carry0_rows_by_class_gap": acc_ci(np.concatenate(shift_ok)),
            "within_class_sd_over_gap": [float(s) for s in spread]}


def gate_table(z_by_k, combos_by_k, fam):
    """6b table from projections z (no labels): 2-means levels per k (B) or pooled (C2), u = (z - lo)/(hi - lo),
    mean and std of u per (a, b, s) combo, entry margins |mean - 0.5| / std."""
    us, cs = [], []
    if fam == "C2":
        lo, hi, _ = two_means(np.concatenate(list(z_by_k.values())))
        levels = {k: (lo, hi) for k in z_by_k}
    else:
        levels = {k: two_means(z)[:2] for k, z in z_by_k.items()}
    for k, z in z_by_k.items():
        lo, hi = levels[k]
        us.append((z - lo) / (hi - lo)); cs.append(combos_by_k[k])
    u, c = np.concatenate(us), np.concatenate(cs)
    mean = np.array([u[c == i].mean() for i in range(8)]); sd = np.array([u[c == i].std() for i in range(8)])
    margins = np.abs(mean - 0.5) / np.maximum(sd, 1e-12)
    return {"table": mean.tolist(), "sd": sd.tolist(), "entry_margins": margins.tolist(),
            "gate_margin": float(margins.min()), "snapped": (mean > 0.5).astype(int).tolist()}


def step_gate_from_transformer(model, lay, fam, U, probe_dir, te, Lte, rand_U, rand_te, n_ripple=100000):
    """6b: the transformer's carry as a gate. Table from the label-free PC1 projection; analog margins vs
    random-init; trilinear gate: static discipline at 0.05 and an analog 64-bit ripple; snapped table
    and snapped ripple as a sanity check; the label-trained probe table as a reference row."""
    ks = range(1, NB + 1)
    combos = {k: 4 * Lte["a"][:, k - 1] + 2 * Lte["b"][:, k - 1] + Lte["s"][:, k - 1] for k in ks}
    X = {k: te["q"]["resid_mid"][:, k].astype(float) for k in ks}
    out = {"pc1": gate_table({k: X[k] @ U[k][:, 0] for k in ks}, combos, fam),
           "probe_reference": gate_table({k: X[k] @ probe_dir[k] for k in ks}, combos, fam)}
    Xr = {k: rand_te["q"]["resid_mid"][:, k].astype(float) for k in ks}
    out["random_init_pc1"] = gate_table({k: Xr[k] @ rand_U[k][:, 0] for k in ks}, combos, fam)
    t = out["pc1"]
    table01 = np.clip(t["table"], 0, 1)
    g = trilinear(table01)
    ok, worst = restores3(g, np.array(t["snapped"], float), 0.05)
    rng = np.random.default_rng(42)
    A = rng.integers(0, 2, (n_ripple, 64)).astype(float); B_ = rng.integers(0, 2, (n_ripple, 64)).astype(float)
    ref = exact_ripple(A, B_)
    S, _ = gate_adder(A, B_, carry_gate=g)
    snap_tab = np.array(t["snapped"], float)
    S2, _ = gate_adder(A, B_, carry_gate=lambda a, b, s: snap_tab[(4 * np.round(a) + 2 * np.round(b) + np.round(s)).astype(int)])
    out["analog_gate"] = {"table_clipped": table01.tolist(), "restores_0.05": bool(ok), "worst_dev_at_0.05": worst,
                          "ripple64_analog": acc_ci((np.round(S) == ref).all(1)),
                          "ripple64_worst_signal_dev": float(np.abs(S - ref).max())}
    out["sanity"] = {"snapped_equals_carry_table": bool((snap_tab == CARRY_MASK).all()),
                     "ripple64_snapped": acc_ci((np.round(S2) == ref).all(1))}
    return out


def step_module_transplant(model, lay, fam, tr, Ltr, n_test=4000):
    """6c: replace the MLP output at every q_k by a hand-built sum module: probes read a_k, b_k, c_k from
    resid_mid, snapped; s_k = XOR(XOR(a_k, b_k), c_k) with library XOR neurons (k = 0: XOR(a_0, b_0);
    top: c_n); write u * m(s_k) with u the logit-difference direction through ln_f."""
    P = model.P
    u = P["ln_f.w"] * (P["W_U"][:, 1] - P["W_U"][:, 0]); u = (u / np.linalg.norm(u)).astype(float)
    u_rand = np.random.default_rng(77).normal(size=u.shape); u_rand /= np.linalg.norm(u_rand)
    Xm = {k: tr["q"]["resid_mid"][:, k].astype(float) for k in range(NB + 1)}
    mlp_out = {k: (tr["q"]["resid_post"][:, k] - tr["q"]["resid_mid"][:, k]).astype(float) @ u for k in range(NB + 1)}

    def fit(lab, d, kr):
        if fam == "B":
            return {k: Probe(*subsample(Xm[k], Ltr[lab][:, k + d], 8000, k)) for k in kr}
        Xa = np.concatenate([Xm[k] for k in kr]); ya = np.concatenate([Ltr[lab][:, k + d] for k in kr])
        pr = Probe(*subsample(Xa, ya, 30000, 1))
        return {k: pr for k in range(0, 21)}
    pa, pb = fit("a", 0, range(0, NB)), fit("b", 0, range(0, NB))
    pc, pa_prev = fit("c", 0, range(1, NB + 1)), fit("a", -1, range(1, NB + 1))
    if fam == "B":
        m = {k: (mlp_out[k][Ltr["s"][:, k] == 0].mean(), mlp_out[k][Ltr["s"][:, k] == 1].mean()) for k in range(NB + 1)}
    else:
        mo = np.concatenate([mlp_out[k] for k in range(NB + 1)]); sl = np.concatenate([Ltr["s"][:, k] for k in range(NB + 1)])
        m = {k: (mo[sl == 0].mean(), mo[sl == 1].mean()) for k in range(0, 21)}
    out = {"m": {str(k): [float(x) for x in m[min(k, NB)]] for k in range(NB + 1)}}
    for n in ([NB] if fam == "B" else [NB, 12, 16, 20]):
        a, b = problems(n_test, n, np.random.default_rng(40000 + n))
        row = {"none": acc_ci(generate(model, lay, a, b, n) == a + b)}
        for mode in ("module", "random_direction", "wrong_wiring"):
            def hooks_fn(k, rows, mode=mode, n=n):
                q = lay.q(k, n); keep = {}
                kp = min(k, NB) if fam == "B" else k

                def grab(x):
                    keep["x"] = x[:, q].astype(float).copy(); return x

                def write(x):
                    xm = keep["x"]
                    XOR = lib_gate("XOR", k)
                    if k == 0:
                        s = XOR(pa[0].predict(xm), pb[0].predict(xm))
                    else:
                        c = (pa_prev if mode == "wrong_wiring" else pc)[kp].predict(xm).astype(float)
                        s = c if k == n else XOR(XOR(pa[kp].predict(xm), pb[kp].predict(xm)), c)
                    m0, m1 = m[min(k, NB) if fam == "B" else k]
                    vec = u_rand if mode == "random_direction" else u
                    x = x.copy(); x[:, q] = xm + (m0 + s * (m1 - m0))[:, None] * vec[None]
                    return x
                return {MID: grab, POST: write}
            row[mode] = acc_ci(generate(model, lay, a, b, n, hooks_fn=hooks_fn) == a + b)
        out[str(n)] = row
    return out


def probe_dirs_c(fam, tr, Ltr):
    """Label-trained c_k probe directions at resid_mid q_k (reference only). C2: pooled over k."""
    ks = range(1, NB + 1)
    if fam == "B":
        return {k: Probe(*subsample(tr["q"]["resid_mid"][:, k], Ltr["c"][:, k], 8000, k)).direction() for k in ks}
    Xa, ya = stack(tr["q"]["resid_mid"], lambda k: Ltr["c"][:, k], lambda k: k, ks)
    d = Probe(*subsample(Xa, ya, 30000, 2)).direction()
    return {k: d for k in ks}


def save(R):
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)


def judge(R):
    """Every pre-registered prediction (PREREG.md), per family, with the per-seed numbers it rests on.
    Only models that pass P0 count."""
    out = {"P0": {n: {"acc_10_bits": M["accuracy"][str(NB)]["acc"], "pass": M["kept"]} for n, M in R["models"].items()}}
    kept = {f: [n for n, M in R["models"].items() if n.split(":")[0] == f and M["kept"]] for f in SEEDS}

    def rule(pname, fams, fn, note=""):
        out[pname] = {"note": note}
        for f in fams:
            per = {n: fn(R["models"][n]) for n in kept[f]}
            out[pname][f] = {"pass": (all(v[0] for v in per.values()) if per else None),
                             "per_seed": {n: {"pass": v[0], **v[1]} for n, v in per.items()}}
    both = list(SEEDS)
    rule("P1", both, lambda M: (all(M["locate"]["P1_max_head"][x] >= 0.25 for x in ("a_k", "b_k", "a_prev", "b_prev")),
                                {"max_head_attention": M["locate"]["P1_max_head"]}))
    rule("P2", ["B"], lambda M: (M["locate"]["trained"]["c_at_q"]["resid_pre"]["pooled"] <= 0.80
                                 and M["locate"]["trained"]["c_at_q"]["resid_mid"]["pooled"] >= 0.99,
                                 {h: M["locate"]["trained"]["c_at_q"][h]["pooled"] for h in HOOKS}))
    out["P2"]["C2_for_reference"] = {n: {h: R["models"][n]["locate"]["trained"]["c_at_q"][h]["pooled"] for h in HOOKS} for n in kept["C2"]}
    rule("P3", both, lambda M: (M["locate"]["random_init"]["c_at_q"]["resid_mid"]["pooled"] >= 0.95
                                and M["locate"]["shuffled_c_mid"] <= M["locate"]["majority_c"] + 0.02,
                                {"random_init_mid": M["locate"]["random_init"]["c_at_q"]["resid_mid"]["pooled"],
                                 "shuffled": M["locate"]["shuffled_c_mid"], "majority": M["locate"]["majority_c"]}))
    rule("P4a", both, lambda M: (M["gate_likeness"]["R2_carry_table"]["n_ge_0.9"] == 0
                                 and (M["gate_likeness"]["units_for_99pct"]["carry"] or 99) >= 3,
                                 {"max_R2_carry_table": M["gate_likeness"]["R2_carry_table"]["max"],
                                  "units_for_99pct_carry": M["gate_likeness"]["units_for_99pct"]["carry"]}))
    rule("P4b", both, lambda M: ((M["gate_likeness"]["best_R2_sum_inputs"]["share_count_threshold_among_ge_0.8"] or 0) >= 0.8,
                                 {"share": M["gate_likeness"]["best_R2_sum_inputs"]["share_count_threshold_among_ge_0.8"],
                                  "n_units_R2_ge_0.8": M["gate_likeness"]["best_R2_sum_inputs"]["n_ge_0.8"]}))
    rule("P5", both, lambda M: (M["interchange"]["pc1"]["acc"] >= 0.8 and M["interchange"]["random1_mean"] <= 0.10
                                and M["interchange_random_init"]["pc1"]["acc"] <= 0.10,
                                {"pc1": M["interchange"]["pc1"]["acc"], "random1_mean": M["interchange"]["random1_mean"],
                                 "random_init_pc1": M["interchange_random_init"]["pc1"]["acc"]}))
    t10 = str(NB)
    rule("P6", both, lambda M: (M["carry_transplant"][t10]["removed"]["acc"] < 0.5 and M["carry_transplant"][t10]["gate"]["acc"] >= 0.99
                                and M["carry_transplant"][t10]["flipped"]["acc"] <= 0.05,
                                {m: M["carry_transplant"][t10][m]["acc"] for m in ("none", "removed", "removed_mean_ablated", "gate", "flipped")}))
    rule("P6-C2", ["C2"], lambda M: (M["carry_transplant"]["16"]["gate"]["acc"] - M["carry_transplant"]["16"]["none"]["acc"] < 0.10,
                                     {"none_16": M["carry_transplant"]["16"]["none"]["acc"], "gate_16": M["carry_transplant"]["16"]["gate"]["acc"]}))
    rule("P7a", both, lambda M: (M["gate_from_transformer"]["pc1"]["gate_margin"] >= 4
                                 and M["gate_from_transformer"]["pc1"]["gate_margin"] >= 3 * M["gate_from_transformer"]["random_init_pc1"]["gate_margin"],
                                 {"margin": M["gate_from_transformer"]["pc1"]["gate_margin"],
                                  "random_init_margin": M["gate_from_transformer"]["random_init_pc1"]["gate_margin"]}))
    rule("P7b", both, lambda M: (M["gate_from_transformer"]["analog_gate"]["restores_0.05"]
                                 and M["gate_from_transformer"]["analog_gate"]["ripple64_analog"]["acc"] == 1.0,
                                 {"restores": M["gate_from_transformer"]["analog_gate"]["restores_0.05"],
                                  "worst_dev": M["gate_from_transformer"]["analog_gate"]["worst_dev_at_0.05"],
                                  "ripple64": M["gate_from_transformer"]["analog_gate"]["ripple64_analog"]["acc"]}),
         note="restoring half expected to fail by construction (PREREG note)")
    rule("P7c", both, lambda M: (M["gate_from_transformer"]["sanity"]["snapped_equals_carry_table"]
                                 and M["gate_from_transformer"]["sanity"]["ripple64_snapped"]["acc"] == 1.0,
                                 {"snapped_table": M["gate_from_transformer"]["pc1"]["snapped"]}), note="sanity only, not evidence")
    rule("P8", both, lambda M: (M["module_transplant"][t10]["module"]["acc"] >= 0.99 and M["module_transplant"][t10]["random_direction"]["acc"] <= 0.05
                                and M["module_transplant"][t10]["wrong_wiring"]["acc"] <= 0.05,
                                {m: M["module_transplant"][t10][m]["acc"] for m in ("none", "module", "random_direction", "wrong_wiring")}))
    rule("P8-C2", ["C2"], lambda M: (M["module_transplant"]["16"]["module"]["acc"] - M["module_transplant"]["16"]["none"]["acc"] < 0.10,
                                     {"none_16": M["module_transplant"]["16"]["none"]["acc"], "module_16": M["module_transplant"]["16"]["module"]["acc"]}),
         note="exploratory")
    return out


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, MUTED, GRIDC, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    C1, C2c, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.8,
                         "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "lines.linewidth": 1.8})
    FIG = os.path.join(OUT, "figures")
    kept = [n for n, M in R["models"].items() if M["kept"]]

    # 1: attention from q_k to the inputs, per head
    names = ["a_k", "b_k", "a_prev", "b_prev", "s_prev"]
    fig, axs = plt.subplots(1, len(kept), figsize=(1.9 * len(kept), 2.4), squeeze=False)
    for ax, n in zip(axs[0], kept):
        M = np.array([R["models"][n]["locate"]["attention"][x] for x in names]).T
        ax.grid(False); ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(["a_k", "b_k", "a_k-1", "b_k-1", "s_k-1"], rotation=90, fontsize=7)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([f"H{h}" for h in range(M.shape[0])]); ax.set_title(n, fontsize=8)
    fig.suptitle("Mean attention from the query of s_k to its inputs (k = 1..10)", fontsize=8, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_attention.png"), dpi=150); plt.close(fig)

    # 2: carry readability by sublayer: trained vs random-init, shuffled
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, fam in zip(axs, SEEDS):
        for n in [x for x in kept if x.startswith(fam + ":")]:
            L = R["models"][n]["locate"]
            ax.plot(range(5), [L["trained"]["c_at_q"][h]["pooled"] for h in HOOKS], color=C1, marker="o", ms=3, alpha=0.8)
            ax.plot(range(5), [L["random_init"]["c_at_q"][h]["pooled"] for h in HOOKS], color=C2c, marker="s", ms=3, alpha=0.8)
            ax.scatter([2], [L["shuffled_c_mid"]], color=MUTED, marker="x", zorder=3)
        ax.plot([], [], color=C1, marker="o", label="trained"); ax.plot([], [], color=C2c, marker="s", label="random-init")
        ax.scatter([], [], color=MUTED, marker="x", label="shuffled labels (resid_mid)")
        ax.axhline(0.99, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(5)); ax.set_xticklabels(list(HOOKS), rotation=20)
        ax.set_title(f"{fam}: probe for c_k at the query of s_k (pooled k)", fontsize=8, loc="left")
    axs[0].set_ylabel("test accuracy"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_probes.png"), dpi=150); plt.close(fig)

    # 3: neuron gallery vs gate neurons (mean activation by (a_{k-1}, b_{k-1}, s_{k-1}))
    combos = [f"{(c >> 2) & 1}{(c >> 1) & 1}{c & 1}" for c in range(8)]
    shown = [n for n in kept if n.startswith("B")][:2] + [n for n in kept if n.startswith("C2")][:2]
    rows = [(n, g) for n in shown for g in R["models"][n]["gate_likeness"]["gallery"][:3]]
    fig, axs = plt.subplots(3, 5, figsize=(10, 5.4))
    axs = axs.ravel()
    ref = R["gate_neuron_reference"]["carry neuron (margin 4)"]
    axs[0].bar(range(8), ref["levels_by_combo"], color=C3)
    axs[0].set_title(f"carry neuron (margin 4)\nR² {ref['best_R2']:.3f}, gap/sd {ref['gap_over_sd']:.0f}", fontsize=7)
    for i, (n, g) in enumerate(rows[:14]):
        ax = axs[i + 1]
        ax.bar(range(8), g["mean_by_combo"], color=[C1 if CARRY_MASK[c] else BASE for c in range(8)])
        ax.set_title(f"{n} unit {g['unit']}\nR² carry {g['R2_carry_table']:.2f}, best {g['best_R2_carry_inputs']:.2f}", fontsize=7)
    for i, ax in enumerate(axs):
        if i > len(rows):
            ax.set_visible(False)
        ax.set_xticks(range(8)); ax.set_xticklabels(combos, rotation=90, fontsize=6)
    fig.suptitle("Mean MLP activation at the query of s_k by (a_k-1, b_k-1, s_k-1); blue = carry 1", fontsize=8, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_neuron_gallery.png"), dpi=150); plt.close(fig)

    # 4: interchange success with 95% CIs
    keys = [("pc1", "PC1"), ("pc1-4", "PC1-4"), ("probe_direction", "probe dir."), ("random1_mean", "random 1-D"),
            ("random4_mean", "random 4-D"), ("rand_init", "random-init PC1")]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    w = 0.8 / len(kept)
    for i, n in enumerate(kept):
        I = R["models"][n]["interchange"]
        vals, los, his = [], [], []
        for k_, _ in keys:
            if k_ == "rand_init":
                d = R["models"][n]["interchange_random_init"]["pc1"]
            else:
                d = I[k_] if isinstance(I[k_], dict) else {"acc": I[k_], "ci95": [I[k_], I[k_]]}
            vals.append(d["acc"]); los.append(max(0.0, d["acc"] - d["ci95"][0])); his.append(max(0.0, d["ci95"][1] - d["acc"]))
        ax.bar(np.arange(len(keys)) + i * w, vals, w, yerr=[los, his], color=C1 if n.startswith("B") else C2c,
               alpha=0.55 + 0.45 * (i % 2), label=n, error_kw={"lw": 0.8})
    ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2); ax.set_xticklabels([l for _, l in keys])
    ax.set_ylabel("share of outputs that flip to\nthe gate adder's flipped-carry bit"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6, ncol=4)
    ax.set_title("Interchange at resid_mid, query of s_k (k = 1..10)", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_interchange.png"), dpi=150); plt.close(fig)

    # 5: C2 accuracy by length, with the exact gate carry and with the hand-built sum module
    c2 = [n for n in kept if n.startswith("C2")]
    if c2:
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        Ls = [10, 12, 16, 20]
        for n in c2:
            M = R["models"][n]
            ax.plot(Ls, [M["accuracy"][str(L)]["acc"] for L in Ls], color=INK2, marker="o", ms=3)
            ax.plot(Ls, [M["carry_transplant"][str(L)]["gate"]["acc"] for L in Ls], color=C1, marker="s", ms=3)
            ax.plot(Ls, [M["module_transplant"][str(L)]["module"]["acc"] for L in Ls], color=C2c, marker="^", ms=3)
        ax.plot([], [], color=INK2, marker="o", label="model alone"); ax.plot([], [], color=C1, marker="s", label="exact gate carry written in (6a)")
        ax.plot([], [], color=C2c, marker="^", label="MLP replaced by the XOR module (6c)")
        ax.set_xlabel("bits per operand (trained at 10)"); ax.set_ylabel("exact-sum accuracy"); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(Ls); ax.legend(frameon=False, fontsize=7)
        ax.set_title("C2 on longer sums", fontsize=8, loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_c2_length.png"), dpi=150); plt.close(fig)


def analyse():
    T0 = time.time()

    def log(*a):
        print(*a, f"[{time.time() - T0:.0f}s]", flush=True)
    R = {"settings": {"train_test_problems": [8000, 4000], "families": SEEDS, "steps": STEPS},
         "gate_neuron_reference": gate_like_reference(), "models": {}}
    log("reference gate neurons done")
    rng0 = np.random.default_rng(7)
    a_tr, b_tr = problems(8000, NB, rng0); a_te, b_te = problems(4000, NB, rng0)
    Ltr, Lte = labels(a_tr, b_tr), labels(a_te, b_te)
    for fam in SEEDS:
        lay = LAYOUT[fam]
        for seed in SEEDS[fam]:
            name = f"{fam}:{seed}"
            model, train_log = load(fam, seed)
            M = {"final_loss": float(train_log[-1][1]), "steps": trained_steps(fam, seed) if not (fam == "B" and seed == 1) else 2500}
            M["accuracy"] = step_accuracy(model, lay, fam)
            M["kept"] = M["accuracy"][str(NB)]["acc"] >= 0.99
            log(name, "accuracy", {n: round(v["acc"], 4) for n, v in M["accuracy"].items()}, "kept" if M["kept"] else "DROPPED")
            R["models"][name] = M
            if not M["kept"]:
                save(R); continue
            rand = make_model(fam, seed)
            tr, te = collect(model, lay, a_tr, b_tr), collect(model, lay, a_te, b_te)
            rt = {"tr": collect(rand, lay, a_tr, b_tr), "te": collect(rand, lay, a_te, b_te)}
            M["locate"] = step_locate(fam, model, rt, tr, te, Ltr, Lte)
            log(name, "locate: c at q pooled", {h: M["locate"]["trained"]["c_at_q"][h]["pooled"] for h in HOOKS},
                "random-init mid", M["locate"]["random_init"]["c_at_q"]["resid_mid"]["pooled"])
            M["gate_likeness"] = step_gate_likeness(tr, te, Ltr, Lte)
            M["gate_likeness_random_init"] = step_gate_likeness(rt["tr"], rt["te"], Ltr, Lte)
            log(name, "gate-likeness", M["gate_likeness"]["R2_carry_table"], M["gate_likeness"]["units_for_99pct"]["carry"],
                M["gate_likeness"]["units_for_99pct"]["sum"])
            U, rand_U = flip_dirs(model, lay, fam, seed), flip_dirs(rand, lay, fam, seed)
            pdir = probe_dirs_c(fam, tr, Ltr)
            M["interchange"] = step_interchange(model, lay, fam, seed, U, pdir, np.random.default_rng(600 + seed))
            rand_pdir = probe_dirs_c(fam, rt["tr"], Ltr)
            M["interchange_random_init"] = step_interchange(rand, lay, fam, seed, rand_U, rand_pdir, np.random.default_rng(600 + seed), n_random=1)
            log(name, "interchange pc1", M["interchange"]["pc1"]["acc"], "pc1-4", M["interchange"]["pc1-4"]["acc"],
                "random1", M["interchange"]["random1_mean"], "random-init pc1", M["interchange_random_init"]["pc1"]["acc"])
            mu = class_means(model, lay, fam, tr, Ltr)
            M["carry_transplant"] = step_carry_transplant(model, lay, fam, U, mu)
            log(name, "6a", {n: {m: round(v["acc"], 4) for m, v in row.items()} for n, row in M["carry_transplant"].items()})
            M["pc1_diagnostic_exploratory"] = pc1_diagnostic(model, lay, U, mu, te, Lte)
            log(name, "6a diagnostic", {k: (v["acc"] if isinstance(v, dict) else np.round(v, 2).tolist())
                                        for k, v in M["pc1_diagnostic_exploratory"].items()})
            M["gate_from_transformer"] = step_gate_from_transformer(model, lay, fam, U, pdir, te, Lte, rand_U, rt["te"])
            log(name, "6b margin", M["gate_from_transformer"]["pc1"]["gate_margin"], "random-init",
                M["gate_from_transformer"]["random_init_pc1"]["gate_margin"], "table", np.round(M["gate_from_transformer"]["pc1"]["table"], 2).tolist(),
                "ripple", M["gate_from_transformer"]["analog_gate"]["ripple64_analog"]["acc"])
            M["module_transplant"] = step_module_transplant(model, lay, fam, tr, Ltr)
            log(name, "6c", {n: {m: round(v["acc"], 4) for m, v in row.items()} for n, row in M["module_transplant"].items() if n != "m"})
            del tr, te, rt
            save(R)
    R["predictions"] = judge(R)
    R["runtime_s"] = round(time.time() - T0, 1)
    save(R)
    figures(R)
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "analyse", "figures"])
    ap.add_argument("--only", default=None, help="family:seed, e.g. B:2")
    args = ap.parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    if args.cmd == "train":
        train_all(args.only)
    elif args.cmd == "figures":                       # redraw from results.json
        figures(json.load(open(os.path.join(OUT, "results.json"))))
    else:
        analyse()
