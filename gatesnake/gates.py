"""Tiny neural networks that act as logic gates.

Every neuron is the same kind of neuron: y = sigmoid(w . x + b), signals live in [0, 1].
A gate network has optional hidden sigmoid units and an output unit that can also see
the raw inputs (a "skip" connection):

    h     = sigmoid(x @ Wh + bh)                (H hidden units, H may be 0)
    logit = x @ wx + h @ wh + bo                (wx = 0 when there is no skip)
    out   = sigmoid(logit)

Training is full-batch Adam on the truth table, one set of weights per seed, and a run
stops (weights freeze) at the first step where every row is on the correct side with
|logit| >= margin. Same rule as xorlab.model.train_batch.
"""
import numpy as np

TRUTH = {
    "NOT":  [1, 0],
    "AND":  [0, 0, 0, 1],
    "OR":   [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0],
    "XNOR": [1, 0, 0, 1],
}
N_IN = {g: (1 if g == "NOT" else 2) for g in TRUTH}
X1 = np.array([[0.], [1.]])
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)

# name -> (hidden units, skip connection from inputs to the output unit)
ARCHS = {
    "neuron": (0, True),     # one sigmoid unit
    "mlp-1":  (1, False),    # 2 -> 1 -> 1
    "skip-1": (1, True),     # 2 -> 1 -> 1 plus inputs -> output
    "mlp-2":  (2, False),    # 2 -> 2 -> 1
    "mlp-3":  (3, False),
    "mlp-4":  (4, False),    # the gate used in run_adder.py (there with tanh hidden units)
}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def n_params(n_in, arch):
    H, skip = ARCHS[arch]
    return n_in * H + H + (n_in if skip else 0) + H + 1


def n_neurons(arch):
    return ARCHS[arch][0] + 1


def truth_inputs(gate):
    return X1 if N_IN[gate] == 1 else X2


def forward(p, x):
    """p: one gate's params. x: (..., n_in) -> outputs in [0, 1], shape (...)."""
    out = x @ p["wx"] + p["bo"]
    if p["wh"].size:
        out = out + sig(x @ p["Wh"] + p["bh"]) @ p["wh"]
    return sig(out)


def logit(p, x):
    out = x @ p["wx"] + p["bo"]
    if p["wh"].size:
        out = out + sig(x @ p["Wh"] + p["bh"]) @ p["wh"]
    return out


def train(gate, arch, margin, seeds, max_steps=20000, lr=0.05, betas=(0.9, 0.999), init_std=1.0):
    """Train one network per seed. Returns list of {seed, params, converged, steps, min_margin}."""
    X = truth_inputs(gate); y = np.array(TRUTH[gate], float); n_in = X.shape[1]; R = len(X)
    H, skip = ARCHS[arch]
    S = len(seeds)
    P = {k: [] for k in ("Wh", "bh", "wx", "wh", "bo")}
    for s in seeds:
        r = np.random.default_rng(s)
        P["Wh"].append(r.normal(0, init_std, (n_in, H)))
        P["bh"].append(r.normal(0, init_std, H))
        P["wx"].append(r.normal(0, init_std, n_in) if skip else np.zeros(n_in))
        P["wh"].append(r.normal(0, init_std / np.sqrt(max(H, 1)), H))
        P["bo"].append(np.zeros(()))
    P = {k: np.stack(v) for k, v in P.items()}
    keys = [k for k in P if (H > 0 or k not in ("Wh", "bh", "wh")) and (skip or k != "wx")]
    m = {k: np.zeros_like(P[k]) for k in keys}
    v = {k: np.zeros_like(P[k]) for k in keys}
    b1, b2 = betas
    active = np.ones(S, bool); conv = np.full(S, -1); t = np.zeros(S)
    sgn = 2 * y - 1
    for step in range(max_steps + 1):
        Hh = sig(np.einsum("ri,sih->srh", X, P["Wh"]) + P["bh"][:, None, :])
        L = X @ P["wx"].T                                   # (R, S)
        L = L.T + np.einsum("srh,sh->sr", Hh, P["wh"]) + P["bo"][:, None]
        signed = L * sgn
        newly = active & np.all(signed >= margin, axis=1)
        conv[newly] = step; active &= ~newly
        if not active.any() or step == max_steps:
            break
        dL = (sig(L) - y) / R                               # (S, R)
        G = {"bo": dL.sum(1), "wx": dL @ X, "wh": np.einsum("sr,srh->sh", dL, Hh)}
        dZ = dL[:, :, None] * P["wh"][:, None, :] * Hh * (1 - Hh)
        G["Wh"] = np.einsum("ri,srh->sih", X, dZ); G["bh"] = dZ.sum(1)
        t += active
        for k in keys:
            mask = active.reshape((S,) + (1,) * (P[k].ndim - 1))
            m[k] = np.where(mask, b1 * m[k] + (1 - b1) * G[k], m[k])
            v[k] = np.where(mask, b2 * v[k] + (1 - b2) * G[k] ** 2, v[k])
            tt = np.maximum(t, 1).reshape(mask.shape)
            upd = lr * (m[k] / (1 - b1 ** tt)) / (np.sqrt(v[k] / (1 - b2 ** tt)) + 1e-8)
            P[k] = np.where(mask, P[k] - upd, P[k])
    runs = []
    for i, s in enumerate(seeds):
        p = {k: P[k][i].copy() for k in P}
        mm = float(np.min(logit(p, X) * sgn))
        runs.append({"seed": int(s), "params": p, "converged": bool(conv[i] >= 0),
                     "steps": int(conv[i]), "min_margin": mm})
    return runs


# ---------------- does the gate still work inside a bigger machine? ----------------
def _box(n_in, eps, pts):
    """For every logic input row: a grid over the box where each input lies within eps of its rail."""
    X = truth_inputs("NOT" if n_in == 1 else "AND")
    g = np.linspace(0, eps, pts)
    boxes = []
    for row in X:
        axes = [g if v == 0 else 1 - g for v in row]
        boxes.append(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, n_in))
    return boxes


def restores(p, gate, eps, pts=9):
    """Static discipline at noise level eps: every input within eps of a rail gives an output
    within eps of the correct rail. If this holds, signals never drift, however many gates
    are chained or fed back, as long as the primary inputs are within eps of the rails."""
    y = TRUTH[gate]
    for box, target in zip(_box(N_IN[gate], eps, pts), y):
        out = forward(p, box)
        dev = np.abs(out - target).max()
        if dev > eps:
            return False
    return True


EPS_GRID = np.round(np.concatenate([np.geomspace(1e-4, 0.01, 9)[:-1], np.arange(0.01, 0.46, 0.01)]), 5)


def restore_range(p, gate):
    """(smallest, largest) eps on EPS_GRID at which the gate restores; (nan, nan) if none."""
    ok = [e for e in EPS_GRID if restores(p, gate, e)]
    return (min(ok), max(ok)) if ok else (float("nan"), float("nan"))


def quantize(p, bits):
    """Round every weight to a signed `bits`-bit grid with one scale per network."""
    vals = np.concatenate([np.ravel(p[k]) for k in p])
    top = np.abs(vals).max()
    levels = 2 ** (bits - 1) - 1
    step = top / levels if levels > 0 else top
    return {k: np.round(p[k] / step) * step for k in p}, step


def gate_cost(gate, arch):
    """Neurons, parameters and multiply-adds for one evaluation of the gate."""
    n_in = N_IN[gate]; H, skip = ARCHS[arch]
    macs = n_in * H + (n_in if skip else 0) + H
    return {"neurons": H + 1, "params": n_params(n_in, arch), "macs": macs,
            "depth": 2 if H else 1}
