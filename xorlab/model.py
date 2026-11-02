"""2 -> H -> 1 networks for XOR, trained in a batch (one set of weights per seed).

Everything is plain numpy float64 on CPU. Given the same seeds and config the
results are bit-for-bit repeatable on the same machine; different BLAS builds
can differ in the last decimal places of einsum sums.
"""
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)   # the whole truth table
Y = np.array([0, 1, 1, 0], dtype=float)
INPUT_NAMES = ["00", "01", "10", "11"]


def act_fn(name):
    if name == "tanh":
        return np.tanh
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    raise ValueError(name)


def act_grad(name, z, h):
    if name == "tanh":
        return 1.0 - h ** 2
    return (z > 0).astype(float)


def init_params(seed, hidden, init_std):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, init_std, (2, hidden)),
        "b1": r.normal(0, init_std, hidden),
        "w2": r.normal(0, init_std / np.sqrt(hidden), hidden),
        "b2": np.zeros(()),
    }


def hidden_acts(p, act, x=X):
    """Hidden activations, shape (inputs, hidden)."""
    return act_fn(act)(x @ p["W1"] + p["b1"])


def logits(p, act, x=X):
    return hidden_acts(p, act, x) @ p["w2"] + p["b2"]


def all_correct(lg, margin=0.0):
    signed = np.where(Y == 1, lg, -lg)
    return bool(np.all(signed > margin))


def train_batch(seeds, act, hidden, tcfg, y=None):
    """Train one network per seed with full-batch Adam on the 4 truth-table rows.

    A run stops (its weights freeze) at the first step where every input is on the
    correct side with |logit| >= margin. Runs that never get there are kept and
    reported as failures.
    """
    Yt = Y if y is None else np.asarray(y, float)
    S = len(seeds)
    inits = [init_params(s, hidden, tcfg["init_std"]) for s in seeds]
    P = {k: np.stack([ip[k] for ip in inits]) for k in inits[0]}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v_) for k, v_ in P.items()}
    b1, b2 = tcfg["betas"]
    lr, margin = tcfg["lr"], tcfg["margin"]
    active = np.ones(S, bool)
    conv_step = np.full(S, -1)
    t_count = np.zeros(S)
    log_steps, loss_log, acc_log = [], [], []
    f = act_fn(act)
    for step in range(tcfg["max_steps"] + 1):
        Z = np.einsum("ni,sih->snh", X, P["W1"]) + P["b1"][:, None, :]
        H = f(Z)
        L = np.einsum("snh,sh->sn", H, P["w2"]) + P["b2"][:, None]
        loss = np.mean(np.logaddexp(0, L) - Yt * L, axis=1)
        signed = np.where(Yt == 1, L, -L)
        acc = np.mean(signed > 0, axis=1)
        newly = active & np.all(signed >= margin, axis=1)
        conv_step[newly] = step
        active &= ~newly
        if step % tcfg["log_every"] == 0 or not active.any():
            log_steps.append(step); loss_log.append(loss.copy()); acc_log.append(acc.copy())
        if not active.any() or step == tcfg["max_steps"]:
            break
        dL = (1 / (1 + np.exp(-L)) - Yt) / 4.0
        G = {"w2": np.einsum("sn,snh->sh", dL, H), "b2": dL.sum(1)}
        dZ = dL[:, :, None] * P["w2"][:, None, :] * act_grad(act, Z, H)
        G["W1"] = np.einsum("ni,snh->sih", X, dZ)
        G["b1"] = dZ.sum(1)
        t_count += active
        for k in P:
            mask = active.reshape((S,) + (1,) * (P[k].ndim - 1))
            m[k] = np.where(mask, b1 * m[k] + (1 - b1) * G[k], m[k])
            v[k] = np.where(mask, b2 * v[k] + (1 - b2) * G[k] ** 2, v[k])
            tt = np.maximum(t_count, 1).reshape(mask.shape)
            upd = lr * (m[k] / (1 - b1 ** tt)) / (np.sqrt(v[k] / (1 - b2 ** tt)) + 1e-8)
            P[k] = np.where(mask, P[k] - upd, P[k])
    runs = []
    for i, s in enumerate(seeds):
        p = {k: P[k][i].copy() for k in P}
        lg = logits(p, act)
        runs.append({
            "seed": int(s), "params": p, "init": inits[i],
            "converged": bool(conv_step[i] >= 0), "converge_step": int(conv_step[i]),
            "logits": lg, "probs": 1 / (1 + np.exp(-lg)),
            "accuracy": float(np.mean(np.where(Yt == 1, lg, -lg) > 0)),
            "min_margin": float(np.min(np.where(Yt == 1, lg, -lg))),
        })
    return runs, np.array(log_steps), np.array(loss_log).T, np.array(acc_log).T
