"""TransformerLens version of run_transformer.py. Same task, models and analyses, using the
real library instead of the NumPy GPT. NOT run by the author (PyTorch could not be installed
in the authoring environment); written against the stable TransformerLens API.

    pip install torch transformer-lens matplotlib
    python run_transformer_tl.py             # trains 3 small GPTs, then analyses (minutes on a laptop)

Hooks used: blocks.{l}.hook_resid_pre, blocks.{l}.attn.hook_pattern, blocks.{l}.attn.hook_z,
ln_final.hook_normalized, through model.run_with_cache and model.run_with_hooks.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
OUT = "results_transformer_tl"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
NB = 10
PLUS, EQ = 2, 3
T = 3 * NB + 3
POS_EQ = 2 * NB + 1
MODELS = {"lsb-L1": dict(layers=1, order="lsb"), "lsb-L2": dict(layers=2, order="lsb"),
          "msb-L2": dict(layers=2, order="msb")}
STEPS, BATCH = 2500, 128
DEV = "cpu"


def bits(v, n):
    return (v[:, None] >> np.arange(n)) & 1


def encode(a, b, order):
    s = bits(a + b, NB + 1)
    if order == "msb":
        s = s[:, ::-1]
    n = len(a)
    return np.concatenate([bits(a, NB), np.full((n, 1), PLUS), bits(b, NB), np.full((n, 1), EQ), s], 1)


def decode_sum(tok_sum, order):
    s = tok_sum[:, ::-1] if order == "msb" else tok_sum
    return (s * (2 ** np.arange(NB + 1))).sum(1)


def qpos(bit, order):
    return POS_EQ + (bit if order == "lsb" else NB - bit)


def travel_dist(a, b):
    best = np.zeros(len(a), int); dist = np.full(len(a), -1)
    for i in range(NB):
        ai, bi = (a >> i) & 1, (b >> i) & 1
        gen, prop = (ai & bi) == 1, (ai ^ bi) == 1
        dist = np.where(gen, 0, np.where(prop & (dist >= 0), dist + 1, -1))
        best = np.maximum(best, dist)
    return best


def carry_into(a, b, k):
    lo = 2 ** k
    return (((a % lo) + (b % lo)) >= lo).astype(int)


def built_distance(n, L, r):
    a, b = r.integers(0, 2 ** NB, n), r.integers(0, 2 ** NB, n)
    start = r.integers(0, max(NB - L - 1, 0) + 1, n)
    for j in range(n):
        s = start[j]; a[j] |= 1 << s; b[j] |= 1 << s
        for i in range(s + 1, s + L + 1):
            if r.random() < 0.5: a[j] |= 1 << i; b[j] &= ~(1 << i)
            else: b[j] |= 1 << i; a[j] &= ~(1 << i)
        if s + L + 1 <= NB - 1:
            a[j] &= ~(1 << (s + L + 1)); b[j] &= ~(1 << (s + L + 1))
    return a, b


def make_model(layers):
    cfg = HookedTransformerConfig(n_layers=layers, d_model=64, d_head=16, n_heads=4, d_mlp=256,
                                  d_vocab=4, n_ctx=T, act_fn="relu", normalization_type="LN", seed=1)
    return HookedTransformer(cfg).to(DEV)


def train(name, cfg):
    model = make_model(cfg["layers"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / STEPS))))
    r = np.random.default_rng(1000)
    log = []
    for step in range(STEPS):
        a, b = r.integers(0, 2 ** NB, BATCH), r.integers(0, 2 ** NB, BATCH)
        tok = torch.tensor(encode(a, b, cfg["order"]), device=DEV)
        logits = model(tok)
        loss = F.cross_entropy(logits[:, POS_EQ:-1].reshape(-1, 4), tok[:, POS_EQ + 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            log.append((step, loss.item())); print(f"{name} step {step} loss {loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT, f"{name}.pt"))
    return model, log


@torch.no_grad()
def generate(model, a, b, order, fwd_hooks=None, bs=2000):
    outs = []
    for i in range(0, len(a), bs):
        pre = torch.tensor(encode(a[i:i + bs], b[i:i + bs], order)[:, :POS_EQ + 1], device=DEV)
        for _ in range(NB + 1):
            lg = model.run_with_hooks(pre, fwd_hooks=fwd_hooks or []) if fwd_hooks else model(pre)
            nxt = lg[:, -1, :2].argmax(1)
            pre = torch.cat([pre, nxt[:, None]], 1)
        outs.append(decode_sum(pre[:, POS_EQ + 1:].cpu().numpy(), order))
    return np.concatenate(outs)


@torch.no_grad()
def analyse(models):
    S = {}
    rng = np.random.default_rng(7)
    a_r, b_r = rng.integers(0, 2 ** NB, 10000), rng.integers(0, 2 ** NB, 10000)
    parts = [(a_r, b_r)] + [built_distance(800, L, np.random.default_rng(300 + L)) for L in range(2, NB)]
    a_p = np.concatenate([x[0] for x in parts]); b_p = np.concatenate([x[1] for x in parts]); td = travel_dist(a_p, b_p)
    S["accuracy"] = {}
    for name, (model, log, cfg) in models.items():
        ok = generate(model, a_p, b_p, cfg["order"]) == a_p + b_p
        S["accuracy"][name] = {"random_exact": float(ok[:10000].mean()),
                               "by_travel_distance": {str(L): float(ok[td == L].mean()) for L in range(NB) if (td == L).sum() >= 50}}
        print(name, S["accuracy"][name], flush=True)

    # attention by relative offset
    a_t, b_t = rng.integers(0, 2 ** NB, 2000), rng.integers(0, 2 ** NB, 2000)
    S["attention"] = {}
    for name, (model, log, cfg) in models.items():
        tok = torch.tensor(encode(a_t, b_t, cfg["order"]), device=DEV)
        _, cache = model.run_with_cache(tok)
        prof = {}
        for l in range(cfg["layers"]):
            A = cache[f"blocks.{l}.attn.hook_pattern"].cpu().numpy()          # (B, H, Q, K)
            for h in range(4):
                tab = {kind: {str(d): [] for d in range(-4, 3)} for kind in ("a", "b", "s")}
                for bit in range(NB + 1):
                    q = qpos(bit, cfg["order"])
                    for d in range(-4, 3):
                        j = bit + d
                        if 0 <= j < NB:
                            tab["a"][str(d)].append(A[:, h, q, j].mean()); tab["b"][str(d)].append(A[:, h, q, NB + 1 + j].mean())
                        if 0 <= j <= NB:
                            kp = qpos(j, cfg["order"]) + 1
                            if kp <= q:
                                tab["s"][str(d)].append(A[:, h, q, kp].mean())
                prof[f"L{l}H{h}"] = {k: {d: (float(np.mean(v)) if v else None) for d, v in t.items()} for k, t in tab.items()}
        S["attention"][name] = prof

    # activation patching of resid_pre at (layer, position)
    S["patching"] = {}
    for name, (model, log, cfg) in models.items():
        res = {}
        for k in [3, 6]:
            n = 1500; lo = 2 ** k
            a0, b0 = rng.integers(0, 2 ** NB, n), rng.integers(0, 2 ** NB, n); c0 = carry_into(a0, b0, k)
            a1, b1 = a0.copy(), b0.copy()
            for j in range(n):
                while True:
                    x, y = rng.integers(0, lo), rng.integers(0, lo)
                    if int((x + y) >= lo) != c0[j]: break
                a1[j] = a0[j] - a0[j] % lo + x; b1[j] = b0[j] - b0[j] % lo + y
            tok0 = torch.tensor(encode(a0, b0, cfg["order"]), device=DEV); tok1 = torch.tensor(encode(a1, b1, cfg["order"]), device=DEV)
            q = qpos(k, cfg["order"])
            _, cache1 = model.run_with_cache(tok1)
            base = model(tok0)[:, q, :2].argmax(1); valid = (base == tok0[:, q + 1]).cpu().numpy()
            target = tok1[:, q + 1].cpu().numpy()
            heat = np.zeros((cfg["layers"], T))
            for l in range(cfg["layers"]):
                src = cache1[f"blocks.{l}.hook_resid_pre"]
                for p in range(q + 1):
                    def hk(act, hook, p=p, src=src):
                        act = act.clone(); act[:, p] = src[:, p]; return act
                    out = model.run_with_hooks(tok0, fwd_hooks=[(f"blocks.{l}.hook_resid_pre", hk)])[:, q, :2].argmax(1).cpu().numpy()
                    heat[l, p] = np.mean(out[valid] == target[valid])
            res[f"bit{k}"] = {"flip_rate": heat.tolist(), "query_pos": int(q)}
        S["patching"][name] = res

    # head ablation
    S["head_ablation"] = {}
    far = td >= 6
    for name, (model, log, cfg) in models.items():
        row = {}
        for l in range(cfg["layers"]):
            for h in range(4):
                def hk(z, hook, h=h):
                    z = z.clone(); z[:, :, h] = 0; return z
                ok = generate(model, a_p, b_p, cfg["order"], fwd_hooks=[(f"blocks.{l}.attn.hook_z", hk)]) == a_p + b_p
                row[f"L{l}H{h}"] = {"random_acc": float(ok[:10000].mean()), "far_acc": float(ok[far].mean())}
        S["head_ablation"][name] = row
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in models:
        byd = S["accuracy"][name]["by_travel_distance"]
        ax.plot([int(k) for k in byd], list(byd.values()), "o-", label=name)
    ax.set_xlabel("carry travel distance"); ax.set_ylabel("exact-sum accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_accuracy.png"), dpi=130)
    return S


if __name__ == "__main__":
    models = {name: (*train(name, cfg), cfg) for name, cfg in MODELS.items()}
    analyse(models)
    print("done")
