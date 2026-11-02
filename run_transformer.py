"""Binary addition in a small GPT, inspected with TransformerLens-style hooks.

    python run_transformer.py            # trains 3 models (~40 min on 2 CPU cores; resumable), then analyses (~20 min)
    python run_transformer.py --analyse  # analysis only, from saved checkpoints

TASK  Two NB-bit numbers written lowest bit first, then the sum:
        a_0 .. a_{NB-1}  +  b_0 .. b_{NB-1}  =  s_0 .. s_NB        ("lsb" order)
      or the same with the sum written highest bit first         ("msb" order).
      Loss is only on the sum tokens. Evaluation generates the sum token by token.

MODELS  lsb-L1: 1 layer, sum lowest-first      lsb-L2: 2 layers, lowest-first
        msb-L2: 2 layers, sum highest-first
      d_model 64, 4 heads, d_mlp 256, pre-LN, learned positions (a tiny GPT-2).

PREDICTION  Lowest-first makes addition local: carry into bit k follows from
      (a_{k-1}, b_{k-1}, s_{k-1}) alone, so the model never needs to look further back
      than one digit, and carry travel distance should not matter. Highest-first forces the
      model to resolve the whole carry chain before emitting its first digit, so travel
      distance should bite, as it did for the MLPs.

ANALYSES  1 accuracy by carry travel distance   2 attention patterns by relative offset
          3 activation patching of the residual stream, per (layer, position)
          4 linear probes for the carry   5 head ablations
"""
import argparse, json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tinygpt import TinyGPT

OUT = "results_transformer" if not os.environ.get("QUICK") else "/tmp/claude-0/rt_quick"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
NB = 10
ZERO, ONE, PLUS, EQ = 0, 1, 2, 3
T = 3 * NB + 3
POS_EQ = 2 * NB + 1
MODELS = {"lsb-L1": dict(layers=1, order="lsb"), "lsb-L2": dict(layers=2, order="lsb"),
          "msb-L2": dict(layers=2, order="msb")}
QUICK = bool(os.environ.get("QUICK"))          # QUICK=1: tiny smoke test of the whole pipeline
STEPS, BATCH = (30, 32) if QUICK else (2500, 128)
SC = 0.05 if QUICK else 1.0                       # scales evaluation set sizes


# ---------------- task ----------------
def bits(v, n):
    return (v[:, None] >> np.arange(n)) & 1


def encode(a, b, order):
    s = bits(a + b, NB + 1)
    if order == "msb":
        s = s[:, ::-1]
    n = len(a)
    return np.concatenate([bits(a, NB), np.full((n, 1), PLUS), bits(b, NB), np.full((n, 1), EQ), s], 1)


def decode_sum(tokens_sum, order):
    s = tokens_sum[:, ::-1] if order == "msb" else tokens_sum
    return (s * (2 ** np.arange(NB + 1))).sum(1)


def qpos(bit, order):
    """Position whose next token is sum bit `bit`."""
    j = bit if order == "lsb" else NB - bit
    return POS_EQ + j


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
        s = start[j]
        a[j] |= 1 << s; b[j] |= 1 << s
        for i in range(s + 1, s + L + 1):
            if r.random() < 0.5:
                a[j] |= 1 << i; b[j] &= ~(1 << i)
            else:
                b[j] |= 1 << i; a[j] &= ~(1 << i)
        if s + L + 1 <= NB - 1:
            a[j] &= ~(1 << (s + L + 1)); b[j] &= ~(1 << (s + L + 1))
    return a, b


def generate(model, a, b, order, hooks=None, bs=2000):
    """Greedy autoregressive decoding of the sum; returns predicted sums (ints)."""
    outs = []
    for i in range(0, len(a), bs):
        pre = encode(a[i:i + bs], b[i:i + bs], order)[:, :POS_EQ + 1]
        for _ in range(NB + 1):
            lg = model.forward(pre, hooks=hooks)
            nxt = np.clip(lg[:, -1, :2].argmax(1), 0, 1)       # only 0/1 are legal sum tokens
            pre = np.concatenate([pre, nxt[:, None]], 1)
        outs.append(decode_sum(pre[:, POS_EQ + 1:], order))
    return np.concatenate(outs)


# ---------------- training ----------------
def train(name, cfg):
    ck = os.path.join(OUT, f"{name}.npz")
    model = TinyGPT(d_vocab=4, n_ctx=T, d_model=64, n_heads=4, n_layers=cfg["layers"], d_mlp=256, seed=1)
    log = []
    step0 = 0
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        model.load({k: z[k] for k in model.P}); step0 = int(z["step"]); log = list(z["log"])
        model.m = {k: z["m_" + k] for k in model.P}; model.v = {k: z["v_" + k] for k in model.P}; model.t = step0
    r = np.random.default_rng(1000 + step0)
    mask = np.zeros((BATCH, T), bool); mask[:, POS_EQ:T - 1] = True
    t0 = time.time()
    for step in range(step0, STEPS):
        a, b = r.integers(0, 2 ** NB, BATCH), r.integers(0, 2 ** NB, BATCH)
        tok = encode(a, b, cfg["order"])
        lr = 1e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / STEPS)))
        loss, G = model.loss_and_grads(tok, mask)
        model.adam_step(G, lr, wd=0.01)
        if step % 50 == 0:
            log.append((step, loss))
            print(f"{name} step {step} loss {loss:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % 250 == 249 or step == STEPS - 1:
            np.savez(ck, step=step + 1, log=np.array(log), **model.P,
                     **{"m_" + k: v for k, v in model.m.items()}, **{"v_" + k: v for k, v in model.v.items()})
    return model, log


def load(name, cfg):
    z = np.load(os.path.join(OUT, f"{name}.npz"), allow_pickle=True)
    model = TinyGPT(d_vocab=4, n_ctx=T, d_model=64, n_heads=4, n_layers=cfg["layers"], d_mlp=256, seed=1)
    model.load({k: z[k] for k in model.P})
    return model, list(z["log"])


# ---------------- analyses ----------------
def analyse(models):
    S = {"settings": dict(bits=NB, steps=STEPS, batch=BATCH, d_model=64, heads=4, d_mlp=256)}
    rng = np.random.default_rng(7)
    # test pool for accuracy by travel distance
    a_r, b_r = rng.integers(0, 2 ** NB, int(4000 * SC)), rng.integers(0, 2 ** NB, int(4000 * SC))
    parts = [(a_r, b_r)] + [built_distance(int(300 * SC), L, np.random.default_rng(300 + L)) for L in range(2, NB)]
    a_p = np.concatenate([x[0] for x in parts]); b_p = np.concatenate([x[1] for x in parts])
    td = travel_dist(a_p, b_p)
    S["accuracy"] = {}
    for name, (model, log, cfg) in models.items():
        pred = generate(model, a_p, b_p, cfg["order"], bs=1000)
        ok = pred == a_p + b_p
        byd = {str(L): {"acc": float(ok[td == L].mean()), "n": int((td == L).sum())} for L in range(NB) if (td == L).sum() >= (50 if not QUICK else 1)}
        S["accuracy"][name] = {"random_exact": float(ok[:int(4000 * SC)].mean()), "by_travel_distance": byd,
                               "final_loss": float(log[-1][1])}
        print(name, "random acc", round(S["accuracy"][name]["random_exact"], 4),
              {k: round(v["acc"], 2) for k, v in byd.items()}, flush=True)

    # 2. attention by relative offset (teacher-forced correct sequences)
    a_t, b_t = rng.integers(0, 2 ** NB, int(2000 * SC)), rng.integers(0, 2 ** NB, int(2000 * SC))
    S["attention"] = {}
    OFFS = list(range(-4, 3))
    for name, (model, log, cfg) in models.items():
        tok = encode(a_t, b_t, cfg["order"])
        _, acts, _ = model.forward(tok, cache=True)
        prof = {}
        for l in range(cfg["layers"]):
            A = acts[f"blocks.{l}.attn.hook_pattern"]                 # (B, H, T, T)
            for h in range(A.shape[1]):
                tab = {kind: {str(d): [] for d in OFFS} for kind in ("a", "b", "s")}
                special = {"PLUS": [], "EQ": []}
                for bit in range(NB + 1):
                    q = qpos(bit, cfg["order"])
                    special["PLUS"].append(A[:, h, q, NB].mean()); special["EQ"].append(A[:, h, q, POS_EQ].mean())
                    for d in OFFS:
                        j = bit + d
                        if 0 <= j < NB:
                            tab["a"][str(d)].append(A[:, h, q, j].mean())
                            tab["b"][str(d)].append(A[:, h, q, NB + 1 + j].mean())
                        if 0 <= j <= NB:
                            kp = qpos(j, cfg["order"]) + 1 if cfg["order"] == "lsb" else POS_EQ + 1 + (NB - j)
                            if kp <= q:                                 # sum digit j already emitted
                                tab["s"][str(d)].append(A[:, h, q, kp].mean())
                prof[f"L{l}H{h}"] = {kind: {d: (float(np.mean(v)) if v else None) for d, v in t.items()} for kind, t in tab.items()}
                prof[f"L{l}H{h}"]["special"] = {k: float(np.mean(v)) for k, v in special.items()}
        S["attention"][name] = prof

    # 3. activation patching: resid_pre at (layer, position), source = opposite carry into bit k
    S["patching"] = {}
    for name, (model, log, cfg) in models.items():
        res = {}
        for k in [3, 6]:
            n = int(1000 * SC)
            a0, b0 = rng.integers(0, 2 ** NB, n), rng.integers(0, 2 ** NB, n)
            lo = 2 ** k; c0 = carry_into(a0, b0, k)
            a1, b1 = a0.copy(), b0.copy()
            for j in range(n):                                       # new low bits with the opposite carry
                while True:
                    x, y = rng.integers(0, lo), rng.integers(0, lo)
                    if int((x + y) >= lo) != c0[j]:
                        break
                a1[j] = a0[j] - a0[j] % lo + x; b1[j] = b0[j] - b0[j] % lo + y
            tok0, tok1 = encode(a0, b0, cfg["order"]), encode(a1, b1, cfg["order"])
            q = qpos(k, cfg["order"])
            _, acts1, _ = model.forward(tok1, cache=True)
            base = model.forward(tok0)[:, q, :2].argmax(1)
            target = tok1[:, q + 1]                                   # source's bit k (= clean bit k flipped)
            valid = base == tok0[:, q + 1]
            heat = np.zeros((cfg["layers"], T))
            for l in range(cfg["layers"]):
                src = acts1[f"blocks.{l}.hook_resid_pre"]
                for p in range(q + 1):
                    def hk(x, l=l, p=p, src=src):
                        x = x.copy(); x[:, p] = src[:, p]; return x
                    out = model.forward(tok0, hooks={f"blocks.{l}.hook_resid_pre": hk})[:, q, :2].argmax(1)
                    heat[l, p] = np.mean(out[valid] == target[valid])
            res[f"bit{k}"] = {"flip_rate": heat.tolist(), "query_pos": int(q), "n_valid": int(valid.sum())}
        S["patching"][name] = res

    # 4. linear probes for carry into bit k at the query position, per layer
    S["probes"] = {}
    a_q, b_q = rng.integers(0, 2 ** NB, int(2400 * SC)), rng.integers(0, 2 ** NB, int(2400 * SC))
    for name, (model, log, cfg) in models.items():
        untrained = TinyGPT(d_vocab=4, n_ctx=T, d_model=64, n_heads=4, n_layers=cfg["layers"], d_mlp=256, seed=99)
        tok = encode(a_q, b_q, cfg["order"])
        row = {}
        for k in [3, 6]:
            q = qpos(k, cfg["order"]); c = carry_into(a_q, b_q, k)
            for kind, mdl in [("trained", model), ("untrained", untrained)]:
                _, acts, tape = mdl.forward(tok, cache=True); del tape
                names = [f"blocks.{l}.hook_resid_pre" for l in range(cfg["layers"])] + ["ln_final.hook_normalized"]
                acts = {nm: acts[nm][:, q].copy() for nm in names}      # keep only the query position (memory)
                accs = []
                for nm in names:
                    Hq = acts[nm]; Hb = np.hstack([Hq, np.ones((len(Hq), 1))])
                    cut = int(1600 * SC)
                    w = np.linalg.lstsq(Hb[:cut], 2 * c[:cut] - 1, rcond=None)[0]
                    accs.append(float(np.mean((Hb[cut:] @ w > 0) == (c[cut:] == 1))))
                row[f"bit{k}_{kind}"] = accs
        S["probes"][name] = row

    # 5. head ablation (zero hook_z of one head), autoregressive accuracy on the pool
    S["head_ablation"] = {}
    for name, (model, log, cfg) in models.items():
        row = {}
        far = td >= 6
        for l in range(cfg["layers"]):
            for h in range(4):
                def hk(z, h=h):
                    z = z.copy(); z[:, :, h] = 0; return z
                pred = generate(model, a_p, b_p, cfg["order"], hooks={f"blocks.{l}.attn.hook_z": hk}, bs=1000)
                ok = pred == a_p + b_p
                row[f"L{l}H{h}"] = {"random_acc": float(ok[:int(4000 * SC)].mean()), "far_acc": float(ok[far].mean())}
        row["none"] = {"random_acc": S["accuracy"][name]["random_exact"],
                       "far_acc": float((generate(model, a_p, b_p, cfg["order"], bs=1000) == a_p + b_p)[far].mean())}
        S["head_ablation"][name] = row
        print(name, "ablation", {k: round(v["random_acc"], 3) for k, v in row.items()}, flush=True)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    figures(S, models)
    return S


def figures(S, models):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = list(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        byd = S["accuracy"][name]["by_travel_distance"]
        axes[0].plot([int(k) for k in byd], [v["acc"] for v in byd.values()], "o-", label=name)
        st, ls = zip(*models[name][1]); axes[1].plot(st, ls, label=name)
    axes[0].set_xlabel("carry travel distance"); axes[0].set_ylabel("exact-sum accuracy (generated)")
    axes[0].set_title("1. Accuracy vs carry travel distance"); axes[0].legend()
    axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("loss"); axes[1].set_title("Training loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_accuracy.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.5))
    for ax, name in zip(np.atleast_1d(axes), names):
        prof = S["attention"][name]; heads = list(prof)
        cols = [(kind, d) for kind in ("a", "b", "s") for d in range(-4, 3)] + [("PLUS", ""), ("EQ", "")]
        M = np.full((len(heads), len(cols)), np.nan)
        for i, hd in enumerate(heads):
            for j, (kind, d) in enumerate(cols):
                v = prof[hd]["special"][kind] if kind in ("PLUS", "EQ") else prof[hd][kind][str(d)]
                M[i, j] = np.nan if v is None else v
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.5, np.nanmax(M)), aspect="auto")
        ax.set_yticks(range(len(heads))); ax.set_yticklabels(heads)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{k}{d:+d}" if d != "" else k for k, d in cols], rotation=90, fontsize=7)
        ax.set_title(f"2. {name}: mean attention from the position predicting bit i\nto a/b/s at offset d (a-1 = a_(i-1))", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_attention.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.2 * len(names)))
    for i, name in enumerate(names):
        for j, k in enumerate([3, 6]):
            P_ = S["patching"][name][f"bit{k}"]; heat = np.array(P_["flip_rate"]); q = P_["query_pos"]
            ax = axes[i, j]
            im = ax.imshow(heat[:, :q + 1], cmap="Reds", vmin=0, vmax=1, aspect="auto")
            labels = [f"a{t}" for t in range(NB)] + ["+"] + [f"b{t}" for t in range(NB)] + ["="] + [f"s{t}" for t in range(NB + 1)]
            if models[name][2]["order"] == "msb":
                labels = labels[:POS_EQ + 1] + [f"s{NB - t}" for t in range(NB + 1)]
            ax.set_xticks(range(q + 1)); ax.set_xticklabels(labels[:q + 1], fontsize=6, rotation=90)
            ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels([f"L{l}" for l in range(heat.shape[0])])
            ax.set_title(f"3. {name}: patch resid_pre at one position from a problem with the opposite carry into bit {k}: "
                         f"P(output bit {k} flips)", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_patching.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, k in zip(axes, [3, 6]):
        for name in names:
            pr = S["probes"][name]
            ax.plot(pr[f"bit{k}_trained"], "o-", label=f"{name} trained")
            ax.plot(pr[f"bit{k}_untrained"], "s--", alpha=0.5, label=f"{name} untrained")
        ax.set_xlabel("layer (resid_pre; last = final LN)"); ax.set_ylabel("probe accuracy")
        ax.set_title(f"4. Linear probe for carry into bit {k} at the position predicting bit {k}"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "4_probes.png"), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        row = S["head_ablation"][name]; keys = list(row)
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [row[k]["random_acc"] for k in keys], 0.4, label="random problems")
        ax.bar(x + 0.2, [row[k]["far_acc"] for k in keys], 0.4, label="travel distance ≥ 6")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=45); ax.set_title(f"5. {name}: accuracy with one head zeroed"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "5_head_ablation.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analyse", action="store_true"); args = ap.parse_args()
    models = {}
    for name, cfg in MODELS.items():
        model, log = load(name, cfg) if args.analyse else train(name, cfg)
        models[name] = (model, log, cfg)
    analyse(models)
    print("done")
