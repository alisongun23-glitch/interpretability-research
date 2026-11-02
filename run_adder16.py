"""16-bit addition: does a bigger problem force carry-like parts?

    python run_adder16.py             # a few minutes, writes results_adder16/

Networks: 32 inputs (two 16-bit numbers, lowest bit first) -> hidden layers -> 17 output
bits, tanh, Adam, trained on random problems. Depth 2 and depth 4, width 256, 3 seeds.

Tests:
  1. Accuracy by CARRY TRAVEL DISTANCE: the longest run of consecutive "propagate"
     positions (exactly one of a_i, b_i set) that a single carry has to pass through.
     A position that generates its own carry (a_i = b_i = 1) resets the distance to 0.
     Random problems rarely contain long travel distances (each extra position halves the
     odds), so problems are also built to have a given distance, then binned by the exact
     distance they turn out to have. A network that resolves carries locally will fail once
     the distance outgrows its reach. (Plain carry-chain length, which counts generate
     positions too, turns out NOT to predict failure; it is reported alongside.)
  2. Same-total swap (the 4-bit "collapse to total" test): replace a whole hidden layer with
     the activity of a different problem with the SAME total. Still right?
  3. Per-position XOR probes (a_i XOR b_i is not readable from raw inputs): trained vs untrained.
  4. Carry insertion at bits 4, 8, 12: copy the carry coordinate from a problem with the
     same high bits but the opposite carry. Does the answer move by exactly 2^k?
"""
import json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_adder16"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
T0 = time.time()
NB = 16
rng = np.random.default_rng(0)


def bits(v, n):
    return ((v[:, None] >> np.arange(n)) & 1).astype(float)


def make(n, r):
    a = r.integers(0, 2 ** NB, n); b = r.integers(0, 2 ** NB, n)
    return a, b


def encode(a, b):
    return np.concatenate([bits(a, NB), bits(b, NB)], 1)


def carries(a, b):
    """carry INTO each bit position 1..NB, shape (n, NB)."""
    out = np.zeros((len(a), NB), int); c = np.zeros(len(a), int)
    for i in range(NB):
        ai, bi = (a >> i) & 1, (b >> i) & 1
        c = (ai & bi) | ((ai ^ bi) & c)
        out[:, i] = c
    return out


def chain_len(a, b):
    C = carries(a, b); best = np.zeros(len(a), int); run = np.zeros(len(a), int)
    for i in range(NB):
        run = np.where(C[:, i] == 1, run + 1, 0); best = np.maximum(best, run)
    return best


def travel_dist(a, b):
    """Longest distance a carry travels: consecutive propagate positions fed by a carry."""
    best = np.zeros(len(a), int); dist = np.full(len(a), -1)      # -1 = no carry in flight
    for i in range(NB):
        ai, bi = (a >> i) & 1, (b >> i) & 1
        gen, prop = (ai & bi) == 1, (ai ^ bi) == 1
        dist = np.where(gen, 0, np.where(prop & (dist >= 0), dist + 1, -1))
        best = np.maximum(best, dist)
    return best


def built_distance(n, L, r):
    """Random problems forced to contain a carry that travels exactly L propagate positions
    (generate at s, propagate s+1..s+L, kill at s+L+1). Other bits stay random, so the true
    distance is recomputed and may be longer; bin by the recomputed value."""
    a, b = make(n, r)
    hi = NB - L - 1                                   # need s + L + 1 <= NB - 1 for the kill
    start = r.integers(0, max(hi, 0) + 1, n)
    for j in range(n):
        s = start[j]
        a[j] |= (1 << s); b[j] |= (1 << s)
        for i in range(s + 1, s + L + 1):
            if r.random() < 0.5:
                a[j] |= (1 << i); b[j] &= ~(1 << i)
            else:
                b[j] |= (1 << i); a[j] &= ~(1 << i)
        if s + L + 1 <= NB - 1:
            a[j] &= ~(1 << (s + L + 1)); b[j] &= ~(1 << (s + L + 1))
    return a, b


class MLP:
    def __init__(self, sizes, seed):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.P = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.P]; self.v = [np.zeros_like(p) for p in self.P]; self.t = 0

    def forward(self, x, edit=None):
        hs, h = [], x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i == len(self.W) - 1:
                return hs, z
            h = np.tanh(z)
            if edit is not None:
                h = edit(i, h)
            hs.append(h)

    def step(self, x, y, lr):
        hs, z = self.forward(x)
        d = (1 / (1 + np.exp(-z)) - y) / len(x)
        acts = [x] + hs; gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d; gb[i] = d.sum(0)
            if i:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        self.t += 1
        for j, (p, g) in enumerate(zip(self.P, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g; self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            p -= lr * (self.m[j] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.t)) + 1e-8)


def predict(net, a, b, edit=None):
    _, z = net.forward(encode(a, b), edit)
    return ((z > 0) * (2 ** np.arange(NB + 1))).sum(1)


# ---------------- data ----------------
a_tr, b_tr = make(60000, rng)
X_tr = encode(a_tr, b_tr); Y_tr = bits(a_tr + b_tr, NB + 1)
a_te, b_te = make(100000, np.random.default_rng(99))      # large, so every chain length has >= 50 cases
cl_te = chain_len(a_te, b_te)
cl_tr = chain_len(a_tr, b_tr)
td_tr = travel_dist(a_tr, b_tr)
parts = [make(20000, np.random.default_rng(98))]
parts += [built_distance(1500, L, np.random.default_rng(300 + L)) for L in range(2, NB)]
a_td = np.concatenate([x[0] for x in parts]); b_td = np.concatenate([x[1] for x in parts])
td_te = travel_dist(a_td, b_td)

DEPTHS, WIDTH, SEEDS, EPOCHS, BATCH = [2, 4], 256, [0, 1, 2], 40, 256
nets = {}
CK = os.path.join(OUT, "nets.npz")
saved = dict(np.load(CK)) if os.path.exists(CK) else {}
for D in DEPTHS:
    for sd in SEEDS:
        net = MLP([2 * NB] + [WIDTH] * D + [NB + 1], sd)
        if f"d{D}s{sd}_W0" in saved:                      # reuse trained weights from an earlier run
            for i in range(len(net.W)):
                net.W[i][...] = saved[f"d{D}s{sd}_W{i}"]; net.b[i][...] = saved[f"d{D}s{sd}_b{i}"]
            nets[(D, sd)] = net
            continue
        r = np.random.default_rng(100 + sd)
        for ep in range(EPOCHS):
            lr = 2e-3 * (0.5 * (1 + np.cos(np.pi * ep / EPOCHS))) + 1e-5
            idx = r.permutation(len(X_tr))
            for k in range(0, len(idx), BATCH):
                j = idx[k:k + BATCH]; net.step(X_tr[j], Y_tr[j], lr)
        nets[(D, sd)] = net
        print(f"trained depth {D} seed {sd}  {time.time() - T0:.0f}s", flush=True)
np.savez(CK, **{f"d{D}s{sd}_{n}{i}": a for (D, sd), net in nets.items()
                for n, arr in [("W", net.W), ("b", net.b)] for i, a in enumerate(arr)})
untrained = {D: MLP([2 * NB] + [WIDTH] * D + [NB + 1], 999) for D in DEPTHS}

S = {"settings": {"bits": NB, "depths": DEPTHS, "width": WIDTH, "seeds": SEEDS, "epochs": EPOCHS,
                  "train_problems": len(X_tr), "test_problems": len(a_te)},
     "chain_length_counts": {"train": {int(k): int(v) for k, v in zip(*np.unique(cl_tr, return_counts=True))},
                             "test": {int(k): int(v) for k, v in zip(*np.unique(cl_te, return_counts=True))}},
     "travel_distance_counts": {"train": {int(k): int(v) for k, v in zip(*np.unique(td_tr, return_counts=True))},
                                "test_pool": {int(k): int(v) for k, v in zip(*np.unique(td_te, return_counts=True))}}}

# ---------------- 1. accuracy ----------------
acc = {}
for (D, sd), net in nets.items():
    pr = predict(net, a_te, b_te); tot = a_te + b_te
    _, z = net.forward(encode(a_te, b_te))
    perbit = np.mean((z > 0) == (bits(tot, NB + 1) == 1), 0)
    bych = {}
    for L in range(0, NB + 1):
        m = cl_te == L
        if m.sum() >= 50:
            bych[str(L)] = float(np.mean(pr[m] == tot[m]))
    prd = predict(net, a_td, b_td); totd = a_td + b_td
    bydist = {}
    for L in range(0, NB):
        m = td_te == L
        if m.sum() >= 50:
            bydist[str(L)] = {"acc": float(np.mean(prd[m] == totd[m])), "n": int(m.sum())}
    acc[f"d{D}_s{sd}"] = {"exact": float(np.mean(pr == tot)), "per_bit": perbit.tolist(), "by_chain": bych,
                          "by_travel_distance": bydist}
S["accuracy"] = acc
print({k: round(v["exact"], 3) for k, v in acc.items()})

# ---------------- 2. does a layer collapse problems to their total? ----------------
# Distance inside a layer between two problems with the SAME total, divided by the distance
# between two random problems. 0 = same-total problems are identical inside (full collapse);
# about 1 = the total plays no special role. (Swapping a whole layer is NOT a valid test:
# the network would then just answer the other problem, which has the same answer.)
r2 = np.random.default_rng(5)
a_s, b_s = make(4000, r2)
tot_s = a_s + b_s
a2 = np.array([r2.integers(max(0, t - (2 ** NB - 1)), min(t, 2 ** NB - 1) + 1) for t in tot_s]); b2 = tot_s - a2
a_r, b_r = make(4000, r2)


def collapse_ratio(net):
    H1, H2, Hr = net.forward(encode(a_s, b_s))[0], net.forward(encode(a2, b2))[0], net.forward(encode(a_r, b_r))[0]
    return [float(np.mean(np.linalg.norm(H1[L] - H2[L], axis=1)) / np.mean(np.linalg.norm(H1[L] - Hr[L], axis=1)))
            for L in range(len(H1))]


swap = {f"d{D}_s{sd}": collapse_ratio(net) for (D, sd), net in nets.items()}
swap.update({f"untrained_d{D}": collapse_ratio(net) for D, net in untrained.items()})
swap["raw_input"] = float(np.mean(np.linalg.norm(encode(a_s, b_s) - encode(a2, b2), axis=1)) /
                          np.mean(np.linalg.norm(encode(a_s, b_s) - encode(a_r, b_r), axis=1)))
S["same_total_distance_ratio"] = swap

# ---------------- 3. XOR probes ----------------
def probe_acc(Htr, ytr, Hte, yte):
    A = np.hstack([Htr, np.ones((len(Htr), 1))])
    w = np.linalg.lstsq(A, 2 * ytr - 1, rcond=None)[0]
    return float(np.mean((np.hstack([Hte, np.ones((len(Hte), 1))]) @ w > 0) == (yte == 1)))


pa, pb = make(6000, np.random.default_rng(11)); qa, qb = make(3000, np.random.default_rng(12))
xor_p, xor_q = ((pa ^ pb)[:, None] >> np.arange(NB)) & 1, ((qa ^ qb)[:, None] >> np.arange(NB)) & 1
probes = {"raw": [probe_acc(encode(pa, pb), xor_p[:, i], encode(qa, qb), xor_q[:, i]) for i in range(NB)]}
for D in DEPTHS:
    for kind, net in [("trained", nets[(D, 0)]), ("untrained", untrained[D])]:
        Hp, Hq = net.forward(encode(pa, pb))[0], net.forward(encode(qa, qb))[0]
        probes[f"{kind}_d{D}"] = [[probe_acc(Hp[L], xor_p[:, i], Hq[L], xor_q[:, i]) for i in range(NB)]
                                  for L in range(D)]
S["xor_probes"] = probes

# ---------------- 4. carry insertion ----------------
def opposite_carry_source(a, b, k, r):
    """Same bits >= k, new low bits chosen so the carry into bit k flips."""
    lo = 2 ** k; c = ((a % lo) + (b % lo)) >= lo
    a2, b2 = a.copy(), b.copy()
    for j in range(len(a)):
        while True:
            x, y = r.integers(0, lo), r.integers(0, lo)
            if (x + y >= lo) != c[j]:
                break
        a2[j] = (a[j] - a[j] % lo) + x; b2[j] = (b[j] - b[j] % lo) + y
    return a2, b2, c


ins = {}
r4 = np.random.default_rng(7)
ka, kb = make(4000, np.random.default_rng(21))
for k in [4, 8, 12]:
    a_src, b_src, c = opposite_carry_source(ka, kb, k, r4)
    target = ka + kb + np.where(c, -1, 1) * 2 ** k        # answer if the carry into bit k flipped
    for (D, sd), net in nets.items():
        if sd != 0:
            continue
        H, Hs = net.forward(encode(ka, kb))[0], net.forward(encode(a_src, b_src))[0]
        Hfit = net.forward(X_tr[:20000])[0]
        cfit = carries(a_tr[:20000], b_tr[:20000])[:, k - 1]
        base_ok = predict(net, ka, kb) == ka + kb
        row = {}
        for L in range(D):
            A = np.hstack([Hfit[L], np.ones((20000, 1))])
            w = np.linalg.lstsq(A, 2 * cfit - 1, rcond=None)[0][:-1]; u = w / np.linalg.norm(w)
            ur = r4.normal(size=u.shape); ur /= np.linalg.norm(ur)
            for nm, dvec in [("probe", u), ("random", ur)]:
                e = lambda i, h, L=L, dvec=dvec: h + ((Hs[L] - h) @ dvec)[:, None] * dvec[None, :] if i == L else h
                got = predict(net, ka, kb, e)
                row[f"L{L + 1}_{nm}"] = float(np.mean(got[base_ok] == target[base_ok]))
                if nm == "probe":
                    row[f"L{L + 1}_unchanged"] = float(np.mean(got[base_ok] == (ka + kb)[base_ok]))
        ins[f"d{D}_bit{k}"] = row
S["carry_insertion"] = ins
S["runtime_seconds"] = time.time() - T0
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for D, col in zip(DEPTHS, ["tab:blue", "tab:orange"]):
    for sd in SEEDS:
        A_ = acc[f"d{D}_s{sd}"]
        kd = list(A_["by_travel_distance"])
        axes[0].plot([int(k) for k in kd], [A_["by_travel_distance"][k]["acc"] for k in kd], "o-", c=col, alpha=0.7,
                     label=f"depth {D}" if sd == 0 else None)
        kc = list(A_["by_chain"])
        axes[1].plot([int(k) for k in kc], [A_["by_chain"][k] for k in kc], "o-", c=col, alpha=0.7,
                     label=f"depth {D}" if sd == 0 else None)
        axes[2].plot(A_["per_bit"], c=col, alpha=0.7, label=f"depth {D}" if sd == 0 else None)
axes[0].set_xlabel("carry travel distance (propagate positions one carry crosses)"); axes[0].set_ylabel("exact-sum accuracy")
axes[0].set_title("1a. Accuracy vs how far one carry travels"); axes[0].legend()
axes[1].set_xlabel("longest carry chain, generates included (random problems)")
axes[1].set_title("1b. Chain length alone does not predict failure"); axes[1].legend()
axes[2].set_xlabel("output bit"); axes[2].set_ylabel("per-bit accuracy"); axes[2].set_title("1c. Per-bit accuracy"); axes[2].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_accuracy.png"), dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, len(DEPTHS), figsize=(13, 3.8), sharey=True)
for ax, D in zip(axes, DEPTHS):
    ax.plot(probes["raw"], "k:", label="raw input")
    for L in range(D):
        ax.plot(probes[f"trained_d{D}"][L], "o-", label=f"trained layer {L + 1}")
    ax.plot(probes[f"untrained_d{D}"][D - 1], "s--", c="gray", label="untrained, last layer")
    ax.set_xlabel("bit position i"); ax.set_title(f"3. Probe for a_i XOR b_i, depth {D}"); ax.legend(fontsize=7)
axes[0].set_ylabel("held-out probe accuracy")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_xor_probes.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.8))
labels, vals = [], []
for key, row in ins.items():
    for kk, v in row.items():
        if kk.endswith("_probe") or kk.endswith("_random"):
            labels.append(f"{key}\n{kk}"); vals.append(v)
ax.bar(range(len(vals)), vals); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_ylabel("answer moved by exactly 2^k"); ax.set_title("4. Carry insertion (seed 0)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "4_carry_insertion.png"), dpi=130); plt.close(fig)

print("same-total distance ratio", {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 2)) for k, v in swap.items()})
for k, v in ins.items():
    print("insert", k, {a: round(b, 2) for a, b in v.items()})
for k, v in acc.items():
    print(k, round(v["exact"], 3), "by distance", {c: round(x["acc"], 2) for c, x in v["by_travel_distance"].items()})
print("train travel-distance counts", S["travel_distance_counts"]["train"])
print("test pool counts", S["travel_distance_counts"]["test_pool"])
print("xor raw", np.round(probes["raw"], 2).tolist())
for k, v in probes.items():
    if k != "raw":
        print(k, [np.round(l, 2).tolist() for l in v])
print(f"done {time.time() - T0:.0f}s")
