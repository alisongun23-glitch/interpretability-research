"""Look at the compiled computer layers of Qwenputer-Filesystem without loading the model.

Usage: python3 -I inspect_weights.py MODEL_DIR
Reads tensors one at a time through safetensors' memory map. Reports, for the 12 compiled layers and
Qwen's first own layer: how many weights are nonzero; whether the DeltaNet keys are one-hot (a RAM
address); whether the decay gate is held open (alpha = 1); the MLP units that are AND gadgets
(gate row = 4 * (sum of literals) - (4n - 2)); and what a trace token's embedding looks like.
"""
import json, os, sys, collections

import torch
from safetensors import safe_open

M = sys.argv[1]
cfg = json.load(open(os.path.join(M, "config.json")))
chip = json.load(open(os.path.join(M, "chipformer.json")))
ix = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
H, QD = cfg["hidden_size"], chip["chip_dims_offset"]          # 4608, Qwen's dims are 0..2559
print("hidden", H, "qwen dims", QD, "chip layers", chip["chip_layers"], "qwen layers from", chip["qwen_layers_from"],
      "chip vocab", chip["chip_vocab"], "at offset", chip["chip_token_offset"], "circuit", chip["circuit"])
handles = {}
def T(name):
    sh = ix[name]
    if sh not in handles:
        handles[sh] = safe_open(os.path.join(M, sh), "pt")
    return handles[sh].get_tensor(name)

def nz(t): return int((t != 0).sum())

print("\n== nonzero weights per layer (bf16 tensors) ==")
rows = []
for l in list(range(chip["chip_layers"])) + [chip["qwen_layers_from"], chip["qwen_layers_from"] + 3]:
    kind = cfg["layer_types"][l]
    names = [k for k in ix if k.startswith(f"model.layers.{l}.")]
    tot = n = 0
    per = {}
    for k in names:
        t = T(k); c = nz(t); per[k.split(f"layers.{l}.")[1]] = (c, t.numel()); tot += t.numel(); n += c
    rows.append((l, kind, n, tot))
    top = sorted(per.items(), key=lambda kv: -kv[1][0])[:4]
    print(f"layer {l:2d} {kind:16s} nonzero {n:>12,d} of {tot:>13,d} ({100*n/tot:6.3f} %)  biggest:",
          ", ".join(f"{k} {c:,d}/{m:,d}" for k, (c, m) in top))

print("\n== DeltaNet decay and write gates in the chip layers ==")
for l in range(chip["chip_layers"]):
    if cfg["layer_types"][l] != "linear_attention": continue
    A = T(f"model.layers.{l}.linear_attn.A_log").float(); dtb = T(f"model.layers.{l}.linear_attn.dt_bias").float()
    # g = -exp(A_log) * softplus(a + dt_bias); alpha = exp(g).  With a = 0 (chip dims) alpha = exp(-exp(A_log) * softplus(dt_bias)).
    alpha0 = torch.exp(-torch.exp(A) * torch.nn.functional.softplus(dtb))
    print(f"layer {l:2d}: A_log min/max {A.min():.2f}/{A.max():.2f}  dt_bias min/max {dtb.min():.2f}/{dtb.max():.2f}  "
          f"alpha at a=0: min {alpha0.min():.4f} max {alpha0.max():.4f}  heads with alpha==1: {int((alpha0 > 0.9999).sum())}/{len(alpha0)}")

print("\n== are the chip layers' keys one-hot addresses? (in_proj_qkv rows, layer 0..2) ==")
nk, dk, nv, dv = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"], cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
for l in range(3):
    W = T(f"model.layers.{l}.linear_attn.in_proj_qkv.weight").float()       # (q + k + v rows, H)
    q, k, v = W[: nk * dk], W[nk * dk: 2 * nk * dk], W[2 * nk * dk:]
    for nm, X in (("q", q), ("k", k), ("v", v)):
        rows_nz = (X != 0).sum(1)
        used = rows_nz > 0
        print(f"layer {l} {nm}: rows used {int(used.sum())}/{X.shape[0]}; nonzeros per used row: median {rows_nz[used].float().median() if used.any() else 0:.0f}, "
              f"max {int(rows_nz.max())}; reads Qwen dims (< {QD}): {int((X[:, :QD] != 0).any(1).sum())} rows; "
              f"distinct |weights|: {len(torch.unique(X[X != 0].abs()))}")

print("\n== MLP units of the chip layers: AND gadgets ==")
# q35 prims.t_and: gate pre-activation 4 * sum(lits) - (4n - 2); a literal is a dim (coef 4) or its negation (one - dim).
# With no bias, the constant comes from a constant dim (the 'one' dim, carried in the ballast).  We look for gate rows whose
# nonzero weights are all +-4 or a multiple of 4 and count them by number of literals.
for l in range(chip["chip_layers"]):
    G = T(f"model.layers.{l}.mlp.gate_proj.weight").float()        # (units, H)
    U = T(f"model.layers.{l}.mlp.up_proj.weight").float()
    Dn = T(f"model.layers.{l}.mlp.down_proj.weight").float()       # (H, units)
    live = (G != 0).any(1) & (Dn != 0).any(0)
    n_live = int(live.sum())
    g = G[live]
    nlit = (g != 0).sum(1)
    # AND signature (prims.t_and): gate = w * (sum of n literals) - (n - 1/2) * w (+ m * w for m negated literals), so the
    # one weight that is not +-w is a HALF-INTEGER multiple of w.  Threshold gadgets (t_ge / t_lt) have an integer constant.
    and_n, and_const, thr = collections.Counter(), collections.Counter(), 0
    for r in g:
        d = torch.nonzero(r)[:, 0]; vals = r[d]
        if len(d) < 2: continue
        w = collections.Counter(vals.abs().tolist()).most_common(1)[0][0]
        rest = [(int(j), float(x)) for j, x in zip(d.tolist(), vals.tolist()) if abs(abs(x) - w) > 1e-6 * w]
        n = len(d) - len(rest)
        if len(rest) == 1:
            ratio = rest[0][1] / w
            if abs(ratio * 2 - round(ratio * 2)) < 1e-3 and abs(ratio - round(ratio)) > 0.25:
                and_n[n] += 1; and_const[rest[0][0]] += 1
            elif abs(ratio - round(ratio)) < 1e-3:
                thr += 1
    cnt = collections.Counter(int(x) for x in nlit.tolist())
    print(f"layer {l:2d}: live units {n_live:,d}/{G.shape[0]:,d}; AND gadgets {sum(and_n.values()):,d} (literals: {sorted(and_n.items())[:8]}), "
          f"constant dim {and_const.most_common(2)}; integer-threshold units {thr:,d}; nonzeros per gate row (top): {sorted(cnt.items(), key=lambda kv: -kv[1])[:5]}; "
          f"up_proj nonzeros per row median {int((U[live] != 0).sum(1).float().median()) if n_live else 0}")
    if l == 0 and n_live:
        for i in torch.nonzero(live)[:5, 0].tolist():
            r = G[i]; d = torch.nonzero(r)[:, 0].tolist()
            print("   unit", i, "gate:", {int(j): float(r[j]) for j in d[:8]}, "| up:", {int(j): float(U[i, j]) for j in torch.nonzero(U[i])[:4, 0].tolist()},
                  "| writes dims:", torch.nonzero(Dn[:, i])[:4, 0].tolist(), "coef", [float(Dn[j, i]) for j in torch.nonzero(Dn[:, i])[:3, 0].tolist()])

print("\n== embeddings ==")
E = T("model.embed_tokens.weight")
off = chip["chip_token_offset"]
for tid, nm in ((0, "qwen token 0"), (off + 5, "trace token 5"), (off + 1000, "trace token 1000"), (off + chip["chip_vocab"] - 1, "last trace token")):
    e = E[tid].float(); d = torch.nonzero(e)[:, 0]
    big = e.abs().max()
    print(f"{nm:18s}: nonzero dims {len(d)} (in Qwen dims {int((d < QD).sum())}, chip dims {int((d >= QD).sum())}); |max| {big:.3e} = 2^{float(torch.log2(big)) if big > 0 else float('nan'):.1f}; "
          f"first chip-dim values {[(int(j), float(e[j])) for j in d[d >= QD][:5].tolist()]}")
w = T("model.norm.weight").float()
print("final norm gains: Qwen dims median %.3f; chip dims nonzero %d, max %.3e (= 2^%.1f) at dim %d" % (
    w[:QD].median(), int((w[QD:] != 0).sum()), w[QD:].abs().max(), float(torch.log2(w[QD:].abs().max())), QD + int(w[QD:].abs().argmax())))
print("tau dim", chip["tau_dim"], "gain %.3e" % w[chip["tau_dim"]], "| mu dim", chip["mu_dim"], "gain %.3e" % w[chip["mu_dim"]],
      "| code dims gain", "%.3e" % w[chip["code_dims"][0]])

print("\n== tokens ==")
tk = json.load(open(os.path.join(M, "tokenizer.json")))
added = {t["id"]: t["content"] for t in tk.get("added_tokens", [])}
print("added tokens:", len(added))
for tid in chip["tool_call"] + [248066, 248067, 248068, 248069] + [off, off + 1, off + 2, off + 3, off + 100, off + chip["chip_vocab"] - 1]:
    print(f"  {tid}: {added.get(tid, '(not an added token)')!r}")
