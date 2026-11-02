"""Calibrate circuit-discovery methods on Qwenputer's compiled layers, whose wiring is known from the weights.

    python3 calibrate.py MODEL_DIR OUT_DIR B L REGIME        REGIME: zero (empty memory) or random (random 0/1 memory)

Runs an own forward pass of the 12 compiled layers (Gated DeltaNet + SwiGLU, Qwen3.5 semantics: zero-centred
RMSNorm, gated head norm, l2-normalised q/k, delta-rule recurrence) on B sequences of L random trace records,
nothing of Qwen's layers. For every live MLP unit the ground truth is its wiring: the dims its gate and up rows
read, excluding dims that never vary. Two discovery methods from this repo are run blind on the activations:
  M2  mean-ablation (results_storyroom inside(): replace a dim by its mean, see whether the unit changes);
  M1  correlation ranking (xorlab bool_label / corr: rank dims by |corr| with the unit, take the top k).
Scores: coverage (unit ever active), M2 recall of the true inputs, M1 recall at k = true fan-in and precision /
recall at |corr| >= 0.5, by layer and by fan-in. Writes OUT_DIR/calib_REGIME.json.
"""
import json, os, sys, time, collections
import torch, torch.nn.functional as F
from safetensors import safe_open

M, OUT, B, L, REGIME = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); torch.set_num_threads(8)
cfg = json.load(open(os.path.join(M, "config.json"))); chip = json.load(open(os.path.join(M, "chipformer.json")))
ix = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
QD, HID = chip["chip_dims_offset"], cfg["hidden_size"]; C = HID - QD
NK, DK, NV, DV = cfg["linear_num_key_heads"], cfg["linear_key_head_dim"], cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
EPS = cfg["rms_norm_eps"]
handles = {}
def T(n):
    s = ix[n]
    if s not in handles: handles[s] = safe_open(os.path.join(M, s), "pt")
    return handles[s].get_tensor(n)

# ---- trace tokens: records of the documented shapes with random payloads ----
tk = json.load(open(os.path.join(M, "tokenizer.json")))
fam = collections.defaultdict(list)
for t in tk["added_tokens"]:
    c = t["content"]
    if c.startswith("<|rv:"): fam[c[5:-2].split(":")[0]].append(t["id"])
for f in fam: fam[f].sort()
def pick(f): return fam[f][int(torch.randint(len(fam[f]), (1,)))]
def record():
    r = int(torch.randint(12, (1,)))
    if r < 3: return [pick("F0")]
    if r < 5: return [pick("V"), pick("F1")]
    if r == 5: return [pick("V"), pick("V"), pick("F2")]
    if r == 6: return [pick("NT")]
    if r == 7: return [pick("J")]
    if r == 8: return [pick("ST0" if torch.rand(1) < 0.5 else "ST1")]
    if r == 9: return [pick("V"), pick("V"), pick("V"), pick("SS0" if torch.rand(1) < 0.5 else "SS1")]
    if r == 10: return [pick("QS")] + [pick("QD") for _ in range(int(torch.randint(1, 4, (1,))))]
    return [pick("BD") for _ in range(int(torch.randint(1, 4, (1,))))] + [pick("V")]
seqs = []
for b in range(B):
    s = [pick("R")] if torch.rand(1) < 0.5 else [pick("GO")]
    while len(s) < L: s += record()
    seqs.append(s[:L])
tok = torch.tensor(seqs)                                       # (B, L)
E = T("model.embed_tokens.weight")
h = E[tok][:, :, QD:].float()                                  # chip dims only: compiled layers never read or write Qwen's dims
assert int((E[tok][:, :, :QD] != 0).sum()) == 0

def rmsnorm(x, w):                                             # zero-centred RMSNorm over the full hidden size (Qwen dims are 0 here)
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) / HID + EPS) * (1.0 + w)

results = {"regime": REGIME, "B": B, "L": L, "positions": B * L, "layers": []}
t0 = time.time()
for l in range(chip["chip_layers"]):
    assert cfg["layer_types"][l] == "linear_attention"
    P = lambda n: T(f"model.layers.{l}.{n}")
    w_in, w_post = P("input_layernorm.weight")[QD:].float(), P("post_attention_layernorm.weight")[QD:].float()
    Wqkv, Wz = P("linear_attn.in_proj_qkv.weight")[:, QD:].float(), P("linear_attn.in_proj_z.weight")[:, QD:].float()
    Wa, Wb = P("linear_attn.in_proj_a.weight")[:, QD:].float(), P("linear_attn.in_proj_b.weight")[:, QD:].float()
    conv = P("linear_attn.conv1d.weight").float(); A_log, dtb = P("linear_attn.A_log").float(), P("linear_attn.dt_bias").float()
    wn = P("linear_attn.norm.weight").float(); Wout = P("linear_attn.out_proj.weight")[QD:].float()
    # ---- Gated DeltaNet ----
    xn = rmsnorm(h, w_in)
    qkv = F.silu(F.conv1d((xn @ Wqkv.T).transpose(1, 2), conv, padding=conv.shape[-1] - 1, groups=conv.shape[0])[:, :, :L]).transpose(1, 2)
    q, k, v = torch.split(qkv, [NK * DK, NK * DK, NV * DV], -1)
    q = F.normalize(q.view(B, L, NK, DK), dim=-1, eps=1e-6).repeat_interleave(NV // NK, 2) * DK ** -0.5
    k = F.normalize(k.view(B, L, NK, DK), dim=-1, eps=1e-6).repeat_interleave(NV // NK, 2)
    v = v.view(B, L, NV, DV); z = (xn @ Wz.T).view(B, L, NV, DV)
    beta = torch.sigmoid(xn @ Wb.T); g = -A_log.exp() * F.softplus(xn @ Wa.T + dtb)      # (B, L, NV)
    S = torch.zeros(B, NV, DK, DV) if REGIME == "zero" else (torch.rand(B, NV, DK, DV) < 0.05).float()
    o = torch.zeros(B, L, NV, DV)
    for t in range(L):
        S = S * g[:, t].exp()[:, :, None, None]
        kv = torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - kv) * beta[:, t][..., None])
        o[:, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, t])
    o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * wn * F.silu(z)
    h = h + o.reshape(B, L, NV * DV) @ Wout.T
    # ---- MLP ----
    Gw, Uw, Dw = P("mlp.gate_proj.weight")[:, QD:].float(), P("mlp.up_proj.weight")[:, QD:].float(), P("mlp.down_proj.weight")[QD:].float()
    live = torch.nonzero((Gw != 0).any(1) & (Dw != 0).any(0))[:, 0]
    Gw, Uw, Dw = Gw[live], Uw[live], Dw[:, live]
    x2 = rmsnorm(h, w_post).reshape(B * L, C)
    G, U = x2 @ Gw.T, x2 @ Uw.T
    act = F.silu(G) * U                                         # (N, nlive)
    h = h + (act @ Dw.T).view(B, L, C)
    # ---- ground truth and the two methods ----
    N = x2.shape[0]
    mean = x2.mean(0); var = x2.var(0)
    varies = (x2.max(0).values - x2.min(0).values) > 0            # payload bits sit at 2^-36 after the ballast norm, so no relative floor
    active = (act != 0).any(0)
    xc = (x2 - mean); xs = xc / (xc.norm(dim=0) + 1e-12)
    ac = act - act.mean(0); as_ = ac / (ac.norm(dim=0) + 1e-12)
    corr = (xs.T @ as_).abs()                                   # (C, nlive)
    corr[~varies] = 0
    rows = []
    per_fanin = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0.0])   # n, active, m2 recall, m1 r@k, m1 prec, m1 rec
    for i in range(len(live)):
        supp = torch.nonzero((Gw[i] != 0) | (Uw[i] != 0))[:, 0]
        truth = [int(d) for d in supp if varies[d]]
        kf = len(truth)
        if kf == 0: continue
        act_i = act[:, i]
        if not active[i]:
            m2 = 0.0; r_k = 0.0; prec = 0.0; rec = 0.0
        else:
            # M2: mean-ablate each true input dim, does the unit change anywhere?
            found = 0
            for d in truth:
                dx = x2[:, d] - mean[d]
                a2 = F.silu(G[:, i] - Gw[i, d] * dx) * (U[:, i] - Uw[i, d] * dx)
                if (a2 - act_i).abs().max() > 1e-6 * (act_i.abs().max() + 1e-30): found += 1
            m2 = found / kf
            c = corr[:, i]
            top = torch.topk(c, kf).indices.tolist()
            r_k = len(set(top) & set(truth)) / kf
            sel = torch.nonzero(c >= 0.5)[:, 0].tolist()
            hit = len(set(sel) & set(truth))
            prec = hit / len(sel) if sel else 0.0; rec = hit / kf
        rows.append((kf, bool(active[i]), m2, r_k, prec, rec))
        pf = per_fanin[min(kf, 8)]; pf[0] += 1; pf[1] += int(active[i]); pf[2] += m2; pf[3] += r_k; pf[4] += prec; pf[5] += rec
    n = len(rows); na = sum(r[1] for r in rows)
    lay = {"layer": l, "live_units": int(len(live)), "scored_units": n, "active_units": na,
           "m2_recall_all": sum(r[2] for r in rows) / max(n, 1), "m2_recall_active": sum(r[2] for r in rows if r[1]) / max(na, 1),
           "m1_recall_at_k_active": sum(r[3] for r in rows if r[1]) / max(na, 1),
           "m1_precision_active": sum(r[4] for r in rows if r[1]) / max(na, 1), "m1_recall_active": sum(r[5] for r in rows if r[1]) / max(na, 1),
           "by_fanin": {k: {"n": v[0], "active": v[1], "m2": v[2] / v[0], "m1_at_k": v[3] / v[0], "m1_prec": v[4] / v[0], "m1_rec": v[5] / v[0]} for k, v in sorted(per_fanin.items())},
           "varying_dims": int(varies.sum()), "resid_abs_max": float(h.abs().max())}
    results["layers"].append(lay)
    print(f"layer {l:2d}: live {len(live):5d} scored {n:5d} active {na:5d} ({100*na/max(n,1):5.1f} %) | M2 recall all {lay['m2_recall_all']:.3f} active {lay['m2_recall_active']:.3f} "
          f"| M1 r@k {lay['m1_recall_at_k_active']:.3f} prec {lay['m1_precision_active']:.3f} rec {lay['m1_recall_active']:.3f} | varying dims {int(varies.sum())} | {time.time()-t0:.0f}s", flush=True)
tot = sum(l["scored_units"] for l in results["layers"]); act_tot = sum(l["active_units"] for l in results["layers"])
results["summary"] = {"scored_units": tot, "active_units": act_tot,
                      "m2_recall_all": sum(l["m2_recall_all"] * l["scored_units"] for l in results["layers"]) / tot,
                      "m2_recall_active": sum(l["m2_recall_active"] * l["active_units"] for l in results["layers"]) / max(act_tot, 1),
                      "m1_recall_at_k_active": sum(l["m1_recall_at_k_active"] * l["active_units"] for l in results["layers"]) / max(act_tot, 1),
                      "m1_precision_active": sum(l["m1_precision_active"] * l["active_units"] for l in results["layers"]) / max(act_tot, 1),
                      "m1_recall_active": sum(l["m1_recall_active"] * l["active_units"] for l in results["layers"]) / max(act_tot, 1)}
print("summary", json.dumps(results["summary"]))
json.dump(results, open(os.path.join(OUT, f"calib_{REGIME}.json"), "w"), indent=1)
