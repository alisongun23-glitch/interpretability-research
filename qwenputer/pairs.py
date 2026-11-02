"""Threshold pairs and the bridge: for each compiled layer, count MLP units that come in +/- pairs writing the same
output dim with gate rows equal except one constant (prims.t_ge / t_lt: [z >= t] = relu(z - t + 1) - relu(z - t)),
and count units whose gate row reads Qwen's own dims (< 2560): the bridge's text classes."""
import json, os, sys, collections, torch
from safetensors import safe_open
M = sys.argv[1]
ix = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
H = {}
def T(n):
    s = ix[n]; H.setdefault(s, safe_open(os.path.join(M, s), "pt")); return H[s].get_tensor(n)
tot_live = tot_pairs = 0
for l in range(12):
    G = T(f"model.layers.{l}.mlp.gate_proj.weight").float(); Dn = T(f"model.layers.{l}.mlp.down_proj.weight").float()
    live = torch.nonzero((G != 0).any(1) & (Dn != 0).any(0))[:, 0]
    reads_qwen = int((G[live][:, :2560] != 0).any(1).sum())
    # group live units by their output dim set and by gate row without its largest-magnitude (constant) entry
    key = collections.defaultdict(list)
    for i in live.tolist():
        r = G[i]; d = torch.nonzero(r)[:, 0]
        if len(d) == 0: continue
        c = d[r[d].abs().argmax()]                      # the constant dim: largest |weight| (2^40 ballast folded)
        rest = tuple((int(j), float(r[j])) for j in d.tolist() if j != int(c))
        outs = tuple(torch.nonzero(Dn[:, i])[:, 0].tolist())
        key[(rest, outs)].append((i, float(r[c]), [float(Dn[o, i]) for o in outs]))
    pairs = sum(1 for v in key.values() if len(v) == 2 and v[0][2] and v[1][2] and all(a == -b for a, b in zip(v[0][2], v[1][2])))
    fan = collections.Counter(len(k[0]) for k, v in key.items() for _ in v)
    tot_live += len(live); tot_pairs += 2 * pairs
    print(f"layer {l:2d}: live {len(live):5d}; +/- threshold pairs {pairs:5d} ({2*pairs} units); read Qwen dims {reads_qwen:4d}; "
          f"fan-in excl. constant (top): {sorted(fan.items(), key=lambda kv: -kv[1])[:5]}")
print("total live units", tot_live, "in threshold pairs", tot_pairs)
# which Qwen embedding dims are read on text positions, and which tokens light them: the text classes
G0 = T("model.layers.0.mlp.gate_proj.weight").float()
qd = torch.nonzero((G0[:, :2560] != 0).any(0))[:, 0]
print("layer 0 gate rows read", len(qd), "Qwen dims, e.g.", qd[:10].tolist())
