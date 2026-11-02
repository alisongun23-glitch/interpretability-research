"""Which of Qwen's own tokens carry computer features in their embedding rows (the bridge's text classes)."""
import json, os, sys, collections, torch
from safetensors import safe_open
M = sys.argv[1]
ix = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
f = safe_open(os.path.join(M, ix["model.embed_tokens.weight"]), "pt")
E = f.get_slice("model.embed_tokens.weight")
tk = json.load(open(os.path.join(M, "tokenizer.json")))
vocab = {v: k for k, v in tk["model"]["vocab"].items()}
for t in tk.get("added_tokens", []): vocab[t["id"]] = t["content"]
NQ, QD = 248320, 2560
count = torch.zeros(4608 - QD, dtype=torch.long); per_tok = {}
for s in range(0, NQ, 16384):
    X = E[s:min(s + 16384, NQ), QD:].float()
    nzm = X != 0
    count += nzm.sum(0)
    for i in torch.nonzero(nzm.sum(1) > 0)[:, 0].tolist():
        per_tok[s + i] = torch.nonzero(nzm[i])[:, 0] + QD
ballast = set((torch.nonzero(count == NQ)[:, 0] + QD).tolist())
print("Qwen tokens with any chip-dim feature:", len(per_tok), "of", NQ, "| dims lit on every Qwen token (ballast):", len(ballast), sorted(ballast)[:8], "...")
cls = collections.defaultdict(list)
for t, d in per_tok.items():
    extra = tuple(x for x in d.tolist() if x not in ballast)
    if extra: cls[extra].append(t)
print("class patterns (sets of extra chip dims):", len(cls))
for extra, toks in sorted(cls.items(), key=lambda kv: -len(kv[1]))[:24]:
    names = [vocab.get(t, "?") for t in toks[:8]]
    vals = [float(E[toks[0]:toks[0] + 1, x:x + 1].float()[0, 0]) for x in extra[:3]]
    print(f"  dims {list(extra)[:6]}{'...' if len(extra) > 6 else ''}: {len(toks)} tokens, e.g. {names}, values {vals}")
