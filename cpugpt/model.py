"""A 2-layer GPT whose second attention layer has three ordinary heads and one CPU head.

The CPU head, at the position that must predict the answer:
  1. points: three slots (A, B, OP), each a softmax over earlier positions from its own trained
     query and key maps; one position is sampled per slot (training) or the argmax taken (test);
  2. sends: the picked tokens go through FROZEN tables (number token -> value, operator word ->
     opcode) into the computer;
  3. runs: the gate computer runs the calculator program (results come from cpu_table, which was
     computed on the gates; or from a live call supplied by the caller);
  4. reads back: if the head decides to call (a yes/no choice from a trained gate, sampled in
     training, gate >= 0.5 at test), the 8 result bits pass a trained projection into the residual.
No gradient passes through the computer or through the yes/no choice; the pointers and the gate
learn by REINFORCE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, n_heads, dh):
        super().__init__()
        self.h, self.dh = n_heads, dh
        self.qkv = nn.Linear(d, 3 * n_heads * dh)
        self.o = nn.Linear(n_heads * dh, d)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        A = s.softmax(-1)
        return self.o((A @ v).transpose(1, 2).reshape(B, T, -1)), A


class CPUHead(nn.Module):
    SLOTS = ("A", "B", "OP")

    def __init__(self, d, dk, numval, opval, table):
        super().__init__()
        self.dk = dk
        self.q = nn.Linear(d, 3 * dk)
        self.k = nn.Linear(d, 3 * dk)
        self.read = nn.Linear(8, d)
        self.gate = nn.Linear(d, 1)
        self.register_buffer("numval", torch.as_tensor(numval))
        self.register_buffer("opval", torch.as_tensor(opval))
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.long))
        self.register_buffer("bitw", 2 ** torch.arange(8))

    def scores(self, h, last):
        """Pointing logits (B, 3, T) at the last position over positions 0..last."""
        B, T, _ = h.shape
        q = self.q(h[torch.arange(B), last]).view(B, 3, self.dk)
        k = self.k(h).view(B, T, 3, self.dk)
        s = torch.einsum("bsd,btsd->bst", q, k) / math.sqrt(self.dk)
        later = torch.arange(T)[None, :] > last[:, None]
        return s.masked_fill(later[:, None, :], float("-inf"))

    def inputs(self, tokens, picks):
        """picks (B, K, 3) positions -> x, y, op (B, K) as the computer receives them."""
        tk = torch.gather(tokens[:, None, :].expand(-1, picks.shape[1], -1), 2, picks)
        return self.numval[tk[..., 0]], self.numval[tk[..., 1]], self.opval[tk[..., 2]]

    def run(self, x, y, op):
        return self.table[op, x, y]

    def bits(self, result):
        return ((result[..., None] & self.bitw) > 0).float()


class GPT(nn.Module):
    def __init__(self, vocab, n_ctx, d=96, n_heads=4, d_mlp=384, cpu=None):
        """cpu: None for the baseline, else dict(numval, opval, table) for a CPU head."""
        super().__init__()
        dh = d // n_heads
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(n_ctx, d)
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.attn0 = Attn(d, n_heads, dh)
        self.attn1 = Attn(d, n_heads - 1 if cpu else n_heads, dh)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d)) for _ in range(2)])
        self.lnf = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, vocab)
        self.cpuhead = CPUHead(d, dh, **cpu) if cpu else None

    def forward(self, tokens, last, K=1, mode="sample", results=None, op_map=None):
        """Logits for the token after position `last`.
        mode: "sample" (K sampled pointings and call decisions), "gated" (argmax pointers, call if
              gate >= 0.5), "always" (argmax pointers, always call), "off" (never call). results: optional (B, K) answers to use instead of the table
              (e.g. from a live analog computer). op_map: optional tensor remapping opcodes (ablation).
        Returns dict with logits (B, K, V) and, for the CPU head, scores, picks, results, gate."""
        B, T = tokens.shape
        x = self.emb(tokens) + self.pos(torch.arange(T))[None]
        a0, A0 = self.attn0(self.ln[0](x)); x = x + a0
        x = x + self.mlp[0](self.ln[1](x))
        h = self.ln[2](x)
        a1, A1 = self.attn1(h); x = x + a1
        xl = x[torch.arange(B), last]                                   # (B, d)
        out = {"attn": (A0, A1)}
        if self.cpuhead is None:
            xk = xl[:, None, :]
        else:
            s = self.cpuhead.scores(h, last)
            out["scores"] = s
            if mode == "sample":
                picks = torch.distributions.Categorical(logits=s[:, None].expand(-1, K, -1, -1)).sample()   # (B, K, 3)
            else:
                picks = s.argmax(-1)[:, None, :]
            xv, yv, ov = self.cpuhead.inputs(tokens, picks)
            if op_map is not None:
                ov = op_map[ov]
            res = self.cpuhead.run(xv, yv, ov) if results is None else results
            g = torch.sigmoid(self.cpuhead.gate(h[torch.arange(B), last]))[:, 0]    # (B,)
            if mode == "sample":
                call = torch.bernoulli(g[:, None].detach().expand(-1, picks.shape[1]))
            elif mode == "gated":
                call = (g[:, None] >= 0.5).float()
            else:
                call = torch.full((B, 1), 1.0 if mode == "always" else 0.0)
            xk = xl[:, None, :] + call[..., None] * self.cpuhead.read(self.cpuhead.bits(res))
            out.update(picks=picks, x=xv, y=yv, op=ov, results=res, gate=g, call=call)
        xk = xk + self.mlp[1](self.ln[3](xk))
        out["logits"] = self.unembed(self.lnf(xk))
        return out


def losses(model, b, K, shaping=0.0, entropy=0.01, call_cost=0.0):
    """Cross-entropy on the answer (averaged over the K samples) plus, for the CPU head, REINFORCE
    with leave-one-out baselines: the pointers are rewarded by the answer's log-likelihood (plus
    `shaping` when the computer's result is the answer); the call decision by the answer's
    log-likelihood minus `call_cost` per call."""
    tok, last, tgt = b["tokens"], b["last"], b["target"]
    o = model(tok, last, K=K if model.cpuhead is not None else 1)
    logp = o["logits"].log_softmax(-1)                                  # (B, K, V)
    lp_t = logp.gather(2, tgt[:, None, None].expand(-1, logp.shape[1], 1))[..., 0]   # (B, K)
    ce = -lp_t.mean()
    info = {"ce": float(ce.detach())}
    if model.cpuhead is None:
        return ce, info
    loo = lambda R: R - (R.sum(1, keepdim=True) - R) / (K - 1)
    with torch.no_grad():
        adv_p = loo(lp_t + shaping * (o["results"] == tgt[:, None]).float())
        adv_g = loo(lp_t - call_cost * o["call"])
    logpi = o["scores"].log_softmax(-1)                                  # (B, 3, T)
    lp_pick = logpi[:, None].expand(-1, K, -1, -1).gather(3, o["picks"][..., None])[..., 0].sum(-1)   # (B, K)
    g = o["gate"][:, None].clamp(1e-6, 1 - 1e-6)
    lp_call = o["call"] * g.log() + (1 - o["call"]) * (1 - g).log()
    pg = -(adv_p * lp_pick).mean() - (adv_g * lp_call).mean()
    p = logpi.exp().nan_to_num(0)
    ent = -(p * logpi.nan_to_num(0)).sum(-1).mean()
    info.update(pg=float(pg.detach()), entropy=float(ent.detach()), call_right=float((o["results"] == tgt[:, None]).float().mean()))
    info["calls"] = float(o["call"].mean())
    return ce + pg - entropy * ent, info
