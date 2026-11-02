"""A story-writing GPT whose last layer has a CPU head that works at every position.

Same CPU head as cpugpt/model.py (three trained pointers, frozen bus tables, the gate computer,
a trained yes/no gate, a trained read-back projection), but it acts at every position of an
ordinary next-token language model: at each position it may point, call, and add the
computer's answer before the next word is predicted. Embedding and unembedding are tied.
With read_emb the read-back also adds the result's own token embedding (times a trained gain), so
the answer does not have to be decoded from its 8 bits by the last MLP; with read_copy a call also
adds a trained bonus to the result token's logit (a copy path that works for tokens rarely written).

Learning:
  text, read-back, everything differentiable   cross-entropy on every next token
  pointers   REINFORCE over P cheap samples per position: +1 when the computer's result is the
             next token (checking costs a table lookup, no forward pass); leave-one-out baseline.
             A learned relative-position bias, started with a lean toward recent tokens, gives the
             search somewhere to begin in a 64-token context.
  call       REINFORCE over K samples per position: log p(next token), +1 when a call returned the
             next token, - call_cost per call. Calls are explored at least `explore` of the time.
"""
import math
import torch
import torch.nn as nn

from .model import Attn


class Block(nn.Module):
    def __init__(self, d, n_heads, dh, d_mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, n_heads, dh)
        self.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d))


class StoryGPT(nn.Module):
    def __init__(self, vocab, n_ctx, d=128, n_layers=4, n_heads=4, d_mlp=512, cpu=None, num0=4,
                 recency=0.3, explore=0.05, read_emb=False, read_copy=False):
        super().__init__()
        dh = d // n_heads
        self.dh, self.num0, self.n_ctx, self.explore = dh, num0, n_ctx, explore
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(n_ctx, d)
        nn.init.normal_(self.emb.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads - (1 if (cpu and l == n_layers - 1) else 0), dh, d_mlp)
                                     for l in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.out_bias = nn.Parameter(torch.zeros(vocab))
        self.has_cpu = cpu is not None
        if cpu:
            self.q = nn.Linear(d, 3 * dh); self.k = nn.Linear(d, 3 * dh)
            self.rel = nn.Parameter(-recency * torch.arange(n_ctx, dtype=torch.float)[None].repeat(3, 1))   # (3, distance)
            self.read = nn.Linear(8, d); self.gate = nn.Linear(d, 1)
            self.read_emb, self.read_copy = read_emb, read_copy
            if read_emb:
                self.read_gain = nn.Parameter(torch.ones(1))
            if read_copy:
                self.copy_gain = nn.Parameter(torch.ones(1))
            self.register_buffer("numval", torch.tensor(cpu["numval"]))          # copies: loading a checkpoint must not
            self.register_buffer("opval", torch.tensor(cpu["opval"]))            # write into the caller's arrays
            self.register_buffer("table", torch.as_tensor(cpu["table"], dtype=torch.long))
            self.register_buffer("bitw", 2 ** torch.arange(8))

    def pointer_scores(self, h):
        B, T, _ = h.shape
        q = self.q(h).view(B, T, 3, self.dh); k = self.k(h).view(B, T, 3, self.dh)
        s = torch.einsum("btsd,busd->btsu", q, k) / math.sqrt(self.dh)            # (B, T, 3, T)
        dist = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0)  # (T, T) query - key
        s = s + self.rel[:, dist].permute(1, 0, 2)[None]
        return s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1)[None, :, None, :], float("-inf"))

    def call_computer(self, tokens, picks, op_map=None):
        """picks (B, T, K, 3) positions -> x, y, op and the computer's results (B, T, K)."""
        T = tokens.shape[1]
        tk = torch.gather(tokens[:, None, None, :].expand(-1, T, picks.shape[2], -1), 3, picks)
        xv, yv, ov = self.numval[tk[..., 0]], self.numval[tk[..., 1]], self.opval[tk[..., 2]]
        if op_map is not None:
            ov = op_map[ov]
        return xv, yv, ov, self.table[ov, xv, yv]

    def forward(self, tokens, K=1, mode="sample", op_map=None):
        """tokens (B, T). mode: "sample" (training), "gated" (argmax pointers, call if gate >= 0.5),
        "always" (argmax pointers, always call), "off" (never call).
        Returns dict with logits (B, T, K, V) and the CPU head's choices."""
        B, T = tokens.shape
        x = self.emb(tokens) + self.pos(torch.arange(T))[None]
        for blk in self.blocks[:-1]:
            x = x + blk.attn(blk.ln1(x))[0]
            x = x + blk.mlp(blk.ln2(x))
        last = self.blocks[-1]
        h = last.ln1(x)
        x = x + last.attn(h)[0]
        out = {}
        if self.has_cpu:
            s = self.pointer_scores(h)
            if mode == "sample":
                picks = torch.distributions.Categorical(logits=s[:, :, None].expand(-1, -1, K, -1, -1)).sample()
            else:
                picks = s.argmax(-1)[:, :, None, :]
            xv, yv, ov, res = self.call_computer(tokens, picks, op_map)
            g = torch.sigmoid(self.gate(h))[..., 0]                                   # (B, T)
            if mode == "sample":
                p_call = self.explore + (1 - 2 * self.explore) * g
                call = torch.bernoulli(p_call[..., None].detach().expand(-1, -1, K))
            elif mode in ("gated", "always"):
                call = (g[..., None] >= 0.5).float() if mode == "gated" else torch.ones(B, T, 1)
            else:
                call = torch.zeros(B, T, 1)
            bits = ((res[..., None] & self.bitw) > 0).float()
            back = self.read(bits)
            if self.read_emb:
                back = back + self.read_gain * self.emb.weight[self.num0 + res]
            xk = x[:, :, None, :] + call[..., None] * back
            out.update(scores=s, picks=picks, x=xv, y=yv, op=ov, results=res, gate=g, call=call)
        else:
            xk = x[:, :, None, :]
        xk = xk + last.mlp(last.ln2(xk))
        out["logits"] = self.lnf(xk) @ self.emb.weight.T + self.out_bias
        if self.has_cpu and self.read_copy:
            out["logits"] = out["logits"].scatter_add(3, (self.num0 + res)[..., None], (call * self.copy_gain)[..., None])
        return out


def lm_losses(model, inp, tgt, K=2, P=64, check=1.0, call_cost=0.05, entropy=0.001, ignore=None, weight=None):
    """ignore: optional (B, T) bool, targets left out of the loss (what the machine wrote, cpugpt.tool).
    weight: optional (B, T) float, how much each target counts (run_storyroom.py: answers count more); the
    cross-entropy is the weighted mean, so weight=None or all ones is the plain mean."""
    o = model(inp, K=K if model.has_cpu else 1)
    logp = o["logits"].log_softmax(-1)                                            # (B, T, K, V)
    lp_t = logp.gather(3, tgt[:, :, None, None].expand(-1, -1, logp.shape[2], 1))[..., 0]   # (B, T, K)
    m = tgt != 0                                                                  # ignore <pad> targets
    if ignore is not None:
        m = m & ~ignore
    m = m.float()
    w = m if weight is None else m * weight
    ce = -(lp_t * w[..., None]).sum() / (w.sum() * lp_t.shape[2])
    info = {"ce": float(ce.detach())}
    if not model.has_cpu:
        return ce, info
    loo = lambda R, n: R - (R.sum(2, keepdim=True) - R) / (n - 1)
    # pointers: P cheap tries per position, reward = the computer returned the next token
    logpi = o["scores"].log_softmax(-1)                                           # (B, T, 3, T)
    picks = torch.distributions.Categorical(logits=o["scores"].detach()[:, :, None].expand(-1, -1, P, -1, -1)).sample()
    with torch.no_grad():
        right_p = (model.call_computer(inp, picks)[3] + model.num0 == tgt[..., None]).float()
        adv_p = loo(right_p, P) * m[..., None]
    lp_pick = logpi[:, :, None].expand(-1, -1, P, -1, -1).gather(4, picks[..., None])[..., 0].sum(-1)   # (B, T, P)
    pg_ptr = -(adv_p * lp_pick).sum() / (m.sum() * P)
    # call decision: K samples that went through the network
    with torch.no_grad():
        right = (o["results"] + model.num0 == tgt[..., None]).float()
        adv_g = loo(lp_t + check * right * o["call"] - call_cost * o["call"], K) * m[..., None]
    p_call = (model.explore + (1 - 2 * model.explore) * o["gate"])[..., None]
    lp_call = o["call"] * p_call.log() + (1 - o["call"]) * (1 - p_call).log()
    pg_call = -(adv_g * lp_call).sum() / (m.sum() * K)
    p = logpi.exp().nan_to_num(0)
    ent = -(p * logpi.nan_to_num(0)).sum(-1).mean()
    info.update(calls=float(o["call"].mean()), ptr_hits=float(right_p.mean() * 1e3))
    return ce + pg_ptr + pg_call - entropy * ent, info
