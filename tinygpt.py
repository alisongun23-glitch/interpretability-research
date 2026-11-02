"""A small GPT in plain NumPy with manual backprop and TransformerLens-style hooks.

Architecture (pre-LayerNorm, as in GPT-2 / TransformerLens):
    x = W_E[tokens] + W_pos
    for each block:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
    logits = LN_f(x) @ W_U + b_U

forward(tokens, hooks=None, cache=False) runs the model. `hooks` maps an activation name
to a function act -> act that replaces it in place (activation patching / ablation):
    blocks.{l}.hook_resid_pre      (B, T, d_model)
    blocks.{l}.attn.hook_pattern   (B, H, T, T)   softmaxed attention
    blocks.{l}.attn.hook_z         (B, T, H, d_head)
    blocks.{l}.hook_resid_mid, blocks.{l}.hook_resid_post
Gradients are checked against finite differences by `gradient_check()`.
"""
import numpy as np


class TinyGPT:
    def __init__(self, d_vocab, n_ctx, d_model, n_heads, n_layers, d_mlp, seed=0, dtype=np.float32):
        r = np.random.default_rng(seed)
        self.cfg = dict(d_vocab=d_vocab, n_ctx=n_ctx, d_model=d_model, n_heads=n_heads,
                        n_layers=n_layers, d_mlp=d_mlp, d_head=d_model // n_heads)
        d, H, dh, dm = d_model, n_heads, self.cfg["d_head"], d_mlp
        s = 0.02
        P = {"W_E": r.normal(0, s, (d_vocab, d)), "W_pos": r.normal(0, s, (n_ctx, d)),
             "W_U": r.normal(0, s, (d, d_vocab)), "b_U": np.zeros(d_vocab),
             "ln_f.w": np.ones(d), "ln_f.b": np.zeros(d)}
        for l in range(n_layers):
            P[f"{l}.ln1.w"] = np.ones(d); P[f"{l}.ln1.b"] = np.zeros(d)
            P[f"{l}.W_Q"] = r.normal(0, s, (H, d, dh)); P[f"{l}.b_Q"] = np.zeros((H, dh))
            P[f"{l}.W_K"] = r.normal(0, s, (H, d, dh)); P[f"{l}.b_K"] = np.zeros((H, dh))
            P[f"{l}.W_V"] = r.normal(0, s, (H, d, dh)); P[f"{l}.b_V"] = np.zeros((H, dh))
            P[f"{l}.W_O"] = r.normal(0, s, (H, dh, d)); P[f"{l}.b_O"] = np.zeros(d)
            P[f"{l}.ln2.w"] = np.ones(d); P[f"{l}.ln2.b"] = np.zeros(d)
            P[f"{l}.W_in"] = r.normal(0, s, (d, dm)); P[f"{l}.b_in"] = np.zeros(dm)
            P[f"{l}.W_out"] = r.normal(0, s, (dm, d)); P[f"{l}.b_out"] = np.zeros(d)
        self.P = {k: v.astype(dtype) for k, v in P.items()}
        self.dtype = dtype
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    # ---------------- forward ----------------
    @staticmethod
    def _ln(x, w, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True); xc = x - mu
        var = (xc ** 2).mean(-1, keepdims=True); inv = 1 / np.sqrt(var + eps)
        xhat = xc * inv
        return xhat * w + b, (xhat, inv, w)

    def forward(self, tokens, hooks=None, cache=False):
        """tokens: (B, T) ints. Returns logits (B, T, d_vocab); with cache=True also a dict
        of activations and the tape needed for backward."""
        P, c = self.P, self.cfg
        hooks = hooks or {}
        B, T = tokens.shape
        H, dh = c["n_heads"], c["d_head"]
        acts, tape = {}, {"tokens": tokens}
        x = P["W_E"][tokens] + P["W_pos"][:T][None]
        mask = np.triu(np.ones((T, T), bool), 1)
        for l in range(c["n_layers"]):
            name = f"blocks.{l}.hook_resid_pre"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            ln1, ln1_tape = self._ln(x, P[f"{l}.ln1.w"], P[f"{l}.ln1.b"])
            ln1h = ln1[:, None]                                       # (B, 1, T, d)
            q = ln1h @ P[f"{l}.W_Q"][None] + P[f"{l}.b_Q"][None, :, None, :]   # (B, H, T, dh)
            k = ln1h @ P[f"{l}.W_K"][None] + P[f"{l}.b_K"][None, :, None, :]
            v = ln1h @ P[f"{l}.W_V"][None] + P[f"{l}.b_V"][None, :, None, :]
            scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
            scores = np.where(mask[None, None], -1e9, scores)
            scores = scores - scores.max(-1, keepdims=True)
            A = np.exp(scores); A = A / A.sum(-1, keepdims=True)
            name = f"blocks.{l}.attn.hook_pattern"
            if name in hooks: A = hooks[name](A)
            acts[name] = A
            z = (A @ v).transpose(0, 2, 1, 3)                 # (B, T, H, dh)
            name = f"blocks.{l}.attn.hook_z"
            if name in hooks: z = hooks[name](z)
            acts[name] = z
            attn_out = z.reshape(B, T, H * dh) @ P[f"{l}.W_O"].reshape(H * dh, -1) + P[f"{l}.b_O"]
            x = x + attn_out
            name = f"blocks.{l}.hook_resid_mid"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            ln2, ln2_tape = self._ln(x, P[f"{l}.ln2.w"], P[f"{l}.ln2.b"])
            pre = ln2 @ P[f"{l}.W_in"] + P[f"{l}.b_in"]
            h = np.maximum(pre, 0)
            acts[f"blocks.{l}.mlp.hook_post"] = h
            x = x + h @ P[f"{l}.W_out"] + P[f"{l}.b_out"]
            name = f"blocks.{l}.hook_resid_post"
            if name in hooks: x = hooks[name](x)
            acts[name] = x
            tape[l] = dict(ln1=ln1, ln1_tape=ln1_tape, q=q, k=k, v=v, A=A, z=z, ln2=ln2,
                           ln2_tape=ln2_tape, pre=pre, h=h)
        lnf, lnf_tape = self._ln(x, P["ln_f.w"], P["ln_f.b"])
        logits = lnf @ P["W_U"] + P["b_U"]
        acts["ln_final.hook_normalized"] = lnf
        tape["lnf"] = lnf; tape["lnf_tape"] = lnf_tape
        if cache:
            return logits, acts, tape
        return logits

    # ---------------- backward ----------------
    @staticmethod
    def _ln_back(dy, tape_):
        xhat, inv, w = tape_
        dxhat = dy * w
        dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
        dw = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
        db = dy.reshape(-1, dy.shape[-1]).sum(0)
        return dx, dw, db

    def loss_and_grads(self, tokens, loss_mask):
        """Mean next-token cross-entropy over positions where loss_mask[b, t] is True."""
        P, c = self.P, self.cfg
        H, dh = c["n_heads"], c["d_head"]
        logits, acts, tape = self.forward(tokens, cache=True)
        B, T, V = logits.shape
        targets = tokens[:, 1:]
        lg = logits[:, :-1]
        m = loss_mask[:, :-1].astype(logits.dtype)
        n = m.sum()
        lg = lg - lg.max(-1, keepdims=True)
        p = np.exp(lg); p /= p.sum(-1, keepdims=True)
        loss = -(np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None], targets] + 1e-12) * m).sum() / n
        dlogits = np.zeros_like(logits)
        dp = p.copy(); dp[np.arange(B)[:, None], np.arange(T - 1)[None], targets] -= 1
        dlogits[:, :-1] = dp * m[..., None] / n
        G = {}
        G["W_U"] = tape["lnf"].reshape(-1, c["d_model"]).T @ dlogits.reshape(-1, V)
        G["b_U"] = dlogits.reshape(-1, V).sum(0)
        dlnf = dlogits @ P["W_U"].T
        dx, G["ln_f.w"], G["ln_f.b"] = self._ln_back(dlnf, tape["lnf_tape"])
        for l in reversed(range(c["n_layers"])):
            tp = tape[l]
            # MLP
            dh_ = dx @ P[f"{l}.W_out"].T
            G[f"{l}.W_out"] = tp["h"].reshape(-1, c["d_mlp"]).T @ dx.reshape(-1, c["d_model"])
            G[f"{l}.b_out"] = dx.reshape(-1, c["d_model"]).sum(0)
            dpre = dh_ * (tp["pre"] > 0)
            G[f"{l}.W_in"] = tp["ln2"].reshape(-1, c["d_model"]).T @ dpre.reshape(-1, c["d_mlp"])
            G[f"{l}.b_in"] = dpre.reshape(-1, c["d_mlp"]).sum(0)
            dln2 = dpre @ P[f"{l}.W_in"].T
            dx2, G[f"{l}.ln2.w"], G[f"{l}.ln2.b"] = self._ln_back(dln2, tp["ln2_tape"])
            dx = dx + dx2
            # attention
            Bq, Tq = dx.shape[:2]
            dz = (dx @ P[f"{l}.W_O"].reshape(H * dh, -1).T).reshape(Bq, Tq, H, dh)
            G[f"{l}.W_O"] = (tp["z"].reshape(-1, H * dh).T @ dx.reshape(-1, c["d_model"])).reshape(H, dh, -1)
            G[f"{l}.b_O"] = dx.reshape(-1, c["d_model"]).sum(0)
            dzh = dz.transpose(0, 2, 1, 3)                              # (B, H, T, dh)
            dA = dzh @ tp["v"].transpose(0, 1, 3, 2)
            dv = tp["A"].transpose(0, 1, 3, 2) @ dzh
            A = tp["A"]
            dscores = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(dh)
            dq = dscores @ tp["k"]
            dk = dscores.transpose(0, 1, 3, 2) @ tp["q"]
            ln1T = tp["ln1"].transpose(0, 2, 1)[:, None]                 # (B, 1, d, T)
            G[f"{l}.W_Q"] = (ln1T @ dq).sum(0); G[f"{l}.b_Q"] = dq.sum((0, 2))
            G[f"{l}.W_K"] = (ln1T @ dk).sum(0); G[f"{l}.b_K"] = dk.sum((0, 2))
            G[f"{l}.W_V"] = (ln1T @ dv).sum(0); G[f"{l}.b_V"] = dv.sum((0, 2))
            dln1 = ((dq @ P[f"{l}.W_Q"].transpose(0, 2, 1)[None]).sum(1) + (dk @ P[f"{l}.W_K"].transpose(0, 2, 1)[None]).sum(1)
                    + (dv @ P[f"{l}.W_V"].transpose(0, 2, 1)[None]).sum(1))
            dx1, G[f"{l}.ln1.w"], G[f"{l}.ln1.b"] = self._ln_back(dln1, tp["ln1_tape"])
            dx = dx + dx1
        G["W_pos"] = dx.sum(0)
        G["W_E"] = np.zeros_like(P["W_E"])
        np.add.at(G["W_E"], tokens.ravel(), dx.reshape(-1, c["d_model"]))
        return float(loss), G

    def adam_step(self, G, lr, wd=0.0, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.P:
            g = G[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            upd = lr * mh / (np.sqrt(vh) + 1e-8)
            if wd and k.startswith(("W_", "0.", "1.", "2.", "3.")) and ".ln" not in k and ".b_" not in k:
                upd = upd + lr * wd * self.P[k]
            self.P[k] = (self.P[k] - upd).astype(self.dtype)

    def state(self):
        return {k: v.copy() for k, v in self.P.items()}

    def load(self, state):
        for k in self.P:
            self.P[k] = state[k].astype(self.dtype)


def gradient_check(seed=0):
    """Compare analytic gradients with central finite differences (float64, tiny model)."""
    m = TinyGPT(d_vocab=5, n_ctx=7, d_model=8, n_heads=2, n_layers=2, d_mlp=12, seed=seed, dtype=np.float64)
    r = np.random.default_rng(seed)
    for k in m.P:                                          # non-trivial LN / bias values
        m.P[k] = m.P[k] + r.normal(0, 0.3, m.P[k].shape)
    tokens = r.integers(0, 5, (3, 7)); mask = r.random((3, 7)) < 0.6; mask[:, -1] = False
    loss, G = m.loss_and_grads(tokens, mask)
    worst = 0.0
    for k in m.P:
        for _ in range(3):
            idx = tuple(r.integers(0, s) for s in m.P[k].shape)
            old = m.P[k][idx]; eps = 1e-5
            m.P[k][idx] = old + eps; lp, _ = m.loss_and_grads(tokens, mask)
            m.P[k][idx] = old - eps; lm, _ = m.loss_and_grads(tokens, mask)
            m.P[k][idx] = old
            num = (lp - lm) / (2 * eps); ana = G[k][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    print("worst relative gradient error:", gradient_check())
