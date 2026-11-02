"""Probes for the capacity experiment (run_storyroom.py): what a story GPT spends its weights on.

Looking inside: cache() records the residual stream, each head's output and the MLP neurons through hooks (the
StoryGPT forward returns no hidden states); ablated() replaces one head's or one neuron's output with its mean
over reference text; ablation_scan() measures what every unit costs the story loss and the arithmetic loss;
linear_probe() reads a number out of the residual stream; dead_units() and effective_rank() say how much of the
model is in use. Skills, all scored by log-probability, never by generation: minimal_pairs() builds pairs of
sentences from the story lexicon that differ in one token (was/were, has/have, she/he and her/his after a
girl's or a boy's name, him/he, bag/bags), recall_pairs() asks for a name introduced one or two sentences
earlier against a distractor, induction_test() repeats a random run of words and measures how much cheaper the
second copy is. loss_by_role() and loss_by_frequency() split the loss by what a token is and how common it is.
"""
import contextlib
import numpy as np
import torch

from . import text as X

FIRST_WORD = X.NUM0 + 256 + len(X.MATH_WORDS)      # id of the most common story word (story words are in frequency order)


# ------------------------------------------------------------------------------------------------ losses
def token_logprobs(model, inp, tgt, chunk=512):
    """(B, T) log-probability of every target under the model, no gradient, in chunks of rows."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(inp), chunk):
            lg = model(inp[s:s + chunk])["logits"][:, :, 0]
            out.append(lg.log_softmax(-1).gather(2, tgt[s:s + chunk, :, None])[..., 0])
    model.train()
    return torch.cat(out)


def nll(model, inp, tgt, mask=None):
    """Mean negative log-probability of the targets where mask is true (pad targets never count)."""
    lp = token_logprobs(model, inp, tgt)
    m = (tgt != 0) if mask is None else (mask & (tgt != 0))
    return float(-lp[m].mean())


def loss_by_role(model, inp, tgt, roles_t, names=("story", "problem", "answer", "machine", "call", "restate")):
    """Loss split by the role of the target token (cpugpt.text.roles)."""
    lp = token_logprobs(model, inp, tgt)
    out = {}
    for k, nm in enumerate(names):
        m = (roles_t == k) & (tgt != 0)
        out[nm] = {"n": int(m.sum()), "loss": float(-lp[m].mean()) if m.any() else None}
    return out


def loss_by_frequency(model, inp, tgt, v, edges=(50, 200, 1000, 3000)):
    """Story loss split by how common the target word is: ranks 0-49, 50-199, ... of the story vocabulary (rank 0 is
    the most common word); numbers, math words, <eos> and tool tokens are 'other'."""
    lp = token_logprobs(model, inp, tgt)
    last = v.CALC if v.CALC is not None else len(v)
    rank = tgt - FIRST_WORD
    story = (rank >= 0) & (tgt < last) & (tgt != 0)
    out, lo = {}, 0
    for hi in list(edges) + [None]:
        m = story & (rank >= lo) & ((rank < hi) if hi is not None else True)
        out[f"rank {lo}-{hi - 1 if hi else 'end'}"] = {"n": int(m.sum()), "loss": float(-lp[m].mean()) if m.any() else None}
        lo = hi
    m = ~story & (tgt != 0)
    out["other"] = {"n": int(m.sum()), "loss": float(-lp[m].mean()) if m.any() else None}
    return out


# ------------------------------------------------------------------------------------------------ looking inside
def _sq(t):
    return t[:, :, 0] if t.dim() == 4 else t          # the last block runs on (B, T, K, d) with K = 1


def cache(model, tokens, want=("resid", "heads", "neurons")):
    """Run the model on tokens (B, T) and record per layer: resid[l] = the residual stream entering block l (and
    resid[L] = after the last block, before the final norm); heads[l] = (B, T, H, dh), what each head adds before
    the output projection; neurons[l] = (B, T, d_mlp), the ReLU activations; attn[l] = (B, H, T, T) patterns.
    Also 'logits' (B, T, V). The hooks are removed afterwards."""
    out = {k: [] for k in want}
    hs, dh = [], model.dh
    for blk in model.blocks:
        if "resid" in want:
            hs.append(blk.ln1.register_forward_pre_hook(lambda m, a: out["resid"].append(_sq(a[0]).detach())))
        if "heads" in want:
            hs.append(blk.attn.o.register_forward_pre_hook(
                lambda m, a: out["heads"].append(_sq(a[0]).detach().reshape(*_sq(a[0]).shape[:2], m.in_features // dh, dh))))
        if "neurons" in want:
            hs.append(blk.mlp[1].register_forward_hook(lambda m, a, o: out["neurons"].append(_sq(o).detach())))
        if "attn" in want:
            hs.append(blk.attn.register_forward_hook(lambda m, a, o: out["attn"].append(o[1].detach())))
    if "resid" in want:
        hs.append(model.lnf.register_forward_pre_hook(lambda m, a: out["resid"].append(_sq(a[0]).detach())))
    model.eval()
    try:
        with torch.no_grad():
            out["logits"] = model(tokens)["logits"][:, :, 0]
    finally:
        for h in hs:
            h.remove()
        model.train()
    return out


def unit_means(c):
    """From cache(): the mean output vector of every head (H, dh) and the mean of every neuron (d_mlp,), per layer."""
    return {"head": [h.reshape(-1, *h.shape[2:]).mean(0) for h in c["heads"]],
            "neuron": [n.reshape(-1, n.shape[-1]).mean(0) for n in c["neurons"]]}


@contextlib.contextmanager
def ablated(model, kind, layer, index, means):
    """Inside the block: head `index` of block `layer` (kind "head") or neuron `index` (kind "neuron") is replaced by
    its mean from means (unit_means; zeros for a zero ablation). The hook is removed on exit."""
    blk = model.blocks[layer]
    if kind == "head":
        dh, mu = model.dh, means["head"][layer][index]

        def pre(m, a):
            x = a[0].clone()
            x[..., index * dh:(index + 1) * dh] = mu
            return (x,)
        h = blk.attn.o.register_forward_pre_hook(pre)
    else:
        mu = means["neuron"][layer][index]

        def fwd(m, a, o):
            o = o.clone()
            o[..., index] = mu
            return o
        h = blk.mlp[1].register_forward_hook(fwd)
    try:
        yield
    finally:
        h.remove()


def ablation_scan(model, means, story, arith, kinds=("head", "neuron"), neuron_step=1):
    """story and arith: (inp, tgt, mask) batches. For every head and (every neuron_step-th) neuron: the story loss and the
    arithmetic loss with that unit replaced by its mean, minus the unablated losses. Rows [kind, layer, index, d_story, d_arith]."""
    base_s, base_a = nll(model, *story), nll(model, *arith)
    rows = []
    for l, blk in enumerate(model.blocks):
        units = []
        if "head" in kinds:
            units += [("head", i) for i in range(blk.attn.h)]
        if "neuron" in kinds:
            units += [("neuron", i) for i in range(0, blk.mlp[0].out_features, neuron_step)]
        for kind, i in units:
            with ablated(model, kind, l, i, means):
                rows.append([kind, l, i, nll(model, *story) - base_s, nll(model, *arith) - base_a])
    return {"base_story": base_s, "base_arith": base_a, "rows": rows}


def linear_probe(feats, y, train, ridge=1e-2):
    """Ridge read-out of the number y (N,) from the features (N, d), fitted on the rows where train is true:
    R^2 on the other rows, and the accuracy of eight ridge read-outs of the bits (sign of a +-1 regression).
    Features are standardised on the training rows."""
    Xf = torch.as_tensor(np.asarray(feats), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    tr = torch.as_tensor(np.asarray(train, bool)); te = ~tr
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
    Z = torch.cat([(Xf - mu) / sd, torch.ones(len(Xf), 1, dtype=torch.float64)], 1)
    A = Z[tr].T @ Z[tr] + ridge * int(tr.sum()) * torch.eye(Z.shape[1], dtype=torch.float64)

    def fit(t):
        return torch.linalg.solve(A, Z[tr].T @ t[tr])
    pred = Z[te] @ fit(y)
    r2 = float(1 - ((pred - y[te]) ** 2).sum() / max(float(((y[te] - y[te].mean()) ** 2).sum()), 1e-9))
    bits = []
    for k in range(8):
        t = (((y.long() >> k) & 1) * 2 - 1).double()
        bits.append(float(((Z[te] @ fit(t)).sign() == t[te]).double().mean()))
    return {"r2": r2, "bits": bits, "bits_mean": float(np.mean(bits)), "n_train": int(tr.sum()), "n_test": int(te.sum())}


def dead_units(c, tol=1e-6):
    """Share of MLP neurons per layer that never fire on the cached text."""
    return [float((n.reshape(-1, n.shape[-1]).max(0).values <= tol).double().mean()) for n in c["neurons"]]


def effective_rank(resid, n_max=4096):
    """exp(entropy of the normalised singular values) of the centred residual matrix (positions x d): how many
    directions the representation really uses, between 1 and d."""
    Z = resid.reshape(-1, resid.shape[-1]).double()
    if len(Z) > n_max:
        Z = Z[torch.randperm(len(Z))[:n_max]]
    Z = Z - Z.mean(0)
    s = torch.linalg.svdvals(Z); p = s / s.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


# ------------------------------------------------------------------------------------------------ skills
def induction_test(model, v, n=500, run=16, seed=0):
    """n rows: <eos>, a random run of `run` story words, the same run again, padding. Returns the mean loss on the
    first copy and on the second, their difference (bigger = more copying) and the share of second-copy words
    predicted right. A model with no induction heads scores about 0 difference."""
    r = np.random.default_rng(seed)
    last = v.CALC if v.CALC is not None else len(v)
    rows = np.zeros((n, model.n_ctx), np.int64)
    for i in range(n):
        w = r.integers(FIRST_WORD, last, run)
        rows[i, 0] = X.EOS; rows[i, 1:1 + run] = w; rows[i, 1 + run:1 + 2 * run] = w
    rows = torch.as_tensor(rows); inp, tgt = rows[:, :-1], rows[:, 1:]
    model.eval()
    with torch.no_grad():
        lg = model(inp)["logits"][:, :, 0]
    model.train()
    lp = lg.log_softmax(-1).gather(2, tgt[..., None])[..., 0]
    first, second = lp[:, :run], lp[:, run:2 * run]
    acc = float((lg[:, run:2 * run].argmax(-1) == tgt[:, run:2 * run]).double().mean())
    return {"first_copy_loss": float(-first.mean()), "second_copy_loss": float(-second.mean()),
            "induction": float(second.mean() - first.mean()),
            "copy_accuracy": acc, "n": n, "run": run}


ADJ = "happy sad tired hungry scared excited sleepy proud angry glad".split()
FILLERS = ["it was a nice day .", "the sun was out .", "mom was in the kitchen .", "the birds sang .", "it was time to play .",
           "the wind blew .", "a dog ran by .", "the sky was blue ."]
PARADIGMS = ("was_were", "has_have", "gender", "case", "number")


def _ok(v, sent):
    return all(w in v.stoi for w in sent.split())


def _ids(v, sent):
    return [X.EOS] + v.encode(sent.split())


def minimal_pairs(v, r, paradigm, n=500):
    """n (good, bad) id rows for one paradigm, each pair differing in exactly one token, every word in the vocabulary.
    was_were: a singular or plural subject with was/were. has_have: the same with has/have. gender: she/he, her/his,
    her/him after a girl's or a boy's name. case: him/he after "to", his/him before a noun, them/they after "with".
    number: bag/bags after each, a, one, or a number above one."""
    sing = [f"{nm}" for nm in X.BOYS + X.GIRLS] + ["mom", "dad", "the boy", "the girl", "the dog", "the cat"]
    plur = [f"{a} and {b}" for a in X.BOYS for b in X.GIRLS] + ["the kids", "the boys", "the girls", "they", "the friends", "mom and dad"]
    pl_nouns = X.OBJECTS["any"]
    pairs_sp = X.CONTAINERS + X.GROUPS + X.BOXES
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        if paradigm == "was_were":
            s = r.random() < 0.5
            subj = X._pick(r, sing if s else plur); adj = X._pick(r, ADJ)
            good, bad = (f"{subj} was {adj} .", f"{subj} were {adj} .") if s else (f"{subj} were {adj} .", f"{subj} was {adj} .")
        elif paradigm == "has_have":
            s = r.random() < 0.5
            subj = X._pick(r, sing if s else plur); k = int(r.integers(2, 13)); noun = X._pick(r, pl_nouns)
            good, bad = (f"{subj} has {k} {noun} .", f"{subj} have {k} {noun} .") if s else (f"{subj} have {k} {noun} .", f"{subj} has {k} {noun} .")
        elif paradigm == "gender":
            girl = r.random() < 0.5
            nm = X._pick(r, X.GIRLS if girl else X.BOYS); noun = X._pick(r, pairs_sp)[1]; adj = X._pick(r, ADJ)
            frame = X._pick(r, ["{N} lost {H} {S} .", "{N} said {P} was {ADJ} .", "{N} wanted a {S} . mom gave it to {O} .", "{N} put {H} {S} down ."])
            fg = {"H": "her" if girl else "his", "P": "she" if girl else "he", "O": "her" if girl else "him"}
            fb = {"H": "his" if girl else "her", "P": "he" if girl else "she", "O": "him" if girl else "her"}
            good = frame.format(N=nm, S=noun, ADJ=adj, **fg); bad = frame.format(N=nm, S=noun, ADJ=adj, **fb)
        elif paradigm == "case":
            nm = X._pick(r, X.BOYS + X.GIRLS); noun = X._pick(r, pairs_sp)[1]
            frame, g, b = X._pick(r, [("{N} gave it to {X} .", "him", "he"), ("{N} put it in {X} {S} .", "his", "him"),
                                      ("{N} played with {X} .", "them", "they"), ("{N} ran to {X} .", "her", "she")])
            good, bad = frame.format(N=nm, S=noun, X=g), frame.format(N=nm, S=noun, X=b)
        elif paradigm == "number":
            nm = X._pick(r, X.BOYS + X.GIRLS); pl, sg = X._pick(r, pairs_sp)
            one = r.random() < 0.5
            det = X._pick(r, ["each", "a", "one"]) if one else str(int(r.integers(2, 13)))
            good, bad = (f"{nm} had {det} {sg} .", f"{nm} had {det} {pl} .") if one else (f"{nm} had {det} {pl} .", f"{nm} had {det} {sg} .")
        else:
            raise ValueError(paradigm)
        if _ok(v, good) and _ok(v, bad):
            gi, bi = _ids(v, good), _ids(v, bad)
            if len(gi) == len(bi) and sum(a != b for a, b in zip(gi, bi)) == 1:
                out.append((gi, bi))
    return out


def recall_pairs(v, r, n=500, gap=1):
    """n (good, bad) id rows: a name is introduced, `gap` filler sentences follow, then "then <name>" against
    "then <distractor>" (same gender, not in the context). The rows end at the name, so only its probability counts."""
    out, tries = [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        girl = r.random() < 0.5
        names = X.GIRLS if girl else X.BOYS
        nm = X._pick(r, names); other = X._pick(r, [x for x in names if x != nm])
        k = int(r.integers(2, 13)); noun = X._pick(r, X.OBJECTS["any"])
        fill = " ".join(X._pick(r, FILLERS) for _ in range(gap))
        pre = f"{nm} had {k} {noun} . {fill} then"
        good, bad = f"{pre} {nm}", f"{pre} {other}"
        if _ok(v, good) and _ok(v, bad):
            out.append((_ids(v, good), _ids(v, bad)))
    return out


def pair_score(model, pairs):
    """Share of pairs whose good row has the higher summed log-probability (ties count half), and the mean margin."""
    L = max(len(x) for g, b in pairs for x in (g, b))
    scores = []
    for idx in (0, 1):
        rw = torch.zeros((len(pairs), L), dtype=torch.long)
        for i, pr in enumerate(pairs):
            rw[i, :len(pr[idx])] = torch.as_tensor(pr[idx])
        inp, tgt = rw[:, :-1], rw[:, 1:]
        lp = token_logprobs(model, inp, tgt)
        lp = lp.masked_fill(tgt == 0, 0.0)
        scores.append(lp.sum(1))
    g, b = scores
    return {"score": float(((g > b).double() + 0.5 * (g == b).double()).mean()), "n": len(pairs), "margin": float((g - b).mean())}
