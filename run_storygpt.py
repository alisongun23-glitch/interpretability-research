"""A story-writing GPT that sends math problems, and story word problems, to the neural-gate computer.

    python run_storygpt.py             # about 1.5 hours on a laptop CPU; writes results_storygpt/, storygpt_ui/
    python run_storygpt.py --reuse     # train only the runs without a saved model, then analyse
    python run_storygpt.py --analyse   # reuse all trained models (a few minutes)

Data: TinyStories (data/TinyStories-valid.txt, 21,990 stories; the last 1,000 are held out) with
math problems from cpugpt.data (about 2 per story) and story word problems from cpugpt.text
("tom had 12 apples . he got 5 more . now he has 17 apples .", about 1.5 per story) inserted after
random sentence ends. Training problems use training operand pairs only (cpugpt.data.is_test).
Models (4 layers, d_model 128, 4 heads, MLP 512, 64-token context, tied embeddings):
  story GPT                            4 ordinary heads in every layer
  story GPT + CPU head                 the last layer has 3 ordinary heads + the CPU head (cpugpt/lm.py),
                                       which may call the gate computer at every position; the read-back
                                       adds a projection of the result bits and the result's token embedding
  story GPT + CPU head, bit read-back  the same with the bits-only read-back of the first version
  story GPT + CPU head, copy read-back the same plus a copy path: a call adds a trained bonus to the
                                       result token's logit
  story GPT, same story words as v1    plain, trained for more steps so it sees as many story words as
                                       the first version (which had no word problems in its stream)
Evaluation: text loss on held-out stories (head on and off), calls in all held-out story text,
math and word-problem accuracy on held-out operand pairs placed after held-out story text, word
problems in three sentence shapes never trained, where the pointers land, ablations, the first
version's models (results_storygpt/v1_templates_only) on the word problems, generated samples, a live
check on the fully analog neural computer, and an export for storygpt_ui/.
"""
import base64, json, os, sys, time
from collections import Counter
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpugpt import data as D, text as X
from cpugpt.lm import StoryGPT, lm_losses
from gatecpu.design import build_cpu
from gatecpu.calc import PROGRAM, ADDR, CAP, call_ticks
from gatecpu.ref import port_inputs
from gatesnake.sim import Machine, load_library

OUT, UI = "results_storygpt", "storygpt_ui"
V1 = os.path.join(OUT, "v1_templates_only")                       # the first version: template math only
for d in (OUT, os.path.join(OUT, "figures"), os.path.join(OUT, "models"), UI):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
T, BATCH, STEPS, EVAL_EVERY, WORDS_PER_STORY = 64, 32, 7000, 500, 1.5
RUNS = [("story GPT", 0), ("story GPT + CPU head", 0), ("story GPT + CPU head", 1), ("story GPT + CPU head, bit read-back", 0),
        ("story GPT + CPU head, copy read-back", 0), ("story GPT + CPU head, copy read-back", 1), ("story GPT, same story words as v1", 0)]
OPS = ((D.ADD, "add"), (D.SUB, "sub"), (D.MUL, "mul"))
summary = {"context": T, "batch": BATCH, "steps": STEPS, "word_problems_per_story": WORDS_PER_STORY, "runs": {}}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def model_path(cname, seed, root=OUT):
    return os.path.join(root, "models", f"{cname.replace(' ', '_').replace('+', 'plus').replace(',', '')}_seed{seed}.pt")


stories = X.load_stories("data/TinyStories-valid.txt")
train_st, test_st = stories[:-1000], stories[-1000:]
V = X.Vocab(train_st)
STREAM, _, SINFO = X.build_stream(V, train_st, np.random.default_rng(0), "train", per_story=2.0, word_per_story=WORDS_PER_STORY)
V1_TOKENS = len(X.build_stream(V, train_st, np.random.default_rng(0), "train", per_story=2.0)[0])   # the first version's stream
TEXT, _, _ = X.build_stream(V, test_st, np.random.default_rng(1), "test", per_story=0.0)
TABLE = np.load("results_cpugpt/cpu_table.npz")["out"]           # the gate computer's answers (run_cpugpt.py)
CPU = dict(numval=V.numval, opval=V.opval, table=TABLE)
summary["data"] = {"stories_train": len(train_st), "stories_test": len(test_st), "vocab": len(V),
                   "train_tokens": int(len(STREAM)), "v1_train_tokens": V1_TOKENS, "test_text_tokens": int(len(TEXT)),
                   "train_math_problems": sum(not i["word"] for i in SINFO), "train_word_problems": sum(i["word"] for i in SINFO)}


def problem_set(n, split, seed, family="math", templates=X.WORD_PROBLEMS):
    """Problems placed after 0-40 tokens of held-out story text: cpugpt.data templates (family "math")
    or word problems (family "word", with the positions of both operands and of the operation word)."""
    r = np.random.default_rng(seed)
    toks = np.zeros((n, T), np.int64)
    keys = ("pos", "target", "kind", "a", "b", "tid") + (("pa", "pb", "pop") if family == "word" else ())
    info = {k: np.zeros(n, np.int64) for k in keys}
    for i in range(n):
        sn, off, inf = X.snippet(V, r, split) if family == "math" else X.word_snippet(V, r, split, templates=templates)
        pre = int(r.integers(0, 41)); j = int(r.integers(0, len(TEXT) - pre))
        seq = list(TEXT[j:j + pre]) + sn
        toks[i, :len(seq)] = seq
        info["pos"][i] = pre + off; info["target"][i] = X.NUM0 + inf["target"]
        for k in ("kind", "a", "b", "tid"):
            info[k][i] = inf[k]
        if family == "word":
            info["pa"][i], info["pb"][i], info["pop"][i] = (pre + o for o in inf["at"])
    return torch.as_tensor(toks), info, [t[0] for t in (D.MATH if family == "math" else templates)]


def head(ms, n):
    toks, info, names = ms
    return toks[:n], {k: v[:n] for k, v in info.items()}, names


MATH_TEST = problem_set(3000, "test", 11)                         # the same 3,000 problems as the first version
MATH_TRAIN = problem_set(1500, "train", 12)
WORD_TEST = problem_set(3000, "test", 21, "word")
WORD_TRAIN = problem_set(1500, "train", 22, "word")
WORD_NEW = problem_set(1500, "test", 23, "word", X.NEW_WORDINGS)
CURVE_SETS = (head(MATH_TEST, 1500), head(WORD_TEST, 1500))
TEXT_WIN = X.windows(TEXT, np.random.default_rng(13), 256, T)
TEXT_ALL = torch.as_tensor(TEXT[:len(TEXT) // T * T].reshape(-1, T))   # all held-out story text, 64-token windows
STORY_OPWORDS = [V.stoi[w] for w in X.OPCODE_OF_WORD if w not in D.OPCODE_OF_WORD]


def forward_at(m, toks, pos, mode="gated", op_map=None, chunk=250):
    """The model's prediction (and, with a CPU head, its call) at one position per row."""
    out = {}
    m.eval()
    with torch.no_grad():
        for s in range(0, len(toks), chunk):
            o = m(toks[s:s + chunk], mode=mode if m.has_cpu else "sample", op_map=op_map)
            ii = (torch.arange(len(o["logits"])), torch.as_tensor(pos[s:s + chunk]))
            part = {"pred": o["logits"][ii][:, 0].argmax(-1)}
            if m.has_cpu:
                part.update({k: o[k][ii][:, 0] for k in ("x", "y", "op", "results", "call", "picks")})
            for k, v in part.items():
                out.setdefault(k, []).append(v)
    m.train()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def eval_set(m, ms, mode="gated", op_map=None):
    toks, info, names = ms
    f = forward_at(m, toks, info["pos"], mode, op_map)
    ok = f["pred"] == info["target"]
    res = {"by_op": {nm: [int(ok[info["kind"] == k].sum()), int((info["kind"] == k).sum())] for k, nm in OPS},
           "by_template": {nm: float(ok[info["tid"] == t].mean()) for t, nm in enumerate(names)}, "ok": ok, "f": f}
    if m.has_cpu:
        x, y, op = f["x"], f["y"], f["op"]
        res["point"] = float(((x == info["a"]) & (y == info["b"]) & (op == info["kind"])).mean())
        res["call_at_answer"] = float(f["call"].mean())
        res["calls"] = (x, y, op)
        res["computer_right"] = f["results"] + X.NUM0 == info["target"]
    return res


def landing(ev, info):
    """Where the three pointers landed at the answer position: A on the first operand, B on the second,
    OP on the operation word (more / ate / lost / gave / each); 'numbers' accepts A and B swapped for + and x."""
    pk = ev["f"]["picks"]
    A, B, O = pk[:, 0] == info["pa"], pk[:, 1] == info["pb"], pk[:, 2] == info["pop"]
    swap = (pk[:, 0] == info["pb"]) & (pk[:, 1] == info["pa"])
    nums = (A & B) | (swap & (info["kind"] != D.SUB))
    return {nm: {"A": float(A[s].mean()), "B": float(B[s].mean()), "swapped": float(swap[s].mean()), "numbers": float(nums[s].mean()),
                 "OP_on_word": float(O[s].mean()), "all_three": float((nums & O)[s].mean())}
            for k, nm in OPS for s in [info["kind"] == k]}


def errors(ev, info):
    """Answer right / called / computer right / called and computer right but answer misread (with the misreads
    of answers 192-255 and the most common ones as [true, written, count])."""
    out = {}
    for k, nm in OPS:
        s = info["kind"] == k
        right, comp, called = ev["ok"][s], ev["computer_right"][s], ev["f"]["call"][s] > 0
        big = info["target"][s] - X.NUM0 >= 192
        mis = called & comp & ~right
        pairs = Counter(zip((info["target"][s][mis] - X.NUM0).tolist(), (ev["f"]["pred"][s][mis] - X.NUM0).tolist()))
        out[nm] = {"n": int(s.sum()), "answer_right": float(right.mean()), "called": float(called.mean()),
                   "computer_right": float(comp.mean()), "misread": int(mis.sum()), "answers_192_255": int(big.sum()),
                   "misread_192_255": int((mis & big).sum()), "right_192_255": float(right[big].mean()) if big.any() else None,
                   "computer_wrong": int((~comp).sum()), "top_misreads": [[a, b, c] for (a, b), c in pairs.most_common(5)]}
    return out


def eval_text(m, mode="gated"):
    a, b = (torch.as_tensor(t) for t in TEXT_WIN)
    m.eval()
    with torch.no_grad():
        o = m(a, mode=mode if m.has_cpu else "sample")
    m.train()
    lp = o["logits"][:, :, 0].log_softmax(-1).gather(2, b[..., None])[..., 0]
    keep = b != 0
    res = {"loss": float(-lp[keep].mean()), "perplexity": float(torch.exp(-lp[keep].mean()))}
    if m.has_cpu:
        res["calls_per_1000_words"] = float(o["call"][..., 0][keep].float().mean() * 1000)
    return res


def text_all(m, n_examples=10):
    """Perplexity and calls over all held-out story text, and the first few places where a call happened."""
    calls, ex, nll, cnt = 0, [], 0.0, 0
    m.eval()
    with torch.no_grad():
        for s in range(0, len(TEXT_ALL), 250):
            w = TEXT_ALL[s:s + 250]
            o = m(w, mode="gated" if m.has_cpu else "sample")
            lp = o["logits"][:, :-1, 0].log_softmax(-1).gather(2, w[:, 1:, None])[..., 0]
            nll -= float(lp.sum()); cnt += lp.numel()
            if not m.has_cpu:
                continue
            c = o["call"][..., 0] > 0
            calls += int(c.sum())
            for bi, t in c.nonzero().tolist()[:max(0, n_examples - len(ex))]:
                ex.append({"context": V.decode(w[bi, max(0, t - 12):t + 1].tolist()),
                           "next_word": V.itos[int(w[bi, t + 1])] if t + 1 < T else None,
                           **{k: int(o[k][bi, t, 0]) for k in ("x", "y", "op", "results")}})
    m.train()
    n = int(TEXT_ALL.numel())
    res = {"words": n, "perplexity": float(np.exp(nll / cnt))}
    if m.has_cpu:
        res.update(calls=calls, per_1000_words=calls / n * 1000, examples=ex,
                   operation_words_in_text=int(np.isin(TEXT_ALL.numpy(), STORY_OPWORDS).sum()))
    return res


def make_model(cname):
    return StoryGPT(len(V), T, cpu=CPU if "CPU" in cname else None, num0=X.NUM0, read_emb="bit read-back" not in cname,
                    read_copy="copy" in cname)


def run_steps(cname):
    return round(STEPS * len(STREAM) / V1_TOKENS) if "same story words" in cname else STEPS


THREADS = 2


def train_run(job):
    cname, seed = job
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed); r = np.random.default_rng(100 + seed)
    m = make_model(cname)
    steps = run_steps(cname)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / steps))))
    curve, t1 = [], time.time()
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0:
            (em, ew), et = (eval_set(m, ms) for ms in CURVE_SETS), eval_text(m)
            row = {"step": step, "text_loss": et["loss"], **{k: v[0] / v[1] for k, v in em["by_op"].items()},
                   **{"word_" + k: v[0] / v[1] for k, v in ew["by_op"].items()}}
            if m.has_cpu:
                row.update(point=em["point"], word_point=ew["point"], call_at_answer=em["call_at_answer"],
                           word_call_at_answer=ew["call_at_answer"], calls_per_1000_words=et["calls_per_1000_words"])
            curve.append(row)
            print(cname, seed, {k: round(v, 3) for k, v in row.items()}, f"{time.time() - t1:.0f}s", flush=True)
        if step == steps:
            break
        a, b = X.windows(STREAM, r, BATCH, T)
        loss, _ = lm_losses(m, torch.as_tensor(a), torch.as_tensor(b), K=2, P=64 if step < 1500 else 16)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    path = model_path(cname, seed)
    torch.save(m.state_dict(), path)
    return cname, seed, path, curve, time.time() - t1


def generate(m, prompt_ids, n=40, temperature=0.0, seed=0, mode="gated"):
    """Returns generated ids and, for each new token, the CPU call that preceded it (or None)."""
    g = torch.Generator().manual_seed(seed)
    ids, calls = list(prompt_ids), []
    m.eval()
    for _ in range(n):
        ctx = ids[-T:]
        with torch.no_grad():
            o = m(torch.as_tensor([ctx]), mode=mode if m.has_cpu else "sample")
        lg = o["logits"][0, -1, 0]
        if temperature > 0:
            top = torch.topk(lg, 20)
            nxt = int(top.indices[torch.multinomial(torch.softmax(top.values / temperature, -1), 1, generator=g)])
        else:
            nxt = int(lg.argmax())
        call = None
        if m.has_cpu and float(o["call"][0, -1, 0]) > 0:
            call = {k: int(o[k][0, -1, 0]) for k in ("x", "y", "op", "results")}
        ids.append(nxt); calls.append(call)
        if nxt == X.EOS:
            break
    m.train()
    return ids, calls


def acc(ev):
    return {k: v[0] / v[1] for k, v in ev["by_op"].items()}


if __name__ == "__main__":
    import multiprocessing as mp
    old = json.load(open(os.path.join(OUT, "summary.json")))["runs"] if os.path.exists(os.path.join(OUT, "summary.json")) else {}
    reuse = "--analyse" in sys.argv or "--reuse" in sys.argv
    done = {(c, s): (c, s, model_path(c, s), old[f"{c} | seed {s}"]["curve"], old[f"{c} | seed {s}"]["train_seconds"])
            for c, s in RUNS if reuse and os.path.exists(model_path(c, s)) and f"{c} | seed {s}" in old}
    todo = [j for j in RUNS if j not in done]
    assert not (todo and "--analyse" in sys.argv), f"no saved model for {todo}"
    if todo:
        THREADS = max(1, 8 // len(todo))
        with mp.get_context("fork").Pool(len(todo)) as pool:
            done.update({r[:2]: r for r in pool.map(train_run, todo)})
    done = [done[j] for j in RUNS]
    torch.set_num_threads(4)
    models = {}
    for cname, seed, path, curve, secs in done:
        m = make_model(cname); m.load_state_dict(torch.load(path)); models[(cname, seed)] = m
        summary["runs"][f"{cname} | seed {seed}"] = {"curve": curve, "train_seconds": secs, "steps": run_steps(cname)}
    log("trained")

    OPMAP_WRONG = torch.tensor([0, 2, 3, 1])
    for (cname, seed), m in models.items():
        row = summary["runs"][f"{cname} | seed {seed}"]
        em, ew, en = eval_set(m, MATH_TEST), eval_set(m, WORD_TEST), eval_set(m, WORD_NEW)
        row["math_heldout"] = {k: {"acc": v[0] / v[1], "ci95": wilson(*v), "n": v[1]} for k, v in em["by_op"].items()}
        row["math_heldout_by_template"] = em["by_template"]
        row["math_train_pairs"] = acc(eval_set(m, MATH_TRAIN))
        row["word_heldout"] = {k: {"acc": v[0] / v[1], "ci95": wilson(*v), "n": v[1]} for k, v in ew["by_op"].items()}
        row["word_heldout_by_template"] = ew["by_template"]
        row["word_train_pairs"] = acc(eval_set(m, WORD_TRAIN))
        row["word_new_wordings"] = {k: {"acc": v[0] / v[1], "ci95": wilson(*v), "n": v[1]} for k, v in en["by_op"].items()}
        row["text"] = eval_text(m)
        row["text_all"] = text_all(m)
        if m.has_cpu:
            row["gains"] = {k: float(getattr(m, k).detach()) for k in ("read_gain", "copy_gain") if hasattr(m, k)}
            row["point"] = em["point"]; row["call_at_answer"] = em["call_at_answer"]
            row["word_point"] = ew["point"]; row["word_call_at_answer"] = ew["call_at_answer"]
            row["word_new_call_at_answer"] = en["call_at_answer"]
            row["word_landing"] = landing(ew, WORD_TEST[1]); row["word_new_landing"] = landing(en, WORD_NEW[1])
            row["math_errors"] = errors(em, MATH_TEST[1]); row["word_errors"] = errors(ew, WORD_TEST[1])
            row["word_new_errors"] = errors(en, WORD_NEW[1])
            row["text_head_off"] = eval_text(m, mode="off")
            row["math_head_off"] = acc(eval_set(m, MATH_TEST, mode="off"))
            row["word_head_off"] = acc(eval_set(m, WORD_TEST, mode="off"))
            row["math_wrong_op"] = acc(eval_set(m, MATH_TEST, op_map=OPMAP_WRONG))
            row["word_wrong_op"] = acc(eval_set(m, WORD_TEST, op_map=OPMAP_WRONG))
        log(cname, seed, "math", {k: round(v["acc"], 3) for k, v in row["math_heldout"].items()},
            "word", {k: round(v["acc"], 3) for k, v in row["word_heldout"].items()},
            "new wordings", {k: round(v["acc"], 3) for k, v in row["word_new_wordings"].items()}, row["text"],
            {k: v for k, v in row["text_all"].items() if k != "examples"})

    # the first version (template math only, bit read-back) on the same tests; its frozen operator table
    # is given the new operation words, so only the learned parts differ
    if os.path.exists(os.path.join(V1, "summary.json")):
        v1 = {}
        for cname, seed in [("story GPT", 0), ("story GPT + CPU head", 0), ("story GPT + CPU head", 1)]:
            m = StoryGPT(len(V), T, cpu=CPU if "CPU" in cname else None, num0=X.NUM0)
            m.load_state_dict(torch.load(model_path(cname, seed, V1)))
            if m.has_cpu:
                m.opval.copy_(torch.as_tensor(V.opval))
            ew, en = eval_set(m, WORD_TEST), eval_set(m, WORD_NEW)
            r1 = {"math_heldout": acc(eval_set(m, MATH_TEST)), "word_heldout": acc(ew), "word_new_wordings": acc(en), "text": eval_text(m),
                  "text_all": text_all(m, 3)}
            if m.has_cpu:
                r1.update(word_call_at_answer=ew["call_at_answer"], word_landing=landing(ew, WORD_TEST[1]),
                          math_errors=errors(eval_set(m, MATH_TEST), MATH_TEST[1]))
            v1[f"{cname} | seed {seed}"] = r1
            log("v1", cname, seed, r1["math_heldout"], r1["word_heldout"], r1["text"])
        summary["v1_templates_only"] = v1

    # generated samples
    score = lambda k: np.mean([summary["runs"][f"{k[0]} | seed {k[1]}"][f][o]["acc"] for f in ("math_heldout", "word_heldout") for o in ("add", "sub", "mul")])
    best = max((k for k in models if models[k].has_cpu and "bit" not in k[0]), key=score)
    PROMPTS = ["once upon a time", "tom had a red ball .", "lily went to the park with her mom . what is 14 times 12 ?",
               "the cat was hungry . 60 + 117 =", "tom had 12 apples . he got 5 more . now he has",
               "mia had 40 cookies and ate 13 . she had", "there were 6 bags with 8 toys each ,",
               "once upon a time , ben found 23 shells and then 9 more . he had",
               "sam picked 30 grapes . then mia gave him 12 more . sam had"]
    samples = {}
    for name, key in (("story GPT + CPU head", best), ("story GPT", ("story GPT", 0))):
        m = models[key]; samples[name] = []
        for i, p in enumerate(PROMPTS):
            pid = V.encode(X.words(p))
            for temp in (0.0, 0.8):
                ids, calls = generate(m, pid, temperature=temp, seed=i)
                samples[name].append({"prompt": p, "temperature": temp, "text": V.decode(ids),
                                      "calls": [{"after_token": V.decode(ids[:len(pid) + j]).split()[-1:], **c}
                                                for j, c in enumerate(calls) if c]})
    summary["samples"] = samples
    summary["best"] = f"{best[0]} | seed {best[1]}"

    # live analog check: 150 template-math calls and 150 word-problem calls of the best model
    mb = models[best]
    calls_m, calls_w = eval_set(mb, MATH_TEST)["calls"], eval_set(mb, WORD_TEST)["calls"]
    xs, ys, ops = (np.concatenate([a[:150], b[:150]]) for a, b in zip(calls_m, calls_w))
    net = build_cpu()[0].finalize(); cpu = Machine(net, mode="neural", library=load_library(), dff="gates")
    halt_i = [i for i, g in enumerate(net.regs) if g["name"] == "halt"][0]
    out_i = [i for i, g in enumerate(net.regs) if g["name"].startswith("out[")]
    for i, byte in enumerate(PROGRAM):
        cpu.step(port_inputs(prog=1, we=1, addr=i, data=byte))
    diff = 0; t1 = time.time()
    for i in range(len(xs)):
        for t in call_ticks(int(xs[i]), int(ys[i]), int(ops[i])):
            cpu.step(port_inputs(**t))
        for _ in range(CAP):
            cpu.step(port_inputs())
            if cpu.Q[halt_i] > 0.5:
                break
        live = sum(int(cpu.Q[j] > 0.5) << k for k, j in enumerate(out_i))
        diff += int(live != TABLE[ops[i], xs[i], ys[i]])
    summary["live_analog_check"] = {"model": summary["best"], "calls": len(xs), "results_differing_from_table": diff,
                                    "seconds": round(time.time() - t1, 1), "worst_signal_dev": float(cpu.worst)}
    log("live analog", summary["live_analog_check"])

    # export for the page: weights as float32 binary + manifest
    def export(m, fname):
        blobs, manifest, off = [], [], 0
        for k, v in m.state_dict().items():
            if k in ("numval", "opval", "table", "bitw"):
                continue
            a = v.detach().numpy().astype(np.float32)
            manifest.append({"name": k, "shape": list(a.shape), "offset": off}); blobs.append(a.ravel()); off += a.size
        raw = np.concatenate(blobs).astype("<f4").tobytes()       # float32 little-endian, base64 so it can be served as text
        open(os.path.join(UI, fname), "w").write(base64.b64encode(raw).decode("ascii"))
        return manifest
    meta = {"vocab": V.itos, "num0": X.NUM0, "n_ctx": T, "d": 128, "heads": 4, "layers": 4,
            "numval": V.numval.tolist(), "opval": V.opval.tolist(), "program": PROGRAM, "addr": ADDR, "cap": CAP,
            "models": {"cpu": {"file": "cpu_model.f32.txt", "label": summary["best"], "tensors": export(mb, "cpu_model.f32.txt")},
                       "plain": {"file": "plain_model.f32.txt", "label": "story GPT, seed 0", "tensors": export(models[("story GPT", 0)], "plain_model.f32.txt")}},
            "split": {"rule": "(a*7919 + b*104729 + k*31337) % 5 == 0 means held out"},
            "results": {k: {"math": {o: v["math_heldout"][o]["acc"] for o in ("add", "sub", "mul")},
                            "word": {o: v["word_heldout"][o]["acc"] for o in ("add", "sub", "mul")},
                            "perplexity": v["text"]["perplexity"]} for k, v in summary["runs"].items()}}
    json.dump(meta, open(os.path.join(UI, "model.json"), "w"), separators=(",", ":"))
    log("exported", {f: round(os.path.getsize(os.path.join(UI, f)) / 1e6, 1) for f in os.listdir(UI)})
    summary["runtime_s"] = round(time.time() - T0, 1)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # figure: learning curves
    INK2, MUTED, GRID, BASE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2})
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.0))
    col = {"story GPT": "#898781", "story GPT + CPU head": "#2a78d6", "story GPT + CPU head, bit read-back": "#d68c14",
           "story GPT + CPU head, copy read-back": "#1d8a4c", "story GPT, same story words as v1": "#52514e"}
    for (cname, seed), m in models.items():
        c = summary["runs"][f"{cname} | seed {seed}"]["curve"]; st = [r["step"] for r in c]
        kw = dict(color=col[cname], alpha=1 if seed == 0 else 0.5, label=f"{cname}, seed {seed}")
        axs[0].plot(st, [np.mean([r["add"], r["sub"], r["mul"]]) for r in c], **kw)
        axs[1].plot(st, [np.mean([r["word_add"], r["word_sub"], r["word_mul"]]) for r in c], **kw)
        axs[2].plot(st[1:], [r["text_loss"] for r in c][1:], lw=3 if cname == "story GPT" else 1.5, **kw)
    for a_, yl in zip(axs, ("held-out accuracy (+, -, x mean)", "held-out accuracy (+, -, x mean)", "loss on held-out stories (nats/word)")):
        a_.set_xlabel("training step"); a_.set_ylabel(yl)
    for a_ in axs[:2]:
        a_.set_ylim(-0.03, 1.05)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    axs[0].set_title("Math templates, numbers never seen in training", fontsize=9, loc="left")
    axs[1].set_title("Story word problems, numbers never seen", fontsize=9, loc="left")
    axs[2].set_title("Story text loss", fontsize=9, loc="left")
    fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(os.path.join(OUT, "figures", "1_learning.png"), dpi=150); plt.close(fig)
    log("done")
