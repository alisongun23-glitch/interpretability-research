"""A story-writing GPT that calls the neural-gate computer in writing: varied sentence shapes.

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 run_storytool.py            # about 45 minutes on a laptop CPU
    ... run_storytool.py --reuse     # train only the runs without a saved model, then analyse
    ... run_storytool.py --analyse   # reuse all trained models (a few minutes)
    ... run_storytool.py --quick     # a smoke test: 200 steps, small sets, written to $STORYTOOL_OUT (or results_storytool_shapes/quick)
    (the OBJC_... variable is for macOS, where torch in a forked pool can die with an "objc ... initialize" message)

The model writes "<calc> 12 + 5 =" into the story, decoding pauses, the gate computer answers "17 </calc>" into
the text, and the model copies 17: "now he has 17 apples". The computer's answer is in the model's input but
left out of the training loss (cpugpt.tool, cpugpt.text.build_tool_stream).
Data: TinyStories (data/TinyStories-valid.txt, the last 1,000 stories held out) with math templates (2 per
story), word problems (1.5), two-step problems (0.5) and no-call sentences (0.5) inserted after random sentence
ends. The word problems, two-step problems and no-call sentences come from the compositional generator of
cpugpt.text (openings, starts, changes with many verbs, questions, distractor numbers; no-call sentences with the
same verbs and no question, or no numbers at all). Some frames and verbs are held out of training
(cpugpt.text: "*" entries) and make the held-out test sets. Operand pairs use the split cpugpt.data.is_test.
Models (4 layers, d_model 128, 4 heads, MLP 512, 64-token context, no CPU head):
  story GPT + calc        the computer's tokens are left out of the loss (3 seeds)
  story GPT, no calls     the same stream without call blocks (perplexity and memorisation baseline)
  previous run            the three seeds of results_storytool/ (trained on 12 fixed shapes), evaluated on the new sets
Evaluation: generation with the machine answering (and off, and answering the wrong operation) on held-out operand
pairs after held-out story text: math templates, word problems in trained shapes, in held-out shapes, with held-out
verbs, the three never-trained wordings of the first run and its 12 fixed shapes, two-step problems (trained and
held-out), no-call sentences (trained and held-out); spurious calls in held-out story text and in free writing;
a decode-time margin on <calc>; perplexity; samples; a live check on the fully analog neural computer; export for
storytool_ui/. Results go to results_storytool_shapes/ (the first run's results_storytool/ is left as it is).
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
from cpugpt.tool import Calculator, decode, call_right
from gatecpu.design import build_cpu
from gatecpu.calc import PROGRAM, ADDR, CAP, call_ticks
from gatecpu.ref import port_inputs
from gatesnake.sim import Machine, load_library

QUICK = "--quick" in sys.argv
PREV = "results_storytool"                       # the first run: 12 fixed shapes
OUT = os.environ.get("STORYTOOL_OUT", os.path.join(PREV + "_shapes", "quick") if QUICK else PREV + "_shapes")
UI = os.path.join(OUT, "ui") if QUICK else "storytool_ui"
for d in (OUT, os.path.join(OUT, "figures"), os.path.join(OUT, "models"), UI):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
T, BATCH, STEPS, EVAL_EVERY, MAX_CALLS, KEEP = 64, 32, (200 if QUICK else 7000), (100 if QUICK else 500), 2, 48
SCALE = 0.1 if QUICK else 1.0
RUNS = [("story GPT + calc", 0), ("story GPT + calc", 1), ("story GPT + calc", 2), ("story GPT, no calls", 0)]
PREV_RUNS = [("previous run, 12 fixed shapes", s) for s in (0, 1, 2)]     # evaluated only, never trained here
MARGINS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
OPS = ((D.ADD, "add"), (D.SUB, "sub"), (D.MUL, "mul"))
OPNAME = {k: nm for k, nm in OPS}
summary = {"context": T, "batch": BATCH, "steps": STEPS, "max_calls": MAX_CALLS, "runs": {}}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def model_path(cname, seed):
    if cname.startswith("previous"):
        return os.path.join(PREV, "models", f"story_GPT_plus_calc_seed{seed}.pt")
    return os.path.join(OUT, "models", f"{cname.replace(' ', '_').replace('+', 'plus').replace(',', '')}_seed{seed}.pt")


stories = X.load_stories("data/TinyStories-valid.txt")
train_st, test_st = stories[:-1000], stories[-1000:]
V = X.Vocab(train_st, tool_tokens=True)
assert all(w in V.stoi for w in X.GENERATOR_WORDS), [w for w in X.GENERATOR_WORDS if w not in V.stoi]
STREAM, ENV, _, SINFO = X.build_tool_stream(V, train_st, np.random.default_rng(0), "train")
STREAM_PLAIN = X.build_tool_stream(V, train_st, np.random.default_rng(0), "train", calls=False)[0]   # same problems, no call blocks
TEXT, _, _ = X.build_stream(V, test_st, np.random.default_rng(1), "test", per_story=0.0)
TABLE = np.load("results_cpugpt/cpu_table.npz")["out"]           # the gate computer's answers (run_cpugpt.py)
CALCS = {mode: Calculator(TABLE, V, mode) for mode in ("on", "off", "wrong")}
summary["data"] = {"stories_train": len(train_st), "stories_test": len(test_st), "vocab": len(V), "train_tokens": int(len(STREAM)),
                   "train_tokens_no_calls": int(len(STREAM_PLAIN)), "machine_tokens": int(ENV.sum()), "test_text_tokens": int(len(TEXT)),
                   "train_problems": dict(zip(("math", "word", "two_step", "nocall"), np.bincount([i["family"] for i in SINFO], minlength=4).tolist())),
                   "nocall_types": dict(Counter(i["nocall_type"] for i in SINFO if i["family"] == 3)),
                   "distinct_word_shapes": len({i["shape"] for i in SINFO if i["family"] == 1}),
                   "distinct_two_step_shapes": len({i["shape"] for i in SINFO if i["family"] == 2})}
summary["generator"] = {"frames": {nm: {"trained": sum(not f["held"] for f in fr), "held_out": sum(f["held"] for f in fr)} for nm, fr in
                                   (("openings", X.OPENINGS), ("starts", X.STARTS), ("add changes", X.ADD_CHANGES), ("sub changes", X.SUB_CHANGES),
                                    ("add combined", X.ADD_COMBINED), ("sub combined", X.SUB_COMBINED), ("mul frames", X.MUL_FRAMES),
                                    ("asks", X.ASKS_ALL + X.ASKS_MORE + X.ASKS_LESS), ("doubles", sum(X.DOUBLES.values(), [])),
                                    ("nocall frames", [f for f, _ in X.NOCALL_FRAMES]), ("endings", [f for e in X.ENDINGS.values() for f, _ in e]),
                                    ("numberless", X.NUMBERLESS))},
                        "trained_verbs": X.TRAINED_VERBS, "held_out_verbs": X.HELD_OUT_VERBS,
                        "held_out_frames": [f["text"] for f in X._all_frames() if f["held"]]}

FAMILIES = {"math": (lambda r, s: X.snippet(V, r, s), 12),
            "word": (lambda r, s: X.gen_word(V, r, s), 12),                                       # trained frames and verbs
            "newshape": (lambda r, s: X.gen_word(V, r, s, held="shape"), 12),                    # one held-out frame
            "newverb": (lambda r, s: X.gen_word(V, r, s, held="verb"), 12),                      # one held-out verb
            "new": (lambda r, s: X.word_snippet(V, r, s, templates=X.NEW_WORDINGS), 12),          # the first run's three never-trained wordings
            "fixed": (lambda r, s: X.word_snippet(V, r, s), 12),                                 # the first run's 12 fixed shapes
            "two": (lambda r, s: X.gen_two(V, r, s, call=False), 20),
            "twonew": (lambda r, s: X.gen_two(V, r, s, held="shape" if r.random() < 0.5 else "verb", call=False), 20),
            "nocall": (lambda r, s: X.gen_nocall(V, r, s, restate_only=True), 10),
            "nocallnew": (lambda r, s: X.gen_nocall(V, r, s, held="shape" if r.random() < 0.5 else "verb", restate_only=True), 10)}


def problem_set(n, split, seed, family):
    """n problems, each after some held-out story text, cut at the cue word (the model must write the call and
    the answer). Rows are left-aligned 64-token windows with room for the answer and MAX_CALLS calls."""
    make, n_new = FAMILIES[family]
    r = np.random.default_rng(seed)
    rows = torch.zeros((n, T), dtype=torch.long); ptr = torch.zeros(n, dtype=torch.long)
    info = {k: np.zeros(n, np.int64) for k in ("target", "kind", "kind2", "a", "b", "c", "r1")}
    calls, held, shape, verb = [], [], [], []
    for i in range(n):
        ids, off, inf = make(r, split)
        prompt = ids[:inf["cue"] + 1]
        pre = int(r.integers(0, max(1, T - len(prompt) - n_new - 2 * MAX_CALLS + 1))); j = int(r.integers(0, len(TEXT) - pre))
        seq = list(TEXT[j:j + pre]) + prompt
        rows[i, :len(seq)] = torch.as_tensor(seq); ptr[i] = len(seq) - 1
        for k in info:
            info[k][i] = inf.get(k, 0)
        calls.append(inf["calls"]); held.append(inf.get("held", "")); shape.append(inf.get("shape", "")); verb.append(inf.get("verb", ""))
    return {"rows": rows, "ptr": ptr, "info": info, "calls": calls, "held": held, "shape": shape, "verb": verb, "n_new": n_new, "family": family}


def head(ms, n):
    return {**ms, "rows": ms["rows"][:n], "ptr": ms["ptr"][:n], "info": {k: v[:n] for k, v in ms["info"].items()},
            "calls": ms["calls"][:n], "held": ms["held"][:n], "shape": ms["shape"][:n], "verb": ms["verb"][:n]}


SIZES = {"math": 2000, "word": 3000, "newshape": 2000, "newverb": 2000, "new": 1500, "fixed": 1500, "two": 1500, "twonew": 1500, "nocall": 1000, "nocallnew": 1000}
SETS = {k: problem_set(max(30, int(n * SCALE)), "test", 11 + 10 * i, k) for i, (k, n) in enumerate(SIZES.items())}
TRAIN_SETS = {k: problem_set(max(30, int(n * SCALE)), "train", 12 + 10 * i, k) for i, (k, n) in enumerate((("word", 1000), ("two", 500)))}
CURVE_SETS = {k: head(SETS[k], 300) for k in ("word", "newshape", "two", "nocall")}
TEXT_WIN = X.windows(TEXT, np.random.default_rng(13), 256, T)
TEXT_ALL = torch.as_tensor(TEXT[:len(TEXT) // T * T].reshape(-1, T))   # all held-out story text, 64-token windows
if QUICK:
    TEXT_ALL = TEXT_ALL[:len(TEXT_ALL) // 10]
rp = np.random.default_rng(14); starts = rp.integers(0, len(TEXT) - 16, 50 if QUICK else 200)
PREFIXES = torch.as_tensor(np.stack([np.pad(TEXT[s:s + 16], (0, T - 16)) for s in starts]))   # free generation starts
THREADS = 2


def eval_set(m, ms, calc, max_calls=MAX_CALLS, margin=0.0):
    """Generate the call(s) and the answer for every problem; per operation (or operation pair, or kind of no-call
    sentence), per held-out piece and per verb: called, well-formed, call right (operands and operation; swapped
    allowed for + and x), machine right, answer right, copy right given the machine was right, calls per problem;
    two-step problems add hand-off (the second call starts from the first result) and second call right."""
    res = decode(m, ms["rows"], ms["ptr"], calc, n_new=ms["n_new"], max_calls=max_calls, calc_margin=margin)
    info, N = ms["info"], len(res)
    rec = {k: np.zeros(N, bool) for k in ("called", "wellformed", "call_right", "machine_right", "answer_right", "handoff", "call2_right", "final_machine_right")}
    rec["n_calls"] = np.array([len(r["calls"]) for r in res]); rec["n_model"] = np.array([r["n_model"] for r in res])
    log_calls = []
    for i, r in enumerate(res):
        c, e = r["calls"], ms["calls"][i]
        rec["called"][i] = len(c) > 0
        rec["wellformed"][i] = len(c) > 0 and all(x["ok"] for x in c)
        rec["answer_right"][i] = r["answer"] == info["target"][i]
        if e and c:
            a, op, b, rr = e[0]
            rec["call_right"][i] = call_right(c[0], a, op, b)
            rec["machine_right"][i] = c[0]["by"] == "machine" and c[0]["r"] == rr
            rec["final_machine_right"][i] = c[-1]["by"] == "machine" and c[-1]["r"] == info["target"][i]
        if len(e) == 2 and len(c) >= 2:
            a2, op2, b2, _ = e[1]
            rec["handoff"][i] = c[1]["ok"] and c[1]["a"] == c[0]["r"]
            rec["call2_right"][i] = rec["handoff"][i] and c[0]["r"] == a2 and c[1]["op"] == op2 and c[1]["b"] == b2
        log_calls += [(x["a"], x["op"], x["b"]) for x in c if x["ok"] and x["by"] == "machine"]
    two = ms["family"] in ("two", "twonew")
    if ms["family"] in ("nocall", "nocallnew"):
        groups = {("fixed frame" if k == 0 else OPNAME[k] + " setup"): info["kind"] == k for k in (0, D.ADD, D.SUB, D.MUL)}
    elif two:
        groups = {f"{OPNAME[k1]},{OPNAME[k2]}": (info["kind"] == k1) & (info["kind2"] == k2) for k1, k2 in X.TWO_STEP_PAIRS}
    else:
        groups = {nm: info["kind"] == k for k, nm in OPS}
    groups = {k: s for k, s in groups.items() if s.any()}

    def stats(s):
        n = int(s.sum()); mr = rec["final_machine_right"][s]
        d = {"n": n, "answer_right": float(rec["answer_right"][s].mean()), "ci95": wilson(int(rec["answer_right"][s].sum()), n),
             "called": float(rec["called"][s].mean()), "wellformed": float(rec["wellformed"][s].mean()),
             "call_right": float(rec["call_right"][s].mean()), "machine_right": float(rec["machine_right"][s].mean()),
             "copy_right_given_machine": float(rec["answer_right"][s][mr].mean()) if mr.any() else None,
             "calls_per_problem": float(rec["n_calls"][s].mean()), "model_tokens": float(rec["n_model"][s].mean())}
        if two:
            d.update(handoff=float(rec["handoff"][s].mean()), call2_right=float(rec["call2_right"][s].mean()))
        return d

    def by(keys):
        ks = np.array(keys)
        return {k: {"n": int((ks == k).sum()), "answer_right": float(rec["answer_right"][ks == k].mean()), "called": float(rec["called"][ks == k].mean()),
                    "call_right": float(rec["call_right"][ks == k].mean())} for k in sorted(set(keys)) if k}
    out = {"all": stats(np.ones(N, bool)), "by_group": {k: stats(s) for k, s in groups.items()}, "by_held": by(ms["held"]),
           "by_verb": by([v.split()[-1] if v else "" for v in ms["verb"]]), "_rec": rec, "_res": res, "_calls": log_calls}
    return out


def eval_text(m):
    a, b = (torch.as_tensor(t) for t in TEXT_WIN)
    m.eval()
    with torch.no_grad():
        lp = m(a)["logits"][:, :, 0].log_softmax(-1).gather(2, b[..., None])[..., 0]
    m.train()
    keep = b != 0
    return {"loss": float(-lp[keep].mean()), "perplexity": float(torch.exp(-lp[keep].mean()))}


def spurious(m, calc, n_examples=8, margins=(0.0, 1.0, 2.0, 3.0)):
    """Calls the model would make in plain held-out story text: the top guess at every position of all 207k words
    (also with a margin: <calc> must beat the best word by that much), and free generation (40 words) from 200
    held-out openings, with and without the margin."""
    gaps, nll, cnt = [], 0.0, 0
    m.eval()
    with torch.no_grad():
        for s in range(0, len(TEXT_ALL), 250):
            w = TEXT_ALL[s:s + 250]
            lg = m(w)["logits"][:, :-1, 0]
            other = lg.clone(); other[:, :, V.CALC] = float("-inf")
            gaps.append((lg[:, :, V.CALC] - other.max(-1).values).flatten())
            lp = lg.log_softmax(-1).gather(2, w[:, 1:, None])[..., 0]
            nll -= float(lp.sum()); cnt += lp.numel()
    m.train()
    gap = torch.cat(gaps)
    out = {"words": int(TEXT_ALL.numel()), "perplexity_all": float(np.exp(nll / cnt)), "argmax_calls_by_margin": {}, "generated_calls_by_margin": {}}
    for mg in MARGINS:
        out["argmax_calls_by_margin"][str(mg)] = int((gap > mg).sum())
    for mg in margins:
        res = decode(m, PREFIXES, torch.full((len(PREFIXES),), 15), calc, n_new=40, max_calls=MAX_CALLS, stop_at_number=False, calc_margin=mg)
        gen_calls = sum(len(r["calls"]) for r in res); gen_tokens = sum(r["n_model"] for r in res)
        out["generated_calls_by_margin"][str(mg)] = {"calls": gen_calls, "words": gen_tokens}
        if mg == 0.0:
            out["argmax_calls"] = int((gap > 0).sum()); out["argmax_calls_per_1000_words"] = out["argmax_calls"] / TEXT_ALL.numel() * 1000
            out["generated_words"] = gen_tokens; out["generated_calls"] = gen_calls; out["generated_calls_per_1000_words"] = gen_calls / max(gen_tokens, 1) * 1000
            out["examples"] = [{"text": V.decode(PREFIXES[i, :16].tolist() + r["written"]), "calls": r["calls"]} for i, r in enumerate(res) if r["calls"]][:n_examples]
            # the words before the top-guess calls
            pos = (gap > 0).nonzero()[:, 0].tolist()
            before = Counter(V.itos[int(TEXT_ALL.flatten()[p // (T - 1) * T + p % (T - 1)])] for p in pos)
            out["argmax_calls_after"] = dict(before.most_common(8))
    return out


def make_model(cname):
    return StoryGPT(len(V), T, n_layers=4, cpu=None, num0=X.NUM0)


def train_run(job):
    cname, seed = job
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed); r = np.random.default_rng(100 + seed)
    m = make_model(cname)
    stream = STREAM_PLAIN if "no calls" in cname else STREAM
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / STEPS))))
    curve, t1 = [], time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            row = {"step": step, "text_loss": eval_text(m)["loss"]}
            for k, ms in CURVE_SETS.items():
                ev = eval_set(m, ms, CALCS["on"], max_calls=0 if "no calls" in cname else MAX_CALLS)["all"]
                row[k] = ev["answer_right"]; row[k + "_called"] = ev["called"]
                if k == "two":
                    row["two_handoff"] = ev["handoff"]
            curve.append(row)
            print(cname, seed, {k: round(v, 3) for k, v in row.items()}, f"{time.time() - t1:.0f}s", flush=True)
        if step == STEPS:
            break
        if "no calls" in cname:
            a, b = X.windows(stream, r, BATCH, T); e = None
        else:
            a, b, e = X.windows(stream, r, BATCH, T, env=ENV); e = torch.as_tensor(e)
        loss, _ = lm_losses(m, torch.as_tensor(a), torch.as_tensor(b), ignore=e)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    path = model_path(cname, seed)
    torch.save(m.state_dict(), path)
    return cname, seed, path, curve, time.time() - t1


def public(ev):
    return {k: v for k, v in ev.items() if not k.startswith("_")}


def evaluate_run(job):
    """Everything measured on one trained model (runs in a pool)."""
    cname, seed = job
    torch.set_num_threads(THREADS)
    m = make_model(cname); m.load_state_dict(torch.load(model_path(cname, seed)))
    mc = 0 if "no calls" in cname else MAX_CALLS
    row = {"text": eval_text(m), "sets": {}, "train_pairs": {}}
    for mode in (("on", "off", "wrong") if mc else ("off",)):
        row["sets"][mode] = {}
        for k, ms in SETS.items():
            if mode != "on" and mc and k not in ("math", "word", "two"):     # the ablations (machine off / wrong) on three sets only
                continue
            ev = eval_set(m, ms, CALCS[mode], max_calls=mc)
            row["sets"][mode][k] = public(ev)
            if mode == "on" and k in ("math", "word"):
                row.setdefault("_calls", []).extend(ev["_calls"][:150])
    for k, ms in TRAIN_SETS.items():
        row["train_pairs"][k] = public(eval_set(m, ms, CALCS["on"], max_calls=mc))["all"]
    row["spurious"] = spurious(m, CALCS["on"]) if mc else {k: v for k, v in spurious(m, CALCS["off"], margins=()).items() if k in ("words", "perplexity_all")}
    on = row["sets"].get("on", row["sets"]["off"])
    log(cname, seed, {k: round(on[k]["all"]["answer_right"], 3) for k in on}, "text", round(row["text"]["perplexity"], 2),
        "spurious", {k: v for k, v in row["spurious"].items() if "per_1000" in k})
    return cname, seed, row


def margin_point(job):
    """One point of the margin sweep (runs in a pool): accuracy and call rate at this margin on 1,000 problems of
    each set and 1,000 no-call sentences of each kind, and the calls made in free writing."""
    cname, seed, mg = job
    torch.set_num_threads(THREADS)
    m = make_model(cname); m.load_state_dict(torch.load(model_path(cname, seed)))
    row = {}
    for k in ("word", "newshape", "newverb", "two", "nocall", "nocallnew"):
        ev = eval_set(m, head(SETS[k], 1000), CALCS["on"], margin=mg)["all"]
        row[k] = {"answer_right": ev["answer_right"], "called": ev["called"]}
    res = decode(m, PREFIXES, torch.full((len(PREFIXES),), 15), CALCS["on"], n_new=40, max_calls=MAX_CALLS, stop_at_number=False, calc_margin=mg)
    row["generated_calls"] = sum(len(r["calls"]) for r in res)
    log("margin", cname, seed, mg, {k: (round(v["answer_right"], 3), round(v["called"], 3)) if isinstance(v, dict) else v for k, v in row.items()})
    return job, row


def margin_sweep(pool, jobs):
    """The decode-time margin on <calc> (the call must beat the best word by this much) for every (model, margin)
    in jobs; the top-guess calls in story text at each margin come from the models' spurious() counts."""
    out = {}
    for (cname, seed, mg), row in pool.map(margin_point, jobs):
        key = f"{cname} | seed {seed}"
        row["argmax_calls_in_text"] = summary["runs"][key]["spurious"]["argmax_calls_by_margin"][str(mg)]
        out.setdefault(key, {})[str(mg)] = row
    return out


def generate(m, prompt, n=40, temperature=0.0, seed=0, calc=None, max_calls=MAX_CALLS, margin=0.0):
    """A continuation with the machine answering the model's calls. Returns (ids, calls)."""
    pid = V.encode(X.words(prompt))[-(T - n - 2 * MAX_CALLS):]
    rows = torch.zeros((1, T), dtype=torch.long); rows[0, :len(pid)] = torch.as_tensor(pid)
    r = decode(m, rows, torch.tensor([len(pid) - 1]), calc or CALCS["on"], n_new=n, max_calls=max_calls, temperature=temperature,
               seed=seed, stop_at_number=False, calc_margin=margin)[0]
    return pid + r["written"], r["calls"]


if __name__ == "__main__":
    import multiprocessing as mp
    old = json.load(open(os.path.join(OUT, "summary.json")))["runs"] if os.path.exists(os.path.join(OUT, "summary.json")) else {}
    reuse = "--analyse" in sys.argv or "--reuse" in sys.argv
    done = {(c, s): (c, s, model_path(c, s), old.get(f"{c} | seed {s}", {}).get("curve", []), old.get(f"{c} | seed {s}", {}).get("train_seconds"))
            for c, s in RUNS if reuse and os.path.exists(model_path(c, s))}
    todo = [j for j in RUNS if j not in done]
    assert not (todo and "--analyse" in sys.argv), f"no saved model for {todo}"
    if todo:
        THREADS = max(1, 10 // len(todo))
        with mp.get_context("fork").Pool(len(todo)) as pool:
            done.update({r[:2]: r for r in pool.map(train_run, todo)})
    for cname, seed, path, curve, secs in (done[j] for j in RUNS):
        summary["runs"][f"{cname} | seed {seed}"] = {"curve": curve, "train_seconds": secs, "layers": 4, "steps": STEPS}
    log("trained")
    prev_jobs = [j for j in PREV_RUNS if os.path.exists(model_path(*j))]
    for cname, seed in prev_jobs:
        summary["runs"][f"{cname} | seed {seed}"] = {"layers": 4, "trained_on": "the 12 fixed shapes of results_storytool/"}
    jobs = RUNS + prev_jobs
    THREADS = max(1, 10 // len(jobs))
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        evaluated = pool.map(evaluate_run, jobs)
    calls_log = {}
    for cname, seed, row in evaluated:
        calls_log[(cname, seed)] = row.pop("_calls", [])
        summary["runs"][f"{cname} | seed {seed}"].update(row)
    torch.set_num_threads(4)

    def dump():
        summary["runtime_s"] = round(time.time() - T0, 1)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    dump()

    # the first run's own numbers (12 fixed shapes), for the comparison table
    if os.path.exists(os.path.join(PREV, "summary.json")):
        pr = json.load(open(os.path.join(PREV, "summary.json")))["runs"]
        summary["previous_summary"] = {k: {"perplexity": v["text"]["perplexity"], "perplexity_all": v["spurious"].get("perplexity_all"),
                                           "word": v["sets"]["on"]["word"]["all"]["answer_right"] if "on" in v["sets"] else None,
                                           "new": v["sets"]["on"]["new"]["all"]["answer_right"] if "on" in v["sets"] else None,
                                           "two": v["sets"]["on"]["two"]["all"]["answer_right"] if "on" in v["sets"] else None,
                                           "argmax_calls": v["spurious"].get("argmax_calls"), "generated_calls": v["spurious"].get("generated_calls")}
                                       for k, v in pr.items()}

    # the page's model: the best masked seed on held-out word problems in trained and held-out shapes and two-step problems
    score = lambda k: np.mean([summary["runs"][f"{k[0]} | seed {k[1]}"]["sets"]["on"][s]["all"]["answer_right"] for s in ("word", "newshape", "newverb", "two")])
    best = max((j for j in RUNS if j[0] == "story GPT + calc"), key=score)
    summary["best"] = f"{best[0]} | seed {best[1]}"
    models = {j: make_model(j[0]) for j in jobs}
    for j, m in models.items():
        m.load_state_dict(torch.load(model_path(*j)))
    mb, mp_ = models[best], models[("story GPT, no calls", 0)]
    sweep_jobs = [(c, s_, mg) for c, s_ in [best] + prev_jobs[:1] for mg in MARGINS]
    THREADS = 1
    with mp.get_context("fork").Pool(min(10, len(sweep_jobs))) as pool:
        summary["margin_sweep"] = margin_sweep(pool, sweep_jobs)
    torch.set_num_threads(4)
    PROMPTS = ["once upon a time", "tom had a red ball .", "tom had 12 apples . he got 5 more . now he has",
               "mia had 40 cookies and ate 13 . she had", "there were 6 bags with 8 toys each ,",
               "lily had 20 stickers . she got 7 more and then 4 more . now she has",
               "ben had 50 grapes . he ate 12 and then got 30 more . he had",
               "lily counted 9 shells . that is", "sam picked 30 grapes . then mia gave him 12 more . sam had",
               "there were 25 cookies on the plate . tom ate 9 of them . then there were",
               "mia has 6 boxes and each box has 7 toys . so mia has",
               "ben found 40 shells . he caught 15 more . now he has", "lily had 30 marbles . she dropped 12 . she had",
               "tom had 5 bags and every bag had 9 stickers . that is", "tom had 12 apples . he got 5 more . he put the",
               "they hugged each other and ate more cake .",
               "lily went to the park with her mom . what is 14 times 12 ?", "the cat was hungry . 60 + 117 ="]
    samples = {}
    for name, m, mc in (("story GPT + calc", mb, MAX_CALLS), ("story GPT, no calls", mp_, 0), (prev_jobs[0][0] if prev_jobs else None, models.get(prev_jobs[0]) if prev_jobs else None, MAX_CALLS)):
        if name is None:
            continue
        samples[name] = []
        for i, p in enumerate(PROMPTS):
            for temp in (0.0, 0.8):
                ids, calls = generate(m, p, temperature=temp, seed=i, max_calls=mc)
                samples[name].append({"prompt": p, "temperature": temp, "text": V.decode(ids), "calls": calls})
    summary["samples"] = samples

    # live analog check: 300 of the page model's calls re-run on the fully analog neural CPU
    xs = calls_log[best][:300]
    net = build_cpu()[0].finalize(); cpu = Machine(net, mode="neural", library=load_library(), dff="gates")
    halt_i = [i for i, g in enumerate(net.regs) if g["name"] == "halt"][0]
    out_i = [i for i, g in enumerate(net.regs) if g["name"].startswith("out[")]
    for i, byte in enumerate(PROGRAM):
        cpu.step(port_inputs(prog=1, we=1, addr=i, data=byte))
    diff = 0; t1 = time.time()
    for a, op, b in xs:
        for t in call_ticks(int(a), int(b), int(op)):
            cpu.step(port_inputs(**t))
        for _ in range(CAP):
            cpu.step(port_inputs())
            if cpu.Q[halt_i] > 0.5:
                break
        live = sum(int(cpu.Q[j] > 0.5) << k for k, j in enumerate(out_i))
        diff += int(live != TABLE[op, a, b])
    summary["live_analog_check"] = {"model": summary["best"], "calls": len(xs), "results_differing_from_table": diff,
                                    "seconds": round(time.time() - t1, 1), "worst_signal_dev": float(cpu.worst)}
    log("live analog", summary["live_analog_check"])

    # export for the page: weights as float32 binary + manifest
    def export(m, fname):
        blobs, manifest, off = [], [], 0
        for k, v in m.state_dict().items():
            a = v.detach().numpy().astype(np.float32)
            manifest.append({"name": k, "shape": list(a.shape), "offset": off}); blobs.append(a.ravel()); off += a.size
        raw = np.concatenate(blobs).astype("<f4").tobytes()
        open(os.path.join(UI, fname), "w").write(base64.b64encode(raw).decode("ascii"))
        return manifest
    def res_row(v):
        on = v["sets"]["on" if "on" in v["sets"] else "off"]
        return {**{k: on[k]["all"]["answer_right"] for k in ("math", "word", "newshape", "newverb", "new", "fixed", "two", "twonew")},
                "nocall_calls": on["nocall"]["all"]["called"], "nocallnew_calls": on["nocallnew"]["all"]["called"],
                "perplexity": v["text"]["perplexity"], "text_calls_per_1000": v["spurious"].get("argmax_calls_per_1000_words"),
                "free_calls_per_1000": v["spurious"].get("generated_calls_per_1000_words")}
    results = {k: res_row(v) for k, v in summary["runs"].items()}
    meta = {"vocab": V.itos, "num0": X.NUM0, "n_ctx": T, "d": 128, "heads": 4, "calc": V.CALC, "calc_end": V.END, "max_calls": MAX_CALLS, "keep": KEEP,
            "numval": V.numval.tolist(), "opval": V.opval.tolist(), "program": PROGRAM, "addr": ADDR, "cap": CAP,
            "models": {"calc": {"file": "calc_model.f32.txt", "label": summary["best"], "layers": 4, "tensors": export(mb, "calc_model.f32.txt")},
                       "plain": {"file": "plain_model.f32.txt", "label": "story GPT, no calls | seed 0", "layers": 4, "tensors": export(mp_, "plain_model.f32.txt")},
                       "previous_label": f"{prev_jobs[0][0]} | seed {prev_jobs[0][1]}" if prev_jobs else None},
            "split": {"rule": "(a*7919 + b*104729 + k*31337) % 5 == 0 means held out"}, "results": results, "margin": 0.0,
            "held_out_verbs": X.HELD_OUT_VERBS, "held_out_frames": summary["generator"]["held_out_frames"]}
    json.dump(meta, open(os.path.join(UI, "model.json"), "w"), separators=(",", ":"))
    if not os.path.exists(os.path.join(UI, "cpu.json")) and os.path.exists(os.path.join("storygpt_ui", "cpu.json")):
        import shutil; shutil.copy(os.path.join("storygpt_ui", "cpu.json"), os.path.join(UI, "cpu.json"))
    log("exported", {f: round(os.path.getsize(os.path.join(UI, f)) / 1e6, 1) for f in os.listdir(UI)})
    dump()

    # figures: learning curves; accuracy per held-out piece (best seed against the previous run)
    INK2, MUTED, GRID, BASE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASE,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2})
    fig, axs = plt.subplots(1, 5, figsize=(16.5, 3.9))
    col = {"story GPT + calc": "#2a78d6", "story GPT, no calls": "#898781"}
    for cname, seed in RUNS:
        c = summary["runs"][f"{cname} | seed {seed}"]["curve"]; st = [r["step"] for r in c]
        if not c:
            continue
        kw = dict(color=col[cname], alpha=1 if seed == 0 else 0.5, label=f"{cname}, seed {seed}")
        axs[0].plot(st, [r["word"] for r in c], **kw); axs[1].plot(st, [r["newshape"] for r in c], **kw); axs[2].plot(st, [r["two"] for r in c], **kw)
        axs[3].plot(st, [r["nocall_called"] for r in c], **kw)
        axs[4].plot(st[1:], [r["text_loss"] for r in c][1:], lw=3 if cname == "story GPT, no calls" else 1.5, **kw)
    for a_, yl, ttl in zip(axs, ("answer right", "answer right", "answer right", "share of sentences with a call", "loss on held-out stories (nats/word)"),
                           ("Word problems, trained shapes", "Word problems, held-out shapes", "Two-step problems", "No-call sentences: calls made", "Story text loss")):
        a_.set_xlabel("training step"); a_.set_ylabel(yl); a_.set_title(ttl, fontsize=9, loc="left")
    for a_ in axs[:4]:
        a_.set_ylim(-0.03, 1.05)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(OUT, "figures", "1_learning.png"), dpi=150); plt.close(fig)
    rows_h = []
    for k in ("newshape", "newverb"):
        bh = summary["runs"][summary["best"]]["sets"]["on"][k]["by_held"]
        ph = summary["runs"][f"{prev_jobs[0][0]} | seed {prev_jobs[0][1]}"]["sets"]["on"][k]["by_held"] if prev_jobs else {}
        rows_h += [(k, h, v["answer_right"], ph.get(h, {}).get("answer_right"), v["n"]) for h, v in bh.items()]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(rows_h) + 1.2))
    y = np.arange(len(rows_h))
    ax.barh(y + 0.2, [r[2] for r in rows_h], 0.4, color="#2a78d6", label="varied-shape GPT (best seed)")
    if prev_jobs:
        ax.barh(y - 0.2, [r[3] or 0 for r in rows_h], 0.4, color="#c3c2b7", label="previous run (12 fixed shapes)")
    ax.set_yticks(y); ax.set_yticklabels([f"{'shape' if r[0] == 'newshape' else 'verb'}: {r[1]}" for r in rows_h], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("answer right, held-out numbers"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Each held-out sentence frame and verb", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_held_out_pieces.png"), dpi=150); plt.close(fig)
    log("done")
