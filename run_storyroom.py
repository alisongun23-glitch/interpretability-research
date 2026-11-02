"""Does handing arithmetic to the computer leave the story GPT's weights freer for language?

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 run_storyroom.py              # every phase, about 6-10 hours on a laptop CPU
    ... run_storyroom.py --phase 0 1          # only some phases (each phase reuses every job already saved)
    ... run_storyroom.py --quick              # a smoke test: 200 steps, two tiny widths, small sets, results_storyroom/quick/
    ... run_storyroom.py --stop-after 1       # stop after phase 1's verdict even if it passed
    ... run_storyroom.py --recipe m64drill    # force the recipe for phases 2-4
    ... run_storyroom.py --retrain            # ignore saved jobs
    (the OBJC_... variable is for macOS, where torch in a forked pool can die with an "objc ... initialize" message)

Four conditions on the same stories with the same problems (same random draws), the same init per seed and width:
  tool       problems with "<calc> a op b = r </calc>", the machine's tokens in the input and out of the loss (as before)
  computes   the same problems with the answer in prose and in the loss, counted k times (it must learn arithmetic)
  masked     the computes stream with the answer token left out of the loss (it sees problems, never answers)
  stories    the stories alone
Every condition trains on the same number of story tokens (the tool model's at 7,000 steps); the others get fewer steps.
Phase 0  the controls at width 128: is the stream's arithmetic text a cost (stories - computes), is an unlearnable
         answer a cost (computes - masked), is the chip a cost (tool - masked); the noise floor from three seeds.
Phase 1  a recipe matrix that makes the computes model learn addition for real (answer weight, drill lines, a smaller
         number range, three times the steps); the cheapest recipe reaching 80% on held-out addition pairs is kept.
Phase 2  the sweep: 4 widths x 4 conditions x 3 seeds with that recipe; story loss against weights per condition,
         fitted as L = E + A_c / N^alpha; the parameter-equivalent factor of each condition against stories only.
Phase 3  looking inside the sweep's models: loss by token role and by word frequency, what each head and neuron
         costs the story loss and the arithmetic loss when ablated, a linear read-out of the result from the residual
         stream, dead neurons, effective rank; and skills scored by log-probability: minimal pairs, name recall, induction.
Phase 4  plasticity at width 128: fine-tuning on a new story skill and on addition from n examples, against from scratch.
Results: results_storyroom/summary.json, jobs/*.json (one per trained model), models/, figures/, ROOM_REPORT.md.
"""
import json, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpugpt import text as X, data as D, probes as PR
from cpugpt.lm import StoryGPT, lm_losses
from cpugpt.tool import Calculator, decode

T0 = time.time()
QUICK = "--quick" in sys.argv
RETRAIN = "--retrain" in sys.argv
OUT = os.environ.get("STORYROOM_OUT", os.path.join("results_storyroom", "quick") if QUICK else "results_storyroom")
for sub in ("models", "jobs", "figures", "logs"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
T, BATCH, MAX_CALLS = 64, 32, 2
STEPS_A = 200 if QUICK else 7000           # the tool model's steps at the base recipe; every run matches its story tokens
EVAL_EVERY = 100 if QUICK else 1000
SCALE = 0.1 if QUICK else 1.0
WIDTHS = (32, 64) if QUICK else (64, 96, 128, 256)
SEEDS = (0, 1) if QUICK else (0, 1, 2)
D0 = 64 if QUICK else 128                  # the reference width: controls, recipe matrix, transfer
CONDS = ("tool", "computes", "masked", "stories")
STREAM_OF = {"tool": "tool", "computes": "plain", "masked": "plain", "stories": "stories"}
THREADS, POOL = 2, 4
PREV = "results_storytool_shapes"           # the sentence-shapes run: its tool seeds are phase 0's tool condition
OPS_OF = {"all": (D.ADD, D.SUB, D.MUL), "add": (D.ADD,)}
RECIPES = {   # per_story = math drill lines per story ("A + B = R ."), k = weight of a one-step answer in the loss, mult = steps
    "base": dict(per_story=2.0, max_result=256, ops="all", k=1, mult=1, wd=0.01),
    "basewd": dict(per_story=2.0, max_result=256, ops="all", k=1, mult=1, wd=0.1),      # phase 5: is it weight decay that costs stories?
    "k8": dict(per_story=2.0, max_result=256, ops="all", k=8, mult=1, wd=0.1),
    "k16": dict(per_story=2.0, max_result=256, ops="all", k=16, mult=1, wd=0.1),
    "drill": dict(per_story=6.0, max_result=256, ops="add", k=8, mult=1, wd=0.1),
    "m64": dict(per_story=2.0, max_result=64, ops="all", k=8, mult=1, wd=0.1),
    "m64drill": dict(per_story=6.0, max_result=64, ops="add", k=8, mult=1, wd=0.1),
    "m64x3": dict(per_story=2.0, max_result=64, ops="all", k=8, mult=3, wd=0.1),
    "m64drillx3": dict(per_story=6.0, max_result=64, ops="add", k=8, mult=3, wd=0.1),
    "m64drillx2": dict(per_story=6.0, max_result=64, ops="add", k=8, mult=2, wd=0.1),
    "drillx3": dict(per_story=6.0, max_result=256, ops="add", k=16, mult=3, wd=0.1),
}
GROK = {   # phase 6: long training with strong weight decay, the regime in which a memorised table can give way to a circuit
    "grok_wd1": dict(per_story=6.0, max_result=64, ops="add", k=8, mult=10, wd=1.0),
    "grok_wd03": dict(per_story=6.0, max_result=64, ops="add", k=8, mult=10, wd=0.3),
    "grok_m32": dict(per_story=6.0, max_result=32, ops="add", k=8, mult=10, wd=1.0),
    "grok_k32": dict(per_story=6.0, max_result=64, ops="add", k=32, mult=10, wd=1.0),
}
RECIPES.update(GROK)
MATRIX = ("k8", "m64drill") if QUICK else ("k8", "k16", "drill", "m64", "m64drill", "m64x3", "m64drillx3", "drillx3")
CRITERION = 0.8
TRANSFER_N = (25, 100) if QUICK else (25, 50, 100, 200, 400, 800, 1600)
TRANSFER_STEPS = 30 if QUICK else 300


def argv_list(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag) + 1; out = []
        while i < len(sys.argv) and not sys.argv[i].startswith("--"):
            out.append(sys.argv[i]); i += 1
        return out
    return default


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def rng(s):
    return np.random.default_rng(s)



# ------------------------------------------------------------------------------------------------ the held-out split
# Until 1 Nov 2026 cpugpt.data.is_test held out pairs with (a * 7919 + b * 104729 + k * 31337) % 5 == 0; both multipliers are
# 4 mod 5, so for addition it was "a + b == 2 (mod 5)": whole RESULT values (2, 7, 12, ...) were held out and no pair with
# another sum was. Phases 1 and 6 therefore asked B for answer tokens it had never seen as an answer. --split pair swapped in a
# symmetric hash of the pair for addition (about 20 % of pairs, every result value seen in training); that hash is now the
# repo's rule for every kind (cpugpt.data.is_test), so --split pair is a no-op alias of the default, kept because phase 7's
# job tags end in _pairsplit and its summary goes to summary_pairsplit.json. Jobs made before the fix (p0-p6) used the old split.
PAIRSPLIT = argv_list("--split", ["hash"])[0] == "pair"
pair_is_test = D.is_test      # the former --split pair rule, now the default; the name is kept for older scripts

# ------------------------------------------------------------------------------------------------ data
stories = X.load_stories("data/TinyStories-valid.txt")
train_st, test_st = stories[:-1000], stories[-1000:]
V = X.Vocab(train_st, tool_tokens=True)
TABLE = np.load("results_cpugpt/cpu_table.npz")["out"]
CALCS = {mode: Calculator(TABLE, V, mode) for mode in ("on", "off")}
TEXT = X.build_stream(V, test_st, rng(1), "test", per_story=0.0)[0]
TEXT_ALL = torch.as_tensor(TEXT[:len(TEXT) // T * T].reshape(-1, T))      # all held-out story text, 64-token rows
if QUICK:
    TEXT_ALL = TEXT_ALL[:len(TEXT_ALL) // 10]
TEXT_WIN = tuple(torch.as_tensor(t) for t in X.windows(TEXT, rng(13), 256, T))   # the earlier runs' perplexity windows
TID_APLUSB = next(i for i, t in enumerate(D.MATH) if t[0] == "a + b")
STREAMS = {}


def stream_key(rec):
    return (rec["per_story"], rec["max_result"], rec["ops"])


def with_range(M, ops):
    class _Ctx:
        def __enter__(self):
            self.old = D.MAX_RESULT, D.OPS; D.MAX_RESULT, D.OPS = M, OPS_OF[ops]
        def __exit__(self, *a):
            D.MAX_RESULT, D.OPS = self.old
    return _Ctx()


def one_step_answers(pos, info, n):
    a1 = np.zeros(n, bool)
    one = np.array([i["family"] in (0, 1) for i in info], bool)
    if one.any():
        a1[np.asarray(pos)[one] + 1] = True
    return a1


def build_streams(key):
    """The train and test streams of one recipe: tool (with chips), plain (answers in prose), stories (train only);
    each with env, roles and the one-step answer positions (ans1, the targets that count k times)."""
    if key in STREAMS:
        return STREAMS[key]
    per_story, M, ops = key
    out = {}
    with with_range(M, ops):
        for split, st, seed in (("train", train_st, 0), ("test", test_st, 2)):
            tok, env, pos, info = X.build_tool_stream(V, st, rng(seed), split, per_story=per_story)
            tokp, envp, posp, infop = X.build_tool_stream(V, st, rng(seed), split, per_story=per_story, calls=False)
            assert [i["family"] for i in info] == [i["family"] for i in infop]
            out[split] = {"tool": dict(tokens=tok, env=env, roles=X.roles(tok, env, pos, info, V.CALC), ans1=one_step_answers(pos, info, len(tok)), pos=pos, info=info),
                          "plain": dict(tokens=tokp, env=envp, roles=X.roles(tokp, envp, posp, infop), ans1=one_step_answers(posp, infop, len(tokp)), pos=posp, info=infop)}
    sto = X.build_stream(V, train_st, rng(0), "train", per_story=0.0)[0]
    out["train"]["stories"] = dict(tokens=sto, env=np.zeros(len(sto), bool), roles=np.zeros(len(sto), np.int8), ans1=np.zeros(len(sto), bool), pos=np.zeros(0, np.int64), info=[])
    for split in out:
        for nm, s in out[split].items():
            s["share"] = float((s["roles"] == 0).mean())
    STREAMS[key] = out
    log("streams", key, {nm: (len(s["tokens"]), round(s["share"], 3)) for nm, s in out["train"].items()})
    return out


BASE_KEY = stream_key(RECIPES["base"])
BASE = build_streams(BASE_KEY)
STORY_TOKENS = STEPS_A * BATCH * T * BASE["train"]["tool"]["share"]      # every run sees this many story tokens (x mult)


def steps_for(cond, rec):
    s = build_streams(stream_key(rec))["train"][STREAM_OF[cond]]["share"]
    return int(round(STORY_TOKENS * rec["mult"] / (BATCH * T * s)))


# ------------------------------------------------------------------------------------------------ problem sets
def add_pairs(n, split, seed, M, lo=2):
    """Addition word problems (trained frames) on pairs of the split with both operands >= lo and results < M, each after
    some held-out story text and cut at the cue word. Rows (n, T), ptr, a, b, target."""
    r = rng(seed)
    rows = torch.zeros((n, T), dtype=torch.long); ptr = torch.zeros(n, dtype=torch.long)
    a_, b_, t_ = (np.zeros(n, np.int64) for _ in range(3))
    with with_range(M, "add"):
        for i in range(n):
            while True:
                a, b = D.sample_pair(D.ADD, r)
                if a >= lo and b >= lo and D.is_test(D.ADD, a, b) == (split == "test"):
                    break
            ids, off, inf = X.gen_word(V, r, split, kind=D.ADD, pair=(a, b), call=False)
            prompt = ids[:inf["cue"] + 1]
            pre = int(r.integers(0, max(1, T - len(prompt) - 12 - 2 * MAX_CALLS + 1))); j = int(r.integers(0, len(TEXT) - pre))
            seq = list(TEXT[j:j + pre]) + prompt
            rows[i, :len(seq)] = torch.as_tensor(seq); ptr[i] = len(seq) - 1
            a_[i], b_[i], t_[i] = a, b, inf["target"]
    return {"rows": rows, "ptr": ptr, "a": a_, "b": b_, "target": t_}


def head(ms, n):
    return {k: v[:n] for k, v in ms.items()}


ADDSETS = {M: {"test": add_pairs(max(40, int(500 * SCALE)), "test", 21, M), "train": add_pairs(max(30, int(200 * SCALE)), "train", 22, M)}
           for M in sorted({r["max_result"] for r in RECIPES.values()})}


def score_add(m, ms, calc, max_calls, keep=200):
    """Answer right on a set of addition problems; the first number the model writes outside a call is its answer."""
    res = decode(m, ms["rows"], ms["ptr"], calc, n_new=12, max_calls=max_calls)
    ans = np.array([-1 if x["answer"] is None else x["answer"] for x in res])
    right = ans == ms["target"]
    return {"n": len(res), "answer_right": float(right.mean()), "copied_operand": float(((ans == ms["a"]) | (ans == ms["b"])).mean()),
            "no_answer": float((ans < 0).mean()), "calls_per_problem": float(np.mean([len(x["calls"]) for x in res])),
            "pairs": [[int(a), int(b), int(t), int(x)] for a, b, t, x in zip(ms["a"], ms["b"], ms["target"], ans)][:keep]}


def restate_set(n, split, seed):
    """No-call sentences with a held-out frame or verb that restate a number (the new skill of phase 4), cut at the cue."""
    r = rng(seed)
    rows = torch.zeros((n, T), dtype=torch.long); ptr = torch.zeros(n, dtype=torch.long); t_ = np.zeros(n, np.int64)
    for i in range(n):
        ids, off, inf = X.gen_nocall(V, r, split, held="shape" if i % 2 else "verb", restate_only=True)
        prompt = ids[:inf["cue"] + 1]
        pre = int(r.integers(0, max(1, T - len(prompt) - 12 + 1))); j = int(r.integers(0, len(TEXT) - pre))
        seq = list(TEXT[j:j + pre]) + prompt
        rows[i, :len(seq)] = torch.as_tensor(seq); ptr[i] = len(seq) - 1; t_[i] = inf["target"]
    return {"rows": rows, "ptr": ptr, "target": t_, "a": np.full(n, -1), "b": np.full(n, -1)}


RESTATE = restate_set(max(40, int(300 * SCALE)), "test", 23)

N_PAIRS = max(60, int(500 * SCALE))
PAIRS = {p: PR.minimal_pairs(V, rng(31 + i), p, N_PAIRS) for i, p in enumerate(PR.PARADIGMS)}
PAIRS["recall_1"] = PR.recall_pairs(V, rng(41), N_PAIRS, gap=1)
PAIRS["recall_2"] = PR.recall_pairs(V, rng(42), N_PAIRS, gap=2)
assert all(len(v) >= 0.5 * N_PAIRS for v in PAIRS.values()), {k: len(v) for k, v in PAIRS.items()}


# ------------------------------------------------------------------------------------------------ models
def make_model(d, seed):
    torch.manual_seed(seed)
    return StoryGPT(len(V), T, d=d, d_mlp=4 * d, n_layers=4, n_heads=4, cpu=None, num0=X.NUM0)


def n_params(m):
    total = sum(p.numel() for p in m.parameters())
    return {"total": total, "non_embedding": total - m.emb.weight.numel() - m.pos.weight.numel()}


def story_loss(m, rows=None):
    rows = TEXT_ALL if rows is None else rows
    return PR.nll(m, rows[:, :-1], rows[:, 1:])


def tag_of(job):
    return job["tag"]


def job_path(job, kind):
    return os.path.join(OUT, kind, tag_of(job) + (".pt" if kind == "models" else ".json"))


def load_model(job):
    m = make_model(job["d"], job["seed"])
    m.load_state_dict(torch.load(job.get("pretrained") or job_path(job, "models")))
    return m


# ------------------------------------------------------------------------------------------------ one job: train + evaluate
def train(job, m):
    rec = RECIPES[job["recipe"]]; st = build_streams(stream_key(rec))["train"][STREAM_OF[job["cond"]]]
    steps = job.get("steps") or steps_for(job["cond"], rec)
    k = rec["k"] if job["cond"] in ("tool", "computes") else 1
    weight = (1 + (k - 1) * st["ans1"]).astype(np.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=job.get("lr", 1e-3), weight_decay=rec["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s, n=steps: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / n))))
    r = rng(100 + job["seed"]); curve = []; t1 = time.time()
    curve_set = head(ADDSETS[rec["max_result"]]["test"], max(30, int(150 * SCALE)))
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0 or step == steps:
            row = {"step": step, "story_tokens": step * BATCH * T * st["share"], "story_loss": story_loss(m)}
            if job["cond"] == "tool":
                row["add_heldout"] = score_add(m, curve_set, CALCS["on"], MAX_CALLS, keep=0)["answer_right"]
                row["add_heldout_off"] = score_add(m, curve_set, CALCS["off"], MAX_CALLS, keep=0)["answer_right"]
            else:
                row["add_heldout"] = score_add(m, curve_set, CALCS["off"], 0, keep=0)["answer_right"]
            curve.append(row)
            print(tag_of(job), {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in row.items()}, f"{time.time() - t1:.0f}s", flush=True)
        if step == steps:
            break
        a, b, e, ro, w = X.windows(st["tokens"], r, BATCH, T, st["env"], st["roles"], weight)
        ignore = e | (ro == X.ROLE_ANSWER) if job["cond"] == "masked" else e
        loss, _ = lm_losses(m, torch.as_tensor(a), torch.as_tensor(b), ignore=torch.as_tensor(ignore), weight=torch.as_tensor(w) if k != 1 else None)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return {"curve": curve, "steps": steps, "share": st["share"], "story_tokens": steps * BATCH * T * st["share"], "k": k, "train_seconds": time.time() - t1}


def evaluate(job, m):
    rec = RECIPES[job["recipe"]]; key = stream_key(rec); M = rec["max_result"]
    row = {"params": n_params(m), "story_loss": story_loss(m)}
    row["perplexity"] = float(np.exp(row["story_loss"]))
    row["story_loss_win"] = PR.nll(m, *TEXT_WIN); row["perplexity_win"] = float(np.exp(row["story_loss_win"]))
    sets = ADDSETS[M]
    if job["cond"] == "tool":
        row["add"] = {"machine_on": score_add(m, sets["test"], CALCS["on"], MAX_CALLS), "machine_off": score_add(m, sets["test"], CALCS["off"], MAX_CALLS),
                      "train_pairs_off": score_add(m, sets["train"], CALCS["off"], MAX_CALLS)}
        row["add"]["own"] = row["add"]["machine_off"]["answer_right"]
    else:
        row["add"] = {"direct": score_add(m, sets["test"], CALCS["off"], 0), "train_pairs": score_add(m, sets["train"], CALCS["off"], 0)}
        row["add"]["own"] = row["add"]["direct"]["answer_right"]
    ts = build_streams(key)["test"]["tool" if job["cond"] == "tool" else "plain"]
    a, b, e, ro = X.windows(ts["tokens"], rng(7), 400, T, ts["env"], ts["roles"])
    row["by_role"] = PR.loss_by_role(m, torch.as_tensor(a), torch.as_tensor(b), torch.as_tensor(ro))
    row["by_frequency"] = PR.loss_by_frequency(m, TEXT_ALL[:, :-1], TEXT_ALL[:, 1:], V)
    row["skills"] = {p: PR.pair_score(m, pr) for p, pr in PAIRS.items()}
    row["skills"]["induction"] = PR.induction_test(m, V, n=N_PAIRS)
    row["restate_new"] = score_add(m, RESTATE, CALCS["off"], 0, keep=0)["answer_right"]
    return row


def run_job(job):
    """Train (or load) one model and measure it; the result is saved under jobs/ and reused next time."""
    path = job_path(job, "jobs")
    if os.path.exists(path) and not RETRAIN:
        return json.load(open(path))
    torch.set_num_threads(THREADS)
    t1 = time.time()
    if job.get("pretrained"):
        m = load_model(job); info = {"curve": [], "steps": job.get("steps"), "share": build_streams(stream_key(RECIPES[job["recipe"]]))["train"][STREAM_OF[job["cond"]]]["share"], "pretrained": job["pretrained"]}
        info["story_tokens"] = (job.get("steps") or 0) * BATCH * T * info["share"]
    elif os.path.exists(job_path(job, "models")) and not RETRAIN:
        m = load_model(job); info = {"curve": [], "steps": job.get("steps") or steps_for(job["cond"], RECIPES[job["recipe"]]), "reloaded": True}
    else:
        m = make_model(job["d"], job["seed"])
        if job.get("init_from"):                                                   # continue training a saved model
            m.load_state_dict(torch.load(os.path.join(OUT, "models", job["init_from"] + ".pt")))
        info = train(job, m)
        torch.save(m.state_dict(), job_path(job, "models"))
    row = {**job, **info, **evaluate(job, m), "seconds": time.time() - t1}
    json.dump(row, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("done", tag_of(job), "story loss", round(row["story_loss"], 4), "add own", round(row["add"]["own"], 3), f"{row['seconds']:.0f}s")
    return row


def run_jobs(jobs, fn=run_job):
    """Run jobs in a forked pool, longest first; results in the jobs' order."""
    import multiprocessing as mp
    jobs = list(jobs)
    order = sorted(range(len(jobs)), key=lambda i: -(jobs[i].get("d", D0) ** 2) * (RECIPES[jobs[i]["recipe"]]["mult"] if "recipe" in jobs[i] else 1))
    if not jobs:
        return []
    with mp.get_context("fork").Pool(min(POOL, len(jobs))) as pool:
        got = dict(zip(order, pool.map(fn, [jobs[i] for i in order], chunksize=1)))
    return [got[i] for i in range(len(jobs))]


def make_job(cond, d, seed, recipe, phase, **kw):
    job = {"cond": cond, "d": d, "seed": seed, "recipe": recipe, "phase": phase, **kw}
    job["tag"] = f"p{phase}_{cond}_d{d}_s{seed}_{recipe}" + ("_" + kw["suffix"] if "suffix" in kw else "")
    return job


# ------------------------------------------------------------------------------------------------ phase 3: inside
def probe_rows(job, n):
    """Rows ending where the result should be readable: the cue token of a one-step problem (computes, masked,
    stories) or the "=" of its call (tool); the residual stream there is probed for the answer."""
    rec = RECIPES[job["recipe"]]; ts = build_streams(stream_key(rec))["test"]["tool" if job["cond"] == "tool" else "plain"]
    rows, ys = [], []
    for p, info in zip(ts["pos"], ts["info"]):
        if info["family"] not in (0, 1):
            continue
        if job["cond"] == "tool":
            at = info["start"] + info["env"].index(True) - 1          # the "=" before the machine's result
        else:
            at = int(p)
        if at < T - 1:
            continue                                                  # a full window of real context, no padding
        rows.append(ts["tokens"][at - T + 1:at + 1]); ys.append(info["target"])
        if len(rows) >= n:
            break
    return torch.as_tensor(np.stack(rows)), np.array(ys)


def inside(job):
    """Phase 3 on one trained model."""
    path = os.path.join(OUT, "jobs", tag_of(job) + "_inside.json")
    if os.path.exists(path) and not RETRAIN:
        return json.load(open(path))
    torch.set_num_threads(THREADS); t1 = time.time()
    m = load_model(job); rec = RECIPES[job["recipe"]]
    n_ref = max(16, int(48 * SCALE * 2))
    ref = TEXT_ALL[:n_ref]; c = PR.cache(m, ref)
    means = PR.unit_means(c)
    out = {"dead_units": PR.dead_units(c), "effective_rank": [PR.effective_rank(r_) for r_ in c["resid"]]}
    story = (TEXT_ALL[n_ref:2 * n_ref, :-1], TEXT_ALL[n_ref:2 * n_ref, 1:], None)
    ts = build_streams(stream_key(rec))["test"]["tool" if job["cond"] == "tool" else "plain"]
    a, b, e, ro = X.windows(ts["tokens"], rng(8), 600, T, ts["env"], ts["roles"])
    amask = (ro == X.ROLE_ANSWER) | ((ro == X.ROLE_CALL) if job["cond"] == "tool" else False)
    keep = np.nonzero(amask.any(1))[0][:n_ref]
    arith = (torch.as_tensor(a[keep]), torch.as_tensor(b[keep]), torch.as_tensor(amask[keep]))
    out["ablation"] = PR.ablation_scan(m, means, story, arith, neuron_step=1 if job["d"] <= 128 else 2)
    rows, y = probe_rows(job, max(80, int(1200 * SCALE)))
    cp = PR.cache(m, rows, want=("resid",))
    tr = rng(9).random(len(y)) < 0.7
    out["result_probe"] = [PR.linear_probe(r_[:, -1].numpy(), y, tr) for r_ in cp["resid"]]
    out["seconds"] = time.time() - t1
    json.dump(out, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("inside", tag_of(job), "best probe r2", round(max(p["r2"] for p in out["result_probe"]), 3), f"{out['seconds']:.0f}s")
    return out


# ------------------------------------------------------------------------------------------------ phase 4: plasticity
def transfer(job):
    """Fine-tune one checkpoint (or a fresh model) on n examples of a task, then score it."""
    path = os.path.join(OUT, "jobs", job["tag"] + ".json")
    if os.path.exists(path) and not RETRAIN:
        return json.load(open(path))
    torch.set_num_threads(THREADS); t1 = time.time()
    rec = RECIPES[job["recipe"]]; r = rng(500 + job["seed"] + job["n"])
    m = make_model(job["d"], job["seed"]) if job["cond"] == "scratch" else load_model(job["source"])
    toks = []
    with with_range(rec["max_result"], "add"):
        for i in range(job["n"]):
            if job["task"] == "addition":
                ids = X.snippet(V, r, "train", tid=TID_APLUSB, call=False)[0]
            else:
                ids = X.gen_nocall(V, r, "train", held="shape" if i % 2 else "verb", restate_only=True)[0]
            toks += ids + [X.EOS]
    stream = np.array(toks * max(1, (4 * T) // len(toks) + 1), np.int64)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
    for step in range(TRANSFER_STEPS):
        a, b = X.windows(stream, r, 16, T)
        loss, _ = lm_losses(m, torch.as_tensor(a), torch.as_tensor(b))
        opt.zero_grad(); loss.backward(); opt.step()
    ms = head(ADDSETS[rec["max_result"]]["test"], max(40, int(300 * SCALE))) if job["task"] == "addition" else RESTATE
    sc = score_add(m, ms, CALCS["off"], 0, keep=0)
    out = {**{k: v for k, v in job.items() if k != "source"}, "answer_right": sc["answer_right"], "story_loss": story_loss(m), "seconds": time.time() - t1}
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------------------------------------ analysis
def fit_curves(points, alphas=np.linspace(0.05, 1.5, 146)):
    """points: {cond: [(N, loss), ...]}. L = E + A_c / N^alpha with E and alpha shared, by a grid over alpha and least
    squares for the rest. The parameter-equivalent factor of a condition against stories is (A_c / A_stories)^(1/alpha):
    how many times the weights it needs for the same story loss."""
    conds = [c for c in CONDS if c in points]
    best = None
    for al in alphas:
        rows, y = [], []
        for ci, c in enumerate(conds):
            for N, L in points[c]:
                row = np.zeros(1 + len(conds)); row[0] = 1; row[1 + ci] = N ** (-al); rows.append(row); y.append(L)
        Xm, yv = np.array(rows), np.array(y)
        coef = np.linalg.lstsq(Xm, yv, rcond=None)[0]; sse = float(((Xm @ coef - yv) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(al), coef)
    sse, al, coef = best
    A = {c: float(coef[1 + i]) for i, c in enumerate(conds)}
    fac = {c: (A[c] / A["stories"]) ** (1 / al) if "stories" in A and A[c] > 0 and A["stories"] > 0 else None for c in conds}
    return {"E": float(coef[0]), "alpha": al, "A": A, "factor_vs_stories": fac, "sse": sse}


def analyse(summary):
    rows = [json.load(open(os.path.join(OUT, "jobs", f))) for f in sorted(os.listdir(os.path.join(OUT, "jobs"))) if f.endswith(".json") and "_inside" not in f and not f.startswith("transfer_")]
    summary["jobs"] = {r["tag"]: {**{k: v for k, v in r.items() if k not in ("by_role", "by_frequency", "skills", "add", "curve")}, "add_own": r.get("add", {}).get("own")} for r in rows if "cond" in r}
    for r in rows:
        if "cond" in r:
            summary["jobs"][r["tag"]].update(by_role=r.get("by_role"), by_frequency=r.get("by_frequency"), skills=r.get("skills"), add=r.get("add"), curve=r.get("curve"))
    # phase 0 contrasts at D0
    p0 = [r for r in rows if r.get("phase") == 0]
    if p0:
        by = {c: [r["story_loss"] for r in p0 if r["cond"] == c and not r.get("equal_steps")] for c in CONDS}
        mean = {c: float(np.mean(v)) for c, v in by.items() if v}; sd = {c: float(np.std(v, ddof=1)) if len(v) > 1 else None for c, v in by.items() if v}
        summary["phase0"] = {"story_loss": by, "mean": mean, "sd": sd, "equal_steps": {r["tag"]: r["story_loss"] for r in p0 if r.get("equal_steps")},
                             "contrasts": {k: (mean[a_] - mean[b_]) if a_ in mean and b_ in mean else None for k, (a_, b_) in
                                           (("tool - computes", ("tool", "computes")), ("computes - masked", ("computes", "masked")),
                                            ("tool - masked", ("tool", "masked")), ("stories - computes", ("stories", "computes")))},
                             "add_own": {c: [r["add"]["own"] for r in p0 if r["cond"] == c] for c in CONDS}}
    # phase 1 verdict
    p1 = [r for r in rows if r.get("phase") == 1]
    if p1:
        summary["phase1"] = {r["tag"]: {"recipe": r["recipe"], "d": r["d"], "add_heldout": r["add"]["own"], "train_pairs": r["add"].get("train_pairs", {}).get("answer_right"),
                                        "copied_operand": r["add"]["direct"]["copied_operand"], "story_loss": r["story_loss"], "steps": r["steps"], "train_seconds": r.get("train_seconds")} for r in p1}
    # phase 2 fit
    p2 = [r for r in rows if r.get("phase") == 2 and not r.get("lr_check")]
    if p2:
        pts = {c: [(r["params"]["non_embedding"], r["story_loss"]) for r in p2 if r["cond"] == c] for c in CONDS}
        fit = fit_curves(pts)
        rs = np.random.default_rng(0); boots = []
        groups = {k: [r for r in p2 if (r["cond"], r["d"]) == k] for k in {(r["cond"], r["d"]) for r in p2}}
        for _ in range(300):
            bp = {c: [] for c in CONDS}
            for (c_, d_), g in groups.items():
                for i in rs.integers(0, len(g), len(g)):
                    bp[c_].append((g[i]["params"]["non_embedding"], g[i]["story_loss"]))
            boots.append(fit_curves({c: v for c, v in bp.items() if v}))
        ci = {c: [float(np.nanpercentile([b["factor_vs_stories"][c] if b["factor_vs_stories"].get(c) is not None else np.nan for b in boots], q)) for q in (2.5, 97.5)] for c in CONDS if c in pts}
        per_width = {}
        for d_ in WIDTHS:
            per_width[d_] = {c: {"mean": float(np.mean([r["story_loss"] for r in p2 if r["cond"] == c and r["d"] == d_])),
                                 "sd": float(np.std([r["story_loss"] for r in p2 if r["cond"] == c and r["d"] == d_], ddof=1)) if sum(1 for r in p2 if r["cond"] == c and r["d"] == d_) > 1 else None,
                                 "add_own": [r["add"]["own"] for r in p2 if r["cond"] == c and r["d"] == d_]} for c in CONDS if any(r["cond"] == c and r["d"] == d_ for r in p2)}
            seeds = sorted({r["seed"] for r in p2 if r["d"] == d_})
            diff = lambda x, y: [next(r["story_loss"] for r in p2 if r["cond"] == x and r["d"] == d_ and r["seed"] == s_) - next(r["story_loss"] for r in p2 if r["cond"] == y and r["d"] == d_ and r["seed"] == s_)
                                 for s_ in seeds if any(r["cond"] == x and r["d"] == d_ and r["seed"] == s_ for r in p2) and any(r["cond"] == y and r["d"] == d_ and r["seed"] == s_ for r in p2)]
            per_width[d_]["paired"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, "n": len(v)} if v else None
                                       for k, v in (("tool - computes", diff("tool", "computes")), ("computes - masked", diff("computes", "masked")),
                                                    ("tool - masked", diff("tool", "masked")), ("stories - computes", diff("stories", "computes")))}
        summary["phase2"] = {"fit": fit, "factor_ci95": ci, "per_width": per_width, "lr": {r["tag"]: r["story_loss"] for r in rows if r.get("lr_check")}}
    ins = {f[:-12]: json.load(open(os.path.join(OUT, "jobs", f))) for f in os.listdir(os.path.join(OUT, "jobs")) if f.endswith("_inside.json")}
    if ins:
        summary["phase3"] = {}
        for tag, o in ins.items():
            ab = o["ablation"]["rows"]
            def count(kind, which):
                sel = [r_ for r_ in ab if r_[0] == kind]
                if not sel: return None
                ds = np.array([r_[3] for r_ in sel]); da = np.array([r_[4] for r_ in sel])
                ts_, ta = max(0.01, 2 * float(np.std(ds))), max(0.01, 2 * float(np.std(da)))
                return {"arith_only": int(((da > ta) & (ds <= ts_)).sum()), "story_only": int(((ds > ts_) & (da <= ta)).sum()), "both": int(((ds > ts_) & (da > ta)).sum()), "n": len(sel)}
            summary["phase3"][tag] = {"dead_units": o["dead_units"], "effective_rank": o["effective_rank"], "base": {"story": o["ablation"]["base_story"], "arith": o["ablation"]["base_arith"]},
                                      "units": {k: count(k, None) for k in ("head", "neuron")}, "result_probe_r2_by_layer": [p["r2"] for p in o["result_probe"]],
                                      "result_probe_bits_by_layer": [p["bits_mean"] for p in o["result_probe"]]}
    p4 = [json.load(open(os.path.join(OUT, "jobs", f))) for f in os.listdir(os.path.join(OUT, "jobs")) if "_p4_" in f]
    if p4:
        summary["phase4"] = {}
        for task in ("addition", "newskill"):
            summary["phase4"][task] = {c: {str(n): [r["answer_right"] for r in p4 if r["task"] == task and r["cond"] == c and r["n"] == n] for n in TRANSFER_N}
                                       for c in CONDS + ("scratch",)}
    return summary


def figures(summary):
    INK2, MUTED, GRID, BASEC = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASEC, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2})
    col = {"tool": "#d6632a", "computes": "#2a78d6", "masked": "#7a5cc4", "stories": "#4a9a5a", "scratch": "#898781"}
    label = {"tool": "A tool", "computes": "B computes", "masked": "C masked answer", "stories": "D stories only"}
    jobs = summary.get("jobs", {})
    p2 = [r for r in jobs.values() if r.get("phase") == 2 and not r.get("lr_check")]
    if p2 and "phase2" in summary:
        fit = summary["phase2"]["fit"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        Ns = np.geomspace(min(r["params"]["non_embedding"] for r in p2) * 0.8, max(r["params"]["non_embedding"] for r in p2) * 1.2, 50)
        for c in CONDS:
            pts = [(r["params"]["non_embedding"], r["story_loss"]) for r in p2 if r["cond"] == c]
            if not pts: continue
            axs[0].scatter(*zip(*pts), color=col[c], s=18, alpha=0.8, label=f"{label[c]} (A = {fit['A'][c]:.2f})" if c in fit["A"] else label[c])
            if c in fit["A"]:
                axs[0].plot(Ns, fit["E"] + fit["A"][c] / Ns ** fit["alpha"], color=col[c], lw=1.2)
        axs[0].set_xscale("log"); axs[0].set_xlabel("weights outside the embedding"); axs[0].set_ylabel("story loss (nats per token, held-out stories)")
        axs[0].set_title(f"Story loss against size, matched story tokens (fit: E = {fit['E']:.3f}, alpha = {fit['alpha']:.2f})", fontsize=9, loc="left")
        axs[0].legend(frameon=False, fontsize=8)
        pw = summary["phase2"]["per_width"]
        keys = ["tool - computes", "computes - masked", "tool - masked", "stories - computes"]
        for i, k in enumerate(keys):
            xs = [d_ for d_ in WIDTHS if str(d_) in pw or d_ in pw]
            vals = [(pw.get(d_) or pw.get(str(d_)))["paired"].get(k) for d_ in xs]
            axs[1].errorbar([x + (i - 1.5) * 2 for x in xs], [v["mean"] if v else np.nan for v in vals], yerr=[(v["se"] or 0) if v else 0 for v in vals], fmt="o-", ms=4, lw=1.2, capsize=2, label=k)
        axs[1].axhline(0, color=BASEC, lw=1); axs[1].set_xlabel("width d"); axs[1].set_ylabel("difference in story loss (nats), paired seeds")
        axs[1].set_title("Contrasts by width (negative = the first is better)", fontsize=9, loc="left"); axs[1].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "1_curves.png"), dpi=150); plt.close(fig)
    if "phase3" in summary:
        tags = [t for t in summary["phase3"] if f"_d{D0}_s0_" in t]
        fig, axs = plt.subplots(1, max(1, len(tags)), figsize=(3.2 * max(1, len(tags)), 3.4), squeeze=False)
        for ax, t in zip(axs[0], tags):
            o = json.load(open(os.path.join(OUT, "jobs", t + "_inside.json")))
            rows = o["ablation"]["rows"]; c = next((cc for cc in CONDS if f"_{cc}_" in t), "tool")
            ne = [r_ for r_ in rows if r_[0] == "neuron"]; he = [r_ for r_ in rows if r_[0] == "head"]
            ax.scatter([r_[4] for r_ in ne], [r_[3] for r_ in ne], s=6, alpha=0.5, color=col[c], label="neurons")
            ax.scatter([r_[4] for r_ in he], [r_[3] for r_ in he], s=30, marker="^", color=INK2, label="heads")
            ax.set_xlabel("change in arithmetic loss when removed"); ax.set_ylabel("change in story loss"); ax.set_title(label[c], fontsize=9, loc="left")
            ax.axhline(0, color=BASEC, lw=1); ax.axvline(0, color=BASEC, lw=1)
        axs[0][0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"What each unit costs when replaced by its mean (width {D0}, seed 0)", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_where.png"), dpi=150); plt.close(fig)
    p0or2 = [r for r in jobs.values() if r.get("d") == D0 and r.get("phase") in (0, 2) and r.get("skills") and not r.get("equal_steps")]
    if p0or2:
        names = list(PR.PARADIGMS) + ["recall_1", "recall_2"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        x = np.arange(len(names)); w = 0.2
        for i, c in enumerate(CONDS):
            rs_ = [r for r in p0or2 if r["cond"] == c]
            if not rs_: continue
            axs[0].bar(x + (i - 1.5) * w, [np.mean([r["skills"][n]["score"] for r in rs_]) for n in names], w, color=col[c], label=label[c])
            axs[1].bar(i, np.mean([r["skills"]["induction"]["induction"] for r in rs_]), color=col[c])
        axs[0].set_xticks(x, labels=[n.replace("_", " ") for n in names], rotation=20, ha="right", fontsize=8); axs[0].set_ylim(0.4, 1.02)
        axs[0].set_ylabel("share of pairs the right sentence wins"); axs[0].set_title(f"Skills scored by log-probability (width {D0}, seeds averaged)", fontsize=9, loc="left")
        axs[0].legend(frameon=False, fontsize=8); axs[1].set_xticks(range(4), labels=[label[c] for c in CONDS], fontsize=8)
        axs[1].set_ylabel("nats saved on the second copy"); axs[1].set_title("Induction: repeating a random run of words", fontsize=9, loc="left")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "3_skills.png"), dpi=150); plt.close(fig)
    if "phase4" in summary:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, task in zip(axs, ("addition", "newskill")):
            for c in CONDS + ("scratch",):
                ys = [np.mean(v) if v else np.nan for n, v in summary["phase4"][task][c].items()]
                ax.plot(TRANSFER_N, ys, "o-", ms=4, color=col[c], label=label.get(c, "from scratch"))
            ax.set_xscale("log"); ax.set_xlabel("fine-tuning examples"); ax.set_ylabel("answer right"); ax.set_title({"addition": "Learning addition without the tool", "newskill": "A new story skill (held-out frames and verbs)"}[task], fontsize=9, loc="left")
        axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "4_transfer.png"), dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
if __name__ == "__main__":
    phases = [int(p) for p in argv_list("--phase", ["0", "1", "2", "3", "4"])]
    stop_after = int(argv_list("--stop-after", ["99"])[0])
    summary_path = os.path.join(OUT, "summary_pairsplit.json" if PAIRSPLIT else "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    summary.update(context=T, batch=BATCH, steps_tool=STEPS_A, story_tokens=STORY_TOKENS, widths=list(WIDTHS), seeds=list(SEEDS), recipes=RECIPES,
                   data={"stories_train": len(train_st), "stories_test": len(test_st), "vocab": len(V), "test_text_tokens": int(len(TEXT)),
                         "streams": {str(k): {nm: {"tokens": len(s["tokens"]), "story_share": s["share"], "answers_one_step": int(s["ans1"].sum())} for nm, s in v["train"].items()}
                                     for k, v in STREAMS.items()}},
                   pairs={k: len(v) for k, v in PAIRS.items()})

    def dump():
        summary["runtime_s"] = round(time.time() - T0, 1)
        json.dump(summary, open(summary_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    if 0 in phases:
        jobs = []
        for s in SEEDS:                                                           # the shapes run's tool seeds are the matched tool runs
            pre = os.path.join(PREV, "models", f"story_GPT_plus_calc_seed{s}.pt")
            if os.path.exists(pre) and not QUICK:
                jobs.append(make_job("tool", D0, s, "base", 0, pretrained=pre, steps=STEPS_A))
            else:
                jobs.append(make_job("tool", D0, s, "base", 0))
        pre = os.path.join(PREV, "models", "story_GPT_no_calls_seed0.pt")
        if os.path.exists(pre) and not QUICK:
            jobs.append(make_job("computes", D0, 0, "base", 0, pretrained=pre, steps=STEPS_A, equal_steps=True, suffix="eqsteps"))
        jobs += [make_job(c, D0, s, "base", 0) for c in ("computes", "masked", "stories") for s in SEEDS]
        log("phase 0:", len(jobs), "jobs", {c: steps_for(c, RECIPES["base"]) for c in CONDS})
        run_jobs(jobs); analyse(summary); dump()
        log("phase 0 contrasts", summary.get("phase0", {}).get("contrasts"), "sd", summary.get("phase0", {}).get("sd"))

    chosen = argv_list("--recipe", [summary.get("chosen_recipe")])[0]
    if 1 in phases:
        for name in MATRIX:
            build_streams(stream_key(RECIPES[name]))
        jobs = [make_job("computes", D0, 0, name, 1) for name in MATRIX]
        log("phase 1:", len(jobs), "jobs", {j["recipe"]: steps_for("computes", RECIPES[j["recipe"]]) for j in jobs})
        res = run_jobs(jobs)
        passed = [r for r in res if r["add"]["own"] >= CRITERION]
        if not passed and not QUICK:
            best2 = sorted(res, key=lambda r: -r["add"]["own"])[:2]
            log("phase 1: nothing passed at width", D0, "- the best two at width", max(WIDTHS), [r["recipe"] for r in best2])
            res2 = run_jobs([make_job("computes", max(WIDTHS), 0, r["recipe"], 1) for r in best2])
            passed = [r for r in res2 if r["add"]["own"] >= CRITERION]; res += res2
        cost = lambda r: r["steps"] * (r["d"] ** 2)
        if passed:
            chosen = min(passed, key=cost)["recipe"]; summary["phase1_verdict"] = "passed"
        else:
            chosen = max(res, key=lambda r: r["add"]["own"])["recipe"]; summary["phase1_verdict"] = "no recipe reached the criterion"
        summary["chosen_recipe"] = chosen; summary["criterion"] = CRITERION
        analyse(summary); dump()
        log("phase 1 verdict:", summary["phase1_verdict"], "recipe", chosen, {r["tag"]: round(r["add"]["own"], 3) for r in res})
        if stop_after <= 1 or (not passed and not QUICK and "--force" not in sys.argv):
            figures(summary); dump(); log("stopping after phase 1"); sys.exit(0)
    chosen = chosen or "base"
    rec = RECIPES[chosen]

    if 2 in phases:
        build_streams(stream_key(rec))
        big = max(WIDTHS)
        if not QUICK:
            lr_jobs = [make_job("stories", big, 0, chosen, 2, lr=lr, lr_check=True, suffix=f"lr{lr:g}") for lr in (1e-3, 5e-4)]
            small = [make_job(c, d, s, chosen, 2) for d in WIDTHS if d < big for c in CONDS for s in SEEDS]
            res = run_jobs(lr_jobs + small)
            lr_big = min(res[:2], key=lambda r: r["story_loss"])["lr"]; summary["lr_big"] = lr_big
            log("lr check at width", big, {r["tag"]: round(r["story_loss"], 4) for r in res[:2]}, "chosen", lr_big)
            run_jobs([make_job(c, big, s, chosen, 2, lr=lr_big) for c in CONDS for s in SEEDS])
        else:
            run_jobs([make_job(c, d, s, chosen, 2) for d in WIDTHS for c in CONDS for s in SEEDS])
        analyse(summary); dump()
        log("phase 2 fit", summary["phase2"]["fit"], "factor CI", summary["phase2"]["factor_ci95"])

    if 3 in phases:
        build_streams(stream_key(rec))
        jobs = [j for j in (make_job(c, d, s, chosen, 2, lr=summary.get("lr_big", 1e-3) if d == max(WIDTHS) and not QUICK else 1e-3) for d in WIDTHS for c in CONDS for s in SEEDS)
                if os.path.exists(job_path(j, "models"))]
        log("phase 3:", len(jobs), "models")
        run_jobs(jobs, inside); analyse(summary); dump()

    if 4 in phases:
        build_streams(stream_key(rec))
        jobs = []
        for s in SEEDS:
            for c in CONDS + ("scratch",):
                src = make_job(c, D0, s, chosen, 2) if c != "scratch" else None
                if src is not None and not os.path.exists(job_path(src, "models")):
                    continue
                for task in ("addition", "newskill"):
                    for n in TRANSFER_N:
                        jobs.append({"cond": c, "d": D0, "seed": s, "recipe": chosen, "task": task, "n": n, "source": src, "tag": f"transfer_p4_{c}_{task}_n{n}_s{s}"})
        log("phase 4:", len(jobs), "fine-tunes")
        run_jobs(jobs, transfer); analyse(summary); dump()

    if 5 in phases:
        # Matched A, C, D at the chosen recipe (width D0, seed 0) when phase 1 found no passing recipe: phase 1's B model at
        # that recipe memorised its training pairs, so this asks whether holding that table in the weights costs story loss.
        # Run as: python3 run_storyroom.py --phase 5 --recipe <recipe>; the B model is phase 1's job at the same recipe.
        build_streams(stream_key(rec))
        jobs = [make_job(c, D0, 0, chosen, 5) for c in ("tool", "masked", "stories")] + [make_job("computes", D0, 0, "basewd", 5)]
        log("phase 5:", len(jobs), "jobs at recipe", chosen, {j["tag"]: steps_for(j["cond"], RECIPES[j["recipe"]]) for j in jobs})
        res = run_jobs(jobs)
        rows = {r["cond"]: r for r in res[:3]}
        rows["computes"] = json.load(open(job_path(make_job("computes", D0, 0, chosen, 1), "jobs")))
        base = json.load(open(job_path(make_job("computes", D0, 0, "base", 0), "jobs")))
        summary["phase5"] = {"recipe": chosen, "story_loss": {c: rows[c]["story_loss"] for c in CONDS}, "add_own": {c: rows[c]["add"]["own"] for c in CONDS},
                             "steps": {c: rows[c]["steps"] for c in CONDS}, "skills": {c: {k: (v["score"] if isinstance(v, dict) and "score" in v else v) for k, v in rows[c]["skills"].items()} for c in CONDS},
                             "restate_new": {c: rows[c]["restate_new"] for c in CONDS},
                             "contrasts": {"tool - computes": rows["tool"]["story_loss"] - rows["computes"]["story_loss"], "computes - masked": rows["computes"]["story_loss"] - rows["masked"]["story_loss"],
                                           "tool - masked": rows["tool"]["story_loss"] - rows["masked"]["story_loss"], "stories - computes": rows["stories"]["story_loss"] - rows["computes"]["story_loss"]},
                             "wd_control": {"basewd": res[3]["story_loss"], "base": base["story_loss"], "basewd - base": res[3]["story_loss"] - base["story_loss"]}}
        analyse(summary); dump()
        log("phase 5 contrasts", summary["phase5"]["contrasts"], "wd control", summary["phase5"]["wd_control"])

    if 6 in phases:
        # The grokking attempt (the user's follow-up to the phase 1 verdict): B at long-training recipes with strong weight
        # decay, the regime in which a memorised table can give way to a circuit. The cheapest recipe that passes the
        # criterion gets matched A, C, D (same init, same recipe), then the inside look (phase 3) and the fine-tunes (phase 4)
        # on those four models. python3 run_storyroom.py --phase 6 [--grok name ...] [--criterion 0.8]
        crit = float(argv_list("--criterion", [CRITERION])[0])
        names = argv_list("--grok", list(GROK))
        for nm in names:
            build_streams(stream_key(RECIPES[nm]))
        jobs = [make_job("computes", D0, 0, nm, 6) for nm in names]
        log("phase 6:", len(jobs), "grokking jobs", {j["tag"]: steps_for("computes", RECIPES[j["recipe"]]) for j in jobs})
        res = run_jobs(jobs)
        p6 = {"criterion": crit, "candidates": {r["tag"]: {"recipe": r["recipe"], "add_heldout": r["add"]["own"], "train_pairs": r["add"]["train_pairs"]["answer_right"],
                                                            "story_loss": r["story_loss"], "steps": r["steps"], "train_seconds": r.get("train_seconds"),
                                                            "curve": [[c["step"], c["add_heldout"]] for c in r["curve"]]} for r in res}}
        log("phase 6 candidates", {r["tag"]: round(r["add"]["own"], 3) for r in res})
        passed = [r for r in res if r["add"]["own"] >= crit]
        if passed:
            g = min(passed, key=lambda r: r["steps"])["recipe"]; p6["recipe"] = g; p6["verdict"] = "passed"
            mj = [make_job(c, D0, 0, g, 6) for c in ("tool", "masked", "stories")]
            log("phase 6 matched at", g, {j["tag"]: steps_for(j["cond"], RECIPES[g]) for j in mj})
            rows = {r["cond"]: r for r in run_jobs(mj)}; rows["computes"] = next(r for r in res if r["recipe"] == g)
            p6.update(story_loss={c: rows[c]["story_loss"] for c in CONDS}, add_own={c: rows[c]["add"]["own"] for c in CONDS}, steps={c: rows[c]["steps"] for c in CONDS},
                      skills={c: {k: (v["score"] if isinstance(v, dict) and "score" in v else v) for k, v in rows[c]["skills"].items()} for c in CONDS},
                      restate_new={c: rows[c]["restate_new"] for c in CONDS}, by_role={c: rows[c]["by_role"] for c in CONDS},
                      contrasts={"tool - computes": rows["tool"]["story_loss"] - rows["computes"]["story_loss"], "computes - masked": rows["computes"]["story_loss"] - rows["masked"]["story_loss"],
                                 "tool - masked": rows["tool"]["story_loss"] - rows["masked"]["story_loss"], "stories - computes": rows["stories"]["story_loss"] - rows["computes"]["story_loss"]})
            summary["phase6"] = p6; analyse(summary); dump()
            log("phase 6 contrasts", p6["contrasts"])
            four = [make_job(c, D0, 0, g, 6) for c in CONDS]
            log("phase 6 inside:", len(four), "models"); run_jobs(four, inside)
            tj = []
            for c in CONDS + ("scratch",):
                src = make_job(c, D0, 0, g, 6) if c != "scratch" else None
                for task in ("addition", "newskill"):
                    for n in TRANSFER_N:
                        tj.append({"cond": c, "d": D0, "seed": 0, "recipe": g, "task": task, "n": n, "source": src, "tag": f"transfer_p6_{c}_{task}_n{n}_s0"})
            log("phase 6 fine-tunes:", len(tj)); tr = run_jobs(tj, transfer)
            summary["phase4"] = {task: {c: {str(n): [r["answer_right"] for r in tr if r["task"] == task and r["cond"] == c and r["n"] == n] for n in TRANSFER_N}
                                        for c in CONDS + ("scratch",)} for task in ("addition", "newskill")}
            p6["phase4_from"] = "the phase 6 models"
        else:
            p6["verdict"] = "no grokking recipe reached the criterion"
        summary["phase6"] = p6
        analyse(summary); figures(summary); dump()
        log("phase 6 verdict:", p6["verdict"], p6.get("contrasts"))

    if 7 in phases:
        # Addition under the fair pair split: python3 run_storyroom.py --phase 7 --split pair [--recipes m64drill m64drillx3].
        # B alone at width D0, seed 0; the question is only whether exact accuracy on unseen PAIRS (results seen) is reachable.
        assert PAIRSPLIT, "phase 7 is the pair-split re-measurement; run it with --split pair"
        names = argv_list("--recipes", ["m64drill", "m64drillx3"])
        for nm in names:
            build_streams(stream_key(RECIPES[nm]))
        jobs = [make_job("computes", D0, 0, nm, 7, suffix="pairsplit") for nm in names]
        log("phase 7:", len(jobs), "pair-split jobs", {j["tag"]: steps_for("computes", RECIPES[j["recipe"]]) for j in jobs})
        res = run_jobs(jobs)
        summary["phase7"] = {"split": "pair", "candidates": {r["tag"]: {"recipe": r["recipe"], "add_heldout": r["add"]["own"], "train_pairs": r["add"]["train_pairs"]["answer_right"],
                                                                        "story_loss": r["story_loss"], "steps": r["steps"], "train_seconds": r.get("train_seconds"),
                                                                        "curve": [[c["step"], c["add_heldout"]] for c in r["curve"]], "pairs": r["add"]["direct"]["pairs"]} for r in res}}
        dump()
        log("phase 7 candidates", {r["tag"]: round(r["add"]["own"], 3) for r in res})

    if 8 in phases:
        # The matched comparison at the recipe where B adds under the fair split (phase 7's one-cycle drill model: 65 % exact on
        # unseen pairs, under the 80 % gate but computing): A, C, D at that recipe from the same init, then the inside look and
        # the fine-tunes on the four models; alongside, two more B runs to bracket the 65 % (a second seed; two cycles).
        # python3 run_storyroom.py --phase 8 --split pair [--recipe m64drill]
        assert PAIRSPLIT, "phase 8 is the pair-split matched comparison; run it with --split pair"
        g = argv_list("--recipe", ["m64drill"])[0]
        for nm in (g, "m64drillx2"):
            build_streams(stream_key(RECIPES[nm]))
        four = [make_job(c, D0, 0, g, 7, suffix="pairsplit") for c in CONDS]
        extra = [make_job("computes", D0, 1, g, 7, suffix="pairsplit"), make_job("computes", D0, 0, "m64drillx2", 7, suffix="pairsplit")]
        log("phase 8:", len(four + extra), "jobs at recipe", g, {j["tag"]: steps_for(j["cond"], RECIPES[j["recipe"]]) for j in four + extra})
        by = {r["tag"]: r for r in run_jobs(four + extra)}
        rows = {j["cond"]: by[j["tag"]] for j in four}
        p8 = {"split": "pair", "recipe": g, "story_loss": {c: rows[c]["story_loss"] for c in CONDS}, "add_own": {c: rows[c]["add"]["own"] for c in CONDS},
              "add": {c: {k: {kk: vv for kk, vv in v.items() if kk != "pairs"} for k, v in rows[c]["add"].items() if isinstance(v, dict)} for c in CONDS},
              "steps": {c: rows[c]["steps"] for c in CONDS}, "skills": {c: {k: (v["score"] if isinstance(v, dict) and "score" in v else v) for k, v in rows[c]["skills"].items()} for c in CONDS},
              "restate_new": {c: rows[c]["restate_new"] for c in CONDS}, "by_role": {c: rows[c]["by_role"] for c in CONDS},
              "contrasts": {"tool - computes": rows["tool"]["story_loss"] - rows["computes"]["story_loss"], "computes - masked": rows["computes"]["story_loss"] - rows["masked"]["story_loss"],
                            "tool - masked": rows["tool"]["story_loss"] - rows["masked"]["story_loss"], "stories - computes": rows["stories"]["story_loss"] - rows["computes"]["story_loss"]},
              "bracket": {j["tag"]: {"recipe": j["recipe"], "seed": j["seed"], "add_heldout": by[j["tag"]]["add"]["own"], "train_pairs": by[j["tag"]]["add"]["train_pairs"]["answer_right"],
                                     "story_loss": by[j["tag"]]["story_loss"], "steps": by[j["tag"]]["steps"], "curve": [[c["step"], c["add_heldout"]] for c in by[j["tag"]]["curve"]]} for j in extra}}
        summary["phase8"] = p8; dump()
        log("phase 8 contrasts", p8["contrasts"], "bracket", {t: round(v["add_heldout"], 3) for t, v in p8["bracket"].items()})
        log("phase 8 inside:", len(four), "models"); run_jobs(four, inside)
        tj = []
        for c in CONDS + ("scratch",):
            src = make_job(c, D0, 0, g, 7, suffix="pairsplit") if c != "scratch" else None
            for task in ("addition", "newskill"):
                for n in TRANSFER_N:
                    tag = f"transfer_p8_{c}_{task}_n{n}_s0" if g == "m64drill" else f"transfer_p8_{g}_{c}_{task}_n{n}_s0"
                    tj.append({"cond": c, "d": D0, "seed": 0, "recipe": g, "task": task, "n": n, "source": src, "tag": tag})
        log("phase 8 fine-tunes:", len(tj)); tr = run_jobs(tj, transfer)
        summary["phase4"] = {task: {c: {str(n): [r["answer_right"] for r in tr if r["task"] == task and r["cond"] == c and r["n"] == n] for n in TRANSFER_N}
                                    for c in CONDS + ("scratch",)} for task in ("addition", "newskill")}
        p8["phase4_from"] = f"the phase 8 models at {g}"
        analyse(summary)
        summary["phase3"] = {t: v for t, v in summary.get("phase3", {}).items() if t.endswith(f"_{g}_pairsplit")}
        p8["phase3"] = summary["phase3"]; p8["phase4"] = summary["phase4"]
        summary.setdefault("phase8_by_recipe", {})[g] = p8
        figures(summary)
        for f in ("2_where", "4_transfer"):                                             # one pair of figures per recipe
            src = os.path.join(OUT, "figures", f + ".png")
            if os.path.exists(src):
                os.replace(src, os.path.join(OUT, "figures", f"{f}_{g}.png"))
        dump()
        log("phase 8 done", {t: {"probe_r2": max(v["result_probe_r2_by_layer"]), "units": v["units"]} for t, v in summary.get("phase3", {}).items()})

    analyse(summary); (None if PAIRSPLIT else figures(summary)); dump()
    log("done")
