"""A story GPT that checks the computer it calls: the machine can be wrong, so the model learns to call again.

    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 run_storycheck.py            # about 45 minutes on a laptop CPU
    ... run_storycheck.py --reuse     # train only the runs without a saved model, then analyse
    ... run_storycheck.py --analyse   # reuse all trained models (a few minutes)
    ... run_storycheck.py --quick     # a smoke test: 200 steps, small sets, written to results_storycheck/quick
    STORYCHECK_NOISE=0.05 sets the noise on the analog neural CPU for the calibration and the live check.

The setup of run_storytool.py (the model writes "<calc> 12 + 5 =" into the story, the computer answers "17 </calc>",
the model copies 17) with one change: the computer is flaky. In the training stream a call's answer is wrong with
probability P_ERR (cpugpt.check.FlakyMachine: one flipped bit, the answer of another operation, a random byte, or 0 as
from a timeout, the last two being what the analog CPU mostly does under noise), and after a wrong
answer the text shows the model calling again, "<calc> 12 + 5 =" once more, until the answer is right (at most 3
blocks). The machine's tokens stay out of the loss; the retry is the model's own token, so the model is trained to
decide, from the numbers it can see, whether to trust an answer or ask again.
Two conditions, two seeds each:
  checks numbers            the answer and the retry are all the model sees
  checks numbers + signal   the machine also says "<unsure>" after an answer it doubts: the hit rate on wrong
                            answers and the false-alarm rate on right ones are those of the fully analog neural
                            CPU's worst signal margin under noise (calibrated on the real machine, see calibrate())
plus the shapes run's "story GPT + calc" seed 2 (results_storytool_shapes/), which was trained with a machine that is
never wrong: "trusts the machine".
Evaluation (held-out operand pairs after held-out story text): math templates, word problems in trained and held-out
shapes, two-step problems and no-call sentences, each with a perfect machine and with the flaky one (same P_ERR;
word problems also at twice the error rate); per machine answer whether the model called again (detection after a
wrong answer, false alarms after a right one), by what went wrong (which bit, which operation) and by the signal;
the final answer; perplexity; spurious calls; a live run of 300 word problems on the fully analog neural CPU under
noise (500 problems), where "<unsure>" comes from the machine's real worst signal margin. Results go to results_storycheck/.
"""
import json, os, sys, time
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
from cpugpt.check import FlakyMachine
from gatecpu.design import build_cpu
from gatecpu.calc import PROGRAM, CAP, call_ticks
from gatecpu.ref import port_inputs
from gatesnake.sim import Machine, load_library

QUICK = "--quick" in sys.argv
SHAPES = "results_storytool_shapes"                   # the run whose models trust the machine
OUT = os.environ.get("STORYCHECK_OUT", os.path.join("results_storycheck", "quick") if QUICK else "results_storycheck")
for d in (OUT, os.path.join(OUT, "figures"), os.path.join(OUT, "models")):
    os.makedirs(d, exist_ok=True)
T0 = time.time()
T, BATCH, STEPS, EVAL_EVERY, MAX_CALLS = 64, 32, (200 if QUICK else 7000), (100 if QUICK else 500), 3
SCALE = 0.1 if QUICK else 1.0
P_ERR = 0.25
NOISE_SD = float(os.environ.get("STORYCHECK_NOISE", "0.045"))
RUNS = [("checks numbers", 0), ("checks numbers", 1), ("checks numbers + signal", 0), ("checks numbers + signal", 1)]
BASE = ("trusts the machine", 2)                      # evaluated only: results_storytool_shapes/models/story_GPT_plus_calc_seed2.pt
OPS = ((D.ADD, "add"), (D.SUB, "sub"), (D.MUL, "mul"))
OPNAME = {k: nm for k, nm in OPS}
summary = {"context": T, "batch": BATCH, "steps": STEPS, "max_calls": MAX_CALLS, "p_err": P_ERR, "noise_sd": NOISE_SD, "runs": {}}


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def model_path(cname, seed):
    if cname == BASE[0]:
        return os.path.join(SHAPES, "models", f"story_GPT_plus_calc_seed{seed}.pt")
    return os.path.join(OUT, "models", f"{cname.replace(' ', '_').replace('+', 'plus')}_seed{seed}.pt")


def key(job):
    return f"{job[0]} | seed {job[1]}"


# ---------------------------------------------------------------- the real machine: how often it errs under noise
def analog_cpu(sd, seed):
    """The fully analog neural CPU under noise. Returns call(a, op, b) -> (answer, worst signal margin, ticks); the
    program is reloaded before every call (a slipped store can corrupt it), so calls are independent."""
    net = build_cpu()[0].finalize()
    cpu = Machine(net, mode="neural", library=load_library(), dff="gates", noise=sd, seed=seed)
    halt_i = [i for i, g in enumerate(net.regs) if g["name"] == "halt"][0]
    out_i = [i for i, g in enumerate(net.regs) if g["name"].startswith("out[")]

    def call(a, op, b):
        for i, byte in enumerate(PROGRAM):
            cpu.step(port_inputs(prog=1, we=1, addr=i, data=byte))
        for t in call_ticks(int(a), int(b), int(op)):
            cpu.step(port_inputs(**t))
        cpu.worst = 0.0
        ticks = 0
        for ticks in range(1, CAP + 1):
            cpu.step(port_inputs())
            if cpu.Q[halt_i] > 0.5:
                break
        return sum(int(cpu.Q[j] > 0.5) << k for k, j in enumerate(out_i)), float(cpu.worst), ticks
    return call


def calibrate(n):
    """n random calls on the analog CPU at NOISE_SD: the share answered wrong, and the worst-signal threshold that best
    separates wrong from right answers (largest hit rate minus false-alarm rate). Cached in OUT."""
    path = os.path.join(OUT, "analog_calibration.json")
    if os.path.exists(path):
        return json.load(open(path))
    call = analog_cpu(NOISE_SD, 7); r = np.random.default_rng(7); rows = []; t1 = time.time()
    for _ in range(n):
        op = int(r.choice([D.ADD, D.SUB, D.MUL])); a, b = D.sample_pair(op, r)
        live, worst, ticks = call(a, op, b)
        rows.append({"op": op, "a": a, "b": b, "live": live, "right": int(TABLE[op, a, b]), "worst": worst, "ticks": ticks})
    wrong = np.array([x["live"] != x["right"] for x in rows]); w = np.array([x["worst"] for x in rows])
    cands = sorted(set(w.tolist()))
    th = max(cands, key=lambda t: (w[wrong] > t).mean() - (w[~wrong] > t).mean()) if wrong.any() and (~wrong).any() else 0.5
    out = {"noise_sd": NOISE_SD, "calls": n, "p_wrong": float(wrong.mean()), "threshold": float(th),
           "hit": float((w[wrong] > th).mean()) if wrong.any() else None, "false_alarm": float((w[~wrong] > th).mean()) if (~wrong).any() else None,
           "by_op": {OPNAME[k]: {"n": int(sum(x["op"] == k for x in rows)), "p_wrong": float(np.mean([x["live"] != x["right"] for x in rows if x["op"] == k]))} for k in OPNAME},
           "wrong_kinds": dict(Counter(("timeout" if x["live"] == 0 and x["right"] != 0 else "one bit" if bin(x["live"] ^ x["right"]).count("1") == 1 else "several bits")
                                       for x in rows if x["live"] != x["right"])),
           "seconds": round(time.time() - t1, 1), "rows": rows}
    json.dump(out, open(path, "w"), indent=1)
    return out


# ---------------------------------------------------------------- data
stories = X.load_stories("data/TinyStories-valid.txt")
train_st, test_st = stories[:-1000], stories[-1000:]
V = X.Vocab(train_st, tool_tokens=True, extra=X.CHECK_TOKENS)
V_OLD = X.Vocab(train_st, tool_tokens=True)           # the shapes run's vocabulary: the same ids without "<unsure>"
assert V.itos[:len(V_OLD)] == V_OLD.itos and len(V) == len(V_OLD) + 1
TABLE = np.load("results_cpugpt/cpu_table.npz")["out"]
TEXT, _, _ = X.build_stream(V, test_st, np.random.default_rng(1), "test", per_story=0.0)
CAL = calibrate(30 if QUICK else 300)
HIT, FA = (CAL["hit"] if CAL["hit"] is not None else 0.8), (CAL["false_alarm"] if CAL["false_alarm"] is not None else 0.1)
summary["analog_calibration"] = {k: v for k, v in CAL.items() if k != "rows"}
log("analog CPU at noise", NOISE_SD, {k: v for k, v in CAL.items() if k not in ("rows", "by_op")})


def machine(p_err=P_ERR, signal=False, seed=0, **kw):
    return FlakyMachine(TABLE, V, p_err=p_err, signal=signal, hit=HIT, false_alarm=FA, seed=seed, **kw)


STREAMS = {}
for signal in (False, True):
    toks, env, _, info = X.build_tool_stream(V, train_st, np.random.default_rng(0), "train", calls=machine(signal=signal, seed=1))
    STREAMS[signal] = (toks, env)
    nm = "signal" if signal else "numbers"
    summary.setdefault("data", {})[nm] = {"train_tokens": int(len(toks)), "machine_tokens": int(env.sum()),
                                          "unsure_tokens": int((toks == V.UNSURE).sum()), "call_blocks": int((toks == V.CALC).sum()),
                                          "problems_with_calls": sum(1 for i in info if i["calls"])}
summary["data"].update(stories_train=len(train_st), stories_test=len(test_st), vocab=len(V), test_text_tokens=int(len(TEXT)))
log("streams", summary["data"])

FAMILIES = {"math": (lambda r, s: X.snippet(V, r, s), 16),
            "word": (lambda r, s: X.gen_word(V, r, s), 16),
            "newshape": (lambda r, s: X.gen_word(V, r, s, held="shape"), 16),
            "newverb": (lambda r, s: X.gen_word(V, r, s, held="verb"), 16),
            "two": (lambda r, s: X.gen_two(V, r, s, call=False), 16),
            "nocall": (lambda r, s: X.gen_nocall(V, r, s, restate_only=True), 10)}


def problem_set(n, split, seed, family):
    """n problems after some held-out story text, cut at the cue word; rows have room for MAX_CALLS blocks (each up
    to 3 machine tokens) and n_new model tokens."""
    make, n_new = FAMILIES[family]
    r = np.random.default_rng(seed)
    rows = torch.zeros((n, T), dtype=torch.long); ptr = torch.zeros(n, dtype=torch.long)
    info = {k: np.zeros(n, np.int64) for k in ("target", "kind", "kind2", "a", "b", "c", "r1")}
    calls, held, cramped = [], [], 0
    for i in range(n):
        ids, off, inf = make(r, split)
        prompt = ids[:inf["cue"] + 1]
        room = T - len(prompt) - n_new - 3 * MAX_CALLS
        cramped += room < 0
        pre = int(r.integers(0, max(1, room + 1))); j = int(r.integers(0, len(TEXT) - pre))
        seq = (list(TEXT[j:j + pre]) + prompt)[-T:]
        rows[i, :len(seq)] = torch.as_tensor(seq); ptr[i] = len(seq) - 1
        for k in info:
            info[k][i] = inf.get(k, 0)
        calls.append(inf["calls"]); held.append(inf.get("held", ""))
    return {"rows": rows, "ptr": ptr, "info": info, "calls": calls, "held": held, "n_new": n_new, "family": family, "cramped": cramped}


def head(ms, n):
    return {**ms, "rows": ms["rows"][:n], "ptr": ms["ptr"][:n], "info": {k: v[:n] for k, v in ms["info"].items()}, "calls": ms["calls"][:n], "held": ms["held"][:n]}


SIZES = {"math": 1000, "word": 3000, "newshape": 1500, "newverb": 1000, "two": 1500, "nocall": 1000}
SETS = {k: problem_set(max(30, int(n * SCALE)), "test", 11 + 10 * i, k) for i, (k, n) in enumerate(SIZES.items())}
summary["data"]["cramped_rows"] = {k: s["cramped"] for k, s in SETS.items()}
CURVE_SETS = {k: head(SETS[k], 300) for k in ("word", "two", "nocall")}
TEXT_WIN = X.windows(TEXT, np.random.default_rng(13), 256, T)
TEXT_ALL = torch.as_tensor(TEXT[:len(TEXT) // T * T].reshape(-1, T))
if QUICK:
    TEXT_ALL = TEXT_ALL[:len(TEXT_ALL) // 10]
rp = np.random.default_rng(14); starts = rp.integers(0, len(TEXT) - 16, 50 if QUICK else 200)
PREFIXES = torch.as_tensor(np.stack([np.pad(TEXT[s:s + 16], (0, T - 16)) for s in starts]))
THREADS = 2
log("sets", {k: len(s["rows"]) for k, s in SETS.items()}, "cramped", summary["data"]["cramped_rows"])


# ---------------------------------------------------------------- evaluation
def eval_set(m, ms, mach, max_calls=MAX_CALLS):
    """Generate the call(s) and the answer for every problem with machine mach. Per problem: called, well-formed, first call
    right, answer right, calls; per machine answer: whether the model called again with the same call (a retry): the
    detection rate after wrong answers and the false-alarm rate after right ones, when it had the chance (not at the
    call cap, row not ended), by what went wrong and by the machine's signal."""
    res = decode(m, ms["rows"], ms["ptr"], mach, n_new=ms["n_new"], max_calls=max_calls)
    info, N = ms["info"], len(res)
    rec = {k: np.zeros(N, bool) for k in ("called", "wellformed", "call_right", "answer_right", "any_wrong", "copied_wrong", "handoff")}
    rec["n_calls"] = np.array([len(r["calls"]) for r in res]); rec["n_model"] = np.array([r["n_model"] for r in res])
    ans = []
    for i, r in enumerate(res):
        c, e = r["calls"], ms["calls"][i]
        rec["called"][i] = len(c) > 0
        rec["wellformed"][i] = len(c) > 0 and all(x["ok"] for x in c)
        rec["answer_right"][i] = r["answer"] == info["target"][i]
        if e and c:
            a, op, b, _ = e[0]
            rec["call_right"][i] = call_right(c[0], a, op, b)
        mc = [x for x in c if x["by"] == "machine"]
        rec["any_wrong"][i] = any(x["wrong"] for x in mc)
        rec["copied_wrong"][i] = r["answer"] is not None and any(x["wrong"] and x["r"] == r["answer"] for x in mc)
        if len(e) == 2:
            a2, op2, b2, _ = e[1]
            good = [x for x in c if x["ok"] and x["by"] == "machine" and not x["wrong"]]
            rec["handoff"][i] = any(x["a"] == a2 and x["op"] == op2 and x["b"] == b2 for x in good)   # a right second step from the right first result
        for j, x in enumerate(c):
            if x["by"] != "machine":
                continue
            nxt = c[j + 1] if j + 1 < len(c) else None
            retried = nxt is not None and nxt["ok"] and call_right(nxt, x["a"], x["op"], x["b"])
            chance = j + 1 < max_calls and not (nxt is None and r["answer"] is None)
            ans.append({"wrong": x["wrong"], "err": x["err"], "bit": x["bit"], "unsure": x["unsure"], "retried": retried, "chance": chance, "op": x["op"]})

    def rate(sel):
        s = [a["retried"] for a in ans if a["chance"] and sel(a)]
        return {"n": len(s), "rate": float(np.mean(s)) if s else None}

    def stats(s):
        n = int(s.sum()); aw = rec["any_wrong"][s]
        d = {"n": n, "answer_right": float(rec["answer_right"][s].mean()), "ci95": wilson(int(rec["answer_right"][s].sum()), n),
             "called": float(rec["called"][s].mean()), "wellformed": float(rec["wellformed"][s].mean()), "call_right": float(rec["call_right"][s].mean()),
             "calls_per_problem": float(rec["n_calls"][s].mean()), "model_tokens": float(rec["n_model"][s].mean()),
             "problems_with_a_wrong_answer": float(aw.mean()),
             "answer_right_given_wrong_answer": float(rec["answer_right"][s][aw].mean()) if aw.any() else None,
             "answer_right_given_no_wrong_answer": float(rec["answer_right"][s][~aw].mean()) if (~aw).any() else None,
             "copied_a_wrong_answer": float(rec["copied_wrong"][s].mean())}
        if ms["family"] == "two":
            d["handoff"] = float(rec["handoff"][s].mean())
        return d

    groups = {nm: info["kind"] == k for k, nm in OPS} if ms["family"] != "nocall" else {}
    out = {"all": stats(np.ones(N, bool)), "by_group": {k: stats(s) for k, s in groups.items() if s.any()},
           "detection": rate(lambda a: a["wrong"]), "false_alarm": rate(lambda a: not a["wrong"]),
           "by_err": {**{e: rate(lambda a, e=e: a["wrong"] and a["err"] == e) for e in ("op", "random", "zero", "bit")},
                      **{f"bit{k}": rate(lambda a, k=k: a["wrong"] and a["err"] == "bit" and a["bit"] == k) for k in range(8)}},
           "by_op": {OPNAME[k]: {"detection": rate(lambda a, k=k: a["wrong"] and a["op"] == k), "false_alarm": rate(lambda a, k=k: not a["wrong"] and a["op"] == k)} for k in OPNAME},
           "by_signal": {"unsure": {"detection": rate(lambda a: a["wrong"] and a["unsure"]), "false_alarm": rate(lambda a: not a["wrong"] and a["unsure"])},
                         "sure": {"detection": rate(lambda a: a["wrong"] and not a["unsure"]), "false_alarm": rate(lambda a: not a["wrong"] and not a["unsure"])}},
           "machine": {"calls": mach.n_calls, "wrong": mach.n_wrong, "unsure": mach.n_unsure}, "_res": res}
    if ms["family"] == "newshape":
        ks = np.array(ms["held"])
        out["by_held"] = {k: {"n": int((ks == k).sum()), "answer_right": float(rec["answer_right"][ks == k].mean())} for k in sorted(set(ms["held"])) if k}
    return out


def public(ev):
    return {k: v for k, v in ev.items() if not k.startswith("_")}


def eval_text(m):
    a, b = (torch.as_tensor(t) for t in TEXT_WIN)
    m.eval()
    with torch.no_grad():
        lp = m(a)["logits"][:, :, 0].log_softmax(-1).gather(2, b[..., None])[..., 0]
    m.train()
    keep = b != 0
    return {"loss": float(-lp[keep].mean()), "perplexity": float(torch.exp(-lp[keep].mean()))}


def spurious(m):
    """Calls in plain held-out story text: the top guess at every position, and free writing (40 words) from 200 openings."""
    gaps = []
    m.eval()
    with torch.no_grad():
        for s in range(0, len(TEXT_ALL), 250):
            w = TEXT_ALL[s:s + 250]
            lg = m(w)["logits"][:, :-1, 0]
            other = lg.clone(); other[:, :, V.CALC] = float("-inf")
            gaps.append((lg[:, :, V.CALC] - other.max(-1).values).flatten())
    m.train()
    gap = torch.cat(gaps)
    res = decode(m, PREFIXES, torch.full((len(PREFIXES),), 15), machine(p_err=0.0, seed=3), n_new=40, max_calls=MAX_CALLS, stop_at_number=False)
    gen_calls = sum(len(r["calls"]) for r in res); gen_tokens = sum(r["n_model"] for r in res)
    return {"words": int(TEXT_ALL.numel()), "argmax_calls": int((gap > 0).sum()), "argmax_calls_per_1000_words": float((gap > 0).sum()) / TEXT_ALL.numel() * 1000,
            "generated_words": gen_tokens, "generated_calls": gen_calls, "generated_calls_per_1000_words": gen_calls / max(gen_tokens, 1) * 1000,
            "examples": [{"text": V.decode(PREFIXES[i, :16].tolist() + r["written"]), "calls": r["calls"]} for i, r in enumerate(res) if r["calls"]][:6]}


def make_model(cname):
    return StoryGPT(len(V_OLD) if cname == BASE[0] else len(V), T, n_layers=4, cpu=None, num0=X.NUM0)


def load(job):
    m = make_model(job[0]); m.load_state_dict(torch.load(model_path(*job)))
    return m


def train_run(job):
    cname, seed = job
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed); r = np.random.default_rng(100 + seed)
    m = make_model(cname)
    stream, env = STREAMS["signal" in cname]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s / STEPS))))
    curve, t1 = [], time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            row = {"step": step, "text_loss": eval_text(m)["loss"]}
            ev = eval_set(m, CURVE_SETS["word"], machine(p_err=0.0, seed=5))
            row["word_perfect"] = ev["all"]["answer_right"]
            ev = eval_set(m, CURVE_SETS["word"], machine(signal="signal" in cname, seed=5))
            row["word_flaky"] = ev["all"]["answer_right"]; row["detection"] = ev["detection"]["rate"]; row["false_alarm"] = ev["false_alarm"]["rate"]
            ev = eval_set(m, CURVE_SETS["two"], machine(signal="signal" in cname, seed=5))
            row["two_flaky"] = ev["all"]["answer_right"]
            row["nocall_called"] = eval_set(m, CURVE_SETS["nocall"], machine(p_err=0.0, seed=5))["all"]["called"]
            curve.append(row)
            print(cname, seed, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, f"{time.time() - t1:.0f}s", flush=True)
        if step == STEPS:
            break
        a, b, e = X.windows(stream, r, BATCH, T, env=env)
        loss, _ = lm_losses(m, torch.as_tensor(a), torch.as_tensor(b), ignore=torch.as_tensor(e))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    path = model_path(cname, seed)
    torch.save(m.state_dict(), path)
    return cname, seed, path, curve, time.time() - t1


def evaluate_run(job):
    """Everything measured on one trained model (runs in a pool)."""
    cname, seed = job
    torch.set_num_threads(THREADS)
    m = load(job)
    sig = "signal" in cname
    machines = {"perfect": dict(p_err=0.0), "flaky": dict(p_err=P_ERR), "flaky, twice the errors": dict(p_err=2 * P_ERR)}
    if sig:
        machines.update({"flaky + signal": dict(p_err=P_ERR, signal=True), "flaky + perfect signal": dict(p_err=P_ERR, signal=True, hit=1.0, false_alarm=0.0),
                         "flaky + signal, twice the errors": dict(p_err=2 * P_ERR, signal=True)})
    row = {"text": eval_text(m), "sets": {}, "trained_with_signal": sig}
    for mname, kw in machines.items():
        row["sets"][mname] = {}
        for k, ms in SETS.items():
            if "twice" in mname and k != "word":
                continue
            if "perfect signal" in mname and k not in ("word", "two"):
                continue
            kw2 = dict(kw); h, f = kw2.pop("hit", None), kw2.pop("false_alarm", None)
            mach = machine(seed=21, **kw2)
            if h is not None:
                mach.hit, mach.false_alarm = h, f
            row["sets"][mname][k] = public(eval_set(m, ms, mach))
    row["spurious"] = spurious(m)
    on = row["sets"]
    log(cname, seed, "perfect", {k: round(on["perfect"][k]["all"]["answer_right"], 3) for k in on["perfect"]},
        "flaky", {k: round(on["flaky"][k]["all"]["answer_right"], 3) for k in on["flaky"]},
        "detect", round(on["flaky"]["word"]["detection"]["rate"] or 0, 3), "false alarm", round(on["flaky"]["word"]["false_alarm"]["rate"] or 0, 3),
        "ppl", round(row["text"]["perplexity"], 2))
    return cname, seed, row


class AnalogMachine(Calculator):
    """The fully analog neural CPU under noise answering the model's calls at decoding time; "<unsure>" when the worst
    signal margin of the call passed the calibrated threshold (signal=True)."""

    def __init__(self, v, sd, seed, signal, threshold):
        super().__init__(TABLE, v, "on")
        self.call, self.signal, self.threshold = analog_cpu(sd, seed), signal, threshold
        self.last = None; self.n_calls = self.n_wrong = self.n_unsure = self.n_doubt = 0

    def __call__(self, a, op, b):
        live, worst, ticks = self.call(a, op, b)
        right = int(self.table[op, a, b]); doubt = worst > self.threshold
        self.last = {"r": live, "right": right, "wrong": live != right, "err": "analog" if live != right else None, "bit": None,
                     "unsure": bool(self.signal and doubt), "worst": worst}
        self.n_calls += 1; self.n_wrong += live != right; self.n_unsure += self.last["unsure"]; self.n_doubt += doubt
        return live


def live_check(job, n, signal):
    """n held-out word problems decoded with the real analog CPU under noise answering."""
    m = load(job)
    mach = AnalogMachine(V, NOISE_SD, 99, signal, CAL["threshold"])
    t1 = time.time()
    ev = public(eval_set(m, head(SETS["word"], n), mach))
    ev.pop("by_err", None); ev.pop("by_held", None)
    ev["machine"].update(doubts=mach.n_doubt, seconds=round(time.time() - t1, 1))
    return ev


def generate(m, prompt, mach, n=40, seed=0):
    pid = V.encode(X.words(prompt))[-(T - n - 3 * MAX_CALLS):]
    rows = torch.zeros((1, T), dtype=torch.long); rows[0, :len(pid)] = torch.as_tensor(pid)
    r = decode(m, rows, torch.tensor([len(pid) - 1]), mach, n_new=n, max_calls=MAX_CALLS, seed=seed, stop_at_number=False)[0]
    return pid + r["written"], r["calls"]


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    import multiprocessing as mp
    old = json.load(open(os.path.join(OUT, "summary.json")))["runs"] if os.path.exists(os.path.join(OUT, "summary.json")) else {}
    reuse = "--analyse" in sys.argv or "--reuse" in sys.argv
    done = {j: (j[0], j[1], model_path(*j), old.get(key(j), {}).get("curve", []), old.get(key(j), {}).get("train_seconds")) for j in RUNS if reuse and os.path.exists(model_path(*j))}
    todo = [j for j in RUNS if j not in done]
    assert not (todo and "--analyse" in sys.argv), f"no saved model for {todo}"
    if todo:
        THREADS = max(1, 10 // len(todo))
        with mp.get_context("fork").Pool(len(todo)) as pool:
            done.update({r[:2]: r for r in pool.map(train_run, todo)})
    for cname, seed, path, curve, secs in (done[j] for j in RUNS):
        summary["runs"][key((cname, seed))] = {"curve": curve, "train_seconds": secs, "steps": STEPS}
    summary["runs"][key(BASE)] = {"trained_on": "results_storytool_shapes/: a machine that is never wrong"}
    log("trained")
    jobs = RUNS + [BASE]
    THREADS = max(1, 10 // len(jobs))
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        for cname, seed, row in pool.map(evaluate_run, jobs):
            summary["runs"][key((cname, seed))].update(row)
    torch.set_num_threads(4)

    def dump():
        summary["runtime_s"] = round(time.time() - T0, 1)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    dump()

    # the best seed of each condition on word problems with the flaky machine (its own kind)
    def own(j):
        r = summary["runs"][key(j)]["sets"]
        return r["flaky + signal" if "signal" in j[0] else "flaky"]["word"]["all"]["answer_right"]
    best = {c: max((j for j in RUNS if j[0] == c), key=own) for c in ("checks numbers", "checks numbers + signal")}
    summary["best"] = {c: key(j) for c, j in best.items()}

    # live: the real analog CPU under noise answering 300 word problems
    n_live = 20 if QUICK else 500
    live = {}
    for name, job, signal in (("checks numbers + signal, signal on", best["checks numbers + signal"], True),
                              ("checks numbers + signal, signal off", best["checks numbers + signal"], False),
                              ("checks numbers", best["checks numbers"], False), ("trusts the machine", BASE, False)):
        live[name] = live_check(job, n_live, signal)
        log("live", name, {k: live[name]["all"][k] for k in ("answer_right", "calls_per_problem", "problems_with_a_wrong_answer", "copied_a_wrong_answer")},
            "detect", live[name]["detection"], "false alarm", live[name]["false_alarm"], live[name]["machine"])
    summary["live_analog"] = live
    dump()

    # samples: the same prompts with the flaky machine answering (seeded, so the three models meet the same answers)
    PROMPTS = ["tom had 12 apples . he got 5 more . now he has", "mia had 40 cookies and ate 13 . she had", "there were 6 bags with 8 toys each ,",
               "lily had 20 stickers . she got 7 more and then 4 more . now she has", "sam picked 30 grapes . then mia gave him 12 more . sam had",
               "ben had 50 grapes . he ate 12 and then got 30 more . he had", "they hugged each other and ate more cake .", "once upon a time"]
    samples = {}
    for name, job in (("checks numbers + signal", best["checks numbers + signal"]), ("checks numbers", best["checks numbers"]), ("trusts the machine", BASE)):
        m = load(job); samples[name] = []
        for i, p in enumerate(PROMPTS):
            mach = machine(p_err=0.5, signal=("signal" in name), seed=40 + i)
            ids, calls = generate(m, p, mach, seed=i)
            samples[name].append({"prompt": p, "text": V.decode(ids), "calls": [{k: c.get(k) for k in ("a", "op", "b", "r", "wrong", "unsure", "err")} for c in calls]})
    summary["samples"] = samples
    dump()

    # figures
    INK2, MUTED, GRID, BASEC = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASEC, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2})
    col = {"checks numbers": "#2a78d6", "checks numbers + signal": "#d6632a"}
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    for cname, seed in RUNS:
        c = summary["runs"][key((cname, seed))]["curve"]
        if not c:
            continue
        st = [r["step"] for r in c]; kw = dict(color=col[cname], alpha=1 if seed == 0 else 0.5, label=f"{cname}, seed {seed}")
        axs[0].plot(st, [r["word_flaky"] for r in c], **kw); axs[1].plot(st, [r["detection"] or 0 for r in c], **kw)
        axs[2].plot(st, [r["false_alarm"] or 0 for r in c], **kw); axs[3].plot(st, [r["word_perfect"] for r in c], **kw)
    base_flaky = summary["runs"][key(BASE)]["sets"]["flaky"]["word"]["all"]["answer_right"]
    axs[0].axhline(base_flaky, color=BASEC, ls="--", label="trusts the machine (shapes run)")
    for a_, yl, ttl in zip(axs, ("answer right", "share of wrong answers retried", "share of right answers retried", "answer right"),
                           ("Word problems, flaky machine (25% wrong)", "Detection: a wrong answer is retried", "False alarms: a right answer is retried", "Word problems, perfect machine")):
        a_.set_xlabel("training step"); a_.set_ylabel(yl); a_.set_title(ttl, fontsize=9, loc="left"); a_.set_ylim(-0.03, 1.05)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(OUT, "figures", "1_learning.png"), dpi=150); plt.close(fig)

    labels = ["op", "random", "zero"] + [f"bit{k}" for k in range(8)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(labels)); w = 0.26
    for i, (name, job, mname) in enumerate((("checks numbers", best["checks numbers"], "flaky"), ("checks numbers + signal (signal on)", best["checks numbers + signal"], "flaky + signal"),
                                            ("trusts the machine", BASE, "flaky"))):
        be = summary["runs"][key(job)]["sets"][mname]["word"]["by_err"]
        ax.bar(x + (i - 1) * w, [be[l]["rate"] or 0 for l in labels], w, color=[col["checks numbers"], col["checks numbers + signal"], BASEC][i], label=name)
    ax.set_xticks(x, labels=["another operation", "random byte", "0 (timeout)"] + [f"bit {k} flipped (+-{1 << k})" for k in range(8)], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of wrong answers retried"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Which wrong answers the model catches (word problems, held-out numbers)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "2_what_is_caught.png"), dpi=150); plt.close(fig)
    dump()
    log("done")
