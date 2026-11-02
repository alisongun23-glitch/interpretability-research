"""Addendum to clean_carry.py: train the narrow adders long enough to learn, then rerun the squeeze.

    python clean_carry_long.py prereg-check    # locks results_clean_carry_long/PREREG.md; must pass first
    python clean_carry_long.py pilot           # held-out seeds 100, 101 at widths 16 and 4 -> pilot.json (sets S)
    python clean_carry_long.py train           # arm L: widths 256, 32, 16, 8, 4 x seeds 0-4, S steps each
    python clean_carry_long.py analyse         # every checkpoint: labels, gate labels, carry direction; V2g
    python clean_carry_long.py report          # V1L-V4g, then H3, H3g, D2, D3 -> results.json, VERDICTS.md, figures/
    python clean_carry_long.py explore         # measures chosen after the verdicts -> exploratory.json (never judged)

Arm L is clean_carry.py's arm A (same model, init and data stream) with the cosine learning-rate decay
stretched over S steps. New here: the gate label, a label that only accepts units that are silent when off,
so that cleanliness becomes a property of single neurons (a rotation of the units is never silent).
Every rule is in results_clean_carry_long/PREREG.md. Changes nothing in clean_carry.py or its results.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
import numpy as np

import run_transformer as RT
import transformer_gates as TG
import clean_carry as CC

OUT = "results_clean_carry_long"
MODELS, AN, FIG = (os.path.join(OUT, d) for d in ("models", "analysis", "figures"))
PREREG, LOCK, PILOT = (os.path.join(OUT, f) for f in ("PREREG.md", "prereg_lock.json", "pilot.json"))
NB, KS, BATCH = CC.NB, CC.KS, 128
WIDTHS, SEEDS = [256, 32, 16, 8, 4], list(range(5))
PILOT_RUNS, PILOT_STEPS = [(16, 100), (16, 101), (4, 100), (4, 101)], 40000


# ======================= pre-registration lock =======================
def prereg_check(lock=False):
    if not os.path.exists(PREREG):
        raise SystemExit(f"{PREREG} is missing")
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if not os.path.exists(LOCK):
        if not lock:
            raise SystemExit("addendum PREREG not locked yet: run  python clean_carry_long.py prereg-check")
        json.dump({"sha256": h, "locked": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(LOCK, "w"), indent=1)
        print("locked PREREG.md", h)
        return h
    L = json.load(open(LOCK))
    if L["sha256"] != h:
        raise SystemExit("addendum PREREG.md changed after it was locked: refusing to run")
    print("PREREG.md unchanged since", L["locked"], h)
    return h


# ======================= schedule, training =======================
def lr_at(step, horizon):
    """The first study's learning rate with the cosine decay stretched over `horizon` steps."""
    return 1e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / horizon)))


def schedule_long(S):
    steps = {0, S} | set(range(25, 1001, 25)) | set(range(1250, min(5000, S) + 1, 250)) | set(range(6000, S + 1, 1000))
    return sorted(s for s in steps if s <= S)


def get_S():
    if not os.path.exists(PILOT):
        raise SystemExit("pilot.json missing: run  python clean_carry_long.py pilot")
    return int(json.load(open(PILOT))["S"])


def ckpt(w, s, step):
    return os.path.join(MODELS, f"L-w{w}-s{s}", f"step{step:05d}.npz")


def load_ckpt(w, s, step):
    m = CC.make(w, s); z = np.load(ckpt(w, s, step), allow_pickle=True)
    m.load({k: z[k] for k in m.P})
    return m


def train_long(w, s, steps, save_steps=(), save_fn=None, eval_steps=(), tag=""):
    """clean_carry.train_width's loop (same init, data stream and optimizer) with the schedule over `steps`."""
    model = CC.make(w, s); r = np.random.default_rng([s, 1000])
    mask = np.zeros((BATCH, RT.T), bool); mask[:, [TG.Flat.q(k) for k in range(NB + 1)]] = True
    save_steps, eval_steps = set(save_steps), set(eval_steps)
    log, evals, t0 = [], [], time.time()
    if 0 in save_steps:
        save_fn(model, 0, log)
    for step in range(steps):
        a, b = TG.problems(BATCH, NB, r)
        loss, Gr = model.loss_and_grads(TG.Flat.tokens(a, b), mask)
        model.adam_step(Gr, lr_at(step, steps), wd=0.01)
        if step % 250 == 0:
            log.append((step, loss))
            print(f"{tag} step {step} loss {loss:.5f}  {time.time() - t0:.0f}s", flush=True)
        if step + 1 in save_steps:
            save_fn(model, step + 1, log)
        if step + 1 in eval_steps:
            evals.append((step + 1, CC.exact_acc(model)))
            print(f"{tag} step {step + 1} exact {evals[-1][1]:.4f}", flush=True)
    return model, evals


def save_main(w, s):
    def fn(model, step, log):
        os.makedirs(os.path.dirname(ckpt(w, s, step)), exist_ok=True)
        np.savez(ckpt(w, s, step), step=step, log=np.array(log if log else [(0, np.nan)]), **model.P)
    return fn


def pilot_run(w, s):
    p = os.path.join(OUT, "pilot", f"w{w}-s{s}.json")
    if os.path.exists(p):
        return
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _, evals = train_long(w, s, PILOT_STEPS, eval_steps=range(1000, PILOT_STEPS + 1, 1000), tag=f"pilot w{w} s{s}")
    json.dump({"width": w, "seed": s, "evals": evals}, open(p, "w"))


def pilot_S():
    runs, Ls = {}, []
    for w, s in PILOT_RUNS:
        r = json.load(open(os.path.join(OUT, "pilot", f"w{w}-s{s}.json")))
        L_i = next((st for st, acc in r["evals"] if acc >= 0.99), None)
        runs[f"w{w}-s{s}"] = {"learned_at": L_i, "final_exact": r["evals"][-1][1]}
        if L_i is not None:
            Ls.append(L_i)
    S = min(40000, max(10000, math.ceil(2 * max(Ls) / 5000) * 5000)) if Ls else 40000
    out = {"runs": runs, "S": S, "rule": "S = min(40000, max(10000, ceil(2 max L / 5000) * 5000)); 40000 if no pilot run learns"}
    json.dump(out, open(PILOT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return S


def pool(args_list, sub):
    """Run `python clean_carry_long.py <sub> --only <arg>` for each arg, min(7, cores - 1) at once."""
    cores = os.cpu_count() or 2
    n_par = max(1, min(7, cores - 1))
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    print(f"{cores} cores -> {n_par} processes; {len(args_list)} jobs", flush=True)
    todo, running = list(args_list), []
    while todo or running:
        while todo and len(running) < n_par:
            arg = todo.pop(0)
            logf = open(os.path.join(OUT, "logs", f"{sub}-{arg.replace(':', '-')}.log"), "a")
            running.append((arg, subprocess.Popen([sys.executable, __file__, sub, "--only", arg], stdout=logf, stderr=subprocess.STDOUT)))
        time.sleep(3)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                print(f"{sub} {item[0]} finished (exit {item[1].returncode})", flush=True)


# ======================= the gate label =======================
def gate_label_k(Hk, combo, min_active=0.05):
    """Gate labels of every unit at one position: -1 inactive, 0 not a gate, otherwise clean_carry's Boolean or
    count-threshold code. A gate is a sharp step (best Boolean R^2 >= 0.8) that is silent when off:
    |mean where f = 0| <= 0.05 * mean where f = 1 (> 0)."""
    U = Hk.shape[1]; codes = np.full(U, -1, int)
    act = ((Hk > 0).mean(0) >= min_active) & (Hk.std(0) > 1e-6)
    if not act.any():
        return codes
    Ha = np.asarray(Hk[:, act], float)
    R2, masks = TG.bool_fit(Ha, combo, 3)
    best, fi = R2.max(0), R2.argmax(0)
    out = np.zeros(Ha.shape[1], int)
    for j in np.where(best >= 0.8)[0]:
        m = masks[fi[j]].astype(int); on = m[combo] == 1
        mon, moff = Ha[on, j].mean(), Ha[~on, j].mean()
        if mon < moff:
            m, mon, moff = 1 - m, moff, mon
        if mon > 0 and abs(moff) <= 0.05 * mon:
            f = CC.fval(m); out[j] = CC.THR_CODE.get(f, f)
    codes[act] = out
    return codes


def gate_label_all(H, combo):
    return np.stack([gate_label_k(H[:, k], combo[:, k]) for k in KS], 1)


def rotation_both(H, combo, count, seed, draws=10):
    """V4 / V4g: cleanliness and gate-cleanliness after mixing the units by random orthogonal matrices
    (the first study's rotations: rng 1100 + seed)."""
    U = H.shape[2]; rng = np.random.default_rng(1100 + seed); c, g = [], []
    for _ in range(draws):
        Q, _ = np.linalg.qr(rng.normal(size=(U, U)))
        Hr = np.einsum("nku,uv->nkv", H.astype(float), Q)
        for codes, acc in ((np.stack([CC.label_k(Hr[:, k], combo[:, k], count[:, k], "A") for k in KS], 1), c),
                           (gate_label_all(Hr, combo), g)):
            v = CC.verdicts(codes, "A"); alive = v != "dead"
            acc.append(float(((v == "one") | (v == "two")).sum() / max(alive.sum(), 1)))
    return float(np.mean(c)), float(np.mean(g))


# ======================= analysis =======================
def analyse_long(w, s):
    path = os.path.join(AN, f"L-w{w}-s{s}.json")
    if os.path.exists(path):
        return
    S = get_S(); D = CC.data(); a, b, L = D["a_te"], D["b_te"], D["Lte"]
    combo, count = CC.inputs("A", a, b, L)
    res = {"width": w, "seed": s, "S": S, "checkpoints": []}; T0 = time.time()
    for step in schedule_long(S):
        model = load_ckpt(w, s, step); final = step == S
        H, X, _ = CC.hidden(model, TG.Flat, a, b)
        codes, gcodes = CC.label_all(H, combo, count, "A"), gate_label_all(H, combo)
        perms = 1000 if final else 200
        row = {"step": step, "exact_acc": CC.exact_acc(model),
               "labels": CC.summarize(codes, "A", perms, seed=s), "gate": CC.summarize(gcodes, "A", perms, seed=s),
               "codes": codes.tolist(), "gate_codes": gcodes.tolist()}
        U, V = CC.directions(model, s, 2000 if final else 500)
        row["interchange"] = CC.interchange(model, V, np.random.default_rng(600 + s), 1000 if final else 300)
        R = CC.read_cos(model, V); row["read_cos"] = np.round(R, 4).tolist()
        row["readers"] = CC.readers(codes, R, model, V, s, with_clamp=False)
        if final:
            row["d5"] = CC.d5(V); row["debiased_margin"] = CC.debiased_margin(U, X, L)
            if row["exact_acc"] >= 0.99:
                row["readers_clamp"] = CC.readers(codes, R, model, V, s, with_clamp=True)
                Htr, _, _ = CC.hidden(model, TG.Flat, D["a_tr"], D["b_tr"])
                _, count_tr = CC.inputs("A", D["a_tr"], D["b_tr"], D["Ltr"])
                row["snaps"] = CC.snaps(model, Htr, count_tr)
            if w == 256:
                row["rotation_cleanliness"], row["rotation_gate_cleanliness"] = rotation_both(H, combo, count, s)
        res["checkpoints"].append(row)
        print(f"L-w{w}-s{s} step {step}: exact {row['exact_acc']:.3f} clean {row['labels']['cleanliness']:.2f} "
              f"gate {row['gate']['cleanliness']:.2f} gpacked {row['gate']['packed_share_multi']} ic {row['interchange']:.3f}"
              f"  [{time.time() - T0:.0f}s]", flush=True)
    os.makedirs(AN, exist_ok=True)
    json.dump(res, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)


def analyse_v2g():
    """V2g: the first study's arm-R models at widths <= 16 that learned, relabelled with gate labels."""
    path = os.path.join(AN, "v2g.json")
    if os.path.exists(path):
        return
    D = CC.data(); a, b, L = D["a_te"], D["b_te"], D["Lte"]
    combo, _ = CC.inputs("R", a, b, L)
    out = {}
    for w in (16, 8):
        for s in CC.SEEDS:
            r = CC.load_an("R", w, s)
            if r is None or not CC.learns("R", CC.final_row(r)):
                continue
            model = CC.load_ckpt("R", w, s, CC.STEPS)
            H, _, _ = CC.hidden(model, CC.FlatTable, a, b)
            out[f"{w}:{s}"] = CC.summarize(gate_label_all(H, combo), "R", 1000, seed=s)
            print("V2g", w, s, out[f"{w}:{s}"]["packed_share_multi"], out[f"{w}:{s}"]["cleanliness"], flush=True)
    os.makedirs(AN, exist_ok=True)
    json.dump(out, open(path, "w"))


# ======================= verdicts =======================
def judge_long():
    S = get_S()
    M = {(w, s): json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json"))) for w in WIDTHS for s in SEEDS}
    fin = {k: r["checkpoints"][-1] for k, r in M.items()}
    learn = {k: f["exact_acc"] >= 0.99 for k, f in fin.items()}
    lw = [w for w in WIDTHS if sum(learn[(w, s)] for s in SEEDS) >= 3]
    first99 = {f"{w}:{s}": next((c["step"] for c in M[(w, s)]["checkpoints"] if c["exact_acc"] >= 0.99), None) for w in WIDTHS for s in SEEDS}
    V, H, Dv, X = {}, {}, {}, {}
    w256 = [fin[(256, s)] for s in SEEDS]
    V["V1L"] = {"pass": all(learn[(256, s)] for s in SEEDS) and sum(f["interchange"] >= 0.95 for f in w256) >= 4,
                "interchange": [f["interchange"] for f in w256]}
    first = json.load(open(os.path.join(CC.OUT, "results.json")))
    V["V2"] = {"pass": first["validity"]["V2"]["pass"], "source": "results_clean_carry/results.json"}
    v2g = json.load(open(os.path.join(AN, "v2g.json")))
    shares = [x["packed_share_multi"] for x in v2g.values()]
    V["V2g"] = {"pass": bool(shares) and sum(p is not None and p >= 0.30 for p in shares) >= len(shares) / 2,
                "packed_share_multi": shares, "models": list(v2g)}
    z = [M[(256, s)]["checkpoints"][0] for s in SEEDS]
    c0 = float(np.mean([r["labels"]["cleanliness_no_dictators"] for r in z])); g0 = float(np.mean([r["gate"]["cleanliness"] for r in z]))
    V["V3"] = {"pass_labels": c0 <= 0.10, "pass_gate": g0 <= 0.10, "init_cleanliness_no_dictators": c0, "init_gate_cleanliness": g0}
    cl, cr = float(np.mean([f["labels"]["cleanliness"] for f in w256])), float(np.mean([f["rotation_cleanliness"] for f in w256]))
    gl, gr = float(np.mean([f["gate"]["cleanliness"] for f in w256])), float(np.mean([f["rotation_gate_cleanliness"] for f in w256]))
    V["V4"] = {"pass": cr <= 0.5 * cl, "cleanliness": cl, "rotated": cr}
    V["V4g"] = {"pass": gr <= 0.5 * gl, "gate_cleanliness": gl, "rotated": gr}

    # ---- H3 (unchanged rule)
    def per_width(key):
        out = {}
        for w in WIDTHS:
            fr = [fin[(w, s)] for s in SEEDS if learn[(w, s)]]
            out[w] = {"learning_seeds": len(fr),
                      "cleanliness": float(np.mean([f[key]["cleanliness"] for f in fr])) if fr else None,
                      "delta": float(np.nanmean([f[key]["delta"] for f in fr])) if fr and not all(np.isnan(f[key]["delta"]) for f in fr) else None,
                      "positions_per_unit": float(np.mean([f[key]["positions_per_unit"] for f in fr])) if fr else None,
                      "packed_share_multi": [f[key]["packed_share_multi"] for f in fr],
                      "deltas": [f[key]["delta"] for f in fr],
                      "shared_thresholds": sorted({t for f in fr for t in f[key]["shared_thresholds"]})}
        return out
    pw = per_width("labels")
    rho = CC.spearman(lw, [pw[w]["positions_per_unit"] for w in lw]) if len(lw) >= 3 else None
    passh3 = (bool(lw) and all(pw[w]["cleanliness"] >= 0.70 and (pw[w]["delta"] or -1) >= 0.20 for w in lw)
              and (rho is None or rho <= -0.8) and set(pw[min(lw)]["shared_thresholds"]) >= {1, 2, 3})
    failh3 = V["V2"]["pass"] and any(sum((p or 0) >= 0.30 and d < 0.05 for p, d in zip(pw[w]["packed_share_multi"], pw[w]["deltas"])) >= 3
                                     for w in lw if w <= 16)
    h3 = "fail" if failh3 else ("pass" if passh3 else "inconclusive")
    if not (V["V3"]["pass_labels"] and V["V1L"]["pass"] and V["V4"]["pass"]):
        h3 = "no verdict (gate failed: " + ", ".join(g for g, ok in (("V1L", V["V1L"]["pass"]), ("V3", V["V3"]["pass_labels"]), ("V4", V["V4"]["pass"])) if not ok) + ")"
    H["H3"] = {"verdict": h3, "learning_widths": lw, "rho_width_positions": rho, "per_width": {str(w): v for w, v in pw.items()}}

    # ---- H3g (gate labels)
    pg = per_width("gate")
    narrow = [w for w in lw if w <= 32]
    rho_g = CC.spearman(lw, [pg[w]["positions_per_unit"] for w in lw]) if len(lw) >= 3 else None
    if not narrow:
        h3g = "no verdict (no width <= 32 learns)"
    else:
        ok_pack = all(sum((p is None) or p <= 0.10 for p in pg[w]["packed_share_multi"]) >= min(4, pg[w]["learning_seeds"]) and
                      pg[w]["learning_seeds"] >= 4 for w in narrow)
        ok_delta = all(pg[w]["delta"] is not None and pg[w]["delta"] >= 0.20 for w in narrow)
        ok_share = len(set(pg[min(lw)]["shared_thresholds"])) >= 2
        fail = V["V2g"]["pass"] and any(sum((p or 0) >= 0.30 and (d < 0.05 if d == d else False)
                                            for p, d in zip(pg[w]["packed_share_multi"], pg[w]["deltas"])) >= 3 for w in lw if w <= 16)
        h3g = "fail" if fail else ("pass" if ok_pack and ok_delta and ok_share else "inconclusive")
        if not (V["V3"]["pass_gate"] and V["V1L"]["pass"] and V["V4g"]["pass"]):
            h3g = "no verdict (gate failed: " + ", ".join(g for g, ok in (("V1L", V["V1L"]["pass"]), ("V3", V["V3"]["pass_gate"]), ("V4g", V["V4g"]["pass"])) if not ok) + ")"
    H["H3g"] = {"verdict": h3g, "learning_widths": lw, "rho_width_positions_gate": rho_g, "per_width": {str(w): v for w, v in pg.items()}}

    # ---- D2 (unchanged rule; widths 256 and 16 judged, 32, 8, 4 exploratory)
    d2 = {}
    for w in WIDTHS:
        d2[w] = {}
        for s in SEEDS:
            cps = M[(w, s)]["checkpoints"]; fin_n = cps[-1]["labels"]["clean_threshold_pairs"]
            t_dir = next((c["step"] for c in cps if c["interchange"] >= 0.8), None)
            t_cl = next((c["step"] for c in cps if c["labels"]["clean_threshold_pairs"] >= 0.5 * fin_n), None) if fin_n >= 20 else None
            C = np.array([c["codes"] for c in cps]); Rc = np.array([c["read_cos"] for c in cps]); st = [c["step"] for c in cps]
            on_l, on_r = [], []
            for u, i in zip(*np.where(C[-1] >= 1000)):
                tl = next((t for t in range(len(cps)) if C[t, u, i] == C[-1][u, i]), None)
                tr_ = next((t for t in range(len(cps)) if abs(Rc[t, u, i]) > 0.2), None)
                if tl is not None and tr_ is not None:
                    on_l.append(st[tl]); on_r.append(st[tr_])
            d2[w][s] = {"t_dir": t_dir, "t_clean": t_cl, "final_clean_pairs": fin_n, "first_99": first99[f"{w}:{s}"],
                        "unit_onset_rho": CC.spearman(on_l, on_r), "unit_pairs": len(on_l)}

    def order(w):
        u = [x for x in d2[w].values() if x["t_dir"] is not None and x["t_clean"] is not None]
        return None if len(u) < 3 else (sum(x["t_dir"] <= x["t_clean"] for x in u), sum(x["t_dir"] > x["t_clean"] for x in u), len(u))
    o = {w: order(w) for w in WIDTHS}
    if any(o[w] is None for w in (256, 16)):
        d2v = "no verdict (fewer than 3 usable seeds at a width)"
    elif all(o[w][0] >= 4 for w in (256, 16)):
        d2v = "pass (writer before readers)"
    elif all(o[w][1] >= 4 for w in (256, 16)):
        d2v = "fail (readers first)"
    else:
        d2v = "mixed"
    Dv["D2"] = {"verdict": d2v if V["V1L"]["pass"] else "no verdict (gate failed: V1L)",
                "per_width": {str(w): {str(s): d2[w][s] for s in SEEDS} for w in WIDTHS},
                "counts_dir_first_clean_first_n": {str(w): o[w] for w in WIDTHS}}

    # ---- D3 (unchanged rules)
    pts = [(w, s, fin[(w, s)]) for w in WIDTHS for s in SEEDS if learn[(w, s)]]
    low = {w: sum(1 for ww, _, f in pts if ww == w and f["interchange"] < 0.8) for w in WIDTHS}
    rho3 = CC.spearman([f["labels"]["cleanliness"] for _, _, f in pts], [f["interchange"] for _, _, f in pts])
    rho3g = CC.spearman([f["gate"]["cleanliness"] for _, _, f in pts], [f["interchange"] for _, _, f in pts])
    gate1 = V["V1L"]["pass"]
    Dv["D3a"] = {"verdict": ("pass" if all(v < 2 for v in low.values()) else "fail") if gate1 else "no verdict (gate failed: V1L)",
                 "seeds_below_0.8_per_width": {str(w): v for w, v in low.items()}, "interchange": {f"{w}:{s}": f["interchange"] for w, s, f in pts}}
    Dv["D3b"] = {"verdict": (("independent" if rho3 is not None and abs(rho3) < 0.3 else "go together" if rho3 is not None and rho3 >= 0.3
                              else "trade off" if rho3 is not None else "no verdict") if gate1 else "no verdict (gate failed: V1L)"),
                 "rho": rho3, "n": len(pts), "rho_gate_cleanliness_exploratory": rho3g}

    # ---- exploratory
    X["first_step_99"] = first99
    X["final"] = {f"{w}:{s}": {"exact_acc": f["exact_acc"], "interchange": f["interchange"],
                               "cleanliness": f["labels"]["cleanliness"], "gate_cleanliness": f["gate"]["cleanliness"],
                               "packed_share_multi": f["labels"]["packed_share_multi"], "gate_packed_share_multi": f["gate"]["packed_share_multi"],
                               "gate_delta": f["gate"]["delta"], "positions_per_unit": f["labels"]["positions_per_unit"],
                               "gate_positions_per_unit": f["gate"]["positions_per_unit"], "gate_shared_thresholds": f["gate"]["shared_thresholds"],
                               "readers_clamp": f.get("readers_clamp"), "snaps": f.get("snaps"),
                               "debiased_margin": f["debiased_margin"]} for (w, s), f in fin.items()}
    X["v2g"] = v2g
    return {"S": S, "validity": V, "cammarata": H, "direction": Dv, "exploratory": X}


def figures(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    pil = json.load(open(PILOT))
    fig, ax = plt.subplots(figsize=(6, 4))
    for w, s in PILOT_RUNS:
        r = json.load(open(os.path.join(OUT, "pilot", f"w{w}-s{s}.json")))
        ax.plot([e[0] for e in r["evals"]], [e[1] for e in r["evals"]], label=f"width {w}, seed {s}")
    ax.axhline(0.99, ls="--", color="grey"); ax.axvline(pil["S"], ls=":", color="k")
    ax.set_xlabel("step"); ax.set_ylabel("exact sums"); ax.set_title(f"pilot (S = {pil['S']})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_pilot.png"), dpi=130); plt.close(fig)

    fins = {}
    fig, axs = plt.subplots(3, len(WIDTHS), figsize=(4 * len(WIDTHS), 9), sharex=True)
    for j, w in enumerate(WIDTHS):
        for s in SEEDS:
            r = json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json"))); cps = r["checkpoints"]; fins[(w, s)] = cps[-1]
            st = [c["step"] for c in cps]
            axs[0, j].plot(st, [c["exact_acc"] for c in cps], color=f"C{s}", label=f"seed {s}")
            axs[1, j].plot(st, [c["interchange"] for c in cps], color=f"C{s}")
            axs[2, j].plot(st, [c["labels"]["clean_threshold_pairs"] for c in cps], color=f"C{s}")
        axs[0, j].set_title(f"width {w}")
        for i, lab in enumerate(("exact sums", "carry-direction interchange", "clean threshold pairs")):
            axs[i, j].set_xscale("symlog", linthresh=100); axs[i, 0].set_ylabel(lab)
        axs[2, j].set_xlabel("step")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_emergence_by_width.png"), dpi=120); plt.close(fig)

    # one dot per model (hollow = did not learn), so learners and non-learners are never averaged together
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    F = R["exploratory"]["final"]; lx = {w: np.log2(w) for w in WIDTHS}
    for ax, keys in ((axs[0], (("cleanliness", "cleanliness (first study's labels)"), ("gate_cleanliness", "gate-cleanliness"))),
                     (axs[2], (("exact_acc", "exact sums"), ("interchange", "carry-direction interchange")))):
        for (key, lab), dx, col, mk in zip(keys, (-0.17, 0.17), ("C0", "C1"), ("o", "s")):
            for i, s in enumerate(SEEDS):
                for w in WIDTHS:
                    f = F[f"{w}:{s}"]
                    ax.scatter(lx[w] + dx + 0.05 * (i - 2), f[key], s=26, marker=mk, edgecolors=col, linewidths=1.2,
                               facecolors=col if f["exact_acc"] >= 0.99 else "none")
            ax.scatter([], [], marker=mk, color=col, label=lab)
        ax.scatter([], [], marker="o", facecolors="none", edgecolors="grey", label="hollow: did not learn")
    kinds = (("one", "one job", "C2"), ("two", "two related jobs", "C8"), ("packed", "packed", "C3"), ("unclear", "unclear", "0.7"))
    for key, hatch, dx in (("labels", None, -0.17), ("gate", "///", 0.17)):
        for w in WIDTHS:
            fl = [fins[(w, s)][key] for s in SEEDS if fins[(w, s)]["exact_acc"] >= 0.99]
            if key == "labels":
                axs[1].text(lx[w], 1.03, f"{len(fl)}/5 learn", ha="center", fontsize=8)
            if not fl:
                continue
            alive, bottom = max(sum(f["units_alive"] for f in fl), 1), 0.0
            for kind, name, col in kinds:
                v = sum(f[kind] for f in fl) / alive
                axs[1].bar(lx[w] + dx, v, 0.3, bottom=bottom, color=col, hatch=hatch, edgecolor="white", linewidth=0.5,
                           label=name if (w == WIDTHS[0] and key == "labels") else None)
                bottom += v
    axs[1].bar([0], [0], 0, color="white", edgecolor="grey", hatch="///", label="right bar: gate labels")
    axs[1].set_ylim(0, 1.1)
    for ax, t, yl, loc in zip(axs, ("cleanliness (one dot per model)", "unit verdicts, models that learn", "learning and the carry direction"),
                              ("share of alive units", "share of alive units", ""), ("upper left", "center left", "lower right")):
        ax.set_xticks([lx[w] for w in WIDTHS]); ax.set_xticklabels([str(w) for w in WIDTHS])
        ax.set_xlabel("MLP width"); ax.set_title(t); ax.set_ylabel(yl); ax.legend(fontsize=7, loc=loc)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_squeeze.png"), dpi=130); plt.close(fig)


def report():
    h = prereg_check()
    R = judge_long(); R["prereg_sha256"] = h; R["prereg_locked"] = json.load(open(LOCK))["locked"]
    R["pilot"] = json.load(open(PILOT))
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    figures(R)
    L = ["# Addendum: train the narrow adders long enough to squeeze: verdicts", "",
         f"Generated by `python clean_carry_long.py report` from `results.json` (PREREG sha256 `{h[:16]}…`, locked {R['prereg_locked']}). S = {R['S']}.", "",
         "## Validity gates", "", "| gate | pass | detail |", "|---|---|---|"]
    for k, v in R["validity"].items():
        p = v.get("pass", (v.get("pass_labels"), v.get("pass_gate")))
        L.append(f"| {k} | {p} | " + ", ".join(f"{a}={b}" for a, b in v.items() if not a.startswith("pass")) + " |")
    L += ["", "## Verdicts", "", "| # | verdict |", "|---|---|"]
    for grp in ("cammarata", "direction"):
        for k, v in R[grp].items():
            L.append(f"| {k} | {v['verdict']} |")
    open(os.path.join(OUT, "VERDICTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ======================= exploratory (defined after the verdicts, never judged) =======================
def explore():
    S = get_S(); X = {"note": "defined after the verdicts were read; never judged"}
    M = {(w, s): json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json")))["checkpoints"] for w in WIDTHS for s in SEEDS}
    # 1. how many active unit-positions carry a label at step S (one unlabelled position makes a unit "unclear")
    X["label_shares"] = {}
    for (w, s), cps in M.items():
        f = cps[-1]; C, G = np.array(f["codes"]), np.array(f["gate_codes"]); act = C >= 0; n = max(int(act.sum()), 1)
        per_unit = [float((row[row >= 0] != 0).mean()) for row in C if (row >= 0).any()]
        X["label_shares"][f"{w}:{s}"] = {
            "learns": f["exact_acc"] >= 0.99, "exact_acc": f["exact_acc"], "active_unit_positions": int(act.sum()),
            "any_label": float((act & (C != 0)).sum() / n), "count_label": float((C >= 1000).sum() / n),
            "other_boolean": float((act & (C > 0) & (C < 1000)).sum() / n), "gate_label": float((G > 0).sum() / n),
            "units_labelled_at_half_their_positions": float(np.mean(np.array(per_unit) >= 0.5)) if per_unit else None}
    # 2. the interchange each model shows in its first 100 steps, before any model can add
    X["early_interchange_max_steps_0_100"] = {f"{w}:{s}": max(c["interchange"] for c in cps if c["step"] <= 100)
                                              for (w, s), cps in M.items()}
    # 3. late learning curves at widths <= 32: still rising, or on a plateau?
    X["exact_acc_late"] = {f"{w}:{s}": {str(c["step"]): c["exact_acc"] for c in cps if c["step"] in (20000, 30000, S)}
                           for (w, s), cps in M.items() if w <= 32}
    # 4. width 256 here and in the first study (arm A): same init and data order, learning rates a few % apart
    A = {s: json.load(open(os.path.join(CC.AN, f"A-w256-s{s}.json")))["checkpoints"] for s in SEEDS}
    X["schedules_w256"] = {"max_relative_lr_gap_steps_0_500": max(abs(lr_at(t, 2500) - lr_at(t, S)) / lr_at(t, S) for t in range(501)),
                           "seeds": {}}
    for s in SEEDS:
        P0 = CC.make(256, s).P; rows = []
        for st in (25, 50, 100, 150, 200, 250, 300, 400, 500, 1000, 2500):
            zA, zL = np.load(CC.ckpt("A", 256, s, st)), np.load(ckpt(256, s, st))
            gap = np.sqrt(sum(((zA[k].astype(float) - zL[k]) ** 2).sum() for k in P0))
            moved = np.sqrt(sum(((zA[k].astype(float) - P0[k]) ** 2).sum() for k in P0))
            cA = next(c for c in A[s] if c["step"] == st); cL = next(c for c in M[(256, s)] if c["step"] == st)
            rows.append({"step": st, "param_gap_over_distance_moved": float(gap / max(moved, 1e-12)),
                         "cleanliness_A": cA["labels"]["cleanliness"], "cleanliness_L": cL["labels"]["cleanliness"],
                         "interchange_A": cA["interchange"], "interchange_L": cL["interchange"]})
        X["schedules_w256"]["seeds"][str(s)] = rows
        print(f"w256 s{s}: " + " ".join(f"{r['step']}:{r['param_gap_over_distance_moved']:.2f}/{r['cleanliness_A']:.2f}/{r['cleanliness_L']:.2f}"
                                        for r in rows), flush=True)
    json.dump(X, open(os.path.join(OUT, "exploratory.json"), "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    unit_functions()
    token_path()


# ---- what the unlabelled units compute (exploratory, added after the verdicts)
def _halves(N, rng):
    perm = rng.permutation(N); h = (perm[: N // 2], perm[N // 2:])
    return (h, h[::-1])


def _cv_table_r2(x, g, m, rng):
    """Two-fold cross-validated R² of the best lookup table over the groups g (N,) in 0..m-1, per column of x (N, U):
    the group means of one half of the problems score the other half. 1.0 means the unit is a function of those
    bits; the gap to 1.0 is what other bits contribute."""
    out = []
    for tr, te in _halves(len(x), rng):
        G = np.zeros((len(tr), m)); G[np.arange(len(tr)), g[tr]] = 1
        cnt = G.sum(0); means = np.where(cnt[:, None] > 0, (G.T @ x[tr]) / np.maximum(cnt, 1)[:, None], x[tr].mean(0))
        res = ((x[te] - means[g[te]]) ** 2).sum(0); tot = ((x[te] - x[te].mean(0)) ** 2).sum(0)
        out.append(1 - res / np.maximum(tot, 1e-12))
    return np.mean(out, 0)


def _cv_linear_r2(x, B, rng):
    """Two-fold cross-validated R² of a linear fit of x (N, U) on the bits B (N, p) plus an intercept."""
    Z = np.hstack([np.asarray(B, float), np.ones((len(x), 1))]); out = []
    for tr, te in _halves(len(x), rng):
        W = np.linalg.lstsq(Z[tr], x[tr], rcond=None)[0]
        res = ((x[te] - Z[te] @ W) ** 2).sum(0); tot = ((x[te] - x[te].mean(0)) ** 2).sum(0)
        out.append(1 - res / np.maximum(tot, 1e-12))
    return np.mean(out, 0)


def _groups(bits):
    g = np.zeros(len(bits[0]), int)
    for b in bits:
        g = 2 * g + np.asarray(b, int)
    return g, 2 ** len(bits)


UNIT_SETS = ("token", "back1", "ahead1", "back1c", "back2c", "window")   # smallest first; the order the category picks


def _bit_sets(L, k):
    """Bit sets around position k. local = the labeller's three inputs a_k, b_k, c_k; token = local plus s_(k-1), the
    answer bit already written, which is the token at q_k itself."""
    def bit(name, j):
        return None if not 0 <= j <= NB else L[name][:, j]
    local = [bit("a", k), bit("b", k), bit("c", k)]
    sets = {"local": local, "token": local + [bit("s", k - 1)],
            "back1": local + [bit("a", k - 1), bit("b", k - 1)], "ahead1": local + [bit("a", k + 1), bit("b", k + 1)],
            "back1c": local + [bit("a", k - 1), bit("b", k - 1), bit("c", k - 1)],
            "back2c": local + [bit("a", k - 1), bit("b", k - 1), bit("c", k - 1), bit("a", k - 2), bit("b", k - 2)],
            "window": local + [bit("a", k - 1), bit("b", k - 1), bit("c", k - 1), bit("a", k - 2), bit("b", k - 2), bit("a", k + 1), bit("b", k + 1)]}
    sets = {n: [x for x in v if x is not None] for n, v in sets.items()}
    singles = {n: local + [bit(n[0], k + d)] for n, d in (("a_k-1", -1), ("b_k-1", -1), ("c_k-1", -1), ("s_k-1", -1), ("a_k-2", -2), ("b_k-2", -2),
                                                           ("a_k+1", 1), ("b_k+1", 1)) if bit(n[0], k + d) is not None}
    return sets, singles


def unit_functions(n_prob=20000, thr=0.9):
    """For every active unit-position at step S: cross-validated R² of the best lookup table over the labeller's three
    inputs (a_k, b_k, c_k) and over wider bit sets, and of linear fits. Categories: label (the labeller's code),
    analog-local (a function of the three inputs the labeller did not accept: count = of the count only, mixed = not),
    neighbours (a function of the three inputs plus bits of positions k-2..k+1), unexplained."""
    rng = np.random.default_rng(11); a, b = TG.problems(n_prob, NB, rng); L = TG.labels(a, b)
    S = get_S(); X = {"n_problems": n_prob, "threshold": thr, "sets": list(UNIT_SETS), "models": {}}
    all_bits = np.concatenate([L["a"], L["b"], L["c"]], 1)
    for w in WIDTHS:
        for s in SEEDS:
            fin = json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json")))["checkpoints"][-1]
            codes = np.array(fin["codes"]); model = load_ckpt(w, s, S); H, _, _ = CC.hidden(model, TG.Flat, a, b)
            cat = np.full(codes.shape, "", dtype=object); R = {}; best_single = np.full(codes.shape, "", dtype=object)
            for i, k in enumerate(KS):
                x = H[:, k].astype(float); sets, singles = _bit_sets(L, k); rk = np.random.default_rng(100 + k)
                r = {n: _cv_table_r2(x, *_groups(v), rk) for n, v in sets.items()}
                r["count"] = _cv_table_r2(x, L["a"][:, k] + L["b"][:, k] + L["c"][:, k], 4, rk)
                r["linear_local"] = _cv_linear_r2(x, np.stack(sets["local"], 1), rk)
                r["linear_window"] = _cv_linear_r2(x, np.stack(sets["window"], 1), rk)
                r["linear_all_bits"] = _cv_linear_r2(x, all_bits, rk)
                sg = {n: _cv_table_r2(x, *_groups(v), rk) for n, v in singles.items()}
                for n, v in r.items():
                    R.setdefault(n, []).append(v)
                for u in range(codes.shape[0]):
                    c = codes[u, i]
                    if c < 0:
                        cat[u, i] = "inactive"; continue
                    if c != 0:
                        cat[u, i] = "label"; continue
                    if r["local"][u] >= thr:
                        cat[u, i] = "analog-local-count" if r["count"][u] >= thr else "analog-local-mixed"; continue
                    hit = next((n for n in UNIT_SETS if r[n][u] >= thr), None)
                    cat[u, i] = f"neighbours-{hit}" if hit else "unexplained"
                    gains = {n: sg[n][u] - r["local"][u] for n in sg}
                    best_single[u, i] = max(gains, key=gains.get)
            R = {n: np.stack(v, 1) for n, v in R.items()}                       # (U, 9)
            act = codes >= 0; n_act = max(int(act.sum()), 1); unl = act & (codes == 0)
            names = ("label", "analog-local-count", "analog-local-mixed") + tuple(f"neighbours-{n}" for n in UNIT_SETS) + ("unexplained",)
            shares = {n: float((cat == n).sum() / n_act) for n in names}
            singles_hist = {}
            for v in best_single[np.char.startswith(cat.astype(str), "neighbours") | (cat == "unexplained")]:
                singles_hist[v] = singles_hist.get(v, 0) + 1
            glyph = {"inactive": ".", "label": "L", "analog-local-count": "c", "analog-local-mixed": "m", "unexplained": "?"}
            X["models"][f"{w}:{s}"] = {
                "learns": fin["exact_acc"] >= 0.99, "exact_acc": fin["exact_acc"], "active_unit_positions": int(act.sum()),
                "unlabelled_unit_positions": int(unl.sum()), "shares_of_active": shares,
                "mean_r2_unlabelled": {n: float(R[n][unl].mean()) if unl.any() else None for n in R},
                "mean_r2_labelled": {n: float(R[n][act & (codes != 0)].mean()) if (act & (codes != 0)).any() else None for n in R},
                "best_extra_bit_counts": singles_hist,
                "units": ["".join(glyph.get(cat[u, i], "n") for i in range(len(KS))) for u in range(codes.shape[0])] if w <= 32 else None,
                "r2": {n: np.round(R[n], 3).tolist() for n in R} if w <= 32 else None}
            print(f"L-w{w}-s{s} acc {fin['exact_acc']:.2f}: " + " ".join(f"{n.split('-')[-1] if n != 'label' else n} {v:.2f}" for n, v in shares.items() if v > 0)
                  + f" | unlabelled: local {X['models'][f'{w}:{s}']['mean_r2_unlabelled']['local']} lin_all {X['models'][f'{w}:{s}']['mean_r2_unlabelled']['linear_all_bits']}", flush=True)
    path = os.path.join(OUT, "exploratory.json")
    E = json.load(open(path)) if os.path.exists(path) else {}
    E["unit_functions"] = X
    json.dump(E, open(path, "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    # figure 4: one stacked bar per model, learners marked
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    groups = (("label", ("label",), "C2"), ("function of the 3 inputs, analog", ("analog-local-count", "analog-local-mixed"), "C8"),
              ("3 inputs + the token s(k-1)", ("neighbours-token",), "C1"),
              ("3 inputs + neighbour bits", tuple(f"neighbours-{n}" for n in UNIT_SETS if n != "token"), "C3"), ("unexplained", ("unexplained",), "0.7"))
    keys = [f"{w}:{s}" for w in WIDTHS for s in SEEDS]
    fig, ax = plt.subplots(figsize=(13, 4.2)); xs = np.arange(len(keys)) + np.repeat(np.arange(len(WIDTHS)), len(SEEDS)) * 0.8
    bottom = np.zeros(len(keys))
    for name, members, col in groups:
        v = np.array([sum(X["models"][k]["shares_of_active"][m] for m in members) for k in keys])
        ax.bar(xs, v, 0.8, bottom=bottom, color=col, label=name); bottom += v
    for x_, k in zip(xs, keys):
        if X["models"][k]["learns"]:
            ax.text(x_, 1.02, "*", ha="center", fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels([k.split(":")[1] for k in keys], fontsize=8)
    for j, w in enumerate(WIDTHS):
        ax.text(xs[j * len(SEEDS) + 2], -0.13, f"width {w}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("share of active unit-positions"); ax.set_title("what each unit-position is a function of at step S  (* = learned)")
    ax.legend(fontsize=7, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True); fig.savefig(os.path.join(FIG, "4_unit_functions.png"), dpi=130); plt.close(fig)
    return X


# ---- the token path (exploratory, added after the unit-function tables): is it the embedding of the token at q_k
# ---- (the answer bit s_(k-1) just written) that the unlabelled units read?
def token_patch_hooks(model, qs, mode, scope):
    """Hooks that replace the token embedding E[s_(k-1)] at the positions qs (k >= 1) by the mean of E[0] and E[1]
    (mode 'remove') or by the other digit's embedding (mode 'flip'), either only in what the MLP reads (scope 'mlp':
    resid_post = resid_mid + mlp(patched resid_mid)) or in the residual stream itself (scope 'all'). The token at
    each q is read off resid_pre, so the hooks also work during generation. Returns (hooks, keep); keep['tok'] has
    the token ids the hooks saw."""
    P = model.P; E = P["W_E"][:2].astype(float); Ebar = E.mean(0); keep = {}

    def pre(x):
        keep["tok"] = {q: np.argmin([np.abs(x[:, q] - P["W_pos"][q] - E[i]).sum(1) for i in (0, 1)], 0) for q in qs}
        return x

    def patched(xm, q):
        t = keep["tok"][q]
        return xm.astype(float) - E[t] + (Ebar if mode == "remove" else E[1 - t])

    def mid(x):
        keep["x"] = x
        if scope == "all":
            x = x.copy()
            for q in qs:
                x[:, q] = patched(keep["x"][:, q], q)
        return x

    def post(x):
        if scope == "mlp":
            x = x.copy()
            for q in qs:
                xm = patched(keep["x"][:, q], q)
                x[:, q] = keep["x"][:, q] + np.maximum(CC._ln2(model, xm) @ P["0.W_in"] + P["0.b_in"], 0) @ P["0.W_out"] + P["0.b_out"]
        return x
    return {TG.HOOKS["resid_pre"]: pre, TG.MID: mid, TG.POST: post}, keep


def _tf_acc(model, a, b, L, hooks_fn=None, bs=2000):
    """Teacher-forced accuracy of s_k, k = 1..9, pooled, and split by whether a_(k-1) = b_(k-1) (when they differ,
    s_(k-1) alone tells the carry into k). hooks_fn() -> (hooks, keep), fresh for every forward pass."""
    qs = [TG.Flat.q(k) for k in KS]; hit = []
    for i in range(0, len(a), bs):
        lg = model.forward(TG.Flat.tokens(a[i:i + bs], b[i:i + bs]), hooks=hooks_fn()[0] if hooks_fn else None)
        hit.append(lg[:, qs, :2].argmax(2) == L["s"][i:i + bs][:, KS])
    hit = np.concatenate(hit); same = L["a"][:, [k - 1 for k in KS]] == L["b"][:, [k - 1 for k in KS]]
    return {"all": float(hit.mean()), "prev_equal": float(hit[same].mean()), "prev_differ": float(hit[~same].mean())}


def _gen_acc(model, a, b, mode=None, scope=None):
    fn = None if mode is None else (lambda k, rows: token_patch_hooks(model, [TG.Flat.q(k)], mode, scope)[0] if k >= 1 else None)
    return float(np.mean(TG.generate(model, TG.Flat, a, b, hooks_fn=fn) == a + b))


def token_path(n_prob=20000, n_gen=4000, thr=0.9):
    """Remove or flip the token embedding at q_k and see (1) whether the unlabelled units stop depending on s_(k-1)
    (unit-level: the table-R² gain of adding s_(k-1), before and after, on activations recomputed from the patched
    resid_mid) and (2) whether the model still adds (teacher-forced per-bit accuracy and greedy exact sums)."""
    rng = np.random.default_rng(11); a, b = TG.problems(n_prob, NB, rng); L = TG.labels(a, b)
    D = CC.data(); ag, bg = D["a_te"][:n_gen], D["b_te"][:n_gen]
    S = get_S(); X = {"n_problems": n_prob, "n_generation": n_gen, "threshold": thr, "models": {}}
    for w in WIDTHS:
        for s in SEEDS:
            fin = json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json")))["checkpoints"][-1]
            codes = np.array(fin["codes"]); model = load_ckpt(w, s, S); P = model.P
            E = P["W_E"][:2].astype(float); Ebar = E.mean(0)
            H, Xm, _ = CC.hidden(model, TG.Flat, a, b)
            R = {"before": {"local": [], "token": []}, "after": {"local": [], "token": []}}
            for i, k in enumerate(KS):
                sets, _ = _bit_sets(L, k)
                xm = Xm[:, k] - E[L["s"][:, k - 1]] + Ebar
                hp = np.maximum(CC._ln2(model, xm) @ P["0.W_in"] + P["0.b_in"], 0)
                for name in ("local", "token"):
                    g, m = _groups(sets[name])
                    R["before"][name].append(_cv_table_r2(H[:, k].astype(float), g, m, np.random.default_rng(100 + k)))
                    R["after"][name].append(_cv_table_r2(hp, g, m, np.random.default_rng(100 + k)))
            R = {t: {n: np.stack(v, 1) for n, v in d.items()} for t, d in R.items()}
            act = codes >= 0; unl = act & (codes == 0); lab = act & (codes != 0)
            tokcat = unl & (R["before"]["local"] < thr) & (R["before"]["token"] >= thr)
            gain = {t: R[t]["token"] - R[t]["local"] for t in R}
            mean = lambda v, m: float(v[m].mean()) if m.any() else None
            units = {"active": int(act.sum()), "unlabelled": int(unl.sum()), "token_category": int(tokcat.sum())}
            for t in ("before", "after"):
                units[f"gain_{t}_token_category"] = mean(gain[t], tokcat); units[f"gain_{t}_unlabelled"] = mean(gain[t], unl)
                units[f"r2_local_{t}_unlabelled"] = mean(R[t]["local"], unl); units[f"r2_local_{t}_labelled"] = mean(R[t]["local"], lab)
                units[f"share_unlabelled_local_ge_thr_{t}"] = mean((R[t]["local"] >= thr).astype(float), unl)
                units[f"share_token_category_local_ge_thr_{t}"] = mean((R[t]["local"] >= thr).astype(float), tokcat)
            tf = {"base": _tf_acc(model, a, b, L)}; gen = {"base": _gen_acc(model, ag, bg)}
            for mode in ("remove", "flip"):
                for scope in ("mlp", "all"):
                    key = f"{mode}_{scope}"; qs = [TG.Flat.q(k) for k in KS]
                    tf[key] = _tf_acc(model, a, b, L, lambda: token_patch_hooks(model, qs, mode, scope))
                    gen[key] = _gen_acc(model, ag, bg, mode, scope)
            X["models"][f"{w}:{s}"] = {"learns": fin["exact_acc"] >= 0.99, "exact_acc": fin["exact_acc"], "units": units,
                                       "teacher_forced": tf, "generation": gen}
            u = units; f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"L-w{w}-s{s} acc {fin['exact_acc']:.2f}: token-cat {u['token_category']}/{u['unlabelled']} gain {f(u['gain_before_token_category'])}->"
                  f"{f(u['gain_after_token_category'])}; unlab local R2 {f(u['r2_local_before_unlabelled'])}->{f(u['r2_local_after_unlabelled'])} "
                  f"(>= thr {f(u['share_unlabelled_local_ge_thr_before'])}->{f(u['share_unlabelled_local_ge_thr_after'])}) | TF base {tf['base']['all']:.3f} "
                  f"remove_mlp {tf['remove_mlp']['all']:.3f} (eq {tf['remove_mlp']['prev_equal']:.3f} diff {tf['remove_mlp']['prev_differ']:.3f}) "
                  f"flip_mlp {tf['flip_mlp']['all']:.3f} (eq {tf['flip_mlp']['prev_equal']:.3f} diff {tf['flip_mlp']['prev_differ']:.3f}) | gen "
                  + " ".join(f"{k} {v:.3f}" for k, v in gen.items()), flush=True)
    path = os.path.join(OUT, "exploratory.json")
    Ex = json.load(open(path)) if os.path.exists(path) else {}
    Ex["token_path"] = X
    json.dump(Ex, open(path, "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    token_path_pre()
    return X


def token_pre_hooks(model, q, mode):
    """Replace the token embedding at one position q at resid_pre, before attention, so that every route from the
    token at q_k (the direct one and the ones through attention) carries the changed digit. mode 'remove' (the mean
    of E[0] and E[1]) or 'flip' (the other digit). One position at a time: a flipped answer token at q_j would also
    change what later positions attend to. Returns (hooks, keep)."""
    P = model.P; E = P["W_E"][:2].astype(float); Ebar = E.mean(0); keep = {}

    def pre(x):
        t = np.argmin([np.abs(x[:, q] - P["W_pos"][q] - E[i]).sum(1) for i in (0, 1)], 0); keep["tok"] = t
        x = x.copy(); x[:, q] = x[:, q].astype(float) - E[t] + (Ebar if mode == "remove" else E[1 - t])
        return x
    return {TG.HOOKS["resid_pre"]: pre}, keep


def token_path_pre(n_prob=20000, n_gen=4000, thr=0.9, bs=2000):
    """token_path again, with the token embedding changed at resid_pre (all routes) instead of at resid_mid (the
    direct route only). Per model and mode: the unlabelled units' dependence on s_(k-1) after the change, the
    teacher-forced per-bit accuracy (one q_k changed per forward pass) and greedy exact sums."""
    rng = np.random.default_rng(11); a, b = TG.problems(n_prob, NB, rng); L = TG.labels(a, b); tok_all = TG.Flat.tokens(a, b)
    D = CC.data(); ag, bg = D["a_te"][:n_gen], D["b_te"][:n_gen]
    S = get_S(); X = {"n_problems": n_prob, "n_generation": n_gen, "threshold": thr, "models": {}}
    same = L["a"][:, [k - 1 for k in KS]] == L["b"][:, [k - 1 for k in KS]]
    for w in WIDTHS:
        for s in SEEDS:
            fin = json.load(open(os.path.join(AN, f"L-w{w}-s{s}.json")))["checkpoints"][-1]
            codes = np.array(fin["codes"]); model = load_ckpt(w, s, S)
            H, _, _ = CC.hidden(model, TG.Flat, a, b)
            R = {"before": {"local": [], "token": []}}
            for i, k in enumerate(KS):
                sets, _ = _bit_sets(L, k)
                for name in ("local", "token"):
                    g, m = _groups(sets[name]); R["before"][name].append(_cv_table_r2(H[:, k].astype(float), g, m, np.random.default_rng(100 + k)))
            tf = {"base": _tf_acc(model, a, b, L)}; gen = {"base": _gen_acc(model, ag, bg)}
            for mode in ("remove", "flip"):
                hit = np.zeros((n_prob, len(KS)), bool); R[mode] = {"local": [], "token": []}
                for i, k in enumerate(KS):
                    q = TG.Flat.q(k); hp = []
                    for j in range(0, n_prob, bs):
                        lg, acts, _ = model.forward(tok_all[j:j + bs], hooks=token_pre_hooks(model, q, mode)[0], cache=True)
                        hit[j:j + bs, i] = lg[:, q, :2].argmax(1) == L["s"][j:j + bs, k]
                        hp.append(acts["blocks.0.mlp.hook_post"][:, q].astype(float))
                    hp = np.concatenate(hp); sets, _ = _bit_sets(L, k)
                    for name in ("local", "token"):
                        g, m = _groups(sets[name]); R[mode][name].append(_cv_table_r2(hp, g, m, np.random.default_rng(100 + k)))
                tf[mode] = {"all": float(hit.mean()), "prev_equal": float(hit[same].mean()), "prev_differ": float(hit[~same].mean())}
                fn = lambda k, rows, mode=mode: token_pre_hooks(model, TG.Flat.q(k), mode)[0] if k >= 1 else None
                gen[mode] = float(np.mean(TG.generate(model, TG.Flat, ag, bg, hooks_fn=fn) == ag + bg))
            R = {t: {n: np.stack(v, 1) for n, v in d.items()} for t, d in R.items()}
            act = codes >= 0; unl = act & (codes == 0); lab = act & (codes != 0)
            tokcat = unl & (R["before"]["local"] < thr) & (R["before"]["token"] >= thr)
            mean = lambda v, m: float(v[m].mean()) if m.any() else None
            units = {"active": int(act.sum()), "unlabelled": int(unl.sum()), "token_category": int(tokcat.sum())}
            for t in R:
                gain = R[t]["token"] - R[t]["local"]
                units[f"gain_{t}_token_category"] = mean(gain, tokcat); units[f"gain_{t}_unlabelled"] = mean(gain, unl)
                units[f"r2_local_{t}_unlabelled"] = mean(R[t]["local"], unl); units[f"r2_local_{t}_labelled"] = mean(R[t]["local"], lab)
                units[f"share_unlabelled_local_ge_thr_{t}"] = mean((R[t]["local"] >= thr).astype(float), unl)
                units[f"share_token_category_local_ge_thr_{t}"] = mean((R[t]["local"] >= thr).astype(float), tokcat)
            X["models"][f"{w}:{s}"] = {"learns": fin["exact_acc"] >= 0.99, "exact_acc": fin["exact_acc"], "units": units,
                                       "teacher_forced": tf, "generation": gen}
            u = units; f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"L-w{w}-s{s} acc {fin['exact_acc']:.2f}: token-cat {u['token_category']}/{u['unlabelled']} gain {f(u['gain_before_token_category'])}"
                  f" -> remove {f(u['gain_remove_token_category'])} flip {f(u['gain_flip_token_category'])}; unlab local R2 {f(u['r2_local_before_unlabelled'])}"
                  f" -> {f(u['r2_local_remove_unlabelled'])} / {f(u['r2_local_flip_unlabelled'])} (>= thr {f(u['share_unlabelled_local_ge_thr_before'])} -> "
                  f"{f(u['share_unlabelled_local_ge_thr_remove'])} / {f(u['share_unlabelled_local_ge_thr_flip'])}) | TF base {tf['base']['all']:.3f} "
                  + " ".join(f"{m} {tf[m]['all']:.3f} (eq {tf[m]['prev_equal']:.3f} diff {tf[m]['prev_differ']:.3f})" for m in ("remove", "flip"))
                  + " | gen " + " ".join(f"{k} {v:.3f}" for k, v in gen.items()), flush=True)
    path = os.path.join(OUT, "exploratory.json")
    Ex = json.load(open(path)) if os.path.exists(path) else {}
    Ex["token_path_pre"] = X
    json.dump(Ex, open(path, "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
    if "token_path" in Ex:
        _figure_token_path(Ex)
    return X


def _figure_token_path(Ex):
    """Figure 5: what changing the digit under a position does, per model."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A, B = Ex["token_path"]["models"], Ex["token_path_pre"]["models"]
    keys = [f"{w}:{s}" for w in WIDTHS for s in SEEDS]
    xs = np.arange(len(keys)) + np.repeat(np.arange(len(WIDTHS)), len(SEEDS)) * 0.8
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 4.4))
    for (mode, split), lab, col, mk, fc in ((("base", "all"), "no change", "0.5", "_", True),
                                            (("remove", "all"), "digit removed before attention", "C0", "s", True),
                                            (("flip", "prev_equal"), "digit flipped before attention, a(k-1) = b(k-1)", "C3", "o", True),
                                            (("flip", "prev_differ"), "digit flipped before attention, a(k-1) ≠ b(k-1)", "C3", "o", False)):
        y = [B[k]["teacher_forced"][mode][split] for k in keys]
        if mk == "_":
            axs[0].scatter(xs, y, marker=mk, s=60, color=col, label=lab)
        else:
            axs[0].scatter(xs, y, marker=mk, s=32, edgecolors=col, facecolors=col if fc else "none", linewidths=1.2, label=lab)
    axs[0].set_ylabel("teacher-forced accuracy of s_k, k = 1..9"); axs[0].set_title("per-bit accuracy when the digit under the position is changed")
    for key, lab, col, mk, src in (("base", "no change", "0.5", "_", B), ("flip_mlp", "flipped on the direct route only (resid_mid)", "C1", "^", A),
                                   ("remove", "removed before attention", "C0", "s", B), ("flip", "flipped before attention", "C3", "o", B)):
        axs[1].scatter(xs, [src[k]["generation"][key] for k in keys], marker=mk, s=32, color=col, label=lab)
    axs[1].set_ylabel("exact sums (greedy generation)"); axs[1].set_title("the whole model")
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([k.split(":")[1] for k in keys], fontsize=8); ax.set_ylim(-0.03, 1.1)
        for j, w in enumerate(WIDTHS):
            ax.text(xs[j * len(SEEDS) + 2], -0.17, f"width {w}", ha="center", fontsize=9)
        for x_, k in zip(xs, keys):
            if B[k]["learns"]:
                ax.text(x_, 1.03, "*", ha="center", fontsize=12)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True); fig.savefig(os.path.join(FIG, "5_token_path.png"), dpi=130); plt.close(fig)


# ======================= main =======================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg-check", "pilot", "train", "analyse", "report", "explore"])
    ap.add_argument("--only", default=None)
    A_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if A_.cmd == "prereg-check":
        prereg_check(lock=True)
    elif A_.cmd == "pilot":
        prereg_check()
        if A_.only:
            w, s = map(int, A_.only.split(":")); pilot_run(w, s)
        else:
            todo = [f"{w}:{s}" for w, s in PILOT_RUNS if not os.path.exists(os.path.join(OUT, "pilot", f"w{w}-s{s}.json"))]
            pool(todo, "pilot")
            pilot_S()
    elif A_.cmd == "train":
        prereg_check(); S = get_S()
        if A_.only:
            w, s = map(int, A_.only.split(":"))
            if not os.path.exists(ckpt(w, s, S)):
                train_long(w, s, S, save_steps=schedule_long(S), save_fn=save_main(w, s), tag=f"L-w{w}-s{s}")
        else:
            pool([f"{w}:{s}" for w in WIDTHS for s in SEEDS if not os.path.exists(ckpt(w, s, S))], "train")
    elif A_.cmd == "analyse":
        prereg_check()
        if A_.only == "v2g":
            analyse_v2g()
        elif A_.only:
            w, s = map(int, A_.only.split(":")); analyse_long(w, s)
        else:
            jobs = ([] if os.path.exists(os.path.join(AN, "v2g.json")) else ["v2g"]) + \
                   [f"{w}:{s}" for w in WIDTHS for s in SEEDS if not os.path.exists(os.path.join(AN, f"L-w{w}-s{s}.json"))]
            pool(jobs, "analyse")
    elif A_.cmd == "explore":
        prereg_check(); explore()
    else:
        report()
