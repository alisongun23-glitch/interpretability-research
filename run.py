"""Run the whole XOR circuit experiment: train -> analyse -> causal tests -> transplants
-> figures -> report.

    python run.py                    # uses config.json
    python run.py --config my.json
    python run.py --seeds 10         # override the number of seeds
"""
import argparse, csv, json, os, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # multithreaded BLAS spins on these small matrices; must precede numpy
from collections import Counter
from itertools import permutations
import numpy as np

from xorlab.model import train_batch, hidden_acts, logits, all_correct, Y, INPUT_NAMES
from xorlab import analysis as A
from xorlab import figures, report


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def analyse_net(run, act, icfg):
    p = run["params"]
    thr = icfg["purity_threshold"]
    Hn = p["w2"].shape[0]
    run["all_units"] = A.describe_units(p, act, range(Hn), thr)
    run["init_units"] = A.describe_units(run["init"], act, range(Hn), thr)
    if not run["converged"]:
        return
    mins = A.minimal_circuits(p, act, icfg["ablation"])
    run["minimal_circuits"] = []
    for c in mins:
        d = A.describe_units(p, act, c["units"], thr)
        run["minimal_circuits"].append({"units": c["units"], "margin": c["margin"],
                                        "signature": A.signature(d), "units_desc": d})
    # Prespecified rule: primary candidate = smallest minimal circuit, ties broken by
    # the largest retained margin.
    prim = run["minimal_circuits"][0] if run["minimal_circuits"] else None
    run["primary"] = prim
    if prim:
        run["causal"] = A.causal_tests(p, act, prim["units"], icfg["ablation"])
        run["zero_ablation_agrees"] = bool(all(
            A.correct(A.forward(p, act, ablate=[i for i in range(Hn) if i not in prim["units"]],
                                mode="zero"))))


def transplant_pairs(donors, d_act, recips, r_act, icfg, tag):
    rows = []
    for d in donors:
        Cd = d["primary"]["units"]
        c_d = A.contribution(d["params"], d_act, Cd)
        c_sham = A.contribution(d["init"], d_act, Cd)        # same units, before training
        for r in recips:
            if r is d:
                continue
            Cr = r["primary"]["units"]
            t = A.transplant(r["params"], r_act, Cr, c_d)
            s = A.transplant(r["params"], r_act, Cr, c_sham)
            ref = A.transplant(r["params"], r_act, Cr,
                               A.reference_contribution(r_act, icfg["reference_logit_scale"]))
            ok = lambda lg: bool(np.all(A.correct(lg)))
            rows.append({
                "pairing": tag, "donor_seed": d["seed"], "recipient_seed": r["seed"],
                "donor_signature": d["primary"]["signature"],
                "recipient_signature": r["primary"]["signature"],
                "same_signature": d["primary"]["signature"] == r["primary"]["signature"],
                "native_ok": ok(t["native"]), "ablated_ok": ok(t["ablated"]),
                "direct_ok": ok(t["direct"]), "aligned_ok": ok(t["aligned"]),
                "sham_direct_ok": ok(s["direct"]), "sham_aligned_ok": ok(s["aligned"]),
                "reference_direct_ok": ok(ref["direct"]), "reference_aligned_ok": ok(ref["aligned"]),
                "direct_acc": float(np.mean(A.correct(t["direct"]))),
                "aligned_acc": float(np.mean(A.correct(t["aligned"]))),
                "direct_mean_abs_dev": float(np.mean(np.abs(t["direct"] - t["native"]))),
                "aligned_mean_abs_dev": float(np.mean(np.abs(t["aligned"] - t["native"]))),
                "direct_logits": t["direct"].tolist(), "aligned_logits": t["aligned"].tolist(),
            })
    return rows


def rate(rows, key, cond=lambda r: True):
    sel = [r[key] for r in rows if cond(r)]
    return (float(np.mean(sel)) if sel else None), len(sel)


def summarise_arch(name, runs, act, trows, disc_seeds):
    ok = [r for r in runs if r["converged"]]
    sigs = Counter(r["primary"]["signature"] for r in ok if r.get("primary"))
    caus = [r["causal"] for r in ok if r.get("causal")]
    mean = lambda k: float(np.mean([c[k] for c in caus if c[k] is not None])) if any(c[k] is not None for c in caus) else None
    s = {
        "architecture": name, "activation": act, "n_runs": len(runs), "n_converged": len(ok),
        "success_rate": len(ok) / len(runs),
        "median_converge_step": float(np.median([r["converge_step"] for r in ok])) if ok else None,
        "failed_seeds": [r["seed"] for r in runs if not r["converged"]],
        "mean_min_margin": float(np.mean([r["min_margin"] for r in ok])) if ok else None,
        "signature_counts": dict(sigs.most_common()),
        "primary_circuit_sizes": dict(Counter(len(r["primary"]["units"]) for r in ok if r.get("primary"))),
        "n_minimal_circuits": dict(Counter(len(r["minimal_circuits"]) for r in ok)),
        "frac_multiple_minimal_circuits": float(np.mean([len(r["minimal_circuits"]) > 1 for r in ok])) if ok else None,
        "frac_no_single_unit_break": float(np.mean([len(c["single_unit_breaks"]) == 0 for c in caus])) if caus else None,
        "zero_ablation_agreement": float(np.mean([r["zero_ablation_agrees"] for r in ok if "zero_ablation_agrees" in r])) if ok else None,
        "causal_means": {k: mean(k) for k in ["intact_acc", "ablate_circuit_acc", "ablate_circuit_zero_acc",
                                              "ablate_random_acc", "ablate_complement_acc",
                                              "patch_circuit_flip", "patch_complement_flip", "patch_random_flip"]},
    }
    tr = {}
    for k in ["native_ok", "ablated_ok", "direct_ok", "aligned_ok", "sham_direct_ok", "sham_aligned_ok",
              "reference_direct_ok", "reference_aligned_ok"]:
        tr[k] = rate(trows, k)[0]
    tr["n_pairs"] = len(trows)
    tr["direct_ok_same_signature"], tr["n_same"] = rate(trows, "direct_ok", lambda r: r["same_signature"])
    tr["direct_ok_diff_signature"], tr["n_diff"] = rate(trows, "direct_ok", lambda r: not r["same_signature"])
    tr["aligned_ok_same_signature"] = rate(trows, "aligned_ok", lambda r: r["same_signature"])[0]
    tr["aligned_ok_diff_signature"] = rate(trows, "aligned_ok", lambda r: not r["same_signature"])[0]
    s["transplant"] = tr
    # discovery vs confirmation
    disc = [r for r in ok if r["seed"] in disc_seeds and r.get("primary")]
    conf = [r for r in ok if r["seed"] not in disc_seeds and r.get("primary")]
    if disc and conf:
        dom = Counter(r["primary"]["signature"] for r in disc).most_common(1)[0][0]
        cs = {r["seed"] for r in conf}
        conf_rows = [t for t in trows if t["donor_seed"] in cs and t["recipient_seed"] in cs]
        s["confirmation"] = {
            "dominant_signature_in_discovery": dom,
            "share_in_discovery": float(np.mean([r["primary"]["signature"] == dom for r in disc])),
            "share_in_confirmation": float(np.mean([r["primary"]["signature"] == dom for r in conf])),
            "n_discovery": len(disc), "n_confirmation": len(conf),
            "confirm_ablate_circuit_acc": float(np.mean([r["causal"]["ablate_circuit_acc"] for r in conf])),
            "confirm_ablate_random_acc": float(np.mean([r["causal"]["ablate_random_acc"] for r in conf
                                                        if r["causal"]["ablate_random_acc"] is not None]))
            if any(r["causal"]["ablate_random_acc"] is not None for r in conf) else None,
            "confirm_direct_ok": rate(conf_rows, "direct_ok")[0],
            "confirm_aligned_ok": rate(conf_rows, "aligned_ok")[0],
            "confirm_sham_aligned_ok": rate(conf_rows, "sham_aligned_ok")[0],
        }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--seeds", type=int)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.seeds:
        cfg["seeds"] = args.seeds
    out = cfg["output_dir"]
    for d in ["", "checkpoints", "training_logs", "figures"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config_used.json"), "w"), indent=2)
    seeds = list(range(cfg["seed_offset"], cfg["seed_offset"] + cfg["seeds"]))
    n_disc = int(round(cfg["discovery_fraction"] * len(seeds)))
    disc_seeds = set(seeds[:n_disc])
    t0 = time.time()

    results = {}
    for arch in cfg["architectures"]:
        name, act, H = arch["name"], arch["activation"], arch["hidden"]
        runs, steps, loss, acc = train_batch(seeds, act, H, cfg["training"])
        np.savez(os.path.join(out, "training_logs", f"{name}.npz"), steps=steps, loss=loss, acc=acc,
                 seeds=np.array(seeds))
        np.savez(os.path.join(out, "checkpoints", f"{name}.npz"),
                 **{f"seed{r['seed']}_{k}": v for r in runs for k, v in r["params"].items()},
                 **{f"seed{r['seed']}_init_{k}": v for r in runs for k, v in r["init"].items()})
        for r in runs:
            analyse_net(r, act, cfg["interventions"])
        ok = [r for r in runs if r["converged"] and r.get("primary")]
        trows = transplant_pairs(ok, act, ok, act, cfg["interventions"], f"{name}->{name}")
        results[name] = {"act": act, "hidden": H, "runs": runs, "steps": steps, "loss": loss,
                         "acc": acc, "transplants": trows,
                         "summary": summarise_arch(name, runs, act, trows, disc_seeds)}
        print(f"{name}: {len(ok)}/{len(runs)} converged  ({time.time() - t0:.1f}s)")

    # cross-activation transplant: tanh-4 circuits into relu-4 networks and back
    cross = []
    if "tanh-4" in results and "relu-4" in results:
        okt = [r for r in results["tanh-4"]["runs"] if r.get("primary")]
        okr = [r for r in results["relu-4"]["runs"] if r.get("primary")]
        cross += transplant_pairs(okt, "tanh", okr, "relu", cfg["interventions"], "tanh-4->relu-4")
        cross += transplant_pairs(okr, "relu", okt, "tanh", cfg["interventions"], "relu-4->tanh-4")

    # representational similarity for the baseline architecture
    base = results[cfg["baseline"]]
    okb = [r for r in base["runs"] if r.get("primary")]
    okb.sort(key=lambda r: (r["primary"]["signature"], r["seed"]))
    Hs = [hidden_acts(r["params"], base["act"]) for r in okb]
    Hi = [hidden_acts(r["init"], base["act"]) for r in okb]
    n = len(okb)
    cka = np.array([[A.linear_cka(Hs[i], Hs[j]) for j in range(n)] for i in range(n)])
    cka_init = np.array([[A.linear_cka(Hi[i], Hi[j]) for j in range(n)] for i in range(n)])
    contrib = [A.contribution(r["params"], base["act"], r["primary"]["units"]) for r in okb]
    ccorr = np.array([[A.corr(contrib[i], contrib[j]) for j in range(n)] for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    sim = {"seeds": [r["seed"] for r in okb], "signatures": [r["primary"]["signature"] for r in okb],
           "cka_trained": cka, "cka_untrained": cka_init, "contribution_corr": ccorr}
    np.savez(os.path.join(out, "similarity.npz"), **{k: np.array(v) for k, v in sim.items()})
    same = np.array([[okb[i]["primary"]["signature"] == okb[j]["primary"]["signature"] for j in range(n)]
                     for i in range(n)])
    simsum = {
        "cka_trained_mean": float(np.nanmean(cka[off])) if n > 1 else None,
        "cka_untrained_mean": float(np.nanmean(cka_init[off])) if n > 1 else None,
        "cka_same_signature_mean": float(np.nanmean(cka[off & same])) if (off & same).any() else None,
        "cka_diff_signature_mean": float(np.nanmean(cka[off & ~same])) if (off & ~same).any() else None,
        "contribution_corr_mean": float(np.nanmean(ccorr[off])) if n > 1 else None,
    }

    # unit-level gate frequencies: trained vs the same networks at initialisation
    gate_freq = {}
    for name, R in results.items():
        tr = Counter(u["activation_gate"] for r in R["runs"] if r["converged"] for u in r["all_units"])
        un = Counter(u["activation_gate"] for r in R["runs"] for u in r["init_units"])
        gate_freq[name] = {"trained": dict(tr), "untrained": dict(un)}

    summary = {
        "config": cfg, "runtime_seconds": time.time() - t0,
        "architectures": {k: v["summary"] for k, v in results.items()},
        "cross_activation_transplant": {
            k: rate(cross, k)[0] for k in ["direct_ok", "aligned_ok", "sham_aligned_ok", "reference_aligned_ok"]
        } | {"n_pairs": len(cross)},
        "similarity_baseline": simsum, "unit_gate_frequencies": gate_freq,
        "discovery_seeds": sorted(disc_seeds),
    }

    # ---------- save machine-readable outputs ----------
    with open(os.path.join(out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["architecture", "seed", "role", "converged", "converge_step", "accuracy", "min_margin"]
                   + [f"prob_{n}" for n in INPUT_NAMES] + [f"logit_{n}" for n in INPUT_NAMES]
                   + ["primary_units", "primary_signature", "n_minimal_circuits"])
        for name, R in results.items():
            for r in R["runs"]:
                w.writerow([name, r["seed"], "discovery" if r["seed"] in disc_seeds else "confirmation",
                            r["converged"], r["converge_step"], r["accuracy"], round(r["min_margin"], 4)]
                           + [round(x, 5) for x in r["probs"]] + [round(x, 4) for x in r["logits"]]
                           + [r["primary"]["units"] if r.get("primary") else "",
                              r["primary"]["signature"] if r.get("primary") else "",
                              len(r.get("minimal_circuits", []))])
    circ = {name: [{"seed": r["seed"], "converged": r["converged"], "all_units": r["all_units"],
                    "minimal_circuits": r.get("minimal_circuits", []), "primary": r.get("primary")}
                   for r in R["runs"]] for name, R in results.items()}
    json.dump(jsonable(circ), open(os.path.join(out, "circuits.json"), "w"), indent=1)
    caus = {name: [{"seed": r["seed"], **r["causal"]} for r in R["runs"] if r.get("causal")]
            for name, R in results.items()}
    json.dump(jsonable(caus), open(os.path.join(out, "causal.json"), "w"), indent=1)
    allrows = [t for R in results.values() for t in R["transplants"]] + cross
    with open(os.path.join(out, "transplants.csv"), "w", newline="") as f:
        keys = [k for k in allrows[0] if not k.endswith("_logits")] if allrows else []
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(allrows)
    json.dump(jsonable(summary), open(os.path.join(out, "summary.json"), "w"), indent=2)

    figures.make_all(results, cfg, sim, out)
    report.write(summary, results, out)
    print(f"Done in {time.time() - t0:.1f}s. Report: {os.path.join(out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
