"""Figure for the calibration: python3 fig_calib.py calib  -> calib/calibration.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = sys.argv[1]
INK2, MUTED, GRID, BASEC = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": BASEC, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
R = {k: json.load(open(os.path.join(D, f"calib_{k}.json"))) for k in ("zero", "random") if os.path.exists(os.path.join(D, f"calib_{k}.json"))}
cols = {"zero": "#3b6ea5", "random": "#d6632a", "real": "#2e8b57"}
real = json.load(open(os.path.join(D, "calib_real.json")))["summary"] if os.path.exists(os.path.join(D, "calib_real.json")) else None
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
ax = axs[0]
for i, (k, r) in enumerate(R.items()):
    ls = [l["layer"] for l in r["layers"]]
    ax.bar([x + (i - 0.5) * 0.4 for x in ls], [l["active_units"] / max(l["scored_units"], 1) for l in r["layers"]], 0.38, color=cols[k], label=f"{k} memory")
if real:
    lv = sorted(int(k) for k in real["by_level"]); ax.plot([k // 2 - 1 for k in lv], [real["by_level"][str(k)]["active"] / real["by_level"][str(k)]["n"] for k in lv], "D-", color=cols["real"], label="real programs (circuit nodes)")
ax.set_xlabel("compiled layer"); ax.set_ylabel("share of wired units that ever fire"); ax.set_title("Coverage: what random trace records exercise", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=8)
ax = axs[1]
for k, r in R.items():
    ls = [l["layer"] for l in r["layers"]]
    ax.plot(ls, [l["m2_recall_all"] for l in r["layers"]], "o-", color=cols[k], label=f"mean-ablation, all wired units ({k})")
    ax.plot(ls, [l["m2_recall_active"] for l in r["layers"]], "s--", color=cols[k], alpha=0.6, label=f"mean-ablation, units that fire ({k})")
if real:
    ax.plot([k // 2 - 1 for k in lv], [real["by_level"][str(k)]["m2"] for k in lv], "D-", color=cols["real"], label="mean-ablation, real programs, nodes that fire")
ax.set_ylim(-0.02, 1.05); ax.set_xlabel("compiled layer"); ax.set_ylabel("share of true inputs found"); ax.set_title("M2: the ablation scan of results_storyroom", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
ax = axs[2]
for k, r in R.items():
    ls = [l["layer"] for l in r["layers"]]
    ax.plot(ls, [l["m1_recall_at_k_active"] for l in r["layers"]], "o-", color=cols[k], label=f"recall at k = true fan-in ({k})")
    ax.plot(ls, [l["m1_precision_active"] for l in r["layers"]], "^:", color=cols[k], alpha=0.7, label=f"precision at |corr| ≥ 0.5 ({k})")
if real:
    ax.plot([k // 2 - 1 for k in lv], [real["by_level"][str(k)]["m1_at_k"] for k in lv], "D-", color=cols["real"], label="recall at k = true fan-in (real programs)")
ax.set_ylim(-0.02, 1.05); ax.set_xlabel("compiled layer"); ax.set_ylabel("score on units that fire"); ax.set_title("M1: correlation ranking of xorlab", fontsize=9, loc="left"); ax.legend(frameon=False, fontsize=7)
fig.suptitle("Circuit discovery scored against the compiled wiring of Qwenputer's 12 computer layers", fontsize=9, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(D, "calibration.png"), dpi=150)
for k, r in R.items():
    print(k, json.dumps(r["summary"]))
    fan = {}
    for l in r["layers"]:
        for f, v in l["by_fanin"].items():
            a = fan.setdefault(int(f), [0, 0, 0.0, 0.0, 0.0]); a[0] += v["n"]; a[1] += v["active"]; a[2] += v["m2"] * v["n"]; a[3] += v["m1_at_k"] * v["n"]; a[4] += v["m1_prec"] * v["n"]
    print("  by fan-in (n, active share, M2 recall over all, M1 r@k over all, M1 prec over all):",
          {f: (a[0], round(a[1] / a[0], 3), round(a[2] / a[0], 3), round(a[3] / a[0], 3), round(a[4] / a[0], 3)) for f, a in sorted(fan.items())})
