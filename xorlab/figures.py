"""Figures A-F. Every panel is drawn from saved experimental results, except the
panel in figure F explicitly titled 'REFERENCE (hand-built, not learned)'."""
import os
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from .model import Y, INPUT_NAMES, hidden_acts

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name), dpi=130)
    plt.close(fig)


def fig_training(results, out):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in names:
        R = results[name]
        L = R["loss"]
        med = np.median(L, 0); lo, hi = np.percentile(L, [10, 90], 0)
        axes[0].plot(R["steps"], med, label=name)
        axes[0].fill_between(R["steps"], lo, hi, alpha=0.15)
        axes[1].plot(R["steps"], R["acc"].mean(0), label=name)
    axes[0].set_yscale("log"); axes[0].set_xscale("symlog", linthresh=100)
    axes[0].set_title("A1. Loss (median, 10-90% band across seeds)"); axes[0].set_xlabel("step")
    axes[1].set_xscale("symlog", linthresh=100)
    axes[1].set_title("A2. Mean accuracy on the 4 inputs"); axes[1].set_xlabel("step")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    sr = [results[n]["summary"]["success_rate"] for n in names]
    steps = [[r["converge_step"] for r in results[n]["runs"] if r["converged"]] for n in names]
    axes[2].boxplot([s if s else [np.nan] for s in steps])
    axes[2].set_xticks(range(1, len(names) + 1)); axes[2].set_xticklabels(names)
    for i, v in enumerate(sr):
        axes[2].text(i + 1, axes[2].get_ylim()[1] * 0.95, f"{v:.0%} ok", ha="center", fontsize=8)
    axes[2].set_title("A3. Steps to converge (successful runs) and success rate")
    _save(fig, out, "A_training.png")


def fig_truth_table(results, out):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        rng = np.random.default_rng(0)
        for r in results[name]["runs"]:
            jit = rng.uniform(-0.15, 0.15, 4)
            ax.scatter(np.arange(4) + jit, r["probs"], s=12, alpha=0.7,
                       c=["tab:blue" if r["converged"] else "tab:red"] * 4)
        ax.scatter(range(4), Y, marker="_", s=600, c="k", label="target")
        ax.axhline(0.5, ls=":", c="gray", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{n}\n→{int(y)}" for n, y in zip(INPUT_NAMES, Y)])
        ax.set_title(f"B. {name}")
    np.atleast_1d(axes)[0].set_ylabel("P(XOR = 1)   blue = converged, red = failed")
    _save(fig, out, "B_truth_table.png")


def fig_activations(results, cfg, out, n_show=8):
    name = cfg["baseline"]; R = results[name]
    runs = [r for r in R["runs"] if r.get("primary")][:n_show]
    if not runs:
        return
    fig, axes = plt.subplots(2, (len(runs) + 1) // 2, figsize=(3.1 * ((len(runs) + 1) // 2), 6.4))
    for ax, r in zip(axes.ravel(), runs):
        H = hidden_acts(r["params"], R["act"])
        im = ax.imshow(H.T, cmap="RdBu_r", vmin=-max(1, np.abs(H).max()), vmax=max(1, np.abs(H).max()))
        ax.set_xticks(range(4)); ax.set_xticklabels(INPUT_NAMES)
        prim = set(r["primary"]["units"])
        ax.set_yticks(range(H.shape[1]))
        ax.set_yticklabels([("★ " if u["unit"] in prim else "") + f"u{u['unit']} {u['activation_gate']}"
                            for u in r["all_units"]], fontsize=7)
        ax.set_title(f"seed {r['seed']}\n{r['primary']['signature']}", fontsize=8)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    fig.suptitle(f"C. Hidden activations, {name} (★ = unit in primary circuit; label = best-matching gate of the activation)")
    _save(fig, out, "C_hidden_activations.png")


def fig_similarity(results, cfg, sim, out):
    n = len(sim["seeds"])
    if n < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = [f"{s} | {g[:22]}" for s, g in zip(sim["seeds"], sim["signatures"])]
    for ax, key, title in [(axes[0], "cka_trained", "D1. Linear CKA, trained"),
                           (axes[1], "cka_untrained", "D2. Linear CKA, same nets untrained")]:
        im = ax.imshow(sim[key], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title + " (sorted by circuit type)")
        if n <= 30:
            ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
            ax.set_xticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    sigs = Counter()
    for name in results:
        for r in results[name]["runs"]:
            if r.get("primary"):
                sigs[(name, r["primary"]["signature"])] += 1
    names = list(results)
    allsig = sorted({s for _, s in sigs}, key=lambda s: -sum(v for (n_, s_), v in sigs.items() if s_ == s))[:10]
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        tot = sum(v for (n_, _), v in sigs.items() if n_ == name) or 1
        axes[2].barh(np.arange(len(allsig)) + i * w, [sigs[(name, s)] / tot for s in allsig], w, label=name)
    axes[2].set_yticks(np.arange(len(allsig)) + 0.4); axes[2].set_yticklabels(allsig, fontsize=7)
    axes[2].invert_yaxis(); axes[2].set_xlabel("share of converged nets")
    axes[2].set_title("D3. Primary-circuit types by architecture"); axes[2].legend(fontsize=7)
    _save(fig, out, "D_cross_model.png")


def fig_causal(results, cfg, out):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    keys = [("intact_acc", "intact"), ("ablate_circuit_acc", "ablate circuit"),
            ("ablate_random_acc", "ablate random\nsame size"), ("ablate_complement_acc", "ablate everything\nelse")]
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        cm = results[name]["summary"]["causal_means"]
        axes[0].bar(np.arange(len(keys)) + i * w, [cm[k] if cm[k] is not None else np.nan for k, _ in keys], w, label=name)
    axes[0].set_xticks(np.arange(len(keys)) + 0.4); axes[0].set_xticklabels([l for _, l in keys])
    axes[0].set_ylabel("XOR accuracy (4 inputs)"); axes[0].set_title("E1. Ablation"); axes[0].legend(fontsize=7)
    pk = [("patch_circuit_flip", "patch circuit"), ("patch_random_flip", "patch random\nsame size"),
          ("patch_complement_flip", "patch everything\nelse")]
    for i, name in enumerate(names):
        cm = results[name]["summary"]["causal_means"]
        axes[1].bar(np.arange(len(pk)) + i * w, [cm[k] if cm[k] is not None else np.nan for k, _ in pk], w, label=name)
    axes[1].set_xticks(np.arange(len(pk)) + 0.4); axes[1].set_xticklabels([l for _, l in pk])
    axes[1].set_ylabel("output follows the source input")
    axes[1].set_title("E2. Activation patching (opposite-label input pairs)")
    tk = [("native_ok", "native"), ("ablated_ok", "ablation\ncontrol"), ("direct_ok", "direct\ntransplant"),
          ("aligned_ok", "aligned\ntransplant"), ("sham_direct_ok", "sham\ndirect"), ("sham_aligned_ok", "sham\naligned"),
          ("reference_direct_ok", "reference\ndirect"), ("reference_aligned_ok", "reference\naligned")]
    for i, name in enumerate(names):
        tr = results[name]["summary"]["transplant"]
        axes[2].bar(np.arange(len(tk)) + i * w, [tr[k] if tr[k] is not None else np.nan for k, _ in tk], w, label=name)
    axes[2].set_xticks(np.arange(len(tk)) + 0.4); axes[2].set_xticklabels([l for _, l in tk], fontsize=7)
    axes[2].set_ylabel("share of donor→recipient pairs fully correct")
    axes[2].set_title("E3. Cross-seed circuit transplants (no retraining)")
    _save(fig, out, "E_causal.png")


def _draw_circuit(ax, units_desc, W1, b1, title, color="tab:blue"):
    ax.set_xlim(0, 3); ax.set_ylim(-0.2, 1.2); ax.axis("off"); ax.set_title(title, fontsize=8)
    k = len(units_desc)
    ins = {"A": (0.3, 0.75), "B": (0.3, 0.25)}
    hid = [(1.5, 0.5 if k == 1 else 1 - (i + 0.5) / k) for i in range(k)]
    outp = (2.7, 0.5)
    mx = max(1e-9, max(abs(W1[j, i]) for i in range(k) for j in range(2)))
    mo = max(1e-9, max(abs(d["w_out"]) for d in units_desc))
    for i, d in enumerate(units_desc):
        for j, nm in enumerate("AB"):
            w = W1[j, i]
            ax.add_patch(FancyArrowPatch(ins[nm], hid[i], arrowstyle="-", lw=0.5 + 3 * abs(w) / mx,
                                         color="tab:red" if w > 0 else "tab:blue", alpha=0.7))
            ax.text(*(np.add(ins[nm], hid[i]) / 2 + [0, 0.04 * (1 if j == 0 else -1)]), f"{w:+.1f}", fontsize=6)
        ax.add_patch(FancyArrowPatch(hid[i], outp, arrowstyle="-|>", mutation_scale=8,
                                     lw=0.5 + 3 * abs(d["w_out"]) / mo,
                                     color="tab:red" if d["w_out"] > 0 else "tab:blue", alpha=0.7))
        ax.add_patch(Circle(hid[i], 0.12, color=color, alpha=0.2))
        ax.text(hid[i][0], hid[i][1] + 0.15, f"{d['activation_gate']}", ha="center", fontsize=7, weight="bold")
        ax.text(hid[i][0], hid[i][1] - 0.2, f"pushes: {d['contribution_gate']}", ha="center", fontsize=6)
        ax.text(*hid[i], f"b{b1[i]:+.1f}", ha="center", va="center", fontsize=6)
    for nm, xy in ins.items():
        ax.add_patch(Circle(xy, 0.1, color="gray", alpha=0.3)); ax.text(*xy, nm, ha="center", va="center")
    ax.add_patch(Circle(outp, 0.11, color="gray", alpha=0.3)); ax.text(*outp, "XOR", ha="center", va="center", fontsize=7)


def fig_circuits(results, cfg, out):
    name = cfg["baseline"]; R = results[name]
    ok = [r for r in R["runs"] if r.get("primary")]
    by = {}
    for r in ok:
        by.setdefault(r["primary"]["signature"], []).append(r)
    common = sorted(by, key=lambda s: -len(by[s]))[:5]
    fig, axes = plt.subplots(1, len(common) + 1, figsize=(3.8 * (len(common) + 1), 5.2))
    axes = np.atleast_1d(axes)
    for ax, sig in zip(axes, common):
        r = by[sig][0]; U = r["primary"]["units"]
        _draw_circuit(ax, r["primary"]["units_desc"], r["params"]["W1"][:, U], r["params"]["b1"][U],
                      f"DISCOVERED ({len(by[sig])}/{len(ok)} nets)\nseed {r['seed']}: {sig}")
    if R["act"] == "tanh":
        W1 = np.array([[10.0, 10.0], [10.0, 10.0]]); b1 = np.array([-5.0, -15.0])
        desc = [{"activation_gate": "OR", "contribution_gate": "OR", "w_out": 1.0},
                {"activation_gate": "AND", "contribution_gate": "NAND", "w_out": -1.0}]
    else:
        W1 = np.array([[1.0, 1.0], [1.0, 1.0]]); b1 = np.array([0.0, -1.0])
        desc = [{"activation_gate": "OR (A+B)", "contribution_gate": "count", "w_out": 1.0},
                {"activation_gate": "AND", "contribution_gate": "NAND", "w_out": -2.0}]
    _draw_circuit(axes[-1], desc, W1, b1, "REFERENCE (hand-built, not learned)\nOR - AND", color="tab:green")
    fig.suptitle(f"F. Circuit diagrams, {name}. Line width = |weight|, red = positive, blue = negative")
    _save(fig, out, "F_circuits.png")


def make_all(results, cfg, sim, out):
    fig_training(results, out)
    fig_truth_table(results, out)
    fig_activations(results, cfg, out)
    fig_similarity(results, cfg, sim, out)
    fig_causal(results, cfg, out)
    fig_circuits(results, cfg, out)
