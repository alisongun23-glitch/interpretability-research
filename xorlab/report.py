"""Writes results/REPORT.md from the experiment summary. All numbers are filled in
from the run; the rule-based verdicts are stated with the thresholds they use."""
import os


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def num(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def verdicts(s, sim):
    """Rule-based evidence for hypotheses A-D on the baseline architecture."""
    sig = s["signature_counts"]; n = sum(sig.values()) or 1
    top = max(sig.values()) / n if sig else 0
    t = s["transplant"]
    v = {}
    a_ok = top >= 0.8 and (t["direct_ok"] or 0) >= 0.8 and (t["sham_aligned_ok"] or 0) <= 0.2
    v["A"] = ("SUPPORTED" if a_ok else
              "PARTLY SUPPORTED" if top >= 0.5 and (t["aligned_ok"] or 0) >= 0.8 else "NOT SUPPORTED",
              f"most common circuit type covers {pct(top)} of converged nets (rule: >=80%); "
              f"direct transplant success {pct(t['direct_ok'])} (rule: >=80%); aligned {pct(t['aligned_ok'])}; "
              f"sham aligned {pct(t['sham_aligned_ok'])} (rule: <=20%).")
    multi = sum(1 for c in sig.values() if c / n >= 0.15)
    gap = None
    if t["direct_ok_same_signature"] is not None and t["direct_ok_diff_signature"] is not None:
        gap = t["direct_ok_same_signature"] - t["direct_ok_diff_signature"]
    v["B"] = ("SUPPORTED" if multi >= 2 else "NOT SUPPORTED",
              f"{len(sig)} distinct circuit types; {multi} each cover >=15% of nets (rule: >=2). "
              f"Direct transplant: same type {pct(t['direct_ok_same_signature'])} vs different type "
              f"{pct(t['direct_ok_diff_signature'])}" + (f" (gap {gap:+.2f})." if gap is not None else "."))
    dsim = sim["cka_diff_signature_mean"]
    hi_sim = dsim is not None and dsim > (sim["cka_untrained_mean"] or 0) + 0.05
    low_tr = (t["direct_ok_diff_signature"] or 0) < 0.5
    v["C"] = ("SUPPORTED" if hi_sim and low_tr else "NOT SUPPORTED",
              f"between nets with DIFFERENT circuit types, mean CKA is {num(dsim)} vs {num(sim['cka_untrained_mean'])} "
              f"for untrained nets (rule: > untrained + 0.05), and direct transplant between them works in "
              f"{pct(t['direct_ok_diff_signature'])} of pairs (rule: < 50%). Overall trained CKA is "
              f"{num(sim['cka_trained_mean'])}.")
    red = s["frac_multiple_minimal_circuits"] or 0
    v["D"] = ("SUPPORTED" if red >= 0.5 else "PARTLY SUPPORTED" if red >= 0.2 else "NOT SUPPORTED",
              f"{pct(red)} of converged nets contain more than one minimal sufficient circuit (rule: >=50% "
              f"supported, >=20% partly). In {pct(1 - (s['frac_no_single_unit_break'] or 0))} of nets at least one single unit is still indispensable, so the redundancy is partial. Ablating the primary circuit leaves {pct(s['causal_means']['ablate_circuit_acc'])} accuracy, against {pct(s['causal_means']['ablate_random_acc'])} for random same-size ablations.")
    return v


def write(summary, results, out):
    cfg = summary["config"]; base = cfg["baseline"]
    S = summary["architectures"]; sb = S[base]; sim = summary["similarity_baseline"]
    L = []
    w = L.append
    w("# XOR circuit discovery: experiment report\n")
    w(f"Generated automatically by `run.py` from an actual run ({summary['runtime_seconds']:.0f} s). "
      f"{cfg['seeds']} seeds per architecture; baseline architecture `{base}`. "
      "Every number below comes from `summary.json`; figures are in `figures/`.\n")
    w("## What is established before running anything\n")
    w("- A single tanh or ReLU unit of (A, B) is a monotone function of one straight line through the input "
      "square, so it can only realise *linearly separable* Boolean functions. XOR is not linearly separable. "
      "Therefore every network that solves XOR here **must** combine at least two hidden units. That XOR is "
      "composed from simpler parts is guaranteed by the architecture, not a discovery. The open questions are "
      "*which* parts, how consistently, and whether they are interchangeable.\n"
      "- Hidden units can be permuted and (for ReLU) rescaled without changing the function, so raw weight "
      "comparison is not meaningful; the analysis compares functions of units instead.\n")
    w("## Method in brief\n")
    w("1. **Training.** 2→H→1 networks, sigmoid output with binary cross-entropy, full-batch Adam "
      f"(lr {cfg['training']['lr']}), weights from N(0, {cfg['training']['init_std']}). A run stops at the first "
      f"step where all four inputs are correct with |logit| ≥ {cfg['training']['margin']} "
      f"(probability ≥ 0.88 / ≤ 0.12), or fails after {cfg['training']['max_steps']} steps. Failures are kept.\n"
      "2. **Unit labels.** Each hidden unit's activation over the 4 inputs is matched (by correlation) to the 14 "
      "non-constant Boolean functions; matches below "
      f"{cfg['interventions']['purity_threshold']} are labelled `mixed(...)`. A unit's *contribution* "
      "(activation × output weight) is labelled the same way; this merges equivalent readings (an AND unit with a "
      "negative output weight contributes the NAND pattern).\n"
      "3. **Circuit search.** For each converged net, all 2^H subsets of hidden units are tested exhaustively. "
      "A subset is *sufficient* if XOR stays fully correct when every other unit is **mean-ablated** (replaced by "
      "its mean over the 4 inputs). *Minimal* circuits are sufficient with no sufficient proper subset. "
      "Prespecified rule: the *primary* circuit is the smallest minimal circuit, ties broken by larger margin. "
      "Its *type* is the sorted list of its units' contribution labels.\n"
      "4. **Causal tests.** Ablating the circuit vs every other same-size subset; ablating everything else; "
      "activation patching (copy the circuit's activations from an input with the opposite XOR label into the "
      "clean run and check whether the output follows the source), with same-size and complement controls.\n"
      "5. **Transplants (no retraining).** The circuit's interface is defined as its *logit contribution* on the 4 "
      "inputs. Recipient output = recipient bias + recipient's non-circuit units + donor circuit's contribution. "
      "*Direct* uses the donor contribution as is. *Aligned* (secondary, reported separately) rescales it to the "
      "recipient circuit's own mean and spread over the 4 inputs, without looking at labels. Controls: native, "
      "ablation (recipient circuit mean-ablated), sham (the donor's **same units at initialisation**, matched in "
      "size and architecture), and a hand-built reference XOR circuit.\n"
      f"6. **Discovery vs confirmation.** Seeds {summary['discovery_seeds'][0]}–{summary['discovery_seeds'][-1]} "
      "are used to name the dominant circuit type; its frequency, ablation effect and transplant success are then "
      "measured on the held-out seeds.\n")

    w("## Results\n")
    w("### Training and behaviour\n")
    w("| architecture | converged | median steps | mean min margin (logit) | failed seeds |")
    w("|---|---|---|---|---|")
    for k, s in S.items():
        w(f"| {k} | {s['n_converged']}/{s['n_runs']} ({pct(s['success_rate'])}) | {num(s['median_converge_step'], 0)} "
          f"| {num(s['mean_min_margin'])} | {s['failed_seeds'] or '-'} |")
    w("\nPer-input probabilities and logits for every run are in `runs.csv`; see figure B.\n")

    w("### Circuit types found\n")
    for k, s in S.items():
        n = s["n_converged"] or 1
        items = ", ".join(f"`{sig}` {c} ({c / n:.0%})" for sig, c in s["signature_counts"].items())
        w(f"- **{k}:** {items or 'none'}. Primary circuit sizes {s['primary_circuit_sizes']}. "
          f"Nets with >1 minimal circuit: {pct(s['frac_multiple_minimal_circuits'])}. "
          f"Zero-ablation gives the same sufficiency verdict in {pct(s['zero_ablation_agreement'])}.")
    w("")
    w("### Causal tests (mean over converged nets)\n")
    w("| architecture | intact | ablate circuit | ablate random same size | ablate all else | patch circuit: follows source | patch random | patch all else |")
    w("|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        c = s["causal_means"]
        w(f"| {k} | {pct(c['intact_acc'])} | {pct(c['ablate_circuit_acc'])} | {pct(c['ablate_random_acc'])} "
          f"| {pct(c['ablate_complement_acc'])} | {pct(c['patch_circuit_flip'])} | {pct(c['patch_random_flip'])} "
          f"| {pct(c['patch_complement_flip'])} |")
    w("\nA constant output gets exactly 2 of 4 inputs right, so 50% means XOR is gone. "
      "Ablating the primary circuit can leave XOR largely intact when *another* minimal circuit exists; compare it "
      "with the random same-size column and the redundancy figures above. When the primary circuit is *all* units, 'random same size' and 'all else' "
      "controls are undefined (n/a).\n")

    w("### Transplants between independently trained nets (share of pairs fully correct)\n")
    w("| architecture | pairs | native | ablation control | direct | aligned | sham direct | sham aligned | reference direct | reference aligned | direct, same type | direct, different type |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        t = s["transplant"]
        w(f"| {k} | {t['n_pairs']} | {pct(t['native_ok'])} | {pct(t['ablated_ok'])} | {pct(t['direct_ok'])} "
          f"| {pct(t['aligned_ok'])} | {pct(t['sham_direct_ok'])} | {pct(t['sham_aligned_ok'])} "
          f"| {pct(t['reference_direct_ok'])} | {pct(t['reference_aligned_ok'])} "
          f"| {pct(t['direct_ok_same_signature'])} (n={t['n_same']}) | {pct(t['direct_ok_diff_signature'])} (n={t['n_diff']}) |")
    w("\nRead each row against its **ablation control**: if the recipient still computes XOR with its own circuit "
      "removed (high ablation control, as in wide networks), a successful transplant shows little. The informative "
      "rows are those where the ablation control is low and the sham control stays low.")
    x = summary["cross_activation_transplant"]
    w(f"\nAcross activation functions (tanh-4 ↔ relu-4, {x['n_pairs']} pairs): direct {pct(x['direct_ok'])}, "
      f"aligned {pct(x['aligned_ok'])}, sham aligned {pct(x['sham_aligned_ok'])}, reference aligned "
      f"{pct(x['reference_aligned_ok'])}.\n")

    w(f"### Representational similarity ({base})\n")
    w(f"- Mean linear CKA between trained nets: **{num(sim['cka_trained_mean'])}**; between the same nets at "
      f"initialisation: {num(sim['cka_untrained_mean'])}.")
    w(f"- Same circuit type: {num(sim['cka_same_signature_mean'])}; different type: {num(sim['cka_diff_signature_mean'])}.")
    w(f"- Mean correlation of primary-circuit logit contributions across nets: {num(sim['contribution_corr_mean'])}.")
    w("- With 4 inputs, centred representations have at most 3 dimensions, so CKA is coarse. It measures "
      "similarity of *shape* only; it does not show that units play the same causal role.\n")
    g = summary["unit_gate_frequencies"][base]
    tot_t = sum(g["trained"].values()) or 1; tot_u = sum(g["untrained"].values()) or 1
    keys = sorted(set(g["trained"]) | set(g["untrained"]), key=lambda k: -g["trained"].get(k, 0))[:10]
    w("Unit activation labels, trained vs the same nets at initialisation (null model):\n")
    w("| label | trained | untrained |\n|---|---|---|")
    for k in keys:
        w(f"| {k} | {g['trained'].get(k, 0) / tot_t:.0%} | {g['untrained'].get(k, 0) / tot_u:.0%} |")
    w("")

    if "confirmation" in sb:
        c = sb["confirmation"]
        w("### Discovery → confirmation\n")
        w(f"Dominant type in the discovery seeds: `{c['dominant_signature_in_discovery']}` "
          f"({pct(c['share_in_discovery'])} of {c['n_discovery']} nets). On {c['n_confirmation']} held-out nets it "
          f"accounts for {pct(c['share_in_confirmation'])}. Held-out causal checks: ablating the circuit leaves "
          f"{pct(c['confirm_ablate_circuit_acc'])} accuracy (random same-size: {pct(c['confirm_ablate_random_acc'])}); "
          f"direct transplant {pct(c['confirm_direct_ok'])}, aligned {pct(c['confirm_aligned_ok'])}, sham aligned "
          f"{pct(c['confirm_sham_aligned_ok'])}.\n")

    w(f"## Evidence for hypotheses A–D ({base})\n")
    w("The thresholds are fixed in `xorlab/report.py`. They summarise the evidence; they are not significance tests.\n")
    names = {"A": "shared, interchangeable circuits", "B": "different mechanisms, same behaviour",
             "C": "shared features, different causal use", "D": "distributed / redundant computation"}
    for k, (verdict, why) in verdicts(sb, sim).items():
        w(f"- **{k} ({names[k]}): {verdict}.** {why}")
    w("\nThese are not exclusive: several can hold at once, and the answer differs by architecture (see the tables).\n")

    w("## Changes from the original specification, and why\n")
    w("- **Circuit interface = logit contribution.** In a one-hidden-layer net a 'circuit' only talks to the "
      "output through its weighted sum, so this is the exact interface. It makes direct transplants well defined.\n"
      "- **Mean ablation by default** (zero ablation reported as a robustness check): zeroing a tanh unit also "
      "removes its constant offset, which can break XOR for reasons unrelated to the computation.\n"
      "- **Sham = donor's own circuit units before training.** Same size, same architecture, same initial "
      "weights; only training differs. This is a tighter control than random units.\n"
      "- **Aligned transplant uses moment matching**, not a fitted map, so it cannot simply learn the answer. "
      "Powerful fitted maps can make any circuit look transplantable (Sutter et al., NeurIPS 2025).\n"
      "- **Contribution-based circuit types** to avoid counting `AND with negative weight` and `NAND with positive "
      "weight` as different mechanisms.\n")
    w("## Limitations\n")
    w("- Four inputs: any similarity metric is low-resolution, and an exhaustive truth table means behaviour "
      "cannot distinguish mechanisms; only the internal analyses can.\n"
      "- Boolean labels are a lens. ReLU units take graded values (e.g. counting A+B), and `mixed(...)` labels "
      "mark units that no gate describes well.\n"
      "- 'Minimal circuit' depends on the ablation choice. Different ablations can give different circuits.\n"
      "- Results hold for these architectures, initialisation and optimiser. Nothing here shows universality "
      "beyond XOR in 2→H→1 networks.\n")
    w("## Files\n")
    w("`summary.json` (all headline numbers), `runs.csv` (per seed, per input), `circuits.json` (every unit, "
      "every minimal circuit), `causal.json`, `transplants.csv` (every donor→recipient pair), `similarity.npz`, "
      "`checkpoints/*.npz` (initial and final weights), `training_logs/*.npz`, `figures/A–F`.\n")
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))
