# Pre-registration: Cammarata's clean-neuron test × the carry direction

Written 2026-10-30, before any model of this study was trained and before stage 0 was run. Nothing below
changes after the runs; deviations go in `REPORT.md` under "Deviations". `clean_carry.py` stores this
file's SHA-256 in `results.json` and refuses to train or analyse if it is missing or has changed.

## Question

Cammarata ("Neural Networks Are Already Clean", Goodfire essay, 15 Oct 2026) claims most neurons do one
main job, that apparent mixing comes from bad tools, bad explanations or "recursive generalization",
that superposition should be the last hypothesis, and that the decisive test is watching neurons grow.
The 1-layer B adders of `results_transformer_gates` carry their carry on a residual-stream direction
(PC1 of counterfactual differences at resid_mid), not on a neuron, while their clean MLP units are count
thresholds (OR3, MAJ, AND3) in 4 of 5 seeds and absent in B:3. Do the clean units read that direction,
which forms first in training, does squeezing the MLP change one without the other, and do clean units
and a clean direction go together across seeds?

## Data and models

- Problems: `TG.problems` with `default_rng(7)`: 8,000 reference ("train") problems, then 4,000 test
  problems, 10-bit operands. Labels and interchange use the test set; snap means and thresholds use the
  reference set.
- Arm A (task): `TinyGPT(d_vocab 4, n_ctx 33, d_model 64, 4 heads, 1 layer, d_mlp w, seed s)`, format
  `a0..a9 + b0..b9 = s0..s10`, the `TG.train` recipe (2,500 steps, batch 128, AdamW wd 0.01, lr 1e-3,
  100 warm-up steps, cosine to 0.1×, loss on the sum tokens, data stream `default_rng([s, 1000])`).
  Widths w ∈ {256, 32, 16, 8, 4} × seeds 0–4, plus w = 256 × seeds 5–9.
- Arm R (positive control): same model and format, but the token after q_k is t_k = T_k(a_k, b_k, t_{k−1})
  (t_{−1} = 0, a_10 = b_10 = 0). T_0..T_10 are 11 distinct balanced 3-input Boolean functions that depend on
  all three inputs, drawn once with `default_rng(4242)`. Widths {256, 32, 16, 8} × seeds 0–4.
- Arm 0 (null): the step-0 checkpoint (initialisation) of every arm-A and arm-R model.
- Saved models (stage 0): `results_transformer_gates/models/B-seed{0,2,3,4}.npz` and
  `results_transformer/lsb-L1.npz` (B:1), read only.
- Checkpoints (parameters only): w ∈ {256, 16}, seeds 0–4: steps 0, 25, 50, …, 1000, then 1250, …, 2500.
  Other models: steps 0, 250, …, 2500.

## Definitions

- **Units**: the w ReLU units (`blocks.0.mlp.hook_post`) at q_k, k = 1..9. **Active at k**: non-zero on
  ≥ 5% of test problems at k and std > 1e-6.
- **Inputs at k**: arm A (a_k, b_k, c_k), combo index 4a + 2b + c; arm R (a_k, b_k, t_{k−1}).
- **Label of an active (unit, k)**, first rule that applies:
  1. Boolean: best R² of act ≈ α + β·f(combo) over the 254 non-constant f (`TG.bool_fit`) ≥ 0.8. If f is a
     count threshold [a + b + c ≥ t] (t = 1, 2, 3) or its complement, the label is threshold (t, sign);
     otherwise the label is f. f depending on one input only is flagged "dictator".
  2. Count ramp (arm A only): η² of the count n = a + b + c (4 groups) ≥ 0.9 and the 4 group means are
     monotone. Label threshold (θ, sign), θ = the count at the largest step between consecutive means
     (rising: sign +; falling: sign −).
  3. Otherwise "unclear".
- **Unit verdict** over its active positions (units active nowhere are dead and excluded): any active
  position unclear → **unclear**; one distinct label → **one job**; arm A only: exactly two labels, both
  thresholds of the same sign with t differing by 1 → **two related jobs**; anything else → **packed**.
- **Cleanliness** = (one job + two related jobs) / non-dead units. **Cleanliness without dictators**:
  the same with dictator labels treated as unclear. **Packed share** = packed / units whose active
  positions are all labelled and number ≥ 2 ("multi-position units"). **Positions per unit** = mean
  number of active positions over non-dead units.
- **Δ** = share of same-label pairs among pairs of labelled positions within units, minus its mean under
  1,000 permutations of the labels across units within each position (200 on intermediate checkpoints).
- **Clean threshold pair**: an active (unit, k) whose label is a threshold.
- **Carry direction** v_k: PC1 of `TG.flip_dirs` (resid_mid at q_k, pairs with opposite carry into k,
  signed so carry 0 → 1 is positive), k = 1..9; 2,000 pairs per k on final models and saved models,
  500 on intermediate checkpoints.
- **Interchange**: rank-1 patch x ← x + v vᵀ(x_src − x) at resid_mid q_k on opposite-carry pairs; success
  = output equals a_k ⊕ b_k ⊕ (1 − c_k), among pairs whose unpatched output is right; pooled over
  k = 1..9. 1,000 pairs per k on final and saved models, 300 on intermediate checkpoints.
- **Read cosine** r_jk = cos(w_j, v_k) after removing the mean component of both, with
  w_j = ln2.w ⊙ W_in[:, j] (the MLP reads LN2(resid_mid)).
- **Debiased PC1 margin**: `followup_tests.t6_margins` "pc1_minus_current_digits" gate margin (the PC1
  coordinate with each (k, a_k, b_k) group mean removed; inputs only, no carry labels).
- **Learning**: arm A exact-sum accuracy ≥ 99% (greedy generation, 2,000 test problems); arm R
  teacher-forced per-bit accuracy ≥ 95% over k = 1..9. A width "learns" if ≥ 3 of its 5 seeds learn.
- **Spearman ρ**: on ranks with average ties.

## Validity gates (a failed gate gives "no verdict" for the hypotheses it covers)

| gate | covers | pass if |
|---|---|---|
| V1 reproducibility | H2, H3, D2, D3, D4 | retrained arm-A w = 256 seeds 0, 2, 3, 4 match the saved models: max parameter difference ≤ 1e-3, or (exact accuracy within 0.5 points, clean-threshold-pair count within 10%, interchange within 2 points) |
| V2 sensitivity | H3 | arm R packed share ≥ 0.30 in ≥ half of its learning models at widths ≤ 16 (no learning model there = fail) |
| V3 specificity | H1, H2, H3, D1a | arm-0 (step-0) w = 256 seeds 0–4: mean cleanliness without dictators ≤ 0.10, and read-cosine AUC (D1a statistic) ≤ 0.6 in every model with ≥ 20 clean threshold pairs |
| V4 privileged basis | H1, H2, H3 | multiplying the post-ReLU activations by a random orthogonal matrix (10 draws) gives mean cleanliness ≤ half the unrotated value, averaged over saved B:0–B:4 |
| V5 clamp sanity | D1c | holding every unit at its own unpatched value leaves teacher-forced accuracy at q_k unchanged (≥ 99.9% agreement) |

## Hypotheses: Cammarata

| # | data | pass | fail | otherwise |
|---|---|---|---|---|
| H1 | saved B:0–B:4 | B:3 cleanliness ≥ 0.70 | B:3 packed share ≥ 0.30 and Δ ≤ 0.05 | inconclusive |
| H2 | arm A, w = 256 and 16, dense checkpoints | in ≥ 4 of 5 seeds at both widths: first label = final label in ≥ 80% of final one-job units and joining positions carry the unit's first label at ≥ 0.20 above null | first = final < 60% or excess < 0.05 in ≥ 2 of 5 seeds at either width | inconclusive |
| H3 | arm A, all widths | every learning width: mean cleanliness ≥ 0.70 and mean Δ ≥ 0.20; ρ(width, positions per unit) ≤ −0.8 over learning widths (if ≥ 3); at the smallest learning width, shared (≥ 2 positions, one job) units for each of t = 1, 2, 3 exist in some seed | packed share ≥ 0.30 with Δ < 0.05 in ≥ 3 of 5 seeds at some learning width ≤ 16 | inconclusive |
| H4 | arm A, every learning model, edits at q_k (k = 1..9) during generation | in ≥ 80% of models: zeroing each unit's weakest 75% of non-zero activations costs ≤ 2 points, zeroing its strongest activations of matched mass costs ≥ 50 points, and snapping each unit to its per-(k, count) mean keeps ≥ 99% | count snap < 95% or weak removal > 5 points in ≥ 50% of models | inconclusive |

H1 notes: "first label" in H2 is the most common label at the first checkpoint where the unit has any
labelled position; "joining positions" are a final one-job unit's labelled positions that were not
labelled at that checkpoint, scored at the checkpoint where each is first labelled; the null is the
share of labelled (unit, k) pairs at that k and checkpoint carrying the same label. Seeds with fewer than
5 final one-job units at a width are skipped. H4 also reports a two-level snap (per unit and k, 2-means
levels): ≥ 99% = digital gates; < 90% while the count snap holds ≥ 99% = clean but analog.

## Hypotheses: the carry direction × clean neurons

| # | data | pass | fail | otherwise |
|---|---|---|---|---|
| D1a readers | saved B:0–B:4 + arm A w = 256 seeds 5–9 ("D1 models"); models with ≥ 20 clean threshold pairs | AUC of \|r_jk\| (clean threshold pairs vs other active pairs) ≥ 0.70 in ≥ 80% of eligible models | AUC < 0.60 in ≥ 50% | inconclusive |
| D1b sign | clean threshold pairs of the eligible D1 models, pooled | sign(r_jk) = threshold sign in ≥ 90% | < 75% | inconclusive |
| D1c mediation | eligible D1 models | interchange with the clean threshold units at q_k held at their unpatched values drops by ≥ 0.50, and holding a size-matched random set of other active units (5 draws, mean) drops by ≤ 0.10, in ≥ 80% of models | clean-unit drop < 0.20 in ≥ 50% | inconclusive |
| D2 order | arm A w = 256 and 16, seeds 0–4 | t_dir ≤ t_clean in ≥ 4 of 5 seeds at each width ("writer before readers") | t_dir > t_clean in ≥ 4 of 5 seeds at each width ("readers first") | mixed |
| D3a survival | arm A, learning models, all widths | no learning width has ≥ 2 seeds with interchange < 0.80 | otherwise | – |
| D3b independence | same | \|ρ(cleanliness, interchange)\| < 0.3: independent | ρ ≥ 0.3: go together; ρ ≤ −0.3: trade off | – |
| D4 seeds | arm A w = 256 seeds 0–9, learning | ρ(cleanliness, debiased PC1 margin) ≥ 0.6: go together | \|ρ\| ≤ 0.2: independent | inconclusive |
| D5 basis | saved models + learning arm-A models, all (model, k) | both max v_i² and participation ratio 1/Σv_i⁴ inside the 2.5–97.5% band of 1,000 random unit vectors in R^64 for ≥ 90% of (model, k): not neuron-aligned | < 50% inside | inconclusive |

D2 definitions: t_dir = first checkpoint with interchange ≥ 0.80; t_clean = first checkpoint with ≥ 50%
of the final clean-threshold-pair count; ties count as t_dir ≤ t_clean; a seed with either undefined, or
fewer than 20 final clean threshold pairs, is skipped (a width with fewer than 3 usable seeds: no
verdict). Reported alongside, not judged: per (unit, k) Spearman ρ between the step its final label first
appears and the step |r_jk| first exceeds 0.2 (pre-registered guess ≥ 0.5).

## Exploratory (reported, never judged)

Everything else in `results.json`: per-k breakdowns, the arm-R squeeze, label maps over training,
reader cosines of non-threshold units, D1 on retrained w = 256 seeds 0–4.
