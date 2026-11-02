# Pre-registration: is a logic gate in a transformer the same as a gate neuron?

Written 2026-10-14 21:38 EDT, before any model in this study was trained or analysed.
Nothing below is changed after the runs; deviations go in `REPORT.md` under "Deviations".

## Question

A 1-layer transformer that adds binary numbers lowest bit first must compute the carry into bit k.
Over the previous digit, (a_{k-1}, b_{k-1}, s_{k-1}), that carry is a single threshold function:
c_k = [a + b − s ≥ 1]. A gate neuron from `results_minigates/library.json` (one sigmoid neuron for
AND/OR/NAND/NOR, two for XOR, trained to margin 4) is the clean version of such a function. Where is
the carry inside the transformer, does it look like a gate neuron, and can the two be swapped?

## Models

- **B** (flat format, learned positions): `TinyGPT(d_vocab 4, n_ctx 33, d_model 64, 4 heads, 1 layer,
  d_mlp 256)`, format `a0..a9 + b0..b9 = s0..s10`. Seed 1 is `results_transformer/lsb-L1.npz` (read only);
  seeds 0, 2, 3, 4 are trained with the same recipe (2,500 steps, batch 128, AdamW wd 0.01, lr 1e-3 with
  100-step warm-up and cosine decay to 0.1×, loss on the sum tokens only).
- **C2** (our own design; the original is unknown): same width, no position embedding, a learned
  per-head bias `rel[h, min(i − j, 63)]` on the attention scores, format of triples
  `a0 b0 s0 a1 b1 s1 … a_n b_n s_n` with a_n = b_n = 0. s_k is predicted at position 3k+1. Trained at
  n = 10 only, seeds 0, 1, 2, up to 4,000 steps, same optimizer recipe.

q_k is the position that predicts s_k. In B its token is s_{k−1} (`=` for k = 0); in C2 it is b_k.

## Measures (fixed now)

- **Accuracy**: 10,000 random problems, free greedy generation, exact sum, Wilson 95% CI. C2 also at
  12, 16 and 20 bits.
- **Probes**: L2 logistic regression (IRLS), 8,000 train / 4,000 test problems, teacher-forced.
  "Readable" = test accuracy pooled over k = 1..10.
- **Controls**: the same model class and seed, untrained ("random-init"); probes on shuffled labels.
- **Flip directions** (no labels): differences x_src − x_base of resid_mid at q_k over base/source pairs
  that agree on every bit ≥ k and have the opposite carry into bit k. Top-4 PCs, with the sign of PC1
  set so base c = 0 → source c = 1 is positive. B uses one set per k, because it has learned positions.
  C2 uses one set pooled over k, because it has no absolute position and must also run past 10 bits.
- **Interchange success**: among pairs whose unpatched base output at q_k is correct, the share whose
  output becomes the gate adder's s_k with carry k flipped.
- **Gate table** (6b): z = resid_mid at q_k projected on PC1. Two levels from 1-D 2-means with no labels;
  threshold = midpoint. Normalise u = (z − lo)/(hi − lo), with lo and hi per k for B and pooled for C2.
  Table = mean u over each of the 8 (a_{k−1}, b_{k−1}, s_{k−1}) combos. Entry margin = |mean u − 0.5| /
  within-combo std of u; gate margin = min over the 8 entries. The trilinear gate interpolates the
  8 entries, clipped to [0, 1] (the two levels are the rails). Restoring is checked as
  `gatesnake.gates.restores` does: every input within ε of a rail must give an output within ε of the
  rail of the snapped table, on a 9-point grid per axis.
- **Gate adder**: ripple adder of library XOR, AND and OR neurons, analog, nothing snapped. s_i =
  XOR(XOR(a_i, b_i), c_i); c_{i+1} = OR(AND(a_i, b_i), AND(XOR(a_i, b_i), c_i)).

## Predictions

| # | prediction | pass if |
|---|---|---|
| P0 | every model exact at 10 bits | ≥ 99% (else dropped) |
| P1 | attention gathers inputs at q_k | for each of a_k, b_k, a_{k-1}, b_{k-1} some head puts ≥ 0.25 on it |
| P2 | c_k first readable at resid_mid (B) | resid_pre ≤ 80% (ceiling 75%), resid_mid ≥ 99% |
| P3 | readability is not evidence | random-init resid_mid probe ≥ 95%; shuffled ≤ majority + 2 pts |
| P4a | no single clean carry neuron | no MLP unit with R² ≥ 0.9 for the carry table; 99% readout needs ≥ 3 units |
| P4b | sum built from count thresholds, not XOR | ≥ 80% of units with R² ≥ 0.8 fit a threshold of a_k + b_k + c_k |
| P5 | model uses the carry direction | rank-1 (PC1) interchange success ≥ 80% every seed; random ≤ 10%; random-init ≤ 10% |
| P6 | exact gate carry can replace the model's | removed: < 50%; gate carry in: ≥ 99%; flipped: ≤ 5% |
| P6-C2 | exact carry does not fix length | 16-bit accuracy gains < 10 points |
| P7a | trained carry gate is cleaner than chance structure | trained PC1 gate margin ≥ 4 every seed and ≥ 3× the random-init margin |
| P7b | the transformer's carry works as an analog gate | trilinear gate restores at ε = 0.05; analog 64-bit ripple 100% on 100k sums |
| P7c | sanity | snapped PC1 table = [a+b−s≥1] (snapped ripple 100%, not counted as evidence) |
| P8 | a hand-built XOR module can replace the MLP | 10 bits ≥ 99% every seed; random direction ≤ 5%; wrong wiring ≤ 5% |
| P8-C2 | module does not fix length | 16-bit gain < 10 points (exploratory) |

Details fixed with the table:

- P1 uses attention from q_k averaged over k = 1..10 and over test problems, per head.
- P2 and P3 use c_k at q_k, pooled over k = 1..10.
- P4 uses mlp.hook_post at q_k on test problems, pooled over k = 1..10, and only units active on at least 1% of them.
  - R² is from the best fit act ≈ α + β·f over all 254 non-constant Boolean f of the three inputs.
  - "Threshold of a_k + b_k + c_k" means f = [count ≥ t] or its negation, t ∈ {1, 2, 3}.
  - The readout count comes from greedy forward selection by least squares on train, read as test
    accuracy ≥ 99%, capped at 40 units.
- P5, P6 and P8 count every seed of each family that passes P0.
- P6 removes the model's own carry by projecting out PC1–4 at every q_k (k ≥ 1) during generation.
  The written value is the class mean of resid_mid (train split) for the gate adder's carry,
  projected on that subspace.
- P8 replaces the MLP output at every q_k with u · m(s_k):
  - u = unit(ln_f.w ⊙ (W_U[:,1] − W_U[:,0])).
  - m(s) interpolates m_0 and m_1, the real MLP output's mean projection on u given s_k.
  - s_k comes from library XOR neurons on snapped probe readings of a_k, b_k and c_k.
  - At k = 0, s_0 = XOR(a_0, b_0). At the top position the output is ĉ_n.

## Notes written before any run

- **P7b, restoring half.** A trilinear interpolation of an exact 0/1 carry table has gain 2 at the
  corners where two neighbouring corners take the other value. Examples: (0,1,1), whose neighbours
  (1,1,1) and (0,1,0) are both 1, and likewise (1,0,1). Inputs ε from such a corner give an output about
  2ε from the rail. So even a perfect table should fail "restores at ε = 0.05". It is still reported as
  pre-registered. The analog 64-bit ripple half is not affected, because the library XOR that feeds the
  next carry restores.
- **P2, the "ceiling 75%".** In B, s_{k−1} alone predicts c_k at 75%: c_k = 1 for 3 of the 4 (a, b, c)
  states with s = 0, and for 1 of the 4 with s = 1.
- **P7c** is a sanity check. Any snapped table that is exact equals the carry table, so it passes
  automatically and is not counted as evidence.
