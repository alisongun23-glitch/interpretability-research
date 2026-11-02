# Is a logic gate in a transformer the same as a gate neuron?

Produced by `python transformer_gates.py train` (7 models, about 4 minutes on 10 cores) and
`python transformer_gates.py analyse` (about 32 minutes). The predictions were written to `PREREG.md`
before any model was trained. Every number below is in `results.json`, and figures 1–5 are in `figures/`.
Checks: `python tests/test_circuits.py` and `python tests/check_all.py`, both all PASS.

## The answer in plain English

**The carry is a direction in the residual stream, not a neuron, and it is not clean enough to be a gate.**

- **The model really uses one direction for the carry.**
  - Take the difference between two problems that have the opposite carry into bit k, and find its first
    principal component (no labels needed).
  - Copy only that one coordinate from one problem into another at the query of s_k. The model's next
    bit then changes exactly as flipping the carry would: 97–100% of the time in all five B models,
    and 88–99% in the three C2 models.
  - Random directions do almost nothing: at most 2% for one random direction, 5% for four.
- **No neuron looks like the carry gate.**
  - A one-neuron carry gate trained the way the library gates are fits the carry's truth table with
    R² = 1.000. Its two output levels are 104 within-level standard deviations apart.
  - The best MLP unit in any model fits the carry table with R² 0.30–0.40.
  - The direction is also a noisy gate. Projected onto it, the eight (a, b, s) input combinations line
    up correctly on average: the snapped table equals the carry table in every model. But within each
    combination the projection spreads by more than the distance to the threshold: the margin is
    0.4–0.7 standard deviations.
- **So gate and transformer cannot simply be swapped.**
  - Writing the gate adder's exact carry into the model, as the class mean along the carry directions,
    breaks every model (≤ 0.2% exact sums).
  - The model reads its carry *relative* to the rest of the residual, not as an absolute level.
    Pinning the coordinate gives chance; shifting it by the class gap flips the bit.
  - Going the other way, the transformer's carry turned into an analog gate does not restore signals.
    Yet a ripple adder that uses it, with the library XOR neuron restoring every sum bit, still adds
    64-bit numbers exactly with the B models' gates. It does not with C2's (69–98%).
- **What the MLP builds instead.**
  - In B, the units that clearly track the sum's inputs (a_k, b_k, c_k) are threshold units of their
    count: OR3, majority and AND3. That is the textbook way to get a 3-input XOR from threshold neurons,
    XOR3 = [n ≥ 1] − [n ≥ 2] + [n ≥ 3].
  - In C2 they mostly detect single input combinations instead.
- **Scorecard.** Of the pre-registered predictions, only P0 (every model accurate) and P4a (no single
  clean carry neuron) pass for both families, plus the sanity check P7c. P6-C2 and P8-C2 pass only
  vacuously. Six predictions fail: P1, P2, P3, P5, P6 and P8. See the scorecard below for what each
  failure means.

## Setup

| family | model | seeds | training | format |
|---|---|---|---|---|
| B | `TinyGPT`, 1 layer, d_model 64, 4 heads, d_mlp 256, learned positions | 0–4 (seed 1 = `results_transformer/lsb-L1.npz`) | 2,500 steps | `a0..a9 + b0..b9 = s0..s10` |
| C2 | same width, **no position embedding**, learned bias per head and relative offset | 0, 1, 2 | 4,000 steps | `a0 b0 s0 a1 b1 s1 … a_n b_n s_n`, a_n = b_n = 0 |

- Both families use the recipe of `run_transformer.py`: batches of 128 random 10-bit problems, AdamW
  (wd 0.01), lr 1e-3 with warm-up and cosine decay, and loss on the sum tokens only.
- q_k is the position that predicts s_k. Its token is s_{k−1} in B and b_k in C2.
- In C2 every input of s_k sits at a fixed offset from q_k (b_k 0, a_k 1, s_{k−1} 2, b_{k−1} 3,
  a_{k−1} 4). This makes a 1-layer, relative-position-only model possible. C2 is our own design; the
  original is unknown.
- Over the previous digit the carry is one threshold function: c_k = [a_{k−1} + b_{k−1} − s_{k−1} ≥ 1]
  (`tests/test_circuits.py` checks this). The gate it is compared with is the library gate neuron of
  `results_minigates/library.json`: one sigmoid neuron, trained to margin 4.

## 1. Every model adds exactly (P0: pass)

Exact-sum accuracy on 10,000 random problems, free greedy generation, with Wilson 95% intervals:

| model | 10 bits | 12 bits | 16 bits | 20 bits |
|---|---|---|---|---|
| B:0 … B:4 | 100.00% [99.96, 100] each | – | – | – |
| C2:0 | 100.00% [99.96, 100] | 76.7% [75.9, 77.5] | 21.1% [20.3, 21.9] | 3.5% [3.2, 3.9] |
| C2:1 | 99.99% [99.94, 100] | 81.1% [80.3, 81.8] | 30.8% [29.9, 31.7] | 7.2% [6.7, 7.7] |
| C2:2 | 99.95% [99.88, 99.98] | 77.2% [76.4, 78.0] | 22.2% [21.4, 23.0] | 4.0% [3.6, 4.4] |

All eight models are kept. C2 fails on longer sums, as expected: relative offsets above 31 never
receive a gradient during training, so their biases stay at 0, and longer inputs leak attention to
far keys. Section 2 shows the same leak already inside the training length.

## 2. Where the carry is (P1, P2, P3: fail)

**Attention (P1, figure 1).** Largest mean attention any head puts from q_k on each input of s_k
(k = 1..10):

| model | a_k | b_k | a_{k−1} | b_{k−1} |
|---|---|---|---|---|
| B:0 / B:1 / B:2 / B:3 / B:4 | 0.30 / 0.27 / 0.30 / 0.23 / 0.34 | 0.30 / 0.27 / 0.29 / 0.22 / 0.36 | 0.14 / 0.14 / 0.17 / 0.13 / 0.20 | 0.14 / 0.14 / 0.18 / 0.12 / 0.21 |
| C2:0 / C2:1 / C2:2 | 0.17 / 0.18 / 0.18 | 0.14 / 0.11 / 0.09 | 0.14 / 0.12 / 0.13 | 0.14 / 0.12 / 0.13 |

- In B one head reads the current digits (about 0.3 each) and spreads a further 0.14–0.21 over the
  previous ones. In C2 the attention is diffuse.
- No model reaches 0.25 on a_{k−1} and b_{k−1}, so P1 fails. Its threshold assumed each input gets
  copied by its own head. The models instead *average* a and b of a digit with equal weight (next
  paragraph).

**Probes for the carry at q_k (P2, P3, figure 2).** L2-logistic probes, 8,000 train / 4,000 test
problems, pooled over k = 1..10. Each cell is trained / random-init:

| hook | B (5 seeds) | C2 (3 seeds) |
|---|---|---|
| resid_pre | 74.8% / 74.8% | 55.5% / 55.5% |
| attn hook_z | 93.5–96.3% / 81–82% | 69–85% / 59% |
| resid_mid | 91.8–95.2% / 78–79% | 73–85% / 59% |
| mlp hook_post | 93.6–99.9% / 82–83% | 90–98% / 60% |
| resid_post | 89.5–95.3% / 78% | 87–97% / 58–59% |

- Shuffled labels at resid_mid give 55.5–55.8%, against a majority rate of 55.2%.
- **P2 fails on its pooled criterion, but not per bit.** In B the query token s_{k−1} alone gives the
  pre-registered 75% ceiling at resid_pre. After attention the carry is 92–95% readable pooled over k,
  short of the 99% bar. Probed separately for each k it is 98–100% at resid_mid (median 100%). With
  learned positions, the carry sits in a slightly different direction at each bit, so a single
  pooled linear readout loses a few percent.
- **P3 fails the other way round from what was predicted.** Readability *does* separate trained from
  untrained models here. Random-init models read the carry at 78–79% (B) and 59% (C2) at resid_mid,
  far below the trained models. The earlier report found no difference; it probed only resid_pre and
  the final layer with small least-squares probes.
- **C2 is less readable, and less so further along the sum.** The per-k carry probe at resid_mid falls
  from 100% at k = 1 to 70–87% at k = 9. It is 100% again at the top bit, where the carry is the
  answer itself. Later queries see more keys, and the relative-position attention leaks more of its
  weight onto them. This is the same failure that grows past 10 bits.

**The attention delivers sums of bits, not bits** (exploratory reading of the same probes):

- At q_k in B, a_{k−1} and b_{k−1} are each only 69–78% readable at resid_mid, and a_k, b_k 80–85%.
  Yet the carry, which needs both previous bits, is 92–95% readable.
- If attention hands over only a + b (it weights a and b equally), each bit is recoverable exactly
  when the sum is 0 or 2. That is 75% of the time, close to what the probes find.
- The carry [a + b − s ≥ 1] is linear in that sum and the query token s. So the model never needs the
  separate bits.

## 3. Does any neuron look like a gate neuron? (P4a: pass, P4b: fail)

256 ReLU units at q_k, test problems pooled over k = 1..10. 213–250 units are active in B, and
122–153 in C2. A unit's R² is that of the best fit act ≈ α + β·f over Boolean functions f of the
three inputs.

| | library AND / OR / NAND / NOR / XOR neuron | 3-input carry neuron (margin 4) | best B unit | best C2 unit |
|---|---|---|---|---|
| R² for the carry table | – | 1.000 | 0.32–0.38 | 0.30–0.40 |
| best R² over all functions of (a_{k−1}, b_{k−1}, s_{k−1}) | 1.000 | 1.000 | 0.34–0.52 (median unit 0.08–0.15) | 0.35–0.40 (median unit 0.10–0.13) |
| level gap / within-level sd | 122 / 233 / 123 / 238 / 158 | 104 | 1.4–2.1 | 1.5–2.1 |

The inputs to the library neurons are drawn within 0.05 of their rails. The carry neuron has weights
(8.5, 8.4, −8.1), bias −4.4, and restores at ε = 0.05.

- **P4a passes.** No unit in any model fits the carry table with R² ≥ 0.9, and no subset of up to 40
  units gives a 99% linear readout of c_k (best 88–96%).
- The mean activations of the best units do follow the carry pattern (figure 3), but each is buried
  in variation from everything else the unit responds to.
- The sum is far easier to read out: 3–4 units give 99% for s_k in four B models (20 in B:3), and
  6–9 in C2.

**What the units compute (P4b).** Units whose best fit on the sum's inputs (a_k, b_k, c_k) reaches
R² ≥ 0.8:

| model | units | of which count thresholds | functions |
|---|---|---|---|
| B:0, B:1, B:2, B:4 | 96, 100, 98, 78 | 100% | OR3 25–40, majority 31–41, AND3 14–31 |
| B:3 | 0 | – | no clean units; its sum is spread thinly (20 units for a 99% readout) |
| C2:0, C2:1, C2:2 | 33, 40, 7 | 64%, 3%, 0% | mostly single-combination detectors, e.g. ¬a ∧ b ∧ ¬c |

The breakdown by function (last column) was checked on the saved models; `results.json` keeps only
the shares.

- Four of five B models build the parity of three bits exactly as a threshold-logic textbook would,
  from units that fire when the count of ones reaches 1, 2 or 3.
- P4b still fails for B, because B:3 has no such units, and for C2.
- Random-init B models have no such units. Random-init C2 models have 87–112, and every one of them
  computes b_k, the token at the query (checked on the saved models).

## 4. Does the model use the carry direction? (P5: fail, on its random-init control only)

Interchange at resid_mid, q_k, k = 1..10, 1,000 base/source pairs per k:

- Each pair agrees on every bit ≥ k and has the opposite carry into k. The patch is
  x ← x + P(x_src − x).
- Success means the output becomes the gate adder's s_k with carry k flipped. It is counted among the
  pairs whose unpatched output is right: about 10,000 pairs per model for the trained models.

| model | PC1 (rank 1) | PC1–4 | label-trained probe direction | random rank 1 / rank 4 (mean of 10, max) | random-init model, own PC1 |
|---|---|---|---|---|---|
| B:0 | 100.0% | 100.0% | 0.0% | 0.0% / 0.0% (max 0.1%) | 12.1% |
| B:1 | 100.0% | 100.0% | 0.0% | 0.0% / 0.0% | 9.1% |
| B:2 | 100.0% | 100.0% | 0.0% | 0.0% / 0.0% (max 0.1%) | 39.9% |
| B:3 | 97.4% [97.1, 97.7] | 100.0% | 0.0% | 0.2% / 0.2% (max 1.7%) | 28.0% |
| B:4 | 100.0% | 100.0% | 0.0% | 0.0% / 0.0% | 19.2% |
| C2:0 | 98.9% [98.7, 99.1] | 100.0% | 0.0% | 0.0% / 1.0% (max 2.7%) | 0.0% |
| C2:1 | 92.9% [92.4, 93.4] | 100.0% | 0.0% | 0.4% / 1.0% (max 4.9%) | 0.0% |
| C2:2 | 88.3% [87.6, 88.9] | 99.97% | 1.9% | 0.2% / 1.0% (max 2.8%) | 25.4% |

In every model one label-free direction carries the carry's causal effect, and random directions do
not. P5 nevertheless fails, on its third clause: in five of the eight models the random-init model's
own PC1 also moves the output, 12–40% of the time.

In an untrained B model, PC1 is the direction of the query token s_{k−1}: its 6b table is exactly
NOT s_{k−1}. Moving that token's representation changes whatever the untrained output depends on,
and by chance that sometimes matches the flipped-carry target. So that control measures "does PC1
matter", not "is PC1 the carry".

- **The label-trained probe direction has no effect (0%).**
  - It is nearly orthogonal to PC1 (cosine 0.02 in B:1).
  - It is dominated by low-variance dimensions where the two classes separate cleanly: a direction
    the carry can be *read* from but that the model does not *use*.
  - Base and source differ along it by 0.01, against 0.5 along PC1.
- **Downstream bits do not fully follow the flipped carry.**
  - After the PC1 patch at q_k, the generated sum matches the gate adder with carry k flipped in
    67–88% of cases (B) and 47–49% (C2).
  - A flipped s_k token is not by itself enough to make the next carries follow the counterfactual.
  - The model evidently also draws on digits further back. In B one head reads s_{k−2} and s_{k−3}
    (`results_transformer/TRANSFORMER_REPORT.md`, section 2).

## 5. Swapping gate and transformer

### 6(a) The exact gate carry written into the model (P6: fail; P6-C2: vacuous pass)

At every q_k (k ≥ 1) during generation, PC1–4 are removed and replaced. Directions and class means are
per k for B and pooled for C2.

| model | 10 bits, no patch | carry removed (projected out) | removed, mean-ablated (extra) | gate carry written | flipped gate carry |
|---|---|---|---|---|---|
| B:0 … B:4 | 100% | 0.2% | 0.0–0.2% | 0.1–0.2% | 0.0% |
| C2:0 … C2:2 | 100% | 0.0% | 0.0% | 0.0–0.1% | 0.0–0.1% |

- At 12, 16 and 20 bits every C2 variant is at 0.0–0.1%, against 20–31% unpatched at 16 bits.
- **P6-C2** ("exact carry does not fix length") passes only because writing the carry breaks the model
  at every length.

**Why it fails** (exploratory, not pre-registered; `pc1_diagnostic_exploratory` in `results.json`):

- Along PC1, the spread within one carry class is 0.5–0.9 times the gap between the two class means
  (up to 1.65 at two bits of B:3, and 0.2–0.35 at the top bit).
- Pinning the PC1 coordinate to the true class mean leaves the bit right only 45–58% of the time.
  That is chance below the top bit, where the answer stays right.
- Shifting every carry-0 row by the class gap flips 93–100% of them.
- So the model reads the carry from where PC1 sits *relative to* other components of the same residual
  vector, not from its absolute value. A class mean, which is what a gate's clean 0/1 becomes in that
  space, lands on the wrong side for about half the problems.

### 6(b) The transformer's carry as a gate (P7a, P7b: fail; P7c: pass, sanity only)

The resid_mid projection on PC1 is normalised by its two 2-means levels (no labels), and the table
holds the mean per (a_{k−1}, b_{k−1}, s_{k−1}) combination. Table order: 000 001 010 011 100 101 110 111.

| model | table | gate margin | random-init margin | trilinear gate: worst deviation at ε = 0.05 | analog 64-bit ripple, 100k sums |
|---|---|---|---|---|---|
| B:0 | .16 .10 .81 .19 .82 .19 .89 .86 | 0.65 | 21.6 | 0.26 | 100% |
| B:1 | .18 .11 .82 .17 .83 .18 .89 .87 | 0.70 | 208 | 0.25 | 100% |
| B:2 | .17 .07 .78 .22 .78 .22 .90 .85 | 0.58 | 52.6 | 0.28 | 100% |
| B:3 | .22 .16 .85 .21 .85 .23 .90 .84 | 0.53 | 154 | 0.29 | 100% |
| B:4 | .17 .04 .76 .23 .76 .24 .94 .82 | 0.55 | 119 | 0.30 | 100% |
| C2:0 | .08 .21 .75 .18 .76 .18 .86 .90 | 0.51 | 0.23 | 0.31 | 94.9% [94.7, 95.0] |
| C2:1 | .01 .23 .72 .26 .74 .27 .94 1.02 | 0.44 | 0.10 | 0.33 | 97.7% [97.6, 97.8] |
| C2:2 | −.03 .23 .76 .27 .78 .29 1.01 1.10 | 0.41 | 0.05 | 0.35 | 69.1% [68.8, 69.4] |

- **P7c, sanity check.** Snapped at 0.5, every table equals the carry table [a + b − s ≥ 1], and the
  snapped ripple is 100%. This passes automatically and is not evidence.
- **P7a fails.** Gate margins are 0.4–0.7 standard deviations, against ≥ 4 required and 104 for the
  trained carry neuron.
  - The random-init comparison is not like for like in B. An untrained B model's PC1 is the query
    token's own direction, so its "table" is NOT s_{k−1} ([1 0 1 0 1 0 1 0]): perfectly clean
    (margin 22–208), but not a carry.
  - In C2, where the query token is b_k, random-init margins are 0.05–0.23 and its table is not the
    carry either.
- The label-trained probe direction, a reference row only, gives cleaner tables in B (margin 1.2–2.5).
- **P7b fails.**
  - The trilinear gate's worst deviation is 0.25–0.35 at ε = 0.05. PREREG noted that even a perfect
    table could not restore here.
  - The table entries themselves sit 0.1–0.25 from the rails.
  - Yet inside a ripple adder whose XOR neurons restore every sum bit, the B gates add all 100,000
    random 64-bit sums exactly (worst signal 0.27–0.44).
  - The C2 gates have entries near 0.75 for combos that must give 1, and do not (69–98%).

### 6(c) The MLP replaced by a hand-built XOR module (P8: fail; P8-C2: vacuous pass)

At every q_k the MLP output is replaced by u · m(s_k):
- u is the logit-difference direction through ln_f.
- s_k = XOR(XOR(a_k, b_k), c_k) uses the library XOR neurons.
- a_k, b_k and c_k are read from resid_mid by linear probes and snapped.
- At k = 0, s_0 = XOR(a_0, b_0). At the top bit the output is ĉ_n.

| model | no patch | module | module writing a random direction | module with a_{k−1} wired in place of c_k |
|---|---|---|---|---|
| B:0 / B:1 / B:2 / B:3 / B:4 | 100% | 68.8 / 84.6 / 70.8 / 96.0 / 82.5% | 0.0 / 0.0 / 0.1 / 11.6 / 0.0% | 6.2 / 5.7 / 6.4 / 5.2 / 5.5% |
| C2:0 / C2:1 / C2:2 | 100% | 5.9 / 1.6 / 6.2% | 1.1 / 0.8 / 0.4% | 1.7 / 0.8 / 1.9% |

- During development the module was exact at every k when given the *true* s_k (B:1 and C2:0, checked
  at k = 0, 3, 9 and 10). So the write side works: u · m_s alone sets every bit.
- The failures come from the read side. In B, per-bit linear probes read a_k, b_k and c_k at about
  99.5–100% each (development check on B:1), and a 10-bit sum needs about 30 such readings to all
  be right. In C2, a_k (84–87%)
  and c_k (73–85%, pooled) are far less linearly readable (section 2).
- Wrong wiring scores 5–6% in B, just above P8's 5% bound.
- **P8-C2** passes only because the module fails at every length.

## Scorecard

| # | prediction | B | C2 | what decided it |
|---|---|---|---|---|
| P0 | every model exact at 10 bits | pass | pass | 99.95–100% |
| P1 | attention gathers each input with ≥ 0.25 | fail | fail | a_{k−1}, b_{k−1} get 0.12–0.21: the model averages them |
| P2 | carry first readable at resid_mid (B) | fail | – | pooled 92–95% (per k 98–100%) |
| P3 | readability is not evidence | fail | fail | random-init reads only 78–79% / 59% |
| P4a | no single clean carry neuron | pass | pass | best unit R² 0.30–0.40 |
| P4b | sum from count thresholds | fail | fail | B: 4 of 5 seeds 100% thresholds, B:3 has none; C2: combination detectors |
| P5 | the model uses the carry direction | fail | fail | PC1 88–100% and random ≤ 2% hold; random-init PC1 9–40% |
| P6 | the exact gate carry can replace the model's | fail | fail | gate carry written: ≤ 0.2% |
| P6-C2 | exact carry does not fix length | – | pass (vacuous) | the patch breaks every length |
| P7a | carry gate cleaner than chance | fail | fail | margin 0.4–0.7 |
| P7b | the carry works as an analog gate | fail | fail | no restoring (as pre-noted); ripple B 100%, C2 69–98% |
| P7c | snapped table = carry table | pass | pass | sanity only |
| P8 | an XOR module can replace the MLP | fail | fail | module 69–96% (B), 2–6% (C2) |
| P8-C2 | module does not fix length | – | pass (vacuous) | the module fails at every length |

## Deviations and notes

- **B seeds 0, 2, 3 and 4** use their seed for both the initialisation and the data stream
  (`default_rng([seed, 1000 + step])`). Seed 1 is the existing model, trained with
  `default_rng(1000 + step)`.
- **P6 "removed"** is the subspace projected out, as pre-registered. Mean ablation (the overall mean
  written instead) was added as an extra column and gives the same result.
- **`pc1_diagnostic_exploratory`** and the unit-function breakdown in section 3 are exploratory, added
  after seeing the 6a and P4b results.
- **Three analysis runs, same seeds, same verdicts.**
  - The first crashed while drawing figure 4 (an error bar of −1e−16).
  - The second exposed a diverging probe fit on near-dead MLP units: one per-k entry of 0.50 at
    mlp hook_post. The probe now gives such units zero weight and halves Newton steps that would
    increase the loss.
  - The numbers here are from the third run. None of the pre-registered verdicts changed between runs;
    no verdict uses a probe on hook_post.

## Limitations

- One width, one layer, 10-bit training, 5 + 3 seeds. C2's design is ours, and its length failure
  depends on its untrained long-offset biases.
- Probes are linear. "Not linearly readable" (C2's a_k, c_k) does not mean absent: the MLP reads
  them non-linearly, since the model is exact.
- The gate table (6b) and the interchange use one direction, PC1 of carry-flip differences. In B,
  PC1–4 always work; the four-dimensional subspace may hold a cleaner gate than PC1 alone.
- The trilinear gate and its restore test follow the pre-registration. A sigmoid fitted to the table
  would restore more easily, but would no longer be the transformer's own response.
