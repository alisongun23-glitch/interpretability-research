# Clean neurons and the carry direction: Cammarata's test on adders with a known answer

Produced by `python clean_carry.py` (`stage0`, `train`, `analyse`, `report`, then `explore`): 50 newly trained
models plus the 5 saved B adders, roughly 1.5 hours on 10 cores. The predictions were written to `PREREG.md`
and locked (sha256 `c3f2e446…`, 2026-10-30 11:56:09 EDT) before any model was trained or analysed. Every
number below is in `results.json` (verdicts, judged by code) or `exploratory.json` (labelled exploratory).
`VERDICTS.md` is the code's own one-page verdict table; figures are in `figures/`.
Checks: `python tests/test_circuits.py` and `python tests/check_all.py`, both ALL PASS.

## The answer in plain English

- **The clean neurons are the carry direction's readers, and its only route.** The count-threshold units
  (OR3, majority, AND3) read the residual-stream carry direction far more than other active units (AUC
  0.90–0.98 in 9 of 10 models), always with the sign their threshold predicts (9,804 of 9,804 unit-positions).
  Holding just those units at their unpatched values erases the direction's causal effect: interchange
  falls from 1.00 to 0.00–0.01 (B:3, the odd seed: 0.97 to 0.35). Holding every other active unit changes
  nothing (1.00 stays 1.00). So the
  two earlier findings, "the carry is a direction, not a neuron" and "the clean neurons are count
  thresholds", are one mechanism: attention writes the direction, the clean neurons read it.
- **The direction itself is not neuron-shaped.** In all 90 model-positions, its coordinates look like a
  random direction in the 64-number residual stream (D5). A neuron-level claim does not reach it.
- **Cammarata's own neuron-cleanliness hypotheses got no verdict (H1–H3), because validity gate V4
  failed.** Mixing the 256 units by a random rotation barely lowers "cleanliness" (0.385 → 0.357). In this
  task almost everything the MLP does is a function of one 4-valued number, the count a_k + b_k + c_k,
  and any mixture of such units is again a clean function of the count. So the labeller measures what the
  layer computes, not something special about individual neurons, and cannot test a claim about neurons.
- **No packing in the task, total packing in the control.** Among the adders' multi-position units, the
  packed share is 0.00 in every learning adder but one (0.01). Units that are not clean are "unclear",
  not packed. In arm R, where every position has an unrelated random job, units pack at every width,
  including 256 (packed share 0.91–1.00). The labeller can see packing (V2 passes).
- **The squeeze could not be run on the task.** No adder with an MLP of 32 units or fewer learned in 2,500
  steps (exact sums 0.3–13%), while arm R learned down to 8 units. So H3, D2 and D3 lost their narrow half.
- **Weak activations are not noise here (H4 fails).** Zeroing each unit's weakest 75% of non-zero
  activations destroys every adder (it costs 99–100 points of accuracy). The units are near-binary: snapping
  each to its two levels keeps 100% in 9 of 10 models. Their "weakest 75%" is most of the ON state, not a
  weak tail. The test was borrowed from long-tailed language-model neurons and does not fit digital units.
- **Timing and seeds (exploratory or inconclusive).** At width 256 the direction and the clean units appear
  together, between steps 250 and 375; the direction is first or level in 4 of 5 seeds. Across seeds,
  clean units and a clean direction go together only through one outlier, seed 3 (D4 inconclusive,
  ρ = 0.59).

## Setup

| arm | model | widths × seeds | data |
|---|---|---|---|
| A (task) | the B adder recipe: `TinyGPT`, 1 layer, d_model 64, 4 heads, 2,500 steps | 256, 32, 16, 8, 4 × seeds 0–4; 256 × seeds 5–9 | `a0..a9 + b0..b9 = s0..s10`, lowest bit first |
| R (control) | same | 256, 32, 16, 8 × seeds 0–4 | answer token t_k = T_k(a_k, b_k, t_{k−1}), 11 different random balanced functions |
| saved | `results_transformer_gates` B:0–B:4 | 256 × 5 | as arm A |

Units are labelled at q_k, k = 1..9, as a Boolean function of the position's three inputs (R² ≥ 0.8), else a
monotone count ramp (η² ≥ 0.9, arm A), else "unclear". A unit does one job if every active position carries
the same label. The carry direction is PC1 of counterfactual differences at resid_mid (`TG.flip_dirs`), and
its use is measured by rank-1 interchange. Full definitions are in `PREREG.md`.

## Validity gates

| gate | result | numbers |
|---|---|---|
| V1 reproducibility | pass | retrained width-256 seeds 0, 2, 3, 4 are bit-identical to the saved models (max parameter difference 0.0) |
| V2 sensitivity | pass | arm R learned in 9 models at widths ≤ 16; packed share 1.00 in all 9 |
| V3 specificity | pass | at initialisation: cleanliness without dictators 0.00; no clean threshold pairs, so no reader AUC |
| V4 privileged basis | **fail** | saved B:0–B:4 cleanliness 0.385; after random rotations 0.357 (needed ≤ 0.192) |
| V5 clamp sanity | pass | holding every unit at its own value: 100% agreement |

V4 covers H1, H2 and H3, so those get no verdict. Their numbers follow, for the record.

## 1. Cammarata's hypotheses

**H1, the odd seed (no verdict, V4).** Cleanliness: B:0 0.47, B:1 0.54, B:2 0.47, B:3 0.05, B:4 0.39. Packed
share among multi-position units: 0.00 in all five. Δ (same label across positions, above chance): 0.68–0.73,
B:3 0.39. Units are active at 7.2–8.2 positions on average (B:3 5.2). Ignoring the gate, B:3 would have been
inconclusive: it is neither clean (≥ 0.70) nor packed (≥ 0.30). Its non-clean units are unclear.

**H2, development (no verdict, V4).** At width 256, a one-job unit's first label equals its final label in
0.91–0.95 of units (seeds 0, 1, 2, 4), and positions that join later carry the unit's label 0.62–0.68 above
the permutation null. Seed 3: 0.50 of its 10 one-job units, excess 0.27. At width 16 nothing learned and no
one-job units formed.

**H3, squeeze (no verdict, V4).** Only width 256 learned. Exact sums at widths 32, 16, 8 and 4 were 1–4%,
0.3–13%, 1–4% and 1–3%. Teacher-forced per-bit accuracy was 0.63–0.86 (exploratory), well above chance but
far from the 99% bar.

**H4, readout (fail).** Edits at every q_k during generation, 10 learning models:

| edit | accuracy after the edit |
|---|---|
| none | 100% |
| zero each unit's weakest 75% of non-zero activations | 0–1% (cost 99–100 points) |
| zero its strongest activations, matched mass | 2.8–18.5% (cost 82–97 points) |
| snap each unit to its mean per count | 100% in 9 models, 98.6% in seed 3 |
| snap each unit to its two 2-means levels | 100% in 9 models, 25.2% in seed 3 |

The rule fails on the weak-removal clause in all 10 models. The rival it was meant to detect, an analog
readout, is not what the data show: the count snap holds and the two-level snap is exact, so these units are
digital gates (9 of 10 models by the pre-registered two-level criterion). Seed 3 is the analog exception.

## 2. The carry direction and the clean neurons

**D1a readers (pass).** AUC of |read cosine| for clean threshold unit-positions against other active ones:
B:0 0.94, B:1 0.98, B:2 0.93, B:4 0.92, seeds 5–9 0.90–0.96; B:3 0.65. Median |r| is 0.35–0.44 for threshold
units against 0.06–0.11 for the rest (B:3: 0.40 against 0.27). Figure 2.

**D1b sign (pass).** Units that fire above a count threshold read the direction positively, units that fire
below it negatively, in 9,804 of 9,804 clean threshold unit-positions.

**D1c mediation (pass).** Rank-1 interchange along the direction, with units held at their unpatched values:

| model | plain | clean threshold units held | all other active units held |
|---|---|---|---|
| B:0, B:1, B:4, seeds 5–9 | 1.00 | 0.00 | 1.00 |
| B:2 | 1.00 | 0.01 | 1.00 |
| B:3 | 0.97 | 0.35 | 0.94 |

The direction acts only through the clean units. Nothing goes around them through the residual stream.

**D2, order of emergence (no verdict: width 16 never learned).** At width 256 (exploratory), interchange
first reaches 0.80 at steps 250–375. Half of the final clean threshold pairs are in place by steps 250–350.
The direction is first or level in 4 of 5 seeds: 325/325, 300/325, 250/275, 275/350. Seed 3 is the other way,
375/250. Per unit, the step a label appears and the step the unit starts reading the direction (|r| > 0.2)
correlate weakly: ρ = 0.08–0.26, below the pre-registered guess of 0.5. Figure 1.

**D3, squeeze (D3a pass, D3b "go together", both thin).** Only width 256 learned, so the test ran across 5
seeds at one width instead of across widths. Interchange was 0.97–1.00. The correlation with cleanliness,
ρ = 0.71, comes from seed 3 alone; the other four tie at 1.00.

**D4, seeds (inconclusive).** Across width-256 seeds 0–9, ρ(cleanliness, debiased direction margin) = 0.59,
just under the 0.6 bar. Seed 3 sits apart (cleanliness 0.05, margin 0.91). The other nine sit at 0.39–0.55 and
2.77–3.90 with no clear trend. Figure 4.

**D5, basis (pass).** In 90 of 90 model-positions, the direction's largest squared coordinate and its
participation ratio lie inside the 2.5–97.5% band of random unit vectors (0.065–0.191 and 14.8–30.0).

## 3. Exploratory (defined after the verdicts)

- **A basis-sensitive variant.** Counting only sharp steps (Boolean R² ≥ 0.8, no ramps), the real units are
  more step-like than rotated mixtures: 0.64–0.76 of active unit-positions against 0.56–0.61 (B:3 0.36
  against 0.16). The neuron basis matters, but only modestly in this task.
- **Arm R everywhere.** Cleanliness is 0.00–0.03 at every width, and packing does not need a squeeze: the
  packed share is 0.91–1.00 at width 256 too. When jobs are unrelated and positions never overlap, sharing a
  unit across positions costs nothing, so units pack even with room to spare.

## Deviations

1. **D1c control size.** The pre-registration asked for a random set of other active units of the same size
   as the clean set. There are fewer of them (in the saved models, 25–89 per position against 100–132 clean
   threshold units outside B:3), so the control held all other active units. Holding all of them still
   leaves interchange at 1.00.
2. **Narrow widths did not learn**, so H3, D2 (width 16) and D3 ran on fewer models than planned (rules
   applied as written).
3. `explore` and its two measures were added after the verdicts and are never judged.
4. Figure 1's width-16 panel shows raw counts (its final count is 0); presentation only.

## Limitations

- One task, one layer, a fixed 2,500-step recipe. Narrow adders might learn with longer training, which is
  where the squeeze would become testable.
- The labeller's cleanliness is not neuron-specific when a layer's activity is a function of one small
  variable (V4). A test of Cammarata's claim needs a task with many independent features, or a labeller that
  rewards axis alignment.
- Correlations across seeds rest on 5–10 models, and one seed (3) drives them.
