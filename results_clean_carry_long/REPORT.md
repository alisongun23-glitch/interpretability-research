# Addendum: training the narrow adders long enough to squeeze

Produced by `python clean_carry_long.py` (`prereg-check`, `pilot`, `train`, `analyse`, `report`, then `explore`):
4 pilot runs and 25 models of 40,000 steps each, about 5.5 hours on 10 cores (pilot 0.5 h, training 3.9 h,
analysis 1.1 h). The rules were written to `PREREG.md` and locked (sha256 `a1d2bb34…`, 2026-10-30 13:08:16 EDT)
before the pilot and before any model of this addendum was trained. Every number below is in `results.json`
(gates and verdicts, judged by code) or `exploratory.json` (labelled exploratory). `VERDICTS.md` is the code's
own verdict table; figures are in `figures/`. The first study (`results_clean_carry/`) is unchanged and its
verdicts stand.

## The answer in plain English

- **Training 16 times longer did not rescue the narrow adders.** Every width trained for 40,000 steps,
  against 2,500 in the first study. Width 256 learned in 5 of 5 seeds, by step 500. Width 32 learned in 2 of
  5, width 16 in 1 of 5, and widths 8 and 4 in none. A width counts as learning only if 3 of its 5 seeds learn,
  so only width 256 does. Most of the models that failed were stuck: 14 of 17 at widths ≤ 32 moved by 2 points
  or less over the last 10,000 steps.
- **So the squeeze still has no verdict.**
  - H3g, the squeeze with the new neuron-specific label, gets no verdict because no width ≤ 32 learns. Its
    validity gates all pass, so learning was the only thing missing.
  - H3 again gets no verdict, because V4 fails again.
  - D2 gets no verdict, because width 16 has no usable seeds.
  - D3a passes and D3b says "go together", but both rest on the three narrow models that learned. The D3 rules
    were written for five learning seeds per width; with one or two, D3a can hardly fail.
- **The strict checker does what it was built for.** Under gate labels a unit counts only if it is off (near
  zero) when not doing its job. Random rotations of the units take gate-cleanliness from 0.36 to 0.00 (V4g
  passes), where the first study's labels only drop from 0.48 to 0.40 (V4 fails). So in these adders the
  single neurons are special. At width 256, 28–42% of the units are clean gates: one neuron, one count
  threshold, silent when off. The checker can also see packing (V2g passes): in the control arm 6 of 9 models pack, though
  each has only 1–4 units that are gates at more than one position.
- **The three narrow adders that did learn have neither clean neurons nor packed ones (exploratory).**
  - Every unit has at least one active position without any label, so cleanliness is 0.00 under both labels.
  - Only 13–47% of their active unit-positions carry a label at all, against 62–83% at width 256. More of
    those labels are Boolean functions other than count thresholds (7–26% of positions, against ≤ 1%).
  - Their carry direction is weaker (interchange 0.92, 0.80, 0.61 against 1.00), and it no longer runs only
    through the count-threshold units.
  - Their units are not two-level gates.

  This rests on three models.
- **The messy units are not noise: they read the answer bit just written (exploratory).** The labeller only
  asks whether a unit is a function of the position's three inputs, a_k, b_k and the carry c_k. Fitting the
  unlabelled units with lookup tables over wider sets of bits shows that most of them are functions of those
  three inputs plus one or two nearby bits: above all s_{k−1}, the sum bit the model wrote one step earlier,
  which is the token at the position itself, and next the previous column's bits. In the narrow learners,
  the three inputs explain 52–63% of an unlabelled unit's variance; adding s_{k−1} gives 72–87%, and the bits of
  columns k−2 to k+1 give 95–98%. The same holds for the 17–38% of unlabelled unit-positions at width 256.
  Linear fits do no better than 0.44–0.60, so these units are nonlinear functions of a few nearby bits, not
  linear mixtures of many. In the narrow adders the unit computing the count also "sees" the digit under it;
  in the wide adders most units have learned not to. Figure 4.
- **They see it because the model uses it: the digit under the position is carry evidence (exploratory).**
  When the previous column's digits differ, the digit written there tells the carry (c_k = 1 − s_{k−1}); when
  they are equal it tells nothing. Flip that digit's embedding before attention and every model follows the
  rule: a narrow learner stays right in 80–94% of the equal cases and falls to 7–24% in the differing ones. The
  wide models fall too, but they have a second source: with the digit removed they still get 94–99% of bits
  (the carry direction attention writes), where the narrow learners get 65–84% of bits and 2–18% of sums. And
  the route differs: in the narrow learners the digit reaches the units only through attention (changing the
  direct route from the token embedding changes nothing), in the wide ones the direct route carries about
  half of the unlabelled units' dependence and flipping it alone breaks the sums. So the narrow adders are not
  wide adders with the neurons crammed: they compute the carry from the digit under them and the previous
  column, and the units that do this are not clean functions of a finished carry. Figure 5.
- **Width 256 repeats the first study under the new schedule.** The count-threshold units read the carry
  direction (AUC 0.86–0.94), always with the predicted sign (5,459 of 5,459). Holding them erases the
  direction's effect (interchange 1.00 to 0.00–0.04), while holding other units leaves it (0.98–1.00). The
  direction itself is not neuron-shaped (45 of 45 positions), and the units are two-level gates (both snaps keep
  100%).
- **The first study's odd seed was luck, not the seed (exploratory).** Seed 3 here starts from the same weights
  and sees the same data as the first study's B:3. Only the learning rate differs, by at most 8.6% over the
  first 500 steps. With that change, seed 3 grows an ordinary model: cleanliness 0.42 at step 2,500, where B:3
  had 0.05. The other four seeds end up much the same under both schedules.

## Setup

| arm | model | widths × seeds | training |
|---|---|---|---|
| L (task, long) | the B adder (`clean_carry.make`: 1 layer, d_model 64, 4 heads), same init and data stream as the first study's arm A | 256, 32, 16, 8, 4 × seeds 0–4 | 40,000 steps, the first study's schedule with its cosine stretched over 40,000 |
| pilot (sets S only) | same | 16, 4 × seeds 100, 101 | 40,000 steps, exact sums every 1,000 |

The pilot's width-16 runs reached 99% exact sums at steps 8,000 and 26,000; its width-4 runs ended at 49% and
0.6%. The rule gives S = min(40,000, max(10,000, 55,000)) = 40,000 (`pilot.json`, figure 1). The pilot was
optimistic: 2 of 2 width-16 pilot runs learned, against 1 of 5 main seeds.

A unit's **gate label** at a position is its best Boolean function of the position's three inputs (R² ≥ 0.8),
accepted only if the unit is silent when off: |mean where f = 0| ≤ 0.05 × mean where f = 1. Gate-cleanliness,
gate-packed share and gate-Δ are the first study's measures computed with gate labels. All other definitions
are the first study's (`results_clean_carry/PREREG.md`).

## Validity gates

| gate | result | numbers |
|---|---|---|
| V1L continuity | pass | width 256 learns in 5 of 5 seeds; interchange 1.00 in all 5 |
| V2 (reused) | pass | the first study's result |
| V2g sensitivity | pass | control-arm models at widths ≤ 16 that learned, gate labels: gate-packed share 1.00 in 6 of 9 (16:2, 16:4, 8:0–8:3; 1–4 multi-position gate units each), undefined in 3 |
| V3 specificity | pass | at initialisation: cleanliness without dictators 0.00, gate-cleanliness 0.00 |
| V4 privileged basis | **fail** | cleanliness 0.479; after random rotations 0.395 (needed ≤ 0.239) |
| V4g privileged basis, gate labels | pass | gate-cleanliness 0.364; after the same rotations 0.000 (needed ≤ 0.182) |

V4 covers H3, so H3 gets no verdict. V4g covers H3g and passes; H3g's only obstacle is learning.

## 1. Learning

| width | seeds that learn | first step at ≥ 99% exact sums | exact sums at step 40,000 of the others |
|---|---|---|---|
| 256 | 5 of 5 | 375–500 | – |
| 32 | 2 of 5 (seeds 2, 4) | 28,000, 23,000 | 1.9%, 2.8%, 11.7% |
| 16 | 1 of 5 (seed 4) | 24,000 | 2.1%, 2.5%, 18.6%, 69.4% |
| 8 | 0 of 5 | – | 5.7%, 10.4%, 24.6%, 37.5%, 49.9% |
| 4 | 0 of 5 | – | 1.4%, 1.5%, 3.5%, 50.1%, 86.6% |

Narrow models learn in steps: they sit on a plateau for thousands of steps and then jump (figure 2). The three
that learned jumped from 49–76% to 100% between steps 20,000 and 30,000. Over the last 10,000 steps the
learning rate falls from 0.23× to 0.10× of its peak. In that stretch 14 of the 17 models that did not learn
moved by 2 points or less. Three were still rising: 16:2 by 13.5 points, 4:0 by 5.3 and 8:2 by 4.0.

## 2. Cammarata's squeeze: H3 and H3g

**H3g, squeeze with gate labels (no verdict: no width ≤ 32 learns).** The rule needs every learning width
≤ 32 to show low gate-packing and a positive gate-Δ, so it needs at least one learning width ≤ 32. There is
none. Reported with H3g, not judged:
- **Gate-cleanliness.** Width 256: 0.28–0.42 (mean 0.36). In all three narrow learners: 0.00.
- **ρ(width, positions per gate unit).** Undefined, because it needs three learning widths.
- **Width 256 in detail.** Gate-packed share 0.00 in 5 of 5. Gate-Δ 0.67–0.72. One-job gate units for all
  three thresholds t = 1, 2, 3.

**H3, squeeze with the first study's labels (no verdict: V4).** For the record, without the gate it would be
inconclusive: only width 256 learns, and its cleanliness (0.48) is below the rule's 0.70.

## 3. The carry direction: D2 and D3

**D2, order of emergence (no verdict: width 16 has no usable seed).** A seed is usable only if it ends with at
least 20 clean threshold pairs. Width-16 models end with 0–16, so none is usable. At width 256 the direction is
first or level in 5 of 5 seeds:

| seed | direction (interchange ≥ 0.8) | half the clean threshold pairs | first ≥ 99% |
|---|---|---|---|
| 0 | 375 | 375 | 500 |
| 1 | 325 | 375 | 450 |
| 2 | 275 | 300 | 375 |
| 3 | 325 | 350 | 450 |
| 4 | 350 | 350 | 400 |

Per unit, the step a label appears and the step the unit starts reading the direction correlate weakly
(ρ = 0.14–0.38, below the pre-registered guess of 0.5), as in the first study. Exploratory, other widths: one
narrow model, 32:2, has both times, and they tie at step 8,000.

**D3a, survival (pass, thin).** The rule fails only if some width has 2 learning seeds with interchange below
0.80. Interchange is 1.00 in all five width-256 seeds. The narrow learners have 0.92 and 0.61 (width 32) and
0.796 (width 16). So 2 of the 3 narrow learners are below 0.80, but no width has two of them. The 0.61 is
about what every model shows in its first 100 steps, before it can add (exploratory, below).

**D3b, independence (go together, thin).** ρ(cleanliness, interchange) = 0.84 over the 8 learning models (the
same with gate-cleanliness, exploratory). All five width-256 models tie at interchange 1.00, and all three
narrow learners tie at cleanliness 0.00. So ρ only says that the narrow learners are lower on both.

## 4. Width 256 under the long schedule (exploratory, a replication)

| measure | first study (saved B:0–B:4, B:3 apart) | arm L width 256, seeds 0–4 |
|---|---|---|
| reader AUC, \|read cosine\| of clean threshold units against other active ones | 0.92–0.98 (B:3 0.65) | 0.86–0.94 |
| sign as the threshold predicts | 9,804 of 9,804 (with seeds 5–9) | 5,459 of 5,459 |
| interchange: plain / clean threshold units held / other units held | 1.00 / 0.00–0.01 / 1.00 | 1.00 / 0.00–0.04 / 0.98–1.00 |
| direction inside the random-vector band (D5) | 90 of 90 positions | 45 of 45 |
| accuracy left by weak removal / count snap / two-level snap (H4) | 0–1% / 100% / 100% | 0–1% / 100% / 100% |

In this table the first study's interchange comes from the saved models, while its sign total, D5 count and H4
numbers also include seeds 5–9. Every finding of the first study's section 2 holds under the stretched schedule.

## 5. Exploratory (defined after the verdicts)

**The narrow models that learned.** Shares are of active unit-positions at step 40,000.

| model | interchange | any label | count label | other Boolean | gate label | clean threshold units held | as many other units held | two-level snap | count snap |
|---|---|---|---|---|---|---|---|---|---|
| width 256 (5 seeds) | 1.00 | 0.62–0.83 | 0.62–0.82 | ≤ 0.01 | 0.50–0.67 | 0.00–0.04 | 0.98–1.00 | 100% | 100% |
| 32:2 | 0.92 | 0.47 | 0.32 | 0.15 | 0.35 | 0.51 | 0.85 | 97% | 70% |
| 32:4 | 0.61 | 0.13 | 0.06 | 0.07 | 0.09 | 0.54 | 0.58 | 67% | 97% |
| 16:4 | 0.80 | 0.29 | 0.03 | 0.26 | 0.15 | 0.76 | 0.74 | 66% | 26% |

"Count label" is a count threshold or count ramp; "other Boolean" is any other function of the three inputs.
- **No one-job units.** Every narrow unit is unlabelled at one or more of its active positions, so none does
  one job. At width 256, 55–79% of units carry a label at half or more of their positions. In the narrow
  learners, 3–48% do.
- **Not packed either.** The packed share only counts units with a label at every active position. No narrow
  unit has one, so the share is undefined. Whether the labelled positions of a narrow unit disagree was not
  measured.
- **A weaker reader circuit.** The clean threshold units still read the direction with the predicted sign
  (74 of 74, 13 of 13, 4 of 4), but the reader AUC is lower (0.76, 0.62, 0.79). Holding them removes only part
  of the direction's effect.

**What the unlabelled units compute.** For every active unit-position at step 40,000, on 20,000 fresh problems:
the two-fold cross-validated R² of the best lookup table over a set of bits (the group means of one half of the
problems score the other half). R² = 1 means the unit is a function of those bits; the gap to 1 is what other
bits contribute. Sets: the three inputs a_k, b_k, c_k; the three plus the token s_{k−1}; the three plus the
previous column (a_{k−1}, b_{k−1}, and c_{k−1}); and a window, the three plus columns k−2, k−1 and k+1 (9 bits).
A unit-position is put in the first category that reaches R² ≥ 0.9.

| model | unlabelled share of active unit-positions | analog function of the 3 inputs | 3 inputs + s_{k−1} | 3 inputs + previous columns | unexplained | mean R²: 3 inputs → + s_{k−1} → + column k−1 → window |
|---|---|---|---|---|---|---|
| width 256 (5 seeds) | 0.17–0.38 | 0.06–0.12 | 0.33–0.47 | 0.22–0.41 | 0.08–0.25 | 0.63–0.71 → 0.80–0.84 → 0.86–0.90 → 0.93–0.97 |
| 32:2 | 0.53 | 0.09 | 0.31 | 0.50 | 0.10 | 0.63 → 0.81 → 0.94 → 0.96 |
| 32:4 | 0.87 | 0.01 | 0.60 | 0.35 | 0.05 | 0.52 → 0.87 → 0.94 → 0.98 |
| 16:4 | 0.71 | 0.13 | 0.16 | 0.56 | 0.15 | 0.60 → 0.72 → 0.91 → 0.95 |

The category shares are among the unlabelled unit-positions. Figure 4 shows them per model, among all active
unit-positions.
- **s_{k−1} is the bit that matters most.** Among the eight single bits tried (a, b at k−2, k−1, k+1; c_{k−1};
  s_{k−1}), adding s_{k−1} gives the largest gain for 42–87% of the unlabelled unit-positions in the narrow
  learners and 54–81% at width 256; c_{k−1} is next. Bits of the column ahead (k+1) never help (share 0.00 in
  every learner). In 32:4, 189 of 219 active unit-positions are such units: nearly its whole MLP computes
  "count threshold, shifted by the digit under it".
- **Nearby, not global.** The 9-bit window explains 93–98% of the unlabelled units' variance, so they are not
  mixtures of many far-away bits. Linear fits on the same window reach only 0.44–0.55, so they are not linear
  mixtures either.
- **The labelled units are nearly pure.** Their R² on the three inputs is 0.91–0.97, and s_{k−1} adds 0.01–0.05.
- **The non-learners look the same**, with more unexplained unit-positions (up to 0.47 of active in 4:1).

**The token path: why the units read s_{k−1}.** The digit s_{k−1} was written one step earlier and sits under
the position: it is the token at q_k, so its embedding E[s_{k−1}] is in the residual stream the MLP reads, and
attention at q_k can read it too. It is also carry evidence: if the previous column's digits differ
(a_{k−1} ≠ b_{k−1}), c_k = 1 − s_{k−1}; if they are equal, c_k = a_{k−1} and s_{k−1} says nothing. Two changes
to the embedding: **flip** (the other digit's embedding) and **remove** (the mean of the two, which the model
never saw), applied on the **direct route only** (at resid_mid: attention is computed with the true digit and
only what the MLP reads changes) or **before attention** (at resid_pre, one position per forward pass, so every
route changes). Measured: teacher-forced accuracy of s_k, k = 1..9, on the 20,000 problems, split by whether the
previous digits differ; exact sums under greedy generation with the change at every step (4,000 problems); and
the unlabelled units' dependence on s_{k−1} after the change (the table-R² gain of adding it, for the
unit-positions of the "3 inputs + s_{k−1}" category).

Per-bit accuracy (teacher-forced):

| model | no change | direct route: removed / flipped | before attention: removed / flipped | flipped before attention: a_{k−1} = b_{k−1} / a_{k−1} ≠ b_{k−1} |
|---|---|---|---|---|
| width 256 (5 seeds) | 1.000 | 0.979–0.999 / 0.62–0.78 | 0.935–0.988 / 0.45–0.51 | 0.52–0.81 / 0.18–0.38 |
| 32:2 | 1.000 | 1.000 / 1.000 | 0.788 / 0.497 | 0.893 / 0.101 |
| 32:4 | 1.000 | 1.000 / 0.996 | 0.650 / 0.505 | 0.941 / 0.070 |
| 16:4 | 1.000 | 1.000 / 0.999 | 0.835 / 0.516 | 0.797 / 0.236 |

Exact sums (greedy generation):

| model | no change | direct route: removed / flipped | before attention: removed / flipped |
|---|---|---|---|
| width 256 (5 seeds) | 1.000 | 0.83–0.99 / 0.02–0.10 | 0.55–0.90 / 0.000–0.001 |
| 32:2 | 1.000 | 1.000 / 1.000 | 0.075 / 0.000 |
| 32:4 | 1.000 | 1.000 / 0.961 | 0.019 / 0.001 |
| 16:4 | 1.000 | 1.000 / 0.992 | 0.178 / 0.004 |

Units' dependence on s_{k−1} (mean gain, "3 inputs + s_{k−1}" unit-positions): no change → direct route removed →
removed before attention → flipped before attention:

| model | gain |
|---|---|
| width 256 (5 seeds) | 0.21–0.29 → 0.08–0.19 → 0.10–0.20 → 0.36–0.49 |
| 32:2 | 0.31 → 0.30 → 0.09 → 0.32 |
| 32:4 | 0.42 → 0.41 → 0.14 → 0.33 |
| 16:4 | 0.13 → 0.12 → 0.05 → 0.23 |

- **Every model uses the digit as carry evidence, in the pattern the rule predicts.** Flipped before attention,
  a narrow learner keeps 80–94% of the bits where the previous digits are equal and 7–24% where they differ:
  it follows the flipped digit. The wide models fall where the digits differ (18–38%) and also lose some where
  they are equal (52–81%).
- **The narrow learners depend on it; the wide models have a second source.** With the digit removed before
  attention, the wide models keep 94–99% of bits and 55–90% of sums (the carry direction that attention
  writes, which the first study found sufficient: interchange 1.00). The narrow learners keep 65–84% of bits
  and 2–18% of sums.
- **In the narrow learners the digit reaches the MLP through attention, not the direct route.** Changing the
  direct route alone changes neither their units (gain 0.31 → 0.30, 0.42 → 0.41, 0.13 → 0.12) nor their sums
  (100%, 96–100%). Changing it before attention does (gain 0.05–0.14 when removed; half the bits wrong when
  flipped). In the wide models the direct route carries about half of the unlabelled units' dependence
  (0.21–0.29 → 0.08–0.19) and flipping it alone leaves 2–10% of sums.
- **What survives removal** (gain 0.05–0.20) is what attention brings from the previous column, whose digits
  also determine s_{k−1}.
- **Caveat on "remove".** The mean embedding is an input the model never saw. In 256:4 one near-dead unit (it
  fires in 0.01% of problems after the change) gets a cross-validated R² of −1,639 and drags that model's mean
  local R² under removal to −1.93 in `exploratory.json`; without it the mean is 0.68, against 0.69 before. The
  flip is an input the model has seen, and the flip results carry the conclusions.

So the guess that the token's embedding shifts the units was half right. It is the route in the wide models,
where it is a side channel next to the carry direction. In the narrow learners the same digit matters more and
arrives through attention: with 16 or 32 units the model computes the carry from the digit under it and the
previous column rather than from a finished carry, and units that do that are functions of five bits, not
three. Why narrow training settles there is not tested; one reading is that this rule needs attention to look
only one column back and at the position itself.

**Interchange has a floor near 0.6.** In its first 100 steps every model reaches a rank-1 interchange of
0.36–0.66 (median 0.63), while it gets at most 0.25% of sums right. An interchange near 0.6 does not need a
working adder.

**The odd seed was luck.** Arm L and the first study's arm A share initial weights and data order. Their
learning rates differ by at most 8.6% over the first 500 steps. The two runs agree until about step 100, then
drift apart (parameter gap 0.01–0.04 of the distance moved at step 100, 0.42–0.74 by step 2,500). Arm A's
seed 3 is the saved B:3 (bit-identical in the first study's V1).

| seed | cleanliness at step 2,500: arm A / arm L | interchange at step 500: arm A / arm L |
|---|---|---|
| 0 | 0.47 / 0.54 | 1.00 / 1.00 |
| 1 | 0.48 / 0.56 | 1.00 / 1.00 |
| 2 | 0.47 / 0.50 | 1.00 / 1.00 |
| 3 | **0.05 / 0.42** | **0.83 / 1.00** |
| 4 | 0.39 / 0.35 | 1.00 / 1.00 |

What made B:3 odd (cleanliness 0.05, and the outlier behind the first study's D4) is not fixed by its initial
weights and data. A small change in training turns it into an ordinary model. Across both studies, 1 of 15
width-256 runs came out odd.

## Deviations

1. **Figure 3 was redrawn after its first render.** The first version averaged over all five seeds, mixing
   models that learned with models that did not. It showed a packed share of 1.00 at width 16 that came from
   two units of one model that did not learn (16:1). It now shows one dot per model, with non-learners hollow,
   and unit verdicts for learning models only. Figure 2 gained a seed legend. This is presentation only;
   `results.json` is byte-identical before and after.
2. **`explore` was added after the verdicts.** Its measures are never judged: label shares, the interchange
   floor, late learning curves, the arm A/arm L comparison, the unit-function tables (section 5, figure 4,
   `exploratory.json` under `unit_functions`) and the token-path tests (figure 5, `token_path` and
   `token_path_pre`). Four planted checks of these pieces are in `tests/test_circuits.py`.
3. **No narrow learning width.** The pre-registration lists D1 readers and H4 snaps "at narrow learning
   widths" as exploratory. There are none, so they are reported for the three narrow models that learned.
4. Label maps over training are stored for every checkpoint (`codes`, `gate_codes` in `analysis/*.json`)
   but not plotted.

## Limitations

- **Too few narrow learners.** Narrow adders learn rarely and only for some seeds, so three narrow learners make
  the section-5 picture an anecdote. A test of the squeeze needs at least three learners at several widths
  ≤ 32. That means many more seeds or a different recipe, and picking the learners out of many seeds may pick
  one way of solving the task.
- **Longer runs alone would not help.** At 40,000 steps most non-learners sit on plateaus while the learning
  rate is low, so more steps of this recipe would likely rescue few of them.
- **"Unclear" is a catch-all.** The labeller only tests functions of a_k, b_k and the carry c_k. Section 5
  shows that most unclear unit-positions are functions of those three plus s_{k−1} or the previous column.
  Whether such a unit should count as "one job" is a judgement the pre-registration did not make, so the
  cleanliness numbers stand as defined and the wider fits stay exploratory.
- **V2g is thin.** It passes on 6 models with 1–4 multi-position gate units each.
- **Interchange near 0.6 is weak evidence.** Values there can come from an untrained model.
- **Narrow scope.** One task, one layer, and 5 seeds per width.
