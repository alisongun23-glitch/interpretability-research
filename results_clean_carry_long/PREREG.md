# Pre-registration addendum: train the narrow adders long enough to squeeze

Written 2026-10-30, before the pilot and before any model of this addendum was trained. Nothing below changes
after the runs; deviations go in `REPORT.md` under "Deviations". `clean_carry_long.py` stores this file's
SHA-256 in `results.json` and refuses to run its pilot, training or analysis if it is missing or has changed.

It extends `results_clean_carry/PREREG.md` (sha256 `c3f2e446110a68805f6c88e94fde717bf6492362fd3b07c3d7f5d04a98c104f7`,
locked 2026-10-30 11:56:09 EDT). Every definition not restated here is that file's, unchanged: units and
"active", labels, unit verdicts, cleanliness, packed share, positions per unit, Δ, clean threshold pairs,
the carry direction, interchange, read cosines, learning for arm A, and the H3, D2, D3 rules. That study's
verdicts stand as they are; this addendum adds results and replaces none.

## Why

In the first study no adder with an MLP of 32 units or fewer learned in 2,500 steps (exact sums 0.3–13%),
so H3, D2 (width 16) and D3 had no narrow models. Those runs were still improving when they stopped. H3 also
got no verdict because its rotation gate V4 failed: in this task a random rotation of the units is still a
clean function of the count, so the labels describe the layer, not single neurons. This addendum (1) trains
long enough to learn, and (2) adds H3g, the same squeeze question asked with a label that is specific to
neurons.

## Arm L (task, long)

- `clean_carry.make(w, s)` (the B adder: 1 layer, d_model 64, 4 heads, MLP width w), the same data stream
  `default_rng([s, 1000])`, batch 128, AdamW (wd 0.01), loss on the sum tokens, learning rate
  lr(step) = 1e-3 · min(1, (step + 1)/100) · (0.1 + 0.45 · (1 + cos(π · step / S))), i.e. the first study's
  schedule with its cosine stretched over S steps.
- Widths 256, 32, 16, 8, 4 × seeds 0–4, all trained for S steps (width 256 is retrained so that every width
  has the same recipe).
- Checkpoints (parameters only): step 0; every 25 steps to 1,000; every 250 steps to 5,000; every 1,000 steps
  to S.

## Pilot (sets S; used for nothing else)

- Widths 16 and 4 × pilot seeds 100 and 101, 40,000 steps, schedule stretched over 40,000; exact-sum accuracy
  on 2,000 test problems (greedy generation) every 1,000 steps.
- L_i = first evaluation step with exact accuracy ≥ 99% in pilot run i.
- If at least one pilot run learns: S = min(40,000, max(10,000, ⌈2 · max_i L_i / 5,000⌉ · 5,000)), the max over
  runs that learn. If none learns: S = 40,000.
- S is written to `pilot.json` before the main training starts.

## New definition: gate label

An active (unit, k) gets a **gate label** if its best Boolean fit over the position's three inputs has
R² ≥ 0.8 (oriented so the unit is high where f = 1) **and** it is silent when off:
|mean activation where f = 0| ≤ 0.05 × mean activation where f = 1, with the latter > 0. The code is the
same as the first study's Boolean or count-threshold label; count ramps are not accepted. Everything built on
labels (unit verdicts one job / two related jobs / packed / unclear, cleanliness, packed share, Δ, shared
thresholds) is computed again with gate labels in place of labels and called **gate-cleanliness**,
**gate-packed share**, **gate-Δ**.

## Learning

Arm L: exact sums ≥ 99% at step S (greedy generation, 2,000 test problems). A width learns if ≥ 3 of its 5
seeds learn.

## Validity gates (a failed gate gives "no verdict" for what it covers)

| gate | covers | pass if |
|---|---|---|
| V1L continuity | H3, H3g, D2, D3 | arm L width 256 learns in 5 of 5 seeds, with interchange ≥ 0.95 in ≥ 4 of 5 |
| V2 (reused) | H3's fail clause | the first study's V2 result (it passed) |
| V2g sensitivity | H3g's fail clause | the first study's arm-R models at widths ≤ 16 that learned (per-bit ≥ 95%), relabelled with gate labels: gate-packed share ≥ 0.30 in ≥ half of them (none with a defined share = fail) |
| V3 | H3 (first part), H3g (second part) | arm L width 256 step 0, seeds 0–4: mean cleanliness without dictators ≤ 0.10; mean gate-cleanliness ≤ 0.10 |
| V4 | H3 | the first study's rotation null on arm L width 256 at step S, seeds 0–4 (10 random orthogonal rotations of the post-ReLU activations each): mean rotated cleanliness ≤ half the unrotated mean |
| V4g | H3g | the same rotations: mean rotated gate-cleanliness ≤ half the unrotated mean |

## Hypotheses

| # | data | rule |
|---|---|---|
| H3 | arm L at step S | the first PREREG's H3 rule, unchanged |
| H3g | arm L at step S | **pass** if every learning width ≤ 32 has gate-packed share ≤ 0.10 in ≥ 4 of its learning seeds (a seed with no multi-position gate units counts as not packed) and mean gate-Δ ≥ 0.20 over its learning seeds, and at the smallest learning width, one-job gate units active at ≥ 2 positions exist for at least two of t = 1, 2, 3 (any seed); **fail** if gate-packed share ≥ 0.30 with gate-Δ < 0.05 in ≥ 3 of 5 seeds at some learning width ≤ 16 (only if V2g passes); **no verdict** if no width ≤ 32 learns; otherwise inconclusive |
| D2 | arm L widths 256 and 16, all checkpoints | the first PREREG's D2 rule, unchanged |
| D3a, D3b | arm L learning models, all widths, at step S | the first PREREG's D3 rules, unchanged |

Reported with H3g, not judged: gate-cleanliness per width and Spearman ρ(width, positions per gate unit).

## Exploratory (reported, never judged)

D2 at widths 32, 8, 4; the step each model first reaches 99%; D1 readers (with clamp mediation) and H4 snaps at
narrow learning widths; D3b with gate-cleanliness; label maps over training.
