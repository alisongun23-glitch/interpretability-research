# Tiled gates vs a trained adder

Produced by `python run_adder.py` (about 30 s). See `results_adder16/ADDER16_REPORT.md` for the 16-bit follow-up. Numbers are from that run; every value is in
`summary.json`, and figures 1–4 are in `figures/`. Seeds are fixed, so a rerun gives the same numbers.

## Question

If trained logic-gate networks can be tiled into a working machine, does a network trained on the
whole job from scratch contain the same parts?

## Part 1–2: tiling trained gates works

XOR, AND and OR networks (2→4→1, tanh) were trained with 20 seeds each, at two safety margins.
All 120 converged. Twenty different 4-bit ripple-carry adders were assembled, each gate position
getting a randomly chosen trained network, so every adder mixes gates built by different internal recipes.

| wiring | margin 2 | margin 6 |
|---|---|---|
| snap each gate's output to 0/1 | 100% of 256 sums | 100% |
| pass raw probabilities | 99.5% on average, worst assembly 93.8% | 100% |

Snapping makes the gates' internal differences irrelevant. Without snapping, small errors pile up
along the carry chain: a mean deviation of 0.06 at bit 0 grows to about 0.13 by the upper bits at
margin 2. Training each gate with a wider margin removes the problem. This matches the transplant
result from the XOR experiment: the interface (scale and offset of the signal between parts) is what
decides whether parts compose.

## Part 3: a single trained network also learns addition

Two-hidden-layer networks (8→W→W→5, tanh, W = 16 and 64, 5 seeds each) were trained on 205 of the
256 problems. All reached 100% on the training problems. On the 51 held-out problems, 9 of 10
networks scored 100% and one scored 96%.

## Part 4: it does not contain the tiled adder's parts

| test | result | meaning |
|---|---|---|
| Probe for per-bit XOR (aᵢ XOR bᵢ) | bit 0: 100% trained vs 60–66% untrained (width 16). Bits 1–3 fall toward chance (bit 3: 53–65%) | Only the lowest bit's XOR is clearly computed, and that one *is* an output bit |
| Probe for carries | 100%, but also 100% from the raw inputs | A carry is "is the low part of the sum big enough", a straight-line function of the inputs, so readable everywhere. Probing for it proves nothing |
| Knock out the units most tied to each carry | at most 9% accuracy lost on one bit, usually under 2% | No small group of units carries the carry |
| Swap in the opposite carry along the carry direction, using activity from a real problem with the opposite carry | 0–5% of answers change as the tiled adder predicts; with the swap in layer 2, 100% of answers stay unchanged | The carry direction exists but the network does not use it |
| Push along that direction 1.5×, 4×, 10× harder | 0–6% | Not a weak-push artefact |

### What it does instead: partly collapse each problem to its total

Two measures of whether problems with the same total a + b look alike inside a layer:
η² (share of the layer's variation explained by the total alone; 1.0 = identical) and the
distance ratio (distance between two same-total problems ÷ distance between two random problems;
0 = identical, 1.0 = the total plays no special role).

| | η² layer 1 | η² layer 2 | distance ratio layer 1 | distance ratio layer 2 |
|---|---|---|---|---|
| raw inputs | 0.14 | | 1.00 | |
| trained, width 16 | 0.27 | **0.73** | 0.88 | **0.50** |
| trained, width 64 | 0.16 | **0.52** | 0.98 | **0.74** |
| untrained, width 16 | 0.14 | 0.14 | 1.01 | 1.00 |
| untrained, width 64 | 0.12 | 0.11 | 1.01 | 1.02 |

By layer 2, the trained networks organise problems largely by their total, and untrained networks
do not. The collapse is partial, not complete.

**Correction.** An earlier version of this report also claimed a causal check: replacing all of
layer 2 with the activity of a different problem that has the same total kept the answer right
99.8–100% of the time. That test was invalid. Once a whole layer is replaced, the network is simply
answering the other problem, whose correct answer is the same, so it passes automatically. It has
been removed from `run_adder.py`. The η² and distance-ratio results above are the evidence.

## Conclusion

- **Tiling works** as engineering: trained gates with mismatched internals compose into a correct
  adder, as long as the signals between them are snapped or the gates have wide margins.
- **A network trained on the whole task does not rediscover the tiling.** It has no carry wire
  it actually uses, and by layer 2 it largely organises problems by their total. Addition of 4-bit numbers fits easily in one
  dimension, so that is the cheaper solution for gradient descent.
- So composition from primitives is one valid description of addition, not the one the network
  uses. This repeats the earlier lesson (many circuits, one function), now at the level of a whole
  machine: a network finds whatever decomposition training makes easiest, which need not be the one
  an engineer would build.

## Limitations and next steps

- The task is small (256 problems) and the "total" fits in one number. With wider numbers (say
  16 bits), a single total would need far finer precision, which may force a carry-like
  decomposition. That is the natural next test.
- The carry tests use one straight-line direction per layer. A carry spread nonlinearly across
  many units would be missed; the strong "collapse to total" result makes that unlikely here.
- Only multilayer perceptrons were tested; transformers on digit sequences behave differently.
