# How small can a neural logic gate be?

Produced by `python run_minigates.py` (about 45 s). Every number is in `summary.json`, figures 1–4
are in `figures/`, and the chosen gates are in `library.json` (used by `run_snake.py`). Seeds are
fixed, so a rerun gives the same numbers.

## Question

What is the smallest neural network that implements each logic gate, and still works when
thousands of them are wired into a machine, with no rounding between them?

Every unit here is the same sigmoid neuron, y = sigmoid(w · x + b), with signals in [0, 1].

## Part A: the smallest architecture that learns each gate

100 seeds per gate and architecture, full-batch Adam on the truth table, stop when every row has
|logit| ≥ 4.

| gate | 1 neuron (3 weights) | 2→1→1 (5) | 2→1→1 + skip (7) | 2→2→1 (9) | 2→3→1 (13) | 2→4→1 (17) |
|---|---|---|---|---|---|---|
| NOT | **100%** (2 weights) | | | | | |
| AND, OR, NAND, NOR | **100%** | 100% | 100% | 100% | 100% | 100% |
| XOR | 0% | 0% | **100%** | 57% | 77% | 94% |
| XNOR | 0% | 0% | **100%** | 52% | 83% | 94% |

- AND, OR, NAND and NOR are one neuron: 3 weights. NOT is one neuron with 2 weights.
- XOR and XNOR cannot be one neuron (not linearly separable) and cannot be a 2→1→1 chain either,
  because a chain of monotone units is still monotone. The smallest that works is one hidden
  neuron plus a skip connection from the inputs to the output: 2 neurons, 7 weights. A hand-written
  version is `sigmoid(k·a + k·b − 2k·AND(a, b) − k/2)`.
- The smallest XOR is also the easiest to train: 100 of 100 seeds, against 57 for the usual 2→2→1
  and 94 for the 2→4→1 used in `run_adder.py`.

## Part B: "learns the truth table" is not "works in a machine"

Inside a machine a gate's inputs are other gates' outputs: 0.03 instead of 0, 0.97 instead of 1.
The test is the static discipline digital chips are designed to: **every input within ε of a rail
must give an output within ε of the correct rail**. When that holds, no error can grow, however
many gates are chained and around any feedback loop. The interface used throughout is ε = 0.05.

Share of converged seeds passing at ε = 0.05, by training margin (figure 2):

| training margin | 2 | 3 | 3.5 | 4 | 6 |
|---|---|---|---|---|---|
| NOT | 0% | 0% | 100% | 100% | 100% |
| every 2-input gate | 0% | 0% | 0% | **100%** | 100% |
| largest weight, NAND | 6.3 | 9.3 | 10.8 | 12.3 | 18.3 |

Training with early stopping lands every seed on nearly the same solution, so the pass rate jumps
from 0 to 100% at one margin. The prespecified rule (smallest margin at which ≥ 90% of seeds of every
gate pass) picks **margin 4**. At margin 4 the range of ε that works is 0.03–0.12 for the 2-input
gates (0.04–0.10 for OR and NOR) and 0.03–0.45 for NOT.

Why: the slope of a sigmoid neuron at its threshold is |w| / 4. A gate restores signals only if
that slope (its gain) is above 1, so a small error going in comes out smaller. At margin 2 the NOT
gate's weight is about 4, a gain of 1: it passes errors on unchanged and its outputs only reach
0.12 / 0.88 (figure 3). At margin 4 the weight is about 8 and the gain is 2.

## Part C: bits per weight

Fewest bits per weight (signed, one scale per gate) that keep each margin-4 gate passing:
AND and NAND 3 bits, NOT 4, OR and NOR 4–5, XOR 5–7, XNOR 6–7. A 3-bit NAND is 9 bits in total.

## Part D: stress tests, fully analog

**400-layer random circuits** (128 wires, every layer random gates on random wires, inputs with
noise up to 0.05, 10 circuits per margin, figure 4):

| gates trained at | wrong signals, layer 10 | layer 400 | worst distance from the exact 0/1 |
|---|---|---|---|
| margin 2 | 21% | 27% | 0.97 by layer 10 |
| margin 3 | 0% | 0% | 0.14 |
| margin 4 | 0% | 0% | 0.027 |

**A NAND latch** (two NANDs in a loop, the core of every flip-flop), left to hold for 2,000 steps:

| gates trained at | starting from 1 / 0 | starting from 0.7 / 0.3 |
|---|---|---|
| margin 2 | collapses to 0.56 / 0.44 | 0.56 / 0.44 |
| margin 3 | holds 0.93 / 0.07 | 0.93 / 0.07 |
| margin 4 | holds 0.98 / 0.02 | 0.98 / 0.02 |

Margin-2 gates, the size used in the adder experiments, cannot store a bit: both states fall
into the same middle point.

Margin 3 survives both tests, yet it fails the static discipline, and the snake machine shows it
was right to require it: with margin-3 gates the snake breaks on its first tick (see
`results_snake/SNAKE_REPORT.md`). Random circuits rarely put a gate in its worst case for many
layers in a row; real machines do it all the time (a long OR tree of zeros).

## The library

| gate | network | neurons | weights | multiply-adds |
|---|---|---|---|---|
| NOT | 1 neuron | 1 | 2 | 1 |
| AND, OR, NAND, NOR | 1 neuron | 1 | 3 | 2 |
| XOR, XNOR | 1 hidden + output with skip | 2 | 7 | 5 |

All at margin 4, 8 trained networks per gate type (each gate in a machine draws one of them, so
a machine mixes networks from different seeds). Against the 2→4→1 gate of `run_adder.py`
(5 neurons, 17 weights) this is 5× fewer neurons and 6× fewer weights per AND gate.

## Limitations

- Sigmoid neurons only. A non-monotone activation (a bump) would make XOR a single neuron.
- Gates have two inputs. One neuron can be an 8-input AND or OR (a threshold gate), which would
  shrink OR trees and decoders a lot; this was left out so that the machine uses ordinary gates.
- The static discipline was checked on a 9 × 9 grid of input points per corner, not proven.
