# XOR circuit discovery: experiment report

Generated automatically by `run.py` from an actual run (8 s). 30 seeds per architecture; baseline architecture `tanh-4`. Every number below comes from `summary.json`; figures are in `figures/`.

## What is established before running anything

- A single tanh or ReLU unit of (A, B) is a monotone function of one straight line through the input square, so it can only realise *linearly separable* Boolean functions. XOR is not linearly separable. Therefore every network that solves XOR here **must** combine at least two hidden units. That XOR is composed from simpler parts is guaranteed by the architecture, not a discovery. The open questions are *which* parts, how consistently, and whether they are interchangeable.
- Hidden units can be permuted and (for ReLU) rescaled without changing the function, so raw weight comparison is not meaningful; the analysis compares functions of units instead.

## Method in brief

1. **Training.** 2→H→1 networks, sigmoid output with binary cross-entropy, full-batch Adam (lr 0.05), weights from N(0, 1.0). A run stops at the first step where all four inputs are correct with |logit| ≥ 2.0 (probability ≥ 0.88 / ≤ 0.12), or fails after 20000 steps. Failures are kept.
2. **Unit labels.** Each hidden unit's activation over the 4 inputs is matched (by correlation) to the 14 non-constant Boolean functions; matches below 0.9 are labelled `mixed(...)`. A unit's *contribution* (activation × output weight) is labelled the same way; this merges equivalent readings (an AND unit with a negative output weight contributes the NAND pattern).
3. **Circuit search.** For each converged net, all 2^H subsets of hidden units are tested exhaustively. A subset is *sufficient* if XOR stays fully correct when every other unit is **mean-ablated** (replaced by its mean over the 4 inputs). *Minimal* circuits are sufficient with no sufficient proper subset. Prespecified rule: the *primary* circuit is the smallest minimal circuit, ties broken by larger margin. Its *type* is the sorted list of its units' contribution labels.
4. **Causal tests.** Ablating the circuit vs every other same-size subset; ablating everything else; activation patching (copy the circuit's activations from an input with the opposite XOR label into the clean run and check whether the output follows the source), with same-size and complement controls.
5. **Transplants (no retraining).** The circuit's interface is defined as its *logit contribution* on the 4 inputs. Recipient output = recipient bias + recipient's non-circuit units + donor circuit's contribution. *Direct* uses the donor contribution as is. *Aligned* (secondary, reported separately) rescales it to the recipient circuit's own mean and spread over the 4 inputs, without looking at labels. Controls: native, ablation (recipient circuit mean-ablated), sham (the donor's **same units at initialisation**, matched in size and architecture), and a hand-built reference XOR circuit.
6. **Discovery vs confirmation.** Seeds 0–14 are used to name the dominant circuit type; its frequency, ablation effect and transplant success are then measured on the held-out seeds.

## Results

### Training and behaviour

| architecture | converged | median steps | mean min margin (logit) | failed seeds |
|---|---|---|---|---|
| tanh-4 | 30/30 (100%) | 53 | 2.03 | - |
| relu-4 | 23/30 (77%) | 65 | 2.02 | [0, 3, 4, 11, 17, 27, 29] |
| tanh-2 | 9/30 (30%) | 77 | 2.04 | [0, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23, 25, 27, 28, 29] |
| tanh-8 | 30/30 (100%) | 36 | 2.05 | - |
| relu-8 | 29/30 (97%) | 51 | 2.04 | [7] |

Per-input probabilities and logits for every run are in `runs.csv`; see figure B.

### Circuit types found

- **tanh-4:** `A AND NOT B + B AND NOT A` 21 (70%), `NAND + OR` 5 (17%), `B AND NOT A + NOT B + OR` 2 (7%), `A AND NOT B + NOT A + OR` 1 (3%), `OR + mixed(NAND)` 1 (3%). Primary circuit sizes {2: 27, 3: 3}. Nets with >1 minimal circuit: 67%. Zero-ablation gives the same sufficiency verdict in 60%.
- **relu-4:** `A AND NOT B + B AND NOT A` 7 (30%), `B AND NOT A + mixed(A AND NOT B)` 3 (13%), `A AND NOT B + mixed(B OR NOT A)` 3 (13%), `A + B AND NOT A` 2 (9%), `B AND NOT A + NOT B + mixed(A)` 1 (4%), `NOT A + OR` 1 (4%), `B AND NOT A + NOT B + OR` 1 (4%), `NAND + OR` 1 (4%), `A + B AND NOT A + NAND` 1 (4%), `A AND NOT B + NOT A + OR` 1 (4%), `B AND NOT A + B AND NOT A + mixed(A OR NOT B)` 1 (4%), `A AND NOT B + mixed(NOT A)` 1 (4%). Primary circuit sizes {2: 18, 3: 5}. Nets with >1 minimal circuit: 17%. Zero-ablation gives the same sufficiency verdict in 74%.
- **tanh-2:** `A AND NOT B + B AND NOT A` 6 (67%), `NAND + OR` 3 (33%). Primary circuit sizes {2: 9}. Nets with >1 minimal circuit: 0%. Zero-ablation gives the same sufficiency verdict in 100%.
- **tanh-8:** `A AND NOT B + B AND NOT A` 23 (77%), `NAND + OR` 4 (13%), `A AND NOT B + B AND NOT A + OR` 1 (3%), `B AND NOT A + NOT B + OR` 1 (3%), `A AND NOT B + NOT A + OR` 1 (3%). Primary circuit sizes {2: 27, 3: 3}. Nets with >1 minimal circuit: 100%. Zero-ablation gives the same sufficiency verdict in 23%.
- **relu-8:** `A AND NOT B + B AND NOT A` 11 (38%), `NAND + OR` 4 (14%), `A AND NOT B + mixed(NOT A)` 3 (10%), `A AND NOT B + B` 2 (7%), `B AND NOT A + mixed(NOT B)` 2 (7%), `B AND NOT A + NOT B` 1 (3%), `B AND NOT A + B AND NOT A + mixed(A OR NOT B)` 1 (3%), `NOT A + OR` 1 (3%), `A AND NOT B + NOT A` 1 (3%), `B AND NOT A + NOT B + OR` 1 (3%), `B AND NOT A + mixed(A OR NOT B) + mixed(NOT B)` 1 (3%), `A AND NOT B + A AND NOT B + NOT A` 1 (3%). Primary circuit sizes {2: 25, 3: 4}. Nets with >1 minimal circuit: 69%. Zero-ablation gives the same sufficiency verdict in 66%.

### Causal tests (mean over converged nets)

| architecture | intact | ablate circuit | ablate random same size | ablate all else | patch circuit: follows source | patch random | patch all else |
|---|---|---|---|---|---|---|---|
| tanh-4 | 100% | 80% | 77% | 100% | 73% | 48% | 27% |
| relu-4 | 100% | 57% | 73% | 100% | 94% | 45% | 6% |
| tanh-2 | 100% | 50% | n/a | 100% | 100% | n/a | n/a |
| tanh-8 | 100% | 94% | 94% | 100% | 37% | 23% | 63% |
| relu-8 | 100% | 72% | 89% | 100% | 72% | 25% | 28% |

A constant output gets exactly 2 of 4 inputs right, so 50% means XOR is gone. Ablating the primary circuit can leave XOR largely intact when *another* minimal circuit exists; compare it with the random same-size column and the redundancy figures above. When the primary circuit is *all* units, 'random same size' and 'all else' controls are undefined (n/a).

### Transplants between independently trained nets (share of pairs fully correct)

| architecture | pairs | native | ablation control | direct | aligned | sham direct | sham aligned | reference direct | reference aligned | direct, same type | direct, different type |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tanh-4 | 870 | 100% | 27% | 56% | 99% | 2% | 10% | 27% | 100% | 96% (n=442) | 14% (n=428) |
| relu-4 | 506 | 100% | 0% | 39% | 95% | 0% | 8% | 52% | 100% | 95% (n=56) | 32% (n=450) |
| tanh-2 | 72 | 100% | 0% | 50% | 100% | 0% | 0% | 33% | 100% | 100% (n=36) | 0% (n=36) |
| tanh-8 | 870 | 100% | 80% | 77% | 100% | 23% | 59% | 20% | 100% | 100% (n=518) | 43% (n=352) |
| relu-8 | 812 | 100% | 28% | 40% | 88% | 7% | 29% | 52% | 100% | 82% (n=132) | 32% (n=680) |

Read each row against its **ablation control**: if the recipient still computes XOR with its own circuit removed (high ablation control, as in wide networks), a successful transplant shows little. The informative rows are those where the ablation control is low and the sham control stays low.

Across activation functions (tanh-4 ↔ relu-4, 1380 pairs): direct 42%, aligned 95%, sham aligned 11%, reference aligned 100%.

### Representational similarity (tanh-4)

- Mean linear CKA between trained nets: **0.84**; between the same nets at initialisation: 0.75.
- Same circuit type: 0.91; different type: 0.77.
- Mean correlation of primary-circuit logit contributions across nets: 0.91.
- With 4 inputs, centred representations have at most 3 dimensions, so CKA is coarse. It measures similarity of *shape* only; it does not show that units play the same causal role.

Unit activation labels, trained vs the same nets at initialisation (null model):

| label | trained | untrained |
|---|---|---|
| B AND NOT A | 18% | 5% |
| B OR NOT A | 13% | 7% |
| OR | 12% | 8% |
| NOR | 12% | 4% |
| A AND NOT B | 12% | 2% |
| A OR NOT B | 11% | 3% |
| NAND | 7% | 0% |
| NOT A | 4% | 15% |
| NOT B | 2% | 10% |
| mixed(AND) | 2% | 1% |

### Discovery → confirmation

Dominant type in the discovery seeds: `A AND NOT B + B AND NOT A` (53% of 15 nets). On 15 held-out nets it accounts for 87%. Held-out causal checks: ablating the circuit leaves 78% accuracy (random same-size: 77%); direct transplant 71%, aligned 97%, sham aligned 8%.

## Evidence for hypotheses A–D (tanh-4)

The thresholds are fixed in `xorlab/report.py`. They summarise the evidence; they are not significance tests.

- **A (shared, interchangeable circuits): PARTLY SUPPORTED.** most common circuit type covers 70% of converged nets (rule: >=80%); direct transplant success 56% (rule: >=80%); aligned 99%; sham aligned 10% (rule: <=20%).
- **B (different mechanisms, same behaviour): SUPPORTED.** 5 distinct circuit types; 2 each cover >=15% of nets (rule: >=2). Direct transplant: same type 96% vs different type 14% (gap +0.82).
- **C (shared features, different causal use): NOT SUPPORTED.** between nets with DIFFERENT circuit types, mean CKA is 0.77 vs 0.75 for untrained nets (rule: > untrained + 0.05), and direct transplant between them works in 14% of pairs (rule: < 50%). Overall trained CKA is 0.84.
- **D (distributed / redundant computation): SUPPORTED.** 67% of converged nets contain more than one minimal sufficient circuit (rule: >=50% supported, >=20% partly). In 100% of nets at least one single unit is still indispensable, so the redundancy is partial. Ablating the primary circuit leaves 80% accuracy, against 77% for random same-size ablations.

These are not exclusive: several can hold at once, and the answer differs by architecture (see the tables).

## Changes from the original specification, and why

- **Circuit interface = logit contribution.** In a one-hidden-layer net a 'circuit' only talks to the output through its weighted sum, so this is the exact interface. It makes direct transplants well defined.
- **Mean ablation by default** (zero ablation reported as a robustness check): zeroing a tanh unit also removes its constant offset, which can break XOR for reasons unrelated to the computation.
- **Sham = donor's own circuit units before training.** Same size, same architecture, same initial weights; only training differs. This is a tighter control than random units.
- **Aligned transplant uses moment matching**, not a fitted map, so it cannot simply learn the answer. Powerful fitted maps can make any circuit look transplantable (Sutter et al., NeurIPS 2025).
- **Contribution-based circuit types** to avoid counting `AND with negative weight` and `NAND with positive weight` as different mechanisms.

## Limitations

- Four inputs: any similarity metric is low-resolution, and an exhaustive truth table means behaviour cannot distinguish mechanisms; only the internal analyses can.
- Boolean labels are a lens. ReLU units take graded values (e.g. counting A+B), and `mixed(...)` labels mark units that no gate describes well.
- 'Minimal circuit' depends on the ablation choice. Different ablations can give different circuits.
- Results hold for these architectures, initialisation and optimiser. Nothing here shows universality beyond XOR in 2→H→1 networks.

## Files

`summary.json` (all headline numbers), `runs.csv` (per seed, per input), `circuits.json` (every unit, every minimal circuit), `causal.json`, `transplants.csv` (every donor→recipient pair), `similarity.npz`, `checkpoints/*.npz` (initial and final weights), `training_logs/*.npz`, `figures/A–F`.
