# 16-bit addition: does a bigger problem force carry-like parts?

Produced by `python run_adder16.py` (about 7 minutes the first time; trained weights are saved
to `nets.npz`, so later runs skip training). All numbers are in `summary.json`; figures in `figures/`.

## Setup

Two 16-bit numbers in (32 bits), 17-bit sum out. Multilayer perceptrons with width 256, tanh,
depth 2 and depth 4, 3 seeds each, trained on 60,000 random problems for 40 epochs. Tested on
100,000 fresh random problems, plus a pool of problems built to contain a carry that travels a
given distance (see below), binned by the distance they actually have.

**Carry travel distance** = the longest run of consecutive *propagate* positions (exactly one of
aᵢ, bᵢ set) that a single carry has to pass through. A position where both bits are 1 generates
its own carry and resets the distance to 0. Adding 1111 1111 + 0000 0001 has travel distance 7:
the carry generated at bit 0 crosses bits 1–7.

**Correction.** An earlier version of this report binned problems by plain carry-chain length
(consecutive positions whose carry-out is 1, generate positions included) and used built test
sets whose true chain lengths were longer than their labels. Chain length turns out not to
predict failure at all: on random problems with chains of 16, depth-2 networks still score 95%,
because most positions in a random chain generate their own carry. What fails the networks is a
carry that has to travel far through propagate positions. The analysis now uses that measure.

## 1. Accuracy hides a limited carry reach

On random problems every network looks nearly perfect: 99.5–99.9% of sums exactly right. Binned
by carry travel distance (exact-sum accuracy, range across 3 seeds):

| travel distance | ≤ 7 | 8 | 10 | 12 | 13 | 15 |
|---|---|---|---|---|---|---|
| depth 2 | 97–100% | 91–94% | 69–76% | 42–49% | 27–37% | 15–28% |
| depth 4 | 97–100% | 96–99% | 88–94% | 76–83% | 56–70% | 39–71% |

Accuracy is flat up to a distance of about 7, then falls steeply, and the deeper networks hold
up better at every distance beyond that. That is the signature of carries being resolved a
limited distance per layer, which a carry chain does and a single global "compute the total"
step would not.

Two cautions. First, long travel distances are rare in random training data: each extra
propagate position halves the odds, so only 473 of the 60,000 training problems (0.8%) have a
distance of 8 or more, and 15 have 12 or more. Accuracy starts falling where training examples
thin out, so scarcity of practice is a confound. Depth still matters on its own: both depths saw
identical data, and depth 4 is better at every long distance. Second, the built test problems
are random apart from the forced run, so the pool is not uniformly random given a distance.

## 2. The total emerges gradually, layer by layer

Distance between two problems with the same total ÷ distance between two random problems
(0 = same-total problems identical inside, 1 = the total plays no special role):

| | layer 1 | layer 2 | layer 3 | layer 4 |
|---|---|---|---|---|
| raw inputs | 0.98 | | | |
| depth 2 (3 seeds) | 0.97 | 0.85 | | |
| depth 4 (3 seeds) | 0.98 | 0.93–0.95 | 0.72–0.81 | **0.36–0.40** |
| untrained | 0.98 | 0.99 | 0.99 | 0.99 |

In the 4-bit task the total was largely in place by layer 2. At 16 bits it builds up across
layers and is only well formed at the last layer of the deep networks. This fits section 1: each
layer settles more of the carries, so problems with the same total only converge once enough
carries are resolved.

## 3. Per-position XOR is computed selectively

A straight-line probe for aᵢ XOR bᵢ scores 47–52% on the raw inputs (chance) and about 70% on
untrained networks at every position. Trained networks are near 100% at bits 0 and 1 and well
above baseline at a few other positions (depth 4, layer 2: bit 6 97%, bits 9–10 86%), but near or
below the untrained baseline at most others. So some XOR-like features form, at particular
positions, rather than a uniform row of XOR gates.

## 4. No single carry direction is used

Copying the carry-into-bit-k coordinate (k = 4, 8, 12) from a problem with the same high bits but
the opposite carry, along the best straight-line carry direction in each layer:

- The answer moved by exactly 2ᵏ in 0–4% of problems (random direction: 0%).
- In most layers the answer did not change at all (95–100% unchanged).
- Exception: depth 4, layer 2, for bits 8 and 12. Patching there disrupted the answer (only 8–33% unchanged) without
  producing the carry-flipped answer, so that direction matters but does not act as "the carry".

So there is no single carry wire that can be read and rewritten, unlike the tiled adder.

## Conclusion

The bigger problem changed the answer from the 4-bit study:

- **4 bits:** a quick partial collapse to the total by layer 2, and no carry-like behaviour.
- **16 bits:** carry-like *behaviour*. Accuracy depends on how far a single carry has to travel
  through propagate positions, deeper networks reach further, and the total is assembled
  gradually across layers. But not carry-like
  *parts*: no single direction acts as a carry bit, and XOR features appear only at some positions.

So the network propagates carry information in a distributed way, without the clean wires a tiled
adder has. The function and its depth-limited structure resemble a carry chain; the
implementation does not.

## Limitations and next steps

- One width (256), one optimiser, 3 seeds per depth. Training on random problems leaves long
  travel distances rare (0.8% of problems at 8 or more). Training on more of them might change
  both accuracy and mechanism, and would separate the depth effect from the data effect.
- The carry test looks for one straight-line direction per layer. A carry spread across a curved
  or multi-dimensional subspace would be missed. Distributed alignment search (learning a small
  subspace with interchange interventions) is the stronger next test.
- Multilayer perceptrons see all 32 bits at once. Transformers reading digits one at a time
  behave differently and are a natural comparison.
