# Binary addition in a small GPT

Produced by `python run_transformer.py` (NumPy GPT with TransformerLens-style hooks; gradients
verified against finite differences, worst relative error 6e-7). `run_transformer_tl.py` is the
same experiment written for the real TransformerLens library; it has not been run by the author
because PyTorch could not be installed in the authoring environment. All numbers below are from
`summary.json`, `patching.json`, `head_ablation.json`, `probes.json` and `msb_errors.json`.

## Setup

Two 10-bit numbers, lowest bit first, then the 11-bit sum, as tokens:

    a_0 … a_9  +  b_0 … b_9  =  s_0 … s_10        (lowest-first, "lsb")
    a_0 … a_9  +  b_0 … b_9  =  s_10 … s_0        (highest-first, "msb")

Models: a tiny GPT-2 (pre-LayerNorm, learned positions, 4 heads of 16, d_model 64, MLP 256),
with 1 or 2 layers. Three models: **lsb-L1**, **lsb-L2**, **msb-L2**. Trained 2,500 steps of
128 random problems with AdamW; loss on the sum tokens only. Evaluated by generating the sum one
token at a time (no teacher forcing).

**Prediction.** Lowest-first makes addition local: carry into bit k = majority(a_{k-1}, b_{k-1},
c_{k-1}) and c_{k-1} = s_{k-1} ⊕ a_{k-1} ⊕ b_{k-1}, so the model only needs the previous digit's
a, b and sum. Carry travel distance should not matter. Highest-first forces the whole carry chain
to be resolved before the first digit is emitted, as for the MLPs.

## 1. Accuracy

| model | random problems | by carry travel distance 0 … 9 |
|---|---|---|
| lsb-L1 | 100% | 100% at every distance |
| lsb-L2 | 100% | 100% at every distance |
| msb-L2 | 96.7–97.2% | 92–93% at 0, 98–100% at 1–7, 95–96% at 8, 100% at 9 |

Both lowest-first models are perfect, and distance is irrelevant, as predicted. The lowest-first
loss fell below 0.002 by step 400; the highest-first model needed the full 2,500 steps to reach
0.0065 and still makes errors. Its errors do not grow with travel distance; they sit in the
middle bits (bit 3 wrong on 3.0% of problems, bits 4–8 on 1–2%, bits 0–2 and 9–10 on under
0.03%) and are worst on problems where no carry passes through a propagate position (92%).
See section 3 for why.

## 2. What the heads attend to

Mean attention from the position that predicts sum bit i, by what the key token is (offset d
means a_{i+d}, b_{i+d}, s_{i+d}):

- **lsb-L1.** Head L0H3 attends to a_i and b_i (0.29, 0.28) and b_{i-1} (0.14). Head L0H1 attends
  to the previous sum digits s_{i-2}, s_{i-1}, s_{i-3} (0.21, 0.17, 0.14). The other two heads
  are diffuse.
- **lsb-L2.** Layer 0 repeats the pattern (L0H3: a_i, b_i, b_{i-1}; L0H2: a_{i-2}, b_{i-2},
  a_{i-1}). Every layer-1 head is nearly uniform (no entry above 0.04).
- **msb-L2.** Layer-0 heads attend to the current and next-higher digits (a_i, b_i, a_{i+1},
  b_{i+1}); layer-1 heads attend to the higher sum digits already emitted (s_{i+1}, s_{i+2}) and
  to `=`; one head (L1H0) looks 3–4 digits *below* the current one, where a carry would come from.

## 3. Activation patching (the causal test)

For bit k ∈ {3, 6}: take a problem, build a second one with the same bits ≥ k but the opposite
carry into bit k, and copy the second problem's residual stream (`hook_resid_pre`) into the first
at **one token position** at one layer. Measure how often output bit k flips to the second
problem's value.

| model | layer | positions that matter for bit k (flip rate) | every other position |
|---|---|---|---|
| lsb-L1 | 0 | s_{k-1} 0.35–0.38, a_{k-1} 0.31–0.38, b_{k-1} 0.14–0.33 | 0.00 |
| lsb-L2 | 0 | a_{k-1}, b_{k-1}, s_{k-1}: 0.30–0.37 each | ≤ 0.01 |
| lsb-L2 | 1 | the query position s_{k-1}: 1.00 | 0.00 |
| msb-L2 | 0 | a_{k-1}, b_{k-1} 0.16–0.22; s_{k+1} 0.24–0.29; s_{k+2} 0.12–0.19; a_{k-2}, b_{k-2} ≈ 0.09 | small |
| msb-L2 | 1 | s_{k+1} 0.25–0.61, s_{k+2} 0.21–0.29, s_{k+3} 0.06–0.07 | small |

The lowest-first models implement exactly the local rule: the only inputs to bit k are the
previous digit's a, b and sum, each necessary and none sufficient alone (the carry is a majority
of three), and positions further back have zero causal effect. By layer 1 of lsb-L2 the whole
decision sits at the query position.

The highest-first model does something the MLPs could not: it reads the carry into bit k
**backwards from the digits it already emitted above k**. Since c_{k+1} = majority(a_k, b_k,
c_k), the emitted s_{k+1} reveals c_k whenever a_k ≠ b_k. That is why patching s_{k+1} and s_{k+2}
flips the answer, why errors concentrate in the middle bits (the top bits are computed by lookahead
from the inputs, the low bits are then inferred from above), and why problems with no propagate
positions are hardest: when a_k = b_k the trick gives nothing.

## 4. Probes are uninformative here

A linear probe for the carry into bit k on the residual stream at the query position scores the
same on trained and untrained models (lowest-first: 0.71–0.75 at layer 0, 0.95–1.00 at the final
layer for both). In a teacher-forced sequence the tokens that determine the carry sit one position
back, and a random layer already mixes them enough for a linear readout. Presence is not use.

## 5. Head ablations (one head's output zeroed, sum generated)

| model | heads whose removal breaks addition | heads that do not matter |
|---|---|---|
| lsb-L1 | L0H3 → 0% (current digits); L0H1 → 81% (previous sum digit) | L0H0 100%, L0H2 99% |
| lsb-L2 | L0H3 → 0%; L0H0 → 59% | all four layer-1 heads: 100% each |
| msb-L2 | L0H2 → 0.4%, L0H3 → 2%, L1H3 → 8%, L1H2 → 14%, L1H1 → 16%, L0H1 → 31% | none; the mildest is L1H0 (55%) |

The lowest-first models use two heads (one reading the current digits, one reading the previous
sum digit); the second layer's attention is unused. The highest-first model needs all eight.

## Conclusion

Graduating from MLPs to a transformer changed what the network does because the task's format
changed what is easy. Written lowest digit first, addition is local, and a one-layer transformer
finds a crisp two-head circuit that activation patching pins to three token positions. Written
highest digit first, the model combines limited lookahead with a learned trick of reading carries
backwards from its own output, and it stays imperfect. The structure found is again the problem's
structure, not an engineer's adder: no model built a carry chain, and the "circuit" is clean only
where the format allows a clean one.

## Limitations

- One seed per model, one width, 10 bits. The attention summaries are averages; individual heads
  may be polysemantic.
- Patching one position at a time cannot reveal information that is spread over several positions
  redundantly (patching all low positions at once would).
- The NumPy GPT matches the TransformerLens architecture (pre-LN GPT-2 style) but not its exact
  initialisation; `run_transformer_tl.py` is provided to replicate with the library.
