# A GPT whose fourth attention head is a computer

Produced by `python run_cpugpt.py` (about 15 minutes; `--analyse` reuses the saved models). Every
number is in `summary.json`; figures in `figures/`; models in `models/`. Seeds are fixed.

## Answer first

Yes, the model has to learn to use the computer, and it can. A 2-layer GPT got one attention head
replaced by the frozen 8-bit neuron-gate CPU. It learned on its own which tokens to send (operand A,
operand B, operator), when to call, and how to read the result back. The best run answers **100% of
held-out problems** (+, −, ×, operand pairs never seen in training). A same-size GPT without the computer
gets 2–34%: it memorizes the training pairs and cannot do new ones.

How it learns matters more than anything else:

- **Rewarded only by its final answer**, the head mostly never works out how to use the computer
  (0 of 3 seeds fully, 2 of 3 partly, after a long flat start).
- **Also rewarded when the computer's own result is right**, every seed has its pointers settled
  within 250–1,000 training steps.
- **A small price per call** teaches it *when* to call: the best run calls on every math problem and
  never on the non-math tasks. Price it too high and it stops calling for multiplication pairs it has
  memorized, then fails new ones.

## Setup

**Model.** PyTorch, 2 layers, d_model 96, 4 heads of 24, MLP 384, learned positions, 277 tokens (every
number 0–255 is one token). Baseline: 4 ordinary heads in layer 2 (278,005 weights). CPU-head model:
3 ordinary heads + the CPU head (283,646 weights, of which 14,929 are the head's pointers, gate and
read-back; the computer itself is 4,389 frozen neurons and is not counted).

**The CPU head**, at the position that must predict the answer:

1. **Points.** Three slots, A, B and OP, each a softmax over earlier positions from its own trained
   query and key maps. One position per slot is sampled in training; the argmax is used at test.
2. **Sends.** The picked tokens go through fixed tables, the computer's bus format: a number token
   becomes its 8 bits; `+ plus sum` → ADD, `- minus subtract` → SUB, `* times` → MUL; anything else → 0.
3. **Decides to call.** A trained gate gives a yes/no (sampled in training, gate ≥ 0.5 at test).
4. **Runs.** The gate CPU (`gatecpu`) runs a 26-instruction calculator program (`gatecpu/calc.py`):
   x and y go in through the program port, op 1/2/3 = add/sub/multiply-by-a-loop, a 600-tick limit.
5. **Reads back.** If it called, the 8 result bits pass a trained projection into the residual stream,
   then the layer's MLP and the unembedding produce the answer.

No gradient passes through the computer or the yes/no choices, so the pointers and the gate learn by
REINFORCE (leave-one-out baseline over K samples per problem). Everything else learns by backprop.

**The computer's answers.** The calculator program was run on the gate netlist for all 262,144
possible calls (4 ops × 256 × 256), 64 machines per machine word (`gatecpu/batch.py`), in 11.5 s.
All match the instruction-level reference. Training looks answers up in that table (the gates are
deterministic). As a check, 500 test problems were re-run with every call going to the **fully analog
neural CPU** (neural gates, NAND-neuron flip-flops, nothing rounded): 0 results and 0 answers differed,
113 clock ticks per call on average, worst neuron output 0.030 from a clean 0/1.

**Problems.** 11 math wordings, chosen so the operands move: `a + b =`, `what is a times b ?`,
`sum of a and b =`, `subtract b from a =` (operands reversed), `c ; a - b =` (a distractor number), and
so on; 3 non-math tasks the computer cannot help with: `copy a =`, `first of a and b =`,
`last of a and b =`. Subtraction keeps a ≥ b; multiplication keeps b ≤ 15 and a·b ≤ 255.
About 20% of operand pairs are held out from training, by a fixed rule.

**Conditions** (3 seeds each, 6,000 steps, batch 256, AdamW):

| condition | pointer reward | call price |
|---|---|---|
| baseline | – (no computer) | – |
| CPU head | answer's log-likelihood | 0 (16 samples, gate starts open: the most favourable pilot settings) |
| CPU head + check | the same + 1 when the computer's result equals the answer | 0.01 |
| CPU head + check, pricier calls | the same | 0.05 |

## Results

Held-out accuracy (operand pairs never seen in training), 7,000 test problems per model, every seed.
Wilson 95% intervals are within ±2.5 points everywhere and are in `summary.json`.

| model | seed | a + b | a − b | a × b | non-math | pointers right | calls on math / non-math |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 2.3% | 1.9% | 18.4% | 100% | – | – |
| | 1 | 2.4% | 10.4% | 18.3% | 100% | – | – |
| | 2 | 3.4% | 1.9% | 18.0% | 100% | – | – |
| CPU head | 0 | 17.1% | 70.1% | 19.5% | 100% | 45% | 100% / 100% |
| | 1 | 67.5% | 100% | 18.4% | 100% | 71% | 100% / 100% |
| | 2 | 2.5% | 2.8% | 20.1% | 100% | 0% | 100% / 100% |
| CPU head + check | 0 | **100%** | **100%** | **100%** | 100% | 100% | 100% / 100% |
| | 1 | 100% | 100% | 86.8% | 100% | 73% | 100% / 100% |
| | 2 | 100% | 100% | 86.5% | 100% | 73% | 100% / 100% |
| … pricier calls | 0 | **100%** | **100%** | **100%** | 100% | 100% | **100% / 0%** |
| | 1 | 100% | 100% | 18.1% | 100% | 82% | 71% / 0% |
| | 2 | 59.4% | 97.0% | 20.4% | 100% | 82% | 70% / 68% |

On training pairs the baseline scores 96–100%: its failure on new pairs is memorization, not capacity.

**The computer is what does the math.** Switching the CPU head off in the best models drops held-out
accuracy to 0% (add), 1.6–1.9% (sub) and 8.1% (mul). Swapping the operation the computer receives
(add→sub→mul→add) gives 0–3.6%.

### Learning to use it (figure 1)

With the result check, the pointers settle within 250–1,000 steps in every seed (at 100%, or at
73–83% where a seed sends some operands backwards, see below), and held-out accuracy follows. With only the answer's likelihood, nothing happens for about 1,750
steps: the rest of the network learns to memorize training pairs, which removes most of the reason to
use the computer. Only then do two of three seeds start pointing usefully, and they stay partial.
Rewarding the tool's own output is what makes the interface learnable in this setup.

### Two things it learned that nobody told it

- **Operands backwards, then repaired.** In seeds 1 and 2 of "+ check", for the three short wordings
  `a + b =`, `a - b =`, `a * b =` (and only those), the head sends the operands swapped (y = a, x = b) in
  97–99% of problems. For addition that is harmless. For subtraction the computer returns b − a, and the
  network learned to turn that into a − b itself: these seeds still get 100% on subtraction. For
  multiplication, a swapped call loops a times instead of b, and when a > 17 it runs past the 600-tick
  limit and returns 0. That is the whole 13% loss on multiplication.
- **Not calling when it thinks it knows.** With a call price of 0.05, seeds 1 and 2 stop calling for
  multiplication. There are only about 1,400 legal multiplication pairs, so the model memorizes the
  training ones, and a call no longer improves its answer there; it then answers held-out
  multiplication from memory and gets about 20%. Seed 0 kept calling and is perfect. With the 0.01 price
  every seed calls for every math problem, but also for every non-math problem, where the call is
  harmless but wasted.

## Cost

| | per problem |
|---|---|
| GPT forward pass | about 1.1 M multiply-adds |
| a call: computer clock ticks (add/sub, multiply) | 32 / 54 + 32·y ticks; 113 on average on the test mix |
| a call on the analog neural CPU in NumPy | about 0.08 s |
| a call in the browser (JavaScript, analog) | about 0.01 s for an add, 0.13 s for 13 × 15 |

## Try it

`cpugpt_ui/index.html` (with `model.json` and `cpu.json`) runs the best model in the browser: type a
problem in the supported wording; the page shows where each pointer looks, whether the head calls, the
computer running live on neuron gates (PC, A, OUT, clock ticks), and the answer, next to the same GPT
with the head switched off and the baseline GPT. It also says whether the numbers were held out.

    python3 -m http.server 8765 --directory cpugpt_ui      # then open http://localhost:8765

## Limitations

- The bus format (token → bits, word → opcode) is given, not learned. Learning it would need gradients
  through the computer or far more exploration.
- One call per answer, always at the last position, with a fixed calculator program. The model does not
  write programs.
- Numbers are single tokens, which makes memorization easy and generalization hard for the baseline;
  a digit-level tokenizer would change the baseline's numbers.
- 3 seeds per condition; the "+ check" and "pricier calls" results differ by seed as described.
