# Qwenputer and the gate computer: the same idea at two sizes

Written 1 Nov 2026 from the Qwenputer model card, its GitHub design notes and its weights
(`qwenputer/model/`, the Filesystem variant, 22.3 GB, downloaded from Hugging Face; the ModelScope
page the reel links to is the author's mirror of the same files). Sources are at the end. The
measurements on the weights are in the section "Inside the weights", made with
`inspect_weights.py` (scratch script; its output is quoted there).

## Answer first, in plain words

**What the reel's model is.** Qwenputer is Qwen3.5-4B with a small computer written into it. Not
trained in: *compiled* in. Someone wrote a 32-bit RISC-V processor the way you would write it for
a chip, then had a program turn every logic gate into weights of ordinary transformer layers, and
glued 12 such layers in front of Qwen's own 32. The computer's memory is not in the weights at all.
It lives in the running state that Qwen3.5's "Gated DeltaNet" layers already keep (a kind of
attention that carries a small matrix forward from token to token): the compiled layers write one
row of that matrix per step, exactly, and read one row back, so the matrix is RAM. Qwen's own
weights are untouched, and the model loads with the stock `transformers` class.

**How Qwen uses it.** Qwen writes an ordinary tool call, `<tool_call>` … `calc` … `</tool_call>`,
exactly as it was trained to do for any tool. Nothing outside the model runs it. The compiled
layers read the call out of the token stream, run the program (a few thousand "trace" tokens that
are the computer's execution log), and then write the `<tool_response>` digit by digit as real
tokens. Qwen reads the answer like any tool response and carries on. So it *is* a tool call in
form, and it is *not* a tool in substance: the tool is more layers of the same network.

**What it scores.** On big multiplications and divisions where base Qwen3.5-4B gets 55-90 %, it
gets 97.5-100 %, the same as base Qwen calling a real Python interpreter. Every answer the
computer printed was exact (checked against a reference machine on hundreds of thousands of
positions); the misses are Qwen's (it re-verifies, loops, drops a sign). Word problems (GSM8K)
are a tie, because those are limited by reasoning, not arithmetic. Qwen's text behaviour is
unchanged to within the noise of the kernels (KL 6e-4 against a floor of 5e-4).

**How it ties to this repo.** This repo built the same thing from the other end and at a tiny
size: logic gates as *trained* neurons (one sigmoid per AND, two for XOR), an 8-bit computer from
4,389 of them, that computer wired into a 2-layer GPT as an attention head (100 % on unseen sums),
then a story GPT that types calls to it in the text (99 % on word problems, two-step chains
99.3-99.8 %), and then the capacity measurement of the last two days (delegating the sums frees
0.048-0.055 nats of language at width 128). Qwenputer is that programme at 4 billion weights with
one big change: the gates are written, not learned, so they are exact and cost Qwen nothing. The
place where the two meet is also the most useful thing to do with the download: Qwenputer's first
12 layers are a transformer in which **every logic gate is known in advance**, which is exactly the
ground truth the circuit-finding work in this repo has never had.

**The business-projection idea.** Yes in principle, and the two projects together say how. The
calculator can be as complex as you like, because the model never has to learn what it computes,
only how to call it, copy numbers in and copy the answer out; phase 8 showed those cost almost
nothing (0.007-0.010 nats). What the model must learn is the *plan*: which numbers to send, in
what order, and what the answer means. At this repo's size (1.35 M weights, a 64-token context,
numbers as single tokens up to 255) that is viable for a fixed procedure of a few chained calls
over small numbers, and the two-step result shows the hand-off from one call to the next already
works perfectly. "Give it a bunch of business numbers and ask it to predict next year's growth"
needs three things this model lacks: numbers with digits and decimals, a longer context, and the
language and reasoning to decide what to compute from a free-form question. The first two are
engineering (below). The third is what a 4B model has and a 1 M model does not, and it is the
reason Qwenputer's author put the computer inside Qwen rather than inside a small model. The plan
below does the small version first, as a measured procedure, and names what it can and cannot show.

## What Qwenputer is, in facts

| | Qwen3.5-4B (base) | Qwenputer Filesystem (this download) |
|---|---|---|
| layers | 32 (3 Gated DeltaNet to 1 full attention) | 12 compiled + the same 32 |
| hidden size | 2,560 | 4,608 (Qwen's 2,560 plus 2,048 computer dims) |
| vocabulary | 248,320 | 484,800 (plus 236,469 trace tokens) |
| what the extra layers hold | | RV32IM + Zba core, a bignum coprocessor (8 registers of 128 decimal digits), a filesystem (1,024-slot hashed directory, 512 files, about 53 k tokens of text) |
| learned weights in the extra layers | | none; a compiler computes every one |
| size on disk | 9.3 GB bf16 | 22.3 GB bf16, almost all zero padding; about 18 GB of GPU memory to run |
| exactness | | bit-exact in bf16; 230,000 output positions replayed against a reference machine with zero mismatches |
| text behaviour | | KL 5.5e-4 against base, noise floor 5.1e-4 |
| arithmetic, DeepMind sets (40 problems each) | mul_big 55, div_big 72.5, add_or_sub_big 87.5 | 97.5 / 97.5 / 97.5; base with a host Python tool: 100 / 95 / 97.5 |
| memory across sessions (90 questions) | 7.8 | 83.3 (host Python tool with the same five file tools: 78.9) |

The RISC-V core in these 12 layers (design notes, P2): 2,679 circuit dims, 99 DeltaNet key heads,
42,012 MLP units, 1,040,595 nonzero weights; it passes the official riscv-tests (rv32ui, rv32um,
rv32uzba) and runs C and MicroPython compiled with gcc. The filesystem layout adds a twelfth layer.
It runs 1.2-1.4 trace tokens per instruction, about 3,200 trace tokens a second on an L4 GPU with
the project's own executor, so a `calc` call costs 0.5-1 s and a multiplication of two 8-digit
numbers is a few thousand trace tokens. Running it here is not practical: the Mac has 24 GB in
all, the checkpoint needs about 18 GB on the device, and the fast executor is CUDA-only. Reading
the weights is practical and is what the download is for.

## How the circuit manifests in the architecture

The whole trick is that a transformer layer already contains the three things a computer needs,
if you are allowed to write the weights by hand.

**Gates are MLP units.** A Qwen MLP unit is `down · (silu(gate · x) × (up · x))`. The compiler
(`chipformer/q35/prims.py`) makes an AND of n bits as one unit whose gate row has weight 4 on each
input bit and −(4n − 2) on a constant dim, so the pre-activation is +2 when all bits are 1 and at
most −2 otherwise; every value the circuit produces is an integer no larger than 256 and the
pre-activations are scaled by 128, so `silu` is an exact ReLU in bf16 and a false gate is exactly
0. Thresholds (`[z ≥ θ]`), ranges and equalities are two such units subtracted; a one-hot of an
integer is one AND per value; a 32-bit add is two MLP stages (nibble sums are free linear forms,
then thresholds, then carries as AND literals). Compare this repo's smallest gates
(`results_minigates/`): an AND there is one *trained* sigmoid neuron with weights near 8 and a
margin of 4, chosen because a gate must restore its signal to within 0.05 of the rails to survive
being chained thousands deep. Qwenputer's AND is the same threshold unit with the margin set by
hand (half a unit, so that bf16 noise cannot flip it) and no sigmoid, so there is nothing to
restore: the output is an exact integer.

**Memory is the delta rule.** A Gated DeltaNet head updates its state matrix as
`S ← α S + β k (v − Sᵀk)ᵀ`. With the decay α held at 1 (nothing fades), the write strength β at 1
and a one-hot key k, that is an exact overwrite of one row with v. A one-hot query then returns
exactly that row. So each head is a small RAM with no positions and no large attention scores,
which is why the earlier CHIP-8 version's float64 is no longer needed. The low bits of an address
pick the head and the rest pick the row; each key head drives two value heads; registers, the
program counter, the call stack and the ROM (pre-decoded, 8 bytes per instruction, loaded as a
snapshot) each get heads of their own. The gate is used too: α = 0 for one token wipes a head,
α = ½ makes a head halve every step, which serves as a counter. The file text of the Filesystem
variant sits in other heads, eight tokens to a row.

**Time is the token stream.** The computer's state changes only through tokens: an instruction
is one record of one to four trace tokens (`F0 · V F1 · V V F2` for a register write, `J(word)`
for a jump, `ST0/ST1` for a store, `HALT`), the key for a write is taken from the previous token
through the causal conv1d, the write gate from the current token, and the value from token
payloads at lags 0-3. Every position computes the instruction its *next* token belongs to, so the
trace is the execution log and the model's only job, as always, is to predict the next token. This
is the same "commit rule" as this repo's gate CPU, where every register is a flip-flop that changes
only at the clock edge, with the token stream as the clock.

**The bridge.** The compiled layers read Qwen's tokens too. Digits, the tool-call tags and the
function names in the spellings Qwen actually writes are "text classes" that the circuit can test
for; at `</tool_call>` the circuit takes over the positions (an owner signal `tau`), copies the
call's text into RAM at 0x8000 (16 bytes per token, up to 80 tokens), runs the firmware (`calc.c`,
built with gcc, parked between calls), and prints the response one byte per token through a
"print" instruction that the output code turns into the matching Qwen token. On text positions
the last compiled layer clears every computer dim except a 21-dim output code, `tau` and a mute
marker `mu`, so Qwen's layers receive their usual input. On trace positions, columns on `mu` push
Qwen's DeltaNet gates and attention output gate to −1024, so Qwen's state passes through a trace
unchanged and pure text stays bit-identical. Trace tokens carry a huge constant (2^40) in a few
"ballast" dims so that every RMSNorm on them divides by an exactly known power of 2, which the
compiler folds into the weights. The final norm's gains on the output code and `tau` (2^20, 2^60)
let whoever owns the position win the argmax by a wide margin.

So the "architecture that allows this circuit to manifest" is Qwen3.5's hybrid layout: the
DeltaNet layers give exact, position-free memory; the MLPs give exact gates; the shared token
stream gives a clock and an execution log; the tool-call format Qwen was trained on gives an
interface the model already knows how to use. Nothing is learned and nothing of Qwen's is used.

## Side by side with this repo

| | this repo | Qwenputer |
|---|---|---|
| logic gate | one trained sigmoid neuron (AND, OR, NAND, NOR), two for XOR; margin 4; static discipline ε = 0.05 | one compiled SiLU unit, weights 4 and −(4n − 2), exact integers in bf16 |
| computer | 8-bit SAP-1: 32 bytes of RAM, 8 instructions, 4,389 neurons, 13,141 weights, 29 gate levels; checked tick by tick against a Python CPU | RV32IM + Zba + bignum coprocessor: 32 KB RAM, 256 KB ROM, 128 KB data ROM, in 11-12 layers, 42 k units, 1.04 M nonzero weights; passes riscv-tests, runs C and MicroPython |
| where memory lives | NAND flip-flops (8 gates per bit) in feedback | rows of Gated DeltaNet state, written by one-hot keys at β = 1 |
| clock | one neural tick per gate level, 32 levels per CPU step | one trace token per write; 1.2-1.4 tokens per instruction |
| how the language model reaches it | (a) an attention head with three trained pointers, frozen bus tables, REINFORCE (`cpugpt`); (b) the model types `<calc> a op b =` and the machine writes the answer into the text (`storytool`) | the model writes Qwen's own `<tool_call>`; the compiled layers read it from the stream and write the `<tool_response>` themselves |
| what the model learns | when to call, what to send, how to copy the answer back; 100 % on unseen sums (cpugpt), 99 % on word problems, 99.3-99.8 % on two-step chains (storytool) | nothing: Qwen's weights are untouched; the prompt tells it the tool exists |
| cost to the language | measured: chips 0.007-0.010 nats; computing instead costs 0.055-0.066, of which delegation recovers 0.048-0.055 (`results_storyroom/`) | zero by construction (KL 6e-4, at the noise floor); never measured as a training question |
| numbers | single tokens 0-255; 8-bit results, wrapping | decimal digits as tokens; 62-digit values, exact rationals, decimals |
| chaining | two calls, the second reading the first's result, 100 % hand-off | the firmware evaluates whole expressions with brackets; Qwen chains calls itself (mul_div_multiple_longer 40/40 at calc v2) |
| context | 64 tokens | 262,144 |
| exactness | gates are analog; wrong-answer modes studied on purpose (the flaky machine, `results_storycheck/`) | bit-exact; the misses are the language model's |

The two projects ask different questions with the same machine. This repo asks *what a small model
can learn about using a computer, and what it is worth to it*; Qwenputer asks *how far a fixed
language model gets when the computer is free and exact*. Each has the answer the other lacks:
Qwenputer never trained a model with the computer present (so it cannot say whether the model
would come to rely on it and spend its own arithmetic circuits on language), and this repo cannot
make the computer exact or large (its gates are analog and its numbers are bytes).

## How complex can our computer be, and what it would take to do business projections

**The computer can be arbitrarily complex; the plan is the limit.** Three results fix this.
Phase 8's chip cost (0.007-0.010 nats) says the model pays nothing for a tool it does not have to
understand. The two-step result (the second call starts from the first result 100 % of the time)
says a chain works when the thing to copy is a few tokens back. Qwenputer's own tables say what
stays hard with a perfect computer: word problems are a tie, and every one of its misses was Qwen
deciding what to compute, not the computer computing it. So the question "could it project next
year's growth from a table of numbers" is really "can the model turn that table and that question
into a sequence of calls", and that is a language task that scales with the language model.

**What this repo's model can do now and cannot.** It can: call a 3-operation calculator on
numbers 0-255, chain two calls, decide when not to call (restating a number 21-62 % of the time in
sentence shapes it never saw). It cannot: write or read a number above 255, a decimal or a
percentage (each number is one token); hold more than about three problems in its 64-token
context; divide; or choose a formula it was not trained on.

**A staged plan on this repo** (each stage measurable, each a day or less on the laptop):

1. **Digit numbers.** Tokenise numbers as digits (or as two tokens, hundreds and units) and let
   the calculator hold a bignum. The calculator can stay a table (`cpu_table.npz`) for 8-bit
   operands and become an evaluator for longer ones; in the repo's own spirit, the gate CPU can
   grow to 16 bits and run division by repeated subtraction, which `gatecpu/calc.py` already does
   for multiplication. Measure: copying accuracy by number length, the thing Qwenputer's bridge
   gets for free and a learned model has to earn.
2. **Longer chains.** Generalise `two_step_snippet` to n steps with named intermediate results
   ("the increase was <calc> 120 − 100 = 20 </calc>, the rate <calc> 20 / 100 = 0.20 </calc>, next
   year <calc> 120 × 1.20 = 144 </calc>"), context 128-256 tokens. Measure: hand-off accuracy by
   chain length and by distance to the value being copied; this is where the "most recent write"
   worry from the research notes gets a real test.
3. **Fixed procedures.** Train on templates that are exactly the business computations (growth
   rate, average growth over k years, compound projection, share of total, break-even) with the
   answers masked out of the loss as in condition A, so the model learns the procedure and never
   the arithmetic. Measure: whole-procedure accuracy on unseen numbers, and the capacity contrasts
   of phase 8 again at this harder task (B computes it all itself, C masked, D stories only). The
   prediction from phase 8 is that the gap grows with the arithmetic load.
4. **Free-form questions.** This is the stage that will fail at 1.35 M weights, and it is worth
   running once to show it: hold out whole question templates and watch accuracy fall to the
   restating rate. The honest conclusion to write is that the plan is a language skill, and the
   model that has it is a Qwen, which is why Qwenputer's computer is inside Qwen.

**What the business tool would look like if built for real.** Qwenputer is already the right
shape: a `calc` tool whose firmware is a C program, prompt-listed like any tool, exact to 62
digits, with decimals and rationals. The thing to add is not a bigger computer but a *procedure
library* (growth, CAGR, projection, with the formula in the tool and only the numbers sent by the
model), so that the language model's job shrinks to naming the computation and copying the numbers,
the two things small models provably do well.

## Inside the weights

Measured on the downloaded checkpoint with `qwenputer/inspect_weights.py`, `pairs.py` and `classes.py`
(outputs in `qwenputer/*_out.txt`). Nothing is loaded into memory but one tensor at a time.

**The computer is 1.29 M nonzero weights in 2.44 billion slots.** The 12 compiled layers hold 20,419
(layer 0) to 302,964 (layer 7) nonzero weights each, 1,289,104 in all, 0.053 % of their 2.44 B
entries; the design notes' count for the RISC-V core alone is 1,040,595, and the rest is the bridge
and the filesystem. Qwen's own 32 layers are 55.6 % nonzero, which is exactly 2,560 / 4,608: the
widening padded them with zeros and changed nothing else. So the 22 GB checkpoint is 9.3 GB of Qwen,
about 2.6 MB of computer, and zeros.

**Memory is held open, as documented.** In every one of the 12 layers all 32 DeltaNet value heads
have `A_log = 0` and `dt_bias = −110`, so the decay α is exactly 1 at rest: nothing fades unless a
token says so. Layer 0 uses 13 of its 2,048 key rows and 64 of 4,096 value rows, with two distinct
weight magnitudes, which is the program counter, counter and previous-token heads of the design's
first layer. Layer 2 is the ROM: all 2,048 query rows are used, each a 12-term sum of powers of two
(the PC bits, making the 2-hot address), and **no key row and no value row at all**. A memory that
is read and never written by any token is exactly a read-only memory whose contents arrive in the
state snapshot, as the design says.

**The gates are visible, and they line up with the datapath.** 51,261 of the 110,592 MLP units are
live (gate row and down-projection both nonzero); layer 4 is full (9,216 of 9,216), layers 2 and 10
nearly empty (21 and 19). The fan-in of the gate rows, excluding the constant, follows the
design's layer table: layer 0 is single-input thresholds (1,549 of 1,793 units read one dim; 1,110
are integer thresholds, 53 of them in ± pairs writing the same output, the `[z ≥ θ]` gadget); layer 1
sums 66-69 bits per unit (1,269 units: the PC word and link); layers 4 and 5 are the adder (fan-in
8-17, then 4-5, with 408 threshold pairs in layer 5); layer 7 has fan-in 11-30 (the multiplier and
division compare columns); layer 9 fan-in 7 (lookahead); layer 11 has 1,784 units with no input but
the constant, which is the output code that prints the response tokens. The textbook AND signature
(equal literal weights, a half-integer constant) is found in only 33 units, and the reason is itself
instructive: the emitter folds the RMSNorm of the 2^40 ballast into the weights, so constants show
up as −2^46 or −2^47 on a per-layer "one" dim (3255 and 3282 in layer 0, 3416 in layer 1, 3708 in
layer 4) and inputs arrive at different scales. The first unit of layer 0 reads as gate
56·x − 2^46·one, up-projection −2^29 on two dims, +1 into dim 2584; the second is the same with 48
and −1. That is a threshold pair with the scaling folded in, and a circuit-finding method that does
not know the scale conventions would have to discover them first.

**The two halves of the residual stream never touch.** Not one compiled weight reads any of Qwen's
2,560 dims: zero key, query, value or MLP rows in all 12 layers. Trace tokens have zeros in all
2,560 Qwen dims and 44-48 nonzero computer dims. The bridge works through the embedding table
instead: every one of Qwen's 248,320 tokens carries 20 ballast dims at 2^40 and **its own 21-bit
code in dims 2710-2730, bits written as ±128**, so the circuit recognises digits, `<tool_call>`,
function names and anything else by decoding the code with ANDs, and copies text into RAM the same
way. The final norm gains say who speaks: Qwen's dims median 1.44, the owner dim `tau` 2^60, the
21 output-code dims 2^20, the mute marker −1. The trace vocabulary is 236,745 added tokens named by
family and payload (`<|rv:V:0|>` … `<|rv:SD:0|>`), and `<tool_call>` / `</tool_call>` are Qwen's
ids 248058 and 248059, as `chipformer.json` lists them.

For the circuit-finding work this means the ground truth is unusually clean: the computer's units
are sparse (median one input per unit in several layers), integer-valued, and separated from the
language model by construction, so a discovery method's false positives and misses can be counted
exactly, and the one real difficulty, the folded scaling, is the same difficulty a trained network
presents.

## Calibration: this repo's circuit finders scored against the compiled wiring (`calib/calibration.png`)

`python3 qwenputer/calibrate.py model calib 32 128 zero|random`, about 7 minutes a run. An own forward pass of
the 12 compiled layers (Gated DeltaNet with the delta-rule recurrence, zero-centred RMSNorm, the gated head norm,
l2-normalised keys and queries, SwiGLU; nothing of Qwen's 32 layers, which the compiled weights never touch) is
run on 32 sequences of 128 random trace records of the documented shapes (`F0`, `V F1`, `V V F2`, `NT`, `J`,
`ST`, `V V V SS`, `QS QD…`, `BD… V`). The author's compiler and emulator were not used, so there is no firmware
and no real program: the memory starts either **empty** or filled with **random bits** (density 0.05 in every
DeltaNet head), and the circuit computes whatever those bits make it compute. The ground truth needs neither:
a unit's inputs are the dims its gate and up rows read, 51,261 wired units in all, scored on the dims that vary
at least once in the run. Two methods from this repo then run blind on the activations:

- **M2, the mean-ablation scan** of `results_storyroom` (`inside()`): replace one dim by its mean over the run,
  recompute the unit, count the dim as an input if the unit changes anywhere. (Dims a unit does not read never
  change it, so the scan's precision is 1 by construction; the question is recall.)
- **M1, the correlation ranking** of `xorlab` (`corr`, `bool_label`): rank every dim by |correlation| with the
  unit's activation; take the top k with k the true fan-in, or every dim with |corr| ≥ 0.5.

| | empty memory | random memory |
|---|---|---|
| wired units with a varying input | 8,118 | 49,803 |
| of which ever fire | 1,573 (19 %) | 46,356 (93 %) |
| M2 recall of true inputs, all wired units | 0.19 | 0.75 |
| M2 recall, units that fire | **1.00** | **0.81** |
| M1 recall at k = fan-in, units that fire | 0.81 | 0.27 |
| M1 precision at \|corr\| ≥ 0.5 | 0.40 | 0.24 |
| M1 recall at \|corr\| ≥ 0.5 | 0.31 | 0.32 |

**What the ablation scan gets right and wrong.** On units that fire, with empty memory, mean ablation finds
every input of every unit: 1,573 units, 100 %. That is the clean case the scan was designed for and it passes.
With random memory, which drives 46,356 units, it finds 81 % of their inputs, and the misses are systematic:
layer 1, the program-counter word, is built of units that sum 66-69 bits against a threshold, and ablating any
one bit rarely crosses it (recall 0.35 on units that fire); the 8-input units of the adder and multiplier
(fan-in 8, 23,943 units) show 62 % of their inputs, the 2-input units 90 %. A dim that only enters the
up-projection, or a literal that is always 1 whenever the unit fires, is invisible to a one-at-a-time scan. So
the scan under-counts a unit's inputs when they are many and redundant, and it under-counts units when the data
does not exercise them: with empty memory only 19 % of the wired units fire at all, and the scan can say nothing
about the other 81 %. For `results_storyroom`'s "arithmetic-only neurons" this means the counts (58 / 50 for B,
0 for A) are lower bounds in two ways, by coverage of the problem set and by redundancy within the circuit; the
direction of the A-against-B comparison is safe, since both models were scanned on the same positions.

**What correlation gets wrong.** Correlation ranking finds the right inputs when a unit has few of them and the
data is simple (81 % at the true fan-in with empty memory, mostly single-input units in layers 0, 1 and 11),
and falls to 27 % once the circuit is busy, with precision 24 %: a compiled gate's output correlates with
every dim that co-varies with its inputs in the trace, and in a datapath almost everything co-varies. The
threshold rule is worse than the top-k rule at both settings. This is the expected failure of a correlational
method on a circuit with shared inputs, now measured on a real one.

**On real programs, with the compiler's own labels.** With permission to run the author's package, the
second script, `qwenputer/calibrate_real.py`, builds the RV32 circuit with Qwenputer's compiler (`core.build`:
11 layers, 2,679 dims, 99 heads, 42,014 units, 2,326 FFN nodes), runs 58 real programs on the reference
machine (the riscv-tests `rv32ui`, `rv32um` and `rv32uzba` suites, crc32, string ops and five small assembly
programs; 25,737 trace positions), evaluates the circuit on every trace exactly in float64 (`ir.evaluate`;
the circuit predicted every one of the 25,737 next tokens), and scores the same methods at the level of
the compiler's nodes, where each node has a name, a known input set and a known function. This also removes
the earlier caveat: the random-record runs used my own forward pass, this one uses the author's evaluator.

| real programs, 2,243 FFN nodes with a varying input | |
|---|---|
| nodes that ever fire | 1,898 (85 %) |
| M2 recall of true inputs, nodes that fire | **0.95** (all nodes 0.80) |
| M2 by layer | 1.00 in the first four layers, 0.87-0.91 in the last three (the multiplier and division stages, the output code) |
| M1 recall at k = fan-in, nodes that fire | 0.28 (0.49 at fan-in 1, 0.19 at fan-in 8 and above) |
| M3 gate labels, nodes with ≤ 4 inputs all of them bits | 221 tested, **130 right** |
| M3 right on the input combinations it saw | **221 of 221** |
| M3 right when the whole truth table was seen | **128 of 128** (two more right by luck) |
| M3 by true function | AND 19 / 19, OR 2 / 2, other 109 / 200 |

So on real programs the ablation scan finds 95 % of a firing node's inputs, better than on random
records (81 %), because real traces put the arithmetic units through their intended states; what it still
misses is in the wide units of the late layers, where one input rarely moves a sum across its threshold.
Correlation ranking stays at 28 %: real data does not help a method whose failure is shared inputs. And the
gate labeller of `xorlab` and `clean_carry` turns out to be exactly as good as its data: it never mislabels
a combination of inputs it has observed, every AND and OR in the circuit is named correctly, and every one
of its 91 errors is a guess about a row of the truth table that the programs never produced (a 4-input
node has 16 rows, and the riscv-tests reach all of them for only 60 of 145 such nodes). That is the number
to carry back to `results_clean_carry`, where units are labelled from the inputs the adder data provides:
a label is only as complete as the truth table the data covers, and the fraction of rows seen should be
reported next to it.

**Limits of this calibration.** The random-record runs score the checkpoint's units with my own forward
pass; the real-program run scores the compiler's nodes (one node sums several units) with the author's
evaluator, so the two are the same wiring at two granularities, not one run twice. Neither run touches
Qwen's layers or a trained circuit; what they calibrate is the method, on a circuit whose answer is known.

## Tying it together: what to do with this

Three experiments, in order of how much they would settle.

1. **Use the compiled layers as a calibration standard for circuit discovery.** Everything this
   repo does to find circuits (exhaustive search over gate labels in `xorlab`, the unit labelling
   of `clean_carry`, the mean-ablation scans and probes of `results_storyroom/`) has been tested
   only on networks whose true circuit is unknown. Qwenputer's layers 0-11 are a real
   transformer, at a real width, in which every unit's function is known from the compiler: this
   AND of those bits, that threshold, that memory row. Run the repo's discovery methods on those
   layers blind and score them against the compiler's labels: precision and recall of "this unit
   is an AND of these inputs", of "this head is memory", of "these 58 neurons are arithmetic-only".
   That turns the arithmetic-only neuron counts in the capacity report from a plausible reading
   into a validated measurement, and it is the one experiment nobody with Qwenputer has done.
   Cost: reading the weights, which the download allows today. **Done in the calibration section above**, on
   random records against the checkpoint's wiring and on 58 real programs against the compiler's own nodes.

2. **Run the capacity experiment at 4 billion weights.** Qwenputer is condition A with the
   training skipped: the computer is present but Qwen never trained with it. Fine-tune Qwen3.5-4B
   on an arithmetic-heavy stream (a) with the compiled calculator present and the `<tool_response>`
   positions masked, (b) computing itself with the answers in the loss, (c) masked, (d) text only,
   on identical text. Then probe for the result at the call position and ablate heads, exactly as
   in phase 8. This answers whether a model that *has* a computer gives up its own adders, and
   what it does with the room, at a size where the arithmetic circuits are real (Qwen's base
   scores 55-90 %, so it does pay for them). It needs a GPU with 24 GB and a few hours per run;
   it is the natural next session, not laptop work.

3. **Compile this repo's gate CPU into the story GPT.** The story GPT uses softmax attention, not
   DeltaNet, so the RAM trick does not apply directly, but the earlier chipformer (before q35)
   kept memory as the trace itself: a read is attention to the most recent `ADDR a` token. The
   repo's `results_transformer/` already measures what attention does with positions. A compiled
   8-bit calculator as extra layers of the story GPT, with the gate CPU's own calculator program,
   would be the small, inspectable twin of Qwenputer: a 1.35 M-weight language model that is
   bit-exact at arithmetic, trained only to call. The capacity contrasts could then be run with a
   computer that is inside the weights rather than outside the loss, which closes the one gap
   between the two projects' definitions of "the tool".

If only one is done, do the first: it is cheap, it uses the download, and it strengthens every
circuit claim this repo has made.

## Sources

- Qwenputer repository and design notes: https://github.com/divagr18/qwenputer (README,
  `docs/q35.md`, `docs/trace_format.md`, `experimental/riscv/docs/rv32.md`,
  `chipformer/q35/prims.py` on the `rv-store` branch)
- Model cards: https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer-Filesystem and
  https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer; the ModelScope mirror
  https://modelscope.ai/models/Spartan1925/Qwen3.5-4B-Qwenputer-Filesystem
- Author's announcement: https://x.com/divagr1925/status/2115394340357116169
- A write-up: https://medium.com/data-science-in-your-pocket/what-is-qwenputer-a-working-risc-v-cpu-inside-qwen3-5-febb522f000b
- This repo: `results_minigates/MINIGATES_REPORT.md`, `results_cpu/CPU_REPORT.md`,
  `results_cpugpt/CPUGPT_REPORT.md`, `results_storytool/STORYTOOL_REPORT.md`,
  `results_storyroom/ROOM_REPORT.md`
