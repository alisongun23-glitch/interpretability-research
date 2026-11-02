# XOR circuit discovery

Trains many small networks on XOR, finds the internal circuits they use, tests
whether those circuits are causal, and transplants circuits between independently
trained networks without retraining.

## Run

    pip install numpy matplotlib
    python run.py                 # about 10 seconds, writes results/
    python run.py --seeds 50      # more seeds

Settings (seeds, architectures, optimizer, learning rate, stopping margin,
ablation type) live in `config.json`.

## Layout

    run.py              pipeline: train -> analyse -> causal tests -> transplants -> figures -> report
    config.json         experiment settings
    xorlab/model.py     2->H->1 networks, batched Adam training, stopping rule
    xorlab/analysis.py  gate labels, exhaustive circuit search, ablation, patching, transplants, CKA
    xorlab/figures.py   figures A-F
    xorlab/report.py    writes results/REPORT.md, including the hypothesis rules
    results/            REPORT.md, summary.json, runs.csv, circuits.json, causal.json,
                        transplants.csv, similarity.npz, checkpoints/, training_logs/, figures/

Start with `results/REPORT.md`.

## Checks and performance

    python tests/check_all.py     # gradient checks and logic checks; prints PASS/FAIL, changes nothing

The NumPy scripts set `OPENBLAS_NUM_THREADS=1` before importing NumPy. On these small
matrices, multithreaded OpenBLAS spends its time spinning, and several scripts run at
once slowed each other down from minutes to over an hour. Set the variable yourself
to override.

## Extension: tiling gates into an adder

    python run_adder.py           # about 20 seconds, writes results_adder/

Trains XOR/AND/OR gate networks, tiles them into a 4-bit ripple-carry adder, trains
ordinary networks on 4-bit addition, and tests whether those contain the adder's
parts (probes, ablation, carry insertion, collapse-to-total). Start with
`results_adder/ADDER_REPORT.md`.

## Extension: 16-bit addition

    python run_adder16.py         # trains once and saves nets.npz; later runs about 40 seconds. Writes results_adder16/

Same questions at 16 bits, plus accuracy by carry travel distance and a layer-by-layer
"collapse to total" measure. Start with `results_adder16/ADDER16_REPORT.md`.

## Extension: the smallest working gates, and a snake game built from them

    python run_minigates.py       # about 1 minute; writes results_minigates/ (incl. library.json)
    python run_snake.py           # about 5 minutes; writes results_snake/ and snake_ui/net8.json, net16.json

`run_minigates.py` searches for the smallest neural network for each logic gate that still
works inside a large machine (one sigmoid neuron; XOR/XNOR two), and finds the training
margin that makes chained and looped gates safe. `run_snake.py` builds Snake as a synchronous
machine of 2-input gates and NAND flip-flops (`gatesnake/design.py`), runs it with those neural
gates against a plain-Python snake tick by tick, and measures its size and cost. Start with
`results_minigates/MINIGATES_REPORT.md`, then `results_snake/SNAKE_REPORT.md`.

To play the machine live, serve the folder and open it in a browser:

    python3 -m http.server 8765 --directory snake_ui     # then open http://localhost:8765

The page runs the exported 16x16 netlist (every gate a trained neuron, nothing rounded) in
JavaScript, and has a self-test that replays a recorded game against the Python reference.

    gatesnake/gates.py    gate architectures, training, static-discipline test, quantisation
    gatesnake/hdl.py      circuit builder (constant folding, de-duplication, NAND lowering, levels)
    gatesnake/design.py   the snake machine
    gatesnake/ref.py      plain-Python snake with the same rules
    gatesnake/sim.py      Boolean and neural simulators, flip-flops as 8 NAND gates
    gatesnake/bot.py      test players (food-seeking, and a board-filling loop)

## Extension: an 8-bit computer from the same gates

    python run_cpu.py             # about 1.5 minutes; writes results_cpu/ and cpu_ui/cpu.json

A control unit, an ALU and 32 bytes of RAM on one 8-bit bus, every gate a trained neuron
(`gatecpu/design.py`), checked tick by tick against a Python CPU (`gatecpu/ref.py`), with an
assembler and demo programs (`gatecpu/asm.py`). Start with `results_cpu/CPU_REPORT.md`. To use the
front panel, serve `cpu_ui` (`python3 -m http.server 8765 --directory cpu_ui`) and open
http://localhost:8765.

## Extension: a GPT whose fourth attention head is the computer

    python run_cpugpt.py          # about 15 minutes; writes results_cpugpt/ and cpugpt_ui/model.json

A 2-layer GPT (PyTorch) whose second layer has three ordinary heads and a CPU head: trained
pointers pick the operands and the operator, frozen tables send them to the gate computer, which
runs a calculator program (`gatecpu/calc.py`), and a trained projection and gate read the answer
back. The computer's answers for all 262,144 possible calls are computed on the gates with a
bit-sliced runner (`gatecpu/batch.py`). Start with `results_cpugpt/CPUGPT_REPORT.md`. To try it,
serve `cpugpt_ui` (`python3 -m http.server 8765 --directory cpugpt_ui`) and open http://localhost:8765.

## Extension: a story-writing GPT that sends math to the computer

    python run_storygpt.py        # about 1.5 hours; needs results_cpugpt/cpu_table.npz and data/TinyStories-valid.txt

The same CPU head inside a 4-layer language model trained on TinyStories (download
`TinyStories-valid.txt` from huggingface.co/datasets/roneneldan/TinyStories into `data/`), with math
problems and story word problems ("tom had 12 apples . he got 5 more . now he has") dropped into the
stories. The head runs at every position and learns to call the computer only for math, pointing at the
numbers and at the word that names the operation (`cpugpt/text.py`, `cpugpt/lm.py`). Start with `results_storygpt/STORYGPT_REPORT.md`; the
page is `storygpt_ui/` (`python3 -m http.server 8765 --directory storygpt_ui`).

## Extension: the story GPT calls the computer in writing

    python run_storytool.py       # about 50 minutes; needs results_cpugpt/cpu_table.npz and data/TinyStories-valid.txt

Instead of a CPU head, the GPT types the call into the story itself ("now he has <calc> 12 + 5 ="), the gate
computer answers "17 </calc>" into the text, and the GPT copies the answer; a second call can use the first
result ("and then 3 more"). The computer's tokens stay out of the training loss (`cpugpt/tool.py`,
`cpugpt/text.py` `build_tool_stream`). Report: `results_storytool/STORYTOOL_REPORT.md`; page: `storytool_ui/`
(`python3 -m http.server 8765 --directory storytool_ui`). Research notes behind the design:
`reports/Small GPT in context tool calls.md`.

## Extension: a small GPT (transformer)

    python run_transformer.py        # NumPy GPT with TransformerLens-style hooks; ~1 h on 2 CPU cores
    python run_transformer_tl.py     # same experiment with the real TransformerLens library (~1 h on a laptop CPU)

Binary addition as a token sequence, written lowest digit first or highest digit first.
Attention profiles, activation patching by (layer, position), probes, head ablations.
Start with `results_transformer/TRANSFORMER_REPORT.md`. `tinygpt.py` is the model
(`python tinygpt.py` runs the finite-difference gradient check).

## Extension: clean neurons and the carry direction (Cammarata's test)

    python clean_carry.py prereg-check   # locks results_clean_carry/PREREG.md; everything below refuses to run without it
    python clean_carry.py stage0         # saved B adders: labels, readers of the carry direction (minutes)
    python clean_carry.py train          # arms A (task) and R (random per-position jobs) at MLP widths 256..4, checkpoints
    python clean_carry.py analyse        # labels every unit at every checkpoint; tracks the carry direction (arm A)
    python clean_carry.py report         # validity gates V1-V5, then H1-H4 (Cammarata) and D1-D5 (direction) by code

Tests Nick Cammarata's "Neural Networks Are Already Clean" on adders whose algorithm is known: do MLP
units do one job, does that survive squeezing the MLP, how do labels grow over training, and are the
clean count-threshold units the readers of the residual-stream carry direction? Pre-registered in
`results_clean_carry/PREREG.md`; results in `results_clean_carry/REPORT.md`.

    python clean_carry_long.py pilot     # addendum: how long do narrow adders need? (sets S by a fixed rule)
    python clean_carry_long.py train     # then train every width S steps, analyse, report (results_clean_carry_long/)
    python clean_carry_long.py explore   # measures chosen after the verdicts (never judged)

The addendum reruns the squeeze with adders trained 16 times longer, and adds a neuron-specific "gate
label" (a unit counts only if it is silent when off). Pre-registered in `results_clean_carry_long/PREREG.md`;
results in `results_clean_carry_long/REPORT.md`. Even at 40,000 steps only 3 of 20 narrow adders learned,
so the squeeze still has no verdict; the gate label passes its rotation test (0.36 to 0.00). The "unclear"
units turn out to be functions of the three inputs plus the answer bit just written (the token at the
position) or the previous column, not noise (exploratory, figure 4): the narrow adders read that bit as
carry evidence (c_k = 1 − s_{k−1} when the previous digits differ), through attention, and need it; the wide
adders have the carry direction as well (figure 5).
