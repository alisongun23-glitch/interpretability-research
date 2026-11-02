# An 8-bit computer built from neural logic gates

Produced by `python run_cpu.py` (about 1.5 minutes; needs `results_minigates/library.json`).
Every number is in `summary.json`, figures are in `figures/`. Seeds are fixed, so a rerun gives the
same numbers (timings vary by machine). Play it in `cpu_ui/` (see the end).

## Question

The snake machine showed that trained gate neurons can run a fixed-purpose machine. Can the same
gates (one sigmoid neuron each, two for XOR, trained to margin 4) build a general computer, with a
control unit, an ALU and memory as separate units talking over a bus, and run any program you load?

## The computer

SAP-1 style, hardwired control, one 8-bit bus (`gatecpu/design.py`).

| unit | what it holds | what it does |
|---|---|---|
| memory | 32 × 8 RAM of flip-flops, MAR (5 bits) | 5→32 decoders, a 32:1 read select per bit, per-word write enable; a program port writes RAM while the CPU is held in reset |
| ALU | B (8), flags C and Z | 8-bit ripple adder/subtractor, 8 NANDs, result select, zero detect |
| control unit | PC (5), IR (8), step counter T, HALT | opcode decoder; 14 control lines from (opcode, step, flags) |
| registers | A (8), OUT (8) | accumulator; address 31 is memory-mapped to OUT |
| bus | – | 8 wires, each an OR over (enable AND value) of PC, RAM, IR address, A, ALU |

Instruction = 3-bit opcode + 5-bit address: LDA, ADD, SUB, NAND, STA, JMP, JZ, JC. Every instruction
starts with the same two fetch steps (T0 MAR ← PC, T1 IR ← RAM and PC + 1); jumps finish at T2,
LDA and STA at T3, ALU instructions at T4. A jump to itself sets HALT. `gatecpu/asm.py` adds `OUT`,
`HLT`, labels and `.byte`. Each bit of every register and of RAM is a master-slave flip-flop of
8 NAND gates; in the analog runs those are neural NANDs in a feedback loop.

## 1. Size

| | minimal gates | NAND neurons only | 2→4→1 gates of `run_adder.py` |
|---|---|---|---|
| gates | 4,352 | 4,914 | 4,352 |
| of which memory (304 flip-flops) | 2,433 | 2,433 | 2,433 |
| neurons | **4,389** | 4,914 | 21,508 |
| weights | **13,141** | 14,742 | 73,039 |
| different trained weights | 224 | 24 | |
| logic depth, gates / neuron layers | 29 / 32 | 41 / 41 | 29 / 56 |
| multiply-adds per tick, every gate once | 8,752 | 9,828 | 51,531 |
| multiply-adds per tick, as simulated | 62,255 | 63,330 | 372,554 |

By unit (figure 1):

| unit | logic gates | memory gates | total | share |
|---|---|---|---|---|
| memory (RAM, MAR, decoders, read select) | 1,463 | 2,088 | 3,551 | 82% |
| control unit (PC, IR, T, HALT, control lines) | 151 | 136 | 287 | 6.6% |
| ALU (adder/subtractor, NAND, flags, B) | 132 | 80 | 212 | 4.9% |
| registers A and OUT | 70 | 128 | 198 | 4.5% |
| bus | 60 | 0 | 60 | 1.4% |
| program port | 43 | 0 | 43 | 1.0% |

The parts that make it a computer, control unit plus ALU, are 500 gates. Memory is 82%: 32 bytes cost
3,551 gates, about 14 gates per stored bit (8 for the flip-flop, the rest to select and write it).

## 2. It works, with nothing rounded anywhere

Every register bit, RAM included, is compared with the Python CPU (`gatecpu/ref.py`) on every tick.
Workloads: the four demos loaded through the program port, and three runs of 4,000 ticks of random
bytes as programs, with random resets and random reloads (15,332 ticks per row).

| machine | ticks | wrong bits | demo outputs | worst signal | ms per tick |
|---|---|---|---|---|---|
| Boolean gates, ideal flip-flops | 15,332 | 0 | all right | 0 | 0.12 |
| Boolean gates, Boolean NAND flip-flops | 15,332 | 0 | all right | 0 | 0.33 |
| neural gates, ideal flip-flops | 15,332 | 0 | all right | 0.030 | 0.30 |
| neural gates, neural NAND flip-flops | 15,332 | 0 | all right | 0.030 | 0.72 |
| NAND-only, fully neural | 15,332 | 0 | all right | 0.026 | 0.66 |

| demo | prints | instructions | clock ticks |
|---|---|---|---|
| Fibonacci | 0 1 1 2 3 5 8 13 21 34 55 89 144 | 125 | 487 |
| Multiply (7 × 6 by adding) | 42 | 50 | 200 |
| Countdown | 10 9 … 1 0 | 62 | 238 |
| XOR from NAND (the CPU has no XOR) | 108 = 202 ⊕ 166 | 10 | 43 |

About 4 clock ticks per instruction.

## 3. Could the gates be smaller?

Same fully neural computer, every gate retrained at a smaller margin, 3 runs of up to 3,000 ticks of
random programs (figure 2):

| gate margin | first wrong bit | worst signal |
|---|---|---|
| 2, 2.5 | tick 1 | 0.5 |
| 3 | tick 34 (the first fetch after loading) | 0.5 |
| 3.25 | none | 0.33 |
| 3.5 | none | 0.10 |
| 4 (library) | none | 0.030 |

The computer tolerates slightly smaller gates than the 16 × 16 snake (which needed 3.5) because its
longest OR tree is the 32-way RAM read, 5 levels deep, against the snake's 256-way crash check at 8
levels. Margin 3.25 works here only just (worst signal 0.33); margin 4 keeps every signal within 0.03.

**Noise** on every neuron output (margin-4 gates): no errors in 3,000 ticks up to a standard deviation
of 0.03; at 0.04 one run of three failed at tick 257; at 0.05 all failed within 53–225 ticks.

## 4. What it costs to run

| | time per clock tick |
|---|---|
| plain Python CPU (`gatecpu/ref.py`) | 0.003 ms |
| neural computer, NumPy, fully analog | 0.72 ms |
| neural computer, JavaScript in the browser, fully analog | about 0.24 ms |

About 250 times slower than an ordinary emulator in NumPy. In the browser it runs about 4,000 ticks,
1,000 instructions, per second; the page caps the clock at 1,000 ticks per second so the lights stay
readable.

## Playing it

`cpu_ui/index.html` runs the exported netlist (`cpu.json`) in the browser: every gate a trained neuron,
every flip-flop eight neural NANDs, nothing rounded; LED brightness is the actual neuron output. The
front panel shows the control unit (PC, IR as a mnemonic, the step, the 14 control lines), the bus (its
value, who drives it and who loads it), the ALU registers and flags, the output, and all 32 bytes of RAM
with the program disassembled. Programs are typed in the editor and loaded through the program port
(33 clock ticks). Controls: run, step one clock, step one instruction, reset, clock speed, gate margin
(4, 3, 2), neuron noise, and a counter of wrong bits against exact logic. The self-test replays all four
demos (2,532 ticks) against the Python reference and passes.

    python3 -m http.server 8765 --directory cpu_ui     # then open http://localhost:8765

## Conclusion

- A working stored-program computer fits in 4,352 gates: 4,389 neurons and 13,141 weights. It runs
  any program loaded into it with zero wrong bits, fully analog, matching a Python CPU tick for tick.
- Control unit and ALU together are 500 gates. As in the snake, memory is the cost: 82% of the gates
  hold 32 bytes.
- The same margin rule holds: gates trained to margin 4 keep every signal within 0.03 of a rail;
  smaller gates fail at the first deep OR tree.

## Limitations and next steps

- 32 bytes of memory, no stack, no subroutine calls, no immediates, no input port. More RAM costs about
  14 gates per bit; a 256-byte machine would be about 30,000 gates.
- Flags change only on ADD, SUB and NAND (LDA does not set Z).
- Flip-flops are simulated with ideal clock edges and all eight gates updating together.
