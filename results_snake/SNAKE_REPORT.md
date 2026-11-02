# Snake built from neural logic gates

Produced by `python run_snake.py` (about 5 minutes; needs `results_minigates/library.json` from
`python run_minigates.py`). Every number is in `summary.json`, figures 1–3 are in `figures/`.
Seeds are fixed, so a rerun gives the same numbers (timings vary by machine).

## Question

Take the smallest logic gates that still work in a machine (`results_minigates/MINIGATES_REPORT.md`:
one sigmoid neuron each, two for XOR/XNOR, trained to margin 4) and build a whole game out of them,
memory included. How big is the result, does it work with no rounding anywhere, and what does it cost?

## The machine

Snake on an N × N board with walls, as a synchronous circuit of 2-input gates and D flip-flops
(`gatesnake/design.py`). One clock tick is one game step.

- **Body.** Every cell holds a countdown of how many more steps the snake stays there. The head cell
  is loaded with the snake's length; every other nonzero cell counts down by one. A cell is lit while
  its counter is nonzero. Eating skips the countdown for one tick, so the tail stays and the snake
  grows. This is the third RAM of [stevomitric/Logisim-Snake](https://github.com/stevomitric/Logisim-Snake),
  spread over the board as gates instead of stored in a RAM that is scanned on a faster clock.
- **Moving.** The head's x or y goes through a ±1 adder; its carry out is the wall. A 180° turn is
  ignored once the game has started.
- **Crash.** One AND per cell (head here, counter > 1), then a 256-input OR tree.
- **Food.** A 16-bit linear-feedback shift register proposes a cell; it is accepted on the first tick
  it lands on an empty cell.
- **Memory.** Every register bit is a master-slave D flip-flop of 8 NAND gates, plus one shared
  clock inverter. In the analog runs these are neural NANDs in a feedback loop, iterated until they
  settle (6 gate delays with the clock low, 6 with it high).

The circuit builder (`gatesnake/hdl.py`) folds constants and merges duplicate gates. A plain-Python
snake with exactly the same rules (`gatesnake/ref.py`) is the reference.

## 1. Size

**16 × 16 board** (the playable one), with three gate libraries on the same netlist:

| | minimal gates | NAND neurons only | 2→4→1 gates of `run_adder.py` |
|---|---|---|---|
| gates | 37,521 | 50,587 | 37,521 |
| of which memory (2,350 flip-flops) | 18,801 | 18,801 | 18,801 |
| neurons | **39,863** | 50,587 | 178,269 |
| weights | **119,597** | 151,761 | 602,847 |
| different trained weights | 224 (8 nets × 7 gate types) | **24** (8 NAND nets) | |
| logic depth, gates | 48 | 82 | 48 |
| logic depth, neuron layers | **51** | 82 | 91 |
| multiply-adds per tick, every gate once | 79,734 | 101,174 | 424,578 |
| multiply-adds per tick, as simulated | 493,333 | 514,772 | 2,906,177 |

"As simulated" counts each flip-flop's 8 NANDs 12 times per tick (the latch iterations).

Viewed as one neural network, the 16 × 16 game is a recurrent network with 39,863 sigmoid
neurons and 119,597 weights, a 2,350-bit recurrent state, and 51 neuron layers between one state
and the next. The minimal library is 4.5× smaller in neurons and 5× in weights than the same
machine built from the 2→4→1 gates of the adder experiments. NAND-only is 27% bigger and 60% deeper in neuron layers,
but the whole machine then uses one gate network: 24 distinct trained weights in total.

**By board size** (minimal gates, figure 1):

| board | gates | flip-flops | neurons | weights | depth (gates) |
|---|---|---|---|---|---|
| 4 × 4 | 1,757 | 114 | 1,859 | 5,577 | 37 |
| 8 × 8 | 7,823 | 488 | 8,301 | 24,907 | 43 |
| 16 × 16 | 37,521 | 2,350 | 39,863 | 119,597 | 48 |
| 32 × 32 | 179,011 | 11,316 | 190,321 | 570,975 | 54 |
| 64 × 64 | 836,661 | 53,306 | 889,963 | 2,669,905 | 59 |

Size grows with the number of cells times the counter width (log of the cells), about
110–200 gates per cell. Depth grows only with the log of the board (decoders and OR trees).
The 32 × 32 board, the size of the Logisim version, takes 179,011 gates.

## 2. It works, with nothing rounded anywhere

Each row is checked against the Python snake on every tick, comparing every register bit. The
player is a bot that heads for food, sometimes moves at random and restarts 3 ticks after dying,
so runs include eating, wall crashes, self crashes, food re-rolls and resets. "Worst signal" is
the furthest any neuron output came from a clean 0 or 1 during the run.

| board | machine | ticks | wrong bits | foods eaten | games | worst signal | ms per tick |
|---|---|---|---|---|---|---|---|
| 8 × 8 | Boolean gates, Boolean NAND flip-flops | 12,000 | 0 | 1,073 | 455 | 0 | 0.43 |
| 8 × 8 | neural gates, neural NAND flip-flops | 12,000 | 0 | 1,073 | 455 | 0.031 | 0.96 |
| 8 × 8 | NAND-only, fully neural | 12,000 | 0 | 1,073 | 455 | 0.026 | 0.94 |
| 16 × 16 | Boolean gates, Boolean NAND flip-flops | 9,000 | 0 | 559 | 141 | 0 | 1.07 |
| 16 × 16 | neural gates, neural NAND flip-flops | 9,000 | 0 | 559 | 141 | 0.031 | 2.46 |
| 16 × 16 | NAND-only, fully neural | 9,000 | 0 | 559 | 141 | 0.026 | 2.60 |
| 32 × 32 | neural gates, neural NAND flip-flops | 400 | 0 | 15 | 4 | 0.031 | 10.7 |
| 32 × 32 | NAND-only, fully neural | 400 | 0 | 15 | 4 | 0.026 | 11.1 |

(Ideal flip-flops, Q := D, give the same results and are in `summary.json`.)

**Filling the board.** A second player follows a loop through every cell, so it never crashes and
eats every food. On the fully neural 16 × 16 machine it ran 18,300 ticks, ate 252 foods and grew the
snake to 255 cells, every counter running through its full range, with 0 wrong bits and no signal
further than 0.031 from a rail. The 8 × 8 machine reached 63 cells the same way. (A 255-cell snake on
256 cells is the end: the one free cell is always the one the head is entering, so no food can be placed.)

The worst signal, 0.031, stays below the 0.05 interface the gates were trained for, as the static
discipline promises: about 21,000 neuron outputs and 226,000 latch updates per tick, and none drift.

## 3. Where the gates go

16 × 16 board (figure 2):

| part | logic gates | memory gates | total | share |
|---|---|---|---|---|
| cells (countdowns, their update and occupancy) | 16,384 | 18,432 | 34,816 | 93% |
| food (placement, compare, screen) | 1,384 | 72 | 1,456 | 3.9% |
| control (keys, direction, crash, game state) | 550 | 32 | 582 | 1.6% |
| head and walls | 371 | 64 | 435 | 1.2% |
| random numbers (LFSR) | 3 | 128 | 131 | 0.3% |
| length | 28 | 72 | 100 | 0.3% |

Everything outside the cells, all the game rules included, is 7% of the machine. The rest is the
board's memory: 256 cells × 9
bits, each bit a flip-flop (8 gates) plus about 7 gates of countdown, select and reset logic. By gate
type the logic is 8,637 AND, 5,407 OR, 2,333 NOT, 2,330 XOR, 12 XNOR and 1 NAND, plus the 18,800 NANDs
of the flip-flops.

## 4. Could the gates be smaller?

The same fully neural 16 × 16 machine with every gate retrained at a smaller margin (3 runs of up
to 1,500 ticks each, figure 3):

| gate margin | first wrong bit | worst signal |
|---|---|---|
| 1.5, 2, 2.5, 3, 3.25 | tick 1, every run | 0.5 (a bit flipped) |
| 3.5 | none in 1,500 ticks | 0.205 |
| 4 (library) | none in 1,500 ticks | 0.031 |

Margin-3 gates passed 400-layer random circuits and held a latch (`MINIGATES_REPORT.md`, Part D), but
the snake breaks them on tick 1. The cause is the crash detector, a 256-input OR tree eight levels
deep whose inputs are all analog zeros. Following one zero up the tree:

| OR-tree level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| margin 3 | 0.047 | 0.086 | 0.141 | 0.258 | 0.639 | 0.997 | 1.0 | 1.0 | 1.0 |
| margin 3.5 | 0.029 | 0.045 | 0.058 | 0.070 | 0.084 | 0.102 | 0.130 | 0.191 | 0.382 |
| margin 4 | 0.018 | 0.024 | 0.027 | 0.029 | 0.030 | 0.030 | 0.030 | 0.030 | 0.030 |

At margin 3 an OR of two small zeros is a bigger zero, and by level 5 it reads as 1: the game sees a
crash that did not happen. Margin 3.5 gets through eight levels at 0.38, but reaches 0.93 at level 9, so it
would break a 32 × 32 board, whose tree has 10 levels. Margin 4 settles at 0.030 at any depth. This is the worst case that the
static-discipline rule guards against and that random circuits rarely produce.

**Noise.** With Gaussian noise added to every neuron output (margin-4 gates): no errors in 1,500
ticks up to a standard deviation of 0.03; first errors after 27–147 ticks at 0.04, after 1–7 ticks at
0.05. With about 250,000 noisy neuron evaluations per tick, the rare large draws decide it.

## 5. What it costs to run

| | time per game step |
|---|---|
| plain Python snake (`gatesnake/ref.py`) | 0.012 ms |
| neural machine, NumPy, 16 × 16, fully analog | 2.5 ms |
| neural machine, JavaScript in the browser, 16 × 16, fully analog | about 1.3 ms |
| neural machine, NumPy, 32 × 32, fully analog | 10.7 ms |

The neural machine is about 200 times slower than an ordinary program and does about 500,000
multiply-adds and 250,000 sigmoids per step. That is still fast enough to play: a snake game runs at
5–15 steps per second.

## Playing it

`snake_ui/index.html` runs the exported netlists (`net16.json`, `net8.json`) in the browser:
every gate a trained neuron, every flip-flop eight neural NANDs in a loop, nothing rounded. The LED
brightness is the actual neuron output. Controls switch the gate margin (4, 3, 2) and add neuron noise
live, and a counter shows wrong bits by comparing each tick with exact logic. A self-test replays a
recorded 2,000-tick game and compares a hash of all register bits with the Python reference on every
tick; it passes on both boards.

    python3 -m http.server 8765 --directory snake_ui     # then open http://localhost:8765

## Conclusion

- A complete game can be built from trained neurons alone: 37,521 gates, 39,863 neurons and
  119,597 weights for a 16 × 16 board, run with no rounding anywhere for tens of thousands of ticks
  with zero wrong bits, matching the reference exactly.
- The smallest gates that work are one sigmoid neuron (XOR: two), and "work" has to mean the static
  discipline, not the truth table. Gates that look fine alone and in random circuits (margin 3) break
  the real machine on its first tick, in one long OR tree.
- The cost is memory. 93% of the gates hold and update the board's countdowns, and half of all
  gates are flip-flops. Everything else, the game rules included, is 7%.

## Limitations and next steps

- Gates have two inputs. A sigmoid neuron can be a k-input AND or OR in one unit; allowing that
  would shrink the OR trees, decoders and the per-cell logic considerably, at the cost of a
  smaller noise margin per extra input.
- The machine stores the whole board. A ring buffer of positions (the Logisim design's X/Y RAMs)
  would need about 2 log N bits per body segment instead of log N² bits per cell, which is cheaper
  for long boards with short snakes but needs addressable memory built from gates.
- Flip-flops are simulated with all eight gates updating together, one gate delay per step, and
  ideal clock edges; real asynchronous timing (races, glitches) is not modelled.
- Food comes from a free-running 16-bit LFSR, so the first food of a fresh power-on is always in the
  same corner.
