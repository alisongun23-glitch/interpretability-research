# Does handing the sums to the calculator leave the story GPT more room for language? (`run_storyroom.py`)

Produced by `python3 run_storyroom.py` (phases 0 and 1: 23 trained or loaded models, 3.7 hours on the laptop
CPU; then `--phase 5 --recipe m64drillx3`, four more models, 1.8 hours; `--phase 6`, the grokking attempt, four
models, 4.2 hours; `--phase 7 --split pair`, the fair split, two models, 1 hour; and `--phase 8 --split pair` at two
recipes, nine models and 140 fine-tunes, 2.2 hours). Numbers are in `summary.json` (phases 0-6),
`summary_pairsplit.json` (phases 7-8) and `jobs/<tag>.json`, figures in `figures/`, models in `models/`, logs in `logs/`. The question,
the reading behind it and the plan are in `../reports/Does tool use free model capacity.md` and
`../research_notes/Does tool use free model capacity/`. The models are the story GPT of
`../results_storytool_shapes/` (4 layers, 4 heads, width 128, 1.35 M weights, 0.80 M outside the embedding).

## Answer first, in plain words

The question was: when the GPT hands every sum to the calculator instead of working it out in its own
weights, does that leave it more room for language? The answer from this run is **yes, and the amount can be
measured: once the self-computing model actually computes, the language it spends on doing sums (0.055-0.066
nats, about 5-7 % of perplexity) comes back almost entirely (0.048-0.055 nats) when the sums are delegated, and
the calculator chips themselves cost 0.007-0.010.** Getting to a self-computing model that actually computes took most of the work,
and for a while it looked as if there was nothing to find.

Here is what happened, step by step.

1. **At the base recipe the sums cost almost nothing, because the self-computing model does not do them.**
   Four models were trained on exactly the same stories, the same problems and the same starting weights: one that calls the calculator (A), one that
   has to write the answers itself (B), one that sees the same problems but is never scored on the answer
   (C), and one trained on stories alone (D). Every one of them read the same amount of story text. The
   tool model A ends up a hair better on new stories than the self-computing model B: 0.009 nats per word,
   which is a perplexity of 16.12 against 16.27, under one per cent. Three seed pairs, all in the same
   direction, so it is probably real, but it is tiny. The earlier finding that the no-calls model is
   *better* (15.33 against 15.7) turns out to be only because it sees 12 % more story text per step; at
   equal story text the order flips.

2. **The self-computing model learns to add only partly, and for a while it looked as if it never did.** The plan
   said: before you can measure what the tool frees, you need a self-computing model that really pays for
   arithmetic, so make it learn addition (80 % right on sums it has never seen). Ten recipes were tried: heavier
   weight on the answers (8 and 16 times), bare drill lines ("12 + 7 = 19 ."), results kept below 64 so each sum
   comes round about 60 times, addition only, three times the training, a bigger model. The best scored **3.2 %**
   on unseen sums, and a further four runs at ten times the training with strong weight decay (phase 6) scored 0 %
   at every checkpoint. Every one of those models memorised the sums it had seen (up to 99.5 %) and landed within
   one of the right number on 97-99.5 % of unseen sums while never being exact. That "never exact" turned out to be
   the data split, not the model: the repo's rule for holding out sums reduces, for addition, to holding out every
   pair whose sum leaves remainder 2 when divided by 5, so the unseen sums were unseen **answers**, numbers the
   model had never once written as a result. Under a fair split that holds out pairs (phase 7), the same one-cycle
   drill recipe gets **65 %** of never-seen sums exactly right (78 % at a second seed, 88 % with two training
   cycles), 80 % when both numbers are at least 10, and 96 % within one. That is real, partial addition, so the matched comparison was
   run at that recipe after all (phase 8).

3. **The big cost is the data, not the sums.** The stories-only model D is 0.055 nats better than B on pure
   story text (perplexity 15.40 against 16.27). But on story words that have a problem somewhere in the
   context, D is the *worst* of the four (2.831 against 2.805 nats). D is better at what it was trained on
   and worse at everything else, which is a matter of matching the test to the training text, not of room
   in the weights. So D minus B is not a bound on what a tool could save.

4. **One real transfer, and no hidden adder.** The tool model restates a number from context in sentence shapes
   it was never trained on 21-62 % of the time, depending on the recipe; the three others manage 0-3 %.
   Fine-tuned on a new story skill it starts far ahead for that reason, and fine-tuned on addition it learns no
   faster than a fresh model: delegating left no sleeping adder behind. On grammar pairs (was/were, has/have,
   he/she, him/her, bag/bags) and name recall the four are within seed noise, and that noise is large (one seed
   moves a score by up to 0.4).

5. **What it means for the idea.** "Delegation frees capacity" needs a model that would otherwise spend
   capacity on the task, and at this size that only happens with a fair split, heavy drilling and the answers
   weighted up. Where it happens (phase 8), the self-computing model pays 0.055 nats of language for answering
   65-78 % of new sums and 90 % of seen ones; the tool model gets 0.048 of it back; the chips cost 0.007; and
   the tool's lead grows with how much arithmetic the other model does (0.009 when it holds nothing, 0.023 when
   it holds a table, 0.048 when it adds). Inside, the self-computing model carries the exact answer in its
   residual stream and in a few dozen neurons and two heads that the tool model does not have; the tool model
   carries only the rough size of the answer before it reads it. The freed room shows up as plain story
   prediction and as number copying, not as better grammar and not as a dormant adder. The same comparison at
   the recipe where the self-computing model passes the plan's 80 % gate (88 % on new sums, two training cycles)
   gives the same picture a little larger: computing costs 0.066, the tool gets 0.055 back, the chips cost 0.010,
   and the tool's lead keeps growing with the arithmetic the other model does (0.009, 0.023, 0.048, 0.055). One
   seed each for the tool, masked and stories-only models at each recipe, so every one of these numbers carries
   about ±0.01 of seed scatter; the two recipes agreeing is the best check of that there is, though they share
   starting weights and data order.

## The four conditions

Same stories, same problem draws, same starting weights per seed (`torch.manual_seed(seed)` before the model is
built), width 128, batch 32, context 64, AdamW at 1e-3 with a cosine schedule to each run's own number of steps.
Every run reads 9.82 M story tokens, the tool model's share at 7,000 steps.

| | stream | in the loss | story share | steps |
|---|---|---|---|---|
| **A, tool** | problems with `<calc> a op b = r </calc>` written in the text | model tokens; the machine's tokens masked | 0.685 | 7,000 |
| **B, computes** | the same problems with the answer in prose | every token (answers weighted k in later recipes) | 0.765 | 6,268 |
| **C, masked answer** | B's stream | every token but the answer | 0.765 | 6,268 |
| **D, stories only** | stories, no problems | every token | 1.0 | 4,797 |

The three A seeds are the models of `../results_storytool_shapes/` (same code path, same seeds), re-evaluated;
B, C and D were trained here, three seeds each. One extra B is the shapes run's no-calls model at 7,000 steps,
the equal-steps comparison. Story loss is the mean next-token loss over the full held-out story set
(207,172 tokens, `TEXT_ALL`); the old 400-window figure is kept as "old windows".

## Phase 0: the controls (`figures/0_controls.png`)

Story loss in nats per token on held-out stories, three seeds, then the mean and the seed spread:

| condition | seed 0 | seed 1 | seed 2 | mean | sd | perplexity | old windows |
|---|---|---|---|---|---|---|---|
| A tool | 2.7800 | 2.7846 | 2.7757 | **2.7801** | 0.0044 | 16.12 | 15.71 |
| B computes | 2.7854 | 2.7924 | 2.7898 | **2.7892** | 0.0035 | 16.27 | 15.90 |
| C masked answer | 2.7857 | 2.7914 | 2.7750 | **2.7840** | 0.0083 | 16.18 | 15.83 |
| D stories only | 2.7370 | 2.7374 | 2.7291 | **2.7345** | 0.0047 | 15.40 | 14.93 |
| B at equal steps (7,000 steps, 10.97 M story tokens; the earlier no-calls model) | 2.7522 | | | | | 15.68 | 15.33 |

Paired contrasts (same seed against same seed; standard error over the three pairs):

| contrast | what it measures | mean | se | per seed |
|---|---|---|---|---|
| A − B | freed capacity, if B had paid for arithmetic | **−0.0091** | 0.0026 | −0.005, −0.008, −0.014 |
| A − C | the chips and the machine text in the context | −0.0039 | 0.0023 | −0.006, −0.007, +0.001 |
| B − C | the gradient on answers B cannot learn | +0.0052 | 0.0048 | −0.000, +0.001, +0.015 |
| D − B | the problem text itself | −0.0547 | 0.0036 | −0.048, −0.055, −0.061 |
| D − A | | −0.0456 | 0.0013 | |

Negative means the first is better. A beats B by 0.009 nats in every seed pair; half of that is the chips
(A − C) and half the unlearnable answer gradient (B − C), each within its own noise. The equal-steps model
beats A by 0.028 nats: more story text, not a better model.

**Where the words are.** Loss by token role on the mixed test stream (problems in the context), mean of seeds:

| | story words | problem text | answer | machine's text | call syntax | restated number |
|---|---|---|---|---|---|---|
| A tool | 2.808 | 1.994 | 0.149 (a copy) | 14.60 (never trained) | 0.363 | 0.755 |
| B computes | 2.805 | 2.001 | 5.468 | | | 4.676 |
| C masked answer | 2.798 | 2.000 | 12.35 (never trained) | | | 5.236 |
| D stories only | 2.831 | 7.525 | 15.74 | | | 14.18 |

On story words inside the mixed stream A, B and C are equal within noise and D is worst; D's 0.055-nat lead
on pure story text is the training text matching the test text. By word frequency (`by_frequency` in the
jobs), A and B tie on the 50 commonest words (1.560 each) and A's small lead sits in the rarer words: 0.027
nats on ranks 200-999, 0.066 on ranks 1,000-2,999, 0.045 past 3,000.

**Skills scored by log-probability** (`figures/3_skills.png`; share of 500 minimal pairs where the right
sentence wins, ties half; induction is the nats saved on the second copy of a random 16-word run):

| | was/were | has/have | he/she | him/her | bag/bags | name recall, gap 1 | gap 2 | induction | restates a number in a new sentence shape |
|---|---|---|---|---|---|---|---|---|---|
| A tool | 1.000 | 0.939 | 0.876 | 1.000 | 0.975 | 0.681 | 0.683 | 0.236 | **0.620** |
| B computes | 1.000 | 0.969 | 0.949 | 0.997 | 0.943 | 0.667 | 0.640 | 0.112 | 0.028 |
| C masked answer | 0.999 | 0.993 | 0.894 | 1.000 | 0.941 | 0.687 | 0.668 | 0.208 | 0.030 |
| D stories only | 0.999 | 0.947 | 0.899 | 1.000 | 0.854 | 0.897 | 0.891 | 0.193 | 0.000 |

Seed spreads are 0.03-0.1 on the grammar pairs and 0.06-0.25 on induction, so A, B and C are tied there. D's
name recall (0.90 against 0.67-0.69) is again the training text: stories only, so names are never pushed out
of the context by a problem. The one clear transfer is the last column: the number-copying the chips teach
(answer loss 0.149) carries over to no-call sentences with held-out frames and verbs.

## Phase 1: ten recipes for making B learn addition (`figures/0_recipes.png`)

All at seed 0; the criterion is 80 % exactly right on 500 unseen addition pairs (both operands at least 2,
results below the recipe's ceiling, pairs held out by hash). "Seen pairs" are 200 training pairs. "Within ±1"
and "within ±3" count answers that close to the right number. Non-base recipes use weight decay 0.1.

| recipe | drills per story | results below | ops | answer weight k | steps | story loss | seen pairs, exact | unseen, exact | unseen, within ±1 | within ±3 |
|---|---|---|---|---|---|---|---|---|---|---|
| base (phase 0 B, wd 0.01) | 2 | 256 | all | 1 | 6,268 | 2.789 | 0.000 | 0.000 | 0.000 | |
| k8 | 2 | 256 | all | 8 | 6,268 | 2.808 | 0.055 | 0.000 | 0.135 | 0.430 |
| k16 | 2 | 256 | all | 16 | 6,268 | 2.858 | 0.110 | 0.000 | 0.155 | 0.525 |
| drill | 6 | 256 | add | 8 | 6,921 | 2.847 | 0.375 | 0.000 | 0.755 | 0.925 |
| m64 | 2 | 64 | all | 8 | 6,266 | 2.815 | 0.500 | 0.000 | 0.530 | 0.925 |
| m64drill | 6 | 64 | add | 8 | 6,928 | 2.853 | 0.875 | 0.000 | 0.985 | 1.000 |
| m64x3 | 2 | 64 | all | 8 | 18,799 | 2.577 | 0.915 | **0.032** | 0.045 | 0.810 |
| m64drillx3 | 6 | 64 | add | 8 | 20,783 | 2.607 | **0.960** | 0.000 | **0.995** | 1.000 |
| drillx3 | 6 | 256 | add | 16 | 20,764 | 2.671 | 0.725 | 0.000 | 0.980 | 0.995 |
| m64x3, width 256 | 2 | 64 | all | 8 | 18,799 | 2.413 | **0.985** | 0.016 | 0.040 | 0.665 |
| k8, width 256 | 2 | 256 | all | 8 | 6,268 | 2.608 | 0.260 | 0.000 | 0.000 | 0.275 |

Chance for an exact unseen answer is about 1.7 % with results below 64 and 0.4 % below 256; the two non-zero
scores are chance. Three things stand out.

- **A table, not an adder.** With results below 64 and three times the steps (about 60 sightings of each
  of the 1,664 training pairs), B answers 92-98.5 % of seen pairs exactly and 2-3 % of unseen ones. The
  bigger model memorises better (98.5 %) and generalises no better.
- **Drills teach a sense of size.** Bare "A + B = R ." lines give answers within one of the truth on
  98-99.5 % of unseen pairs after one pass (m64drill) and on 75 % even with results up to 255 (drill); the
  word-problem-only recipes (k8, k16, m64x3) give a coarse sense at best (within three on 43-81 %). The
  model's errors are ±1 or ±2 both ways, never a copied operand (0-3 %): it has a number line, not an
  algorithm. Samples from m64drillx3 on unseen pairs: 18 + 29 → 46 (47), 14 + 13 → 26 (27), 25 + 22 → 48
  (47), 36 + 6 → 43 (42); on seen pairs 49 + 5 → 54, 30 + 30 → 60, 33 + 20 → 53, all right.
- **Making it care costs stories.** Against base B (2.789) at the same story tokens, k8 costs 0.019 nats,
  m64 0.026, drill 0.058, m64drill 0.064 and k16 0.069. Part of that is weight decay 0.1 against 0.01
  (phase 5's `basewd` control separates the two); the x3 rows are not comparable (three times the training).

Story losses on the curve at 9.8 M story tokens for the x3 runs (2.86-2.93, mid-schedule) are not finished
models and are not compared.

## Phase 5: does the memorised table cost stories? (`figures/0_table_cost.png`)

Added after the verdict: A, C and D trained at the strongest memorising recipe (m64drillx3: six drill lines per
story, results below 64, addition only, answers weighted 8, three times the steps, weight decay 0.1), width 128,
seed 0, the same starting weights as phase 1's B at that recipe. Every model reads 29.5 M story tokens. One
seed, so a contrast has to clear about 0.01 nats to count (phase 0's single-seed contrasts scatter by 0.005).

| condition | steps | story loss | perplexity | story words in the mixed stream | unseen sums, own weights | seen pairs | restates a number in a new shape | name recall, gap 1 / 2 |
|---|---|---|---|---|---|---|---|---|
| A tool | 24,760 | 2.584 | 13.25 | **2.584** | 0 % (98.6 % with the machine) | | **0.243** | 0.93 / 0.92 |
| B computes | 20,783 | 2.607 | 13.56 | 2.633 | 0 % (within ±1: 99.5 %) | 96 % | 0.060 | 0.88 / 0.88 |
| C masked answer | 20,783 | 2.563 | 12.97 | 2.596 | 0 % (writes no number 82 %) | | 0.027 | 0.85 / 0.86 |
| D stories only | 14,390 | **2.486** | 12.02 | 2.733 | 0 % | | 0.000 | 1.00 / 1.00 |

Contrasts on held-out story text: **A − B = −0.023**, **B − C = +0.044**, A − C = +0.021, D − B = −0.121. On the
story words inside the mixed stream: A − B = −0.049, B − C = +0.037, A − C = −0.012, D − B = +0.100.

Weight-decay control: B at the base recipe with weight decay 0.1 scores 2.790 against 2.785 at 0.01 (+0.005),
so the 0.02-0.07 nats that the phase 1 recipes cost is the answer weighting and the drills, not the decay.

- **The table costs language.** B is 0.044 nats worse than C on the same stream with the answers masked; the
  only difference between them is that B must produce the answers, and what it produces is a memorised table
  and a number line. That is 4.5 % of perplexity for holding about 1,700 facts and a size sense in 0.8 M
  weights, more than the "few hundred neurons" estimate for an adder would suggest, because B's gradient on
  answers it cannot compute keeps pulling at the shared weights.
- **The tool model gets most of that back.** A is 0.023 nats better than B on new stories and 0.049 better on
  story words that follow a problem.
- **But at this density the chips are no longer free.** A is 0.021 worse than C on pure story text (it was
  0.004 better at two drills per story in phase 0). With six call blocks per story, 42 % of the tool stream is
  problem text, call syntax and machine text, and the model trains 24,760 steps to see the same stories; on
  in-context story words A is still 0.012 ahead of C.
- D's lead on pure story text grows to 0.121 with the longer training while on in-context story words it is
  0.100 behind B: the same text-match effect as in phase 0, larger.
- Skills: name recall rises for everyone with three times the training (A 0.93, B 0.88, C 0.85, D 1.00);
  A keeps the number-restating transfer (0.24 against 0.03-0.06); induction and the grammar pairs are within
  one-seed noise except bag/bags, where all four drop (0.63-0.85) on this number-heavy stream.

So the phase 5 answer is: when the self-computing model is pushed to hold the answers in its weights, it pays
about 0.04 nats of language for them, and delegating to the calculator recovers about half of that at this
call density.

## Phase 6: the grokking attempt (`figures/0_grokking.png`)

`python3 run_storyroom.py --phase 6`: B at four long recipes (six drill lines per story, addition only, results below
64, ten times the base steps: 69,277), width 128, seed 0, 4.2 hours each, four at once. The levers were weight decay
1.0 or 0.3, results below 32 instead of 64, and answers weighted 32 instead of 8. Exact accuracy on 150 unseen sums
was logged every 1,000 steps; at the end each model answered 500 unseen and 200 seen sums.

| recipe (all: 6 drills, add only, 10× steps) | story loss (perplexity) | exact, unseen sums | within ±1, unseen | exact, seen sums |
|---|---|---|---|---|
| weight decay 0.3, results < 64, answers × 8 | 2.605 (13.53) | 0 % | 97.5 % | 99.5 % |
| weight decay 1.0, results < 64, answers × 8 | 3.306 (27.28) | 0 % | 99.0 % | 95.0 % |
| weight decay 1.0, results < 32 | 3.287 (26.77) | 0 % | 98.5 % | 99.5 % |
| weight decay 1.0, answers × 32 | 3.455 (31.64) | 0 % | 99.5 % | 98.0 % |

**Nothing grokked.** Exact accuracy on unseen sums was 0 at every one of the 70 checkpoints of every run; no curve
rose at any point. The runner's verdict was "no grokking recipe reached the criterion", so the matched A, C and D
runs, the probes and the fine-tunes did not start. Two side readings: weight decay 1.0 is too strong for this model at
this learning rate (story loss stalls at 3.3-3.45, double the perplexity of the 0.3 run), yet the seen-sums table
survives it at 95-99.5 %, which says how cheap the table is to hold; and the 0.3 run ends at 2.605, where phase 5's
B had got in a third of the steps (2.607), so more training buys this width nothing on stories either.

### Why exactly zero: the held-out split holds out results, not pairs

The errors are the tell. On unseen sums every model's answer is the true sum minus one or plus one: −1 or +1 in
97-99.5 % of problems and 0 in none, for the four grokking runs and for phase 1's two drill runs alike (right-hand
panel of the figure). A noisy adder that lands within one on 99 % of sums would land exactly on about a third of them.
Zero is below chance, and it comes from the data split.

The repo's rule `is_test(kind, a, b)` in `cpugpt/data.py` held out (until the fix of 1 Nov 2026 below) a pair when
(a · 7919 + b · 104729 + k · 31337) mod 5 = 0. Both multipliers are 4 mod 5, so for addition (k = 1) the rule is simply **a + b ≡ 2 (mod 5)**. Checked by
enumeration below 64: all 429 pairs with sums 2, 7, 12, ..., 62 are held out and none of the other 1,651. (For
subtraction the same rule holds out 78-91 pairs in each residue class of a − b, a proper pair split; only addition
collapses.) So every "unseen sum" in phases 1 and 6 asked the model for an answer token that had never once been the
answer to an addition in training. The model had never written 17 as a sum, so it wrote 16 or 18, the sums it had
seen for the neighbouring pairs (a ± 1, b) and (a, b ± 1), all of which were in training. That is what "within ±1 but
never exact" means: a table and a number line, read off faithfully, with the correct output token missing from the
table by construction.

Consequences for this report:

- The 80 % criterion of phases 1 and 6 was a test of generalisation to never-seen *outputs*, far harder than the
  unseen *pairs* the plan intended, and one a lookup table cannot pass at all. A real adder could in principle pass
  it (the number tokens are trained as operands and in stories), so "no adder formed" still stands as a reading of
  the 0 %, but the recipes were never given the fair version of the test.
- The "approximate adder" wording of phase 1 was too generous: the ±1 behaviour needs no arithmetic beyond knowing
  that 17 sits between 16 and 18.
- The earlier story-tool report's "0 % on unseen addition" for the no-calls model was measured on the same held-out
  set, so it has the same reading. The tool model's 99 % is unaffected: copying the machine's result works for any
  token.
- The fix (1 Nov 2026): `is_test` in `cpugpt/data.py` is now a symmetric hash of (min(a, b), max(a, b)) and the
  operation, for every operation: about 20 % of pairs held out, every result value in training (below 64 every sum
  from 19 up has held-out pairs as well; sums 0-6, 8, 9 and 18 have none, they have too few pairs). For addition it
  is the hash the runner's `--split pair` flag had swapped in, so that flag is now a no-op alias of the default,
  kept for phase 7's `_pairsplit` job tags and `summary_pairsplit.json`. Models saved before the fix (phases 0-6
  here, and the earlier story-tool, shapes and check runs) were trained under the old split: they saw every pair
  whose sum is not 2 mod 5, which includes most of the new held-out pairs, so they cannot be re-scored fairly on
  it; a fair own-weights number needs a retrain. Phase 7 below is that retrain for B.

## Phase 7: addition under a fair pair split (`figures/0_pairsplit.png`)

`python3 run_storyroom.py --phase 7 --split pair`: B again at the two drill recipes (six drill lines per story,
results below 64, addition only, answers × 8; one and three times the base steps), width 128, seed 0, but with the
held-out set now 500 unseen **pairs** whose sums the model has seen for other pairs. Same stories, same starting
weights; only the 20 % of pairs kept back changed.

| recipe | steps | story loss (perplexity) | exact, unseen pairs | within ±1 (of 200 logged) | exact, seen pairs | same recipe, result split |
|---|---|---|---|---|---|---|
| 6 drills, 1× steps | 6,939 | 2.854 (17.36) | **65.0 %** | 96.5 % | 90.5 % | 0 % exact, 98.5 % within ±1 |
| 6 drills, 3× steps | 20,818 | 2.617 (13.69) | 40.6 % | 98.0 % | 95.5 % | 0 % exact, 99.5 % within ±1 |

**The self-computing model does learn to add, partly.** With every result value present in training, the one-cycle
drill model answers two thirds of never-seen pairs exactly (chance is under 2 %), 80 % when both operands are at
least 10 and 55 % when one is a single digit, and nearly all of its misses are off by one. Its exact accuracy was
still climbing at the end of its schedule (12 → 25 → 40 → 49 → 58 → 68 % on the 150 training-time pairs over the
last 6,000 steps). The three-cycle model is lower, 41 %: it memorises the seen pairs better (95.5 % against 90.5 %)
and sat at 25-30 % through its long high-learning-rate stretch before climbing to 43 % in the final third. One seed
each, so the gap between the two is not established; what is established is that the "never exact" of phases 1
and 6 was the split, not the model. Story loss is the same under either split (2.854 against 2.853; 2.617 against
2.607), so the phase 5 contrasts stand as measured.

By the plan's rule (80 % exact on the 500-pair set) neither run passes, so the gate stays formally shut. But a
model that is right on 65 % of new sums and within one on 96 % is computing something, and the question this
experiment was built for, what that computation costs in language and whether the tool gets it back, now has a
self-computing model worth asking it of. Phase 8 runs the matched comparison at the one-cycle drill recipe.

## Phase 8: the matched comparison at the recipes where B adds (`figures/0_table_cost.png`, `2_where_m64drill.png`, `4_transfer_m64drill.png`, and the `_m64drillx2` pair)

`python3 run_storyroom.py --phase 8 --split pair`: A, C and D at the one-cycle drill recipe (six drill lines per
story, results below 64, addition only, answers × 8, weight decay 0.1) from the same starting weights as phase 7's B,
each reading the same 9.82 M story tokens (steps 8,271 / 6,939 / 6,942 / 4,813), width 128, seed 0. Then the inside
look and the fine-tunes on the four models, and two more B runs to bracket the 65 %.

| | A tool | B computes | C masked answer | D stories only |
|---|---|---|---|---|
| story loss on held-out text, nats (perplexity) | 2.806 (16.54) | 2.854 (17.36) | 2.799 (16.42) | 2.743 (15.53) |
| loss on story words with a problem in the context | 2.832 | 2.887 | 2.856 | 2.928 |
| unseen sums right, own weights | 0 % (98.8 % with the machine) | 65.0 % | 0.6 % | 0 % |
| seen sums right, own weights | 1 % | 90.5 % | 0 % | 0 % |
| restating a number in never-seen sentence shapes | 21 % | 1 % | 0.7 % | 0 % |

| contrast | on held-out story text | on story words after a problem | base recipe (phase 0, 3 seeds) | memorising recipe (phase 5) |
|---|---|---|---|---|
| A − B, what the tool gets back | **−0.048** | −0.055 | −0.009 | −0.023 |
| B − C, what computing costs | **+0.055** | +0.031 | +0.005 | +0.044 |
| A − C, what the chips cost | +0.007 | −0.024 | −0.004 | +0.021 |
| D − B | −0.111 | +0.041 | −0.055 | −0.121 |

**Computing the sums costs the self-computing model 0.055 nats of language, and the tool model gets 0.048 of it
back.** Against the same stream with the answers masked out of its loss, B is 5.5 % worse in perplexity on new
stories (17.36 against 16.42); that is the price, at this width, of answering 65 % of new sums and 90 % of seen
ones. A, which reads the answers from the machine, is within 0.007 of the masked model, so the chips cost almost
nothing on pure text, and on story words that follow a problem A is the best of the four (it reads the result
instead of guessing it). The tool's lead over B grows with how much arithmetic B does: 0.009 when B holds nothing
(base recipe), 0.023 when it holds a table (phase 5), 0.048 when it adds. D is 0.111 better on pure stories and
0.041 worse on story words after a problem, the text-match effect of phase 0 again.

**The bracket.** A second B seed at the same recipe reaches 78.4 % on unseen pairs with story loss 2.868 (seed 0:
65.0 %, 2.854), so B sits at 2.85-2.87 against A's 2.806 (one seed). A two-cycle B (`m64drillx2`, 13,883 steps)
reaches **88.2 %**, above the plan's 80 % gate, with story loss 2.701; its exact accuracy peaked at 91 % and ended
at 88 %; a second seed there reaches 77.6 % (story loss 2.692). The matched A, C and D at that recipe, with the
probes and fine-tunes, are in the two-cycle section below.

### Inside the four models (`2_where_m64drill.png`)

- **The result probe.** A ridge probe reads the answer from the residual stream at the problem's cue word (in A, at
  the "=" of the call, before the machine has answered), scored on held-out pairs. R² of the value, best layer: B
  0.95, A 0.88, C 0.68, D −0.02. Mean accuracy of eight bit-probes (chance about 0.7 given the bit priors): B 0.93,
  A 0.79, C 0.75, D 0.71. So the *size* of the answer is linearly present in every model that has seen problems,
  the tool model included, and the exact answer only in B. The tool model knows roughly what is coming before it
  reads it; it does not compute it.
- **Unit ablations.** Each head and each of the 2,048 neurons replaced by its mean, scoring the change in story
  loss and in arithmetic loss (answer tokens; for A the copied answer and call syntax). Neurons whose removal hurts
  arithmetic beyond two standard deviations and stories not at all: B 58, C 40, A 0 (D's 30 are noise on an
  arithmetic loss of 15 nats). B has two heads that carry arithmetic (removing either adds 2.1-2.8 nats to the
  arithmetic loss) and both also cost 0.24-0.39 on stories; A has one such head, the copying head (1.05 on
  arithmetic, 0.35 on stories). Single-unit effects are small everywhere (hundredths of a nat), as a 0.055-nat
  total predicts. No dead units in any model; effective rank of the residual stream 88-107 in all four, B lowest
  at layer 1.

### Fine-tuning the four models (`4_transfer_m64drill.png`)

300 steps on n examples, one run per point, scored on 300 held-out problems.

- **Addition without the tool** (bare "a + b = r" lines): A is at 2-3 % for every n from 25 to 1,600, the same as a
  model trained from scratch (0-2 %), C 1-4 %, D 0-2 %. B, which already adds, drops to 15-22 % (the fine-tune
  interferes with it). There is no dormant adder in the tool model: it learns addition no faster than fresh weights.
- **A new story skill** (restating a number in held-out sentence frames and verbs, no calls): A 24 % from 25
  examples and 42 % from 100, starting from the 21 % it manages with no fine-tuning at all; D 5 % → 26 % at 100 →
  44 % at 400; B 5 % → 17 % at 100 → 36 % at 1,600; C 3 % → 13 % at 1,600. The tool model's head start is the
  copying skill it already has; the stories-only model catches it by 400 examples; the self-computing model trails
  D at every n above 50; the masked model, trained to leave the answer slot alone, is the slowest learner.

### The two-cycle recipe, where B passes the gate (`2_where_m64drillx2.png`, `4_transfer_m64drillx2.png`)

`python3 run_storyroom.py --phase 8 --split pair --recipe m64drillx2`: the same four conditions at twice the steps
(A 16,542 / B 13,883 / C 13,883 / D 9,627; 19.6 M story tokens each), width 128, seed 0, from the same starting
weights; a second B seed; the inside look and the 70 fine-tunes. 72 minutes. B here is the model that passes the
plan's 80 % gate (88.2 % on unseen pairs; the second seed gets 77.6 %).

| | A tool | B computes | C masked answer | D stories only |
|---|---|---|---|---|
| story loss on held-out text, nats (perplexity) | 2.646 (14.09) | 2.701 (14.89) | 2.635 (13.95) | 2.556 (12.88) |
| loss on story words with a problem in the context | 2.691 | 2.759 | 2.705 | 2.809 |
| unseen sums right, own weights | 0 % (99.6 % with the machine) | 88.2 % (seed 1: 77.6 %) | 0.4 % | 0 % |
| seen sums right, own weights | 0 % | 94.5 % | 0.5 % | 0 % |
| restating a number in never-seen sentence shapes | 19 % | 4 % | 2 % | 0 % |

| contrast | on held-out story text | on story words after a problem | one-cycle recipe (above) |
|---|---|---|---|
| A − B, what the tool gets back | **−0.055** | −0.067 | −0.048 |
| B − C, what computing costs | **+0.066** | +0.054 | +0.055 |
| A − C, what the chips cost | +0.010 | −0.013 | +0.007 |
| D − B | −0.145 | +0.050 | −0.111 |

**Same picture, a little larger.** With B above the gate, every contrast moves the way it did at one cycle and a
bit further: computing costs B 0.066 nats against the masked stream (perplexity 14.89 against 13.95, 6.8 %), the
tool gets 0.055 of it back, the chips cost 0.010 on pure text, and A is again the best of the four on story words
that follow a problem. That completes the chain: the tool's lead over B is 0.009 when B does no arithmetic, 0.023
when it holds a table, 0.048 when it adds 65 % of new sums and 0.055 when it adds 88 %. B's two seeds at this
recipe differ by 0.009 in story loss (2.701 and 2.692), so the 0.055 is about six times the one seed scatter that
was measured, with the caveat that the two recipes share starting weights and data order and are not two
independent draws. D's lead on pure stories widens to 0.145 with the longer training, and D is still the worst on
story words after a problem.

**Inside.** The result probe reads the exact answer from B and the rough size from the rest: R² of the value, best
layer, B 0.97, A 0.85, C 0.65, D 0.15; bit-probes B 0.95, A 0.79, C 0.74, D 0.72. Neurons whose removal hurts
arithmetic and not stories: B 50, C 41, A 0 (D's 42 are noise on an arithmetic loss of 17 nats). B now has two
heads in layer 1 that are almost pure arithmetic (removing either adds 3.5-4.0 nats to the arithmetic loss and
0.02 to stories) and one head in layer 0 that carries both (4.6 on arithmetic, 0.60 on stories); A has the copying
head (1.7 on arithmetic, 0.36 on stories) and a layer-1 head that routes the operands (1.3 on arithmetic, 0.02 on
stories). So B's cost in language does not sit in its arithmetic-only units, which can be removed at no story
cost; it sits in the shared head and spread thinly across the rest, which is what a 0.066-nat total made of
hundredths-of-a-nat units looks like. Two neurons of B's layer 1 are dead (none elsewhere); effective rank 93-112
in all four, B lowest at layer 1 again.

**Fine-tunes.** Addition without the tool: A 1-5 % at every n, the same as from scratch (0-2 %), C 1-3 %, D 0-2 %;
B, which already adds, sits at 27-31 % from 25-50 examples, falls to 6 % at 800 and jumps to 47 % at 1,600 (one run
per point; the fine-tune fights the adder it already has). No dormant adder in the tool model at this recipe
either. The new story skill: A 16 % from 25 examples and 39 % from 100 (19 % untuned); D 4 % → 27 % at 100 →
41 % at 400 → 54 % at 1,600; B 5 % → 26 % at 100 → 41 % at 1,600; C 2 % → 7 % at 400 → 29 % at 1,600. A's
head start is smaller than at one cycle (16 against 24 % from 25 examples), D passes it by 400 examples and ends
highest, B keeps pace with D to 100 examples and then trails, C is last.

**Skills.** was/were and pronoun case 1.00 for all four; has/have A 1.00, B 0.81, C 1.00, D 0.99; gender
0.94-0.98; number A 0.65, B 0.69, C 0.74, D 0.75; name recall A 0.84/0.78, B 0.78/0.77, C 0.86/0.84, D 0.99/0.99;
induction 0.20-0.28 for all. One thing repeats across the recipes, and it is one starting point rather than one
condition: B's seed 0 is low on has/have at both (0.57, 0.81) while B's seed 1 is at 0.99-1.00. D's name recall
is highest at both recipes (0.83, 0.99), which fits a model that only ever reads stories. Nothing else repeats.

## What this says about the question

The plan fixed its rules in advance. Freed capacity would need, all at once: A better than B on story loss where B
learned addition and not elsewhere; more arithmetic-only units in B than in A; the result readable from B's
residual stream but not A's; A picking up a new skill or recovering addition faster. Applied to phase 8 at both
recipes (B at 65 % on new sums at one cycle and 88 % at two, seed 0; 78 % at the second seed of each):

1. **A beats B where B adds: yes**, by 0.048 nats at one cycle and 0.055 at two, and the lead scales with B's
   arithmetic (0.009 at the base recipe where B does none, 0.023 where it holds a table, 0.048 where it adds 65 % of
   new sums, 0.055 where it adds 88 %). "Not elsewhere" holds in the sense that
   the base-recipe gap is a fifth of the drill-recipe gap.
2. **More arithmetic-only units in B: yes.** 58 neurons and two heads against none and one (the copying head) at
   one cycle; 50 neurons and two heads against none and one at two.
3. **The result readable in B but not A: half.** Its size is linearly readable in both (R² 0.95 against 0.88 at one
   cycle, 0.97 against 0.85 at two); its exact bits only in B (0.93 against 0.79; 0.95 against 0.79).
4. **A learns faster: half.** The new story skill, yes, from the copying skill it already has (24 % from 25
   examples against 5 % at one cycle, 16 % against 4-5 % at two); addition, no, it is at the from-scratch rate.

So the hypothesis survives in its measurable form: the language a small model spends on computing is recovered when
the computation is delegated, nearly in full, at a chip cost near zero. The plan's four-way test says *where* the
recovered room is not: it is not an adder kept in reserve (fine-tuning finds none) and not a measurable gain on
grammar (seed noise swamps it); it is lower loss on ordinary story text and a model that copies numbers well.

What the whole run establishes, in order of confidence:

- **The chips are free at two calls per story and nearly free at six.** A − C is −0.004 ± 0.002 at the base recipe
  (three seeds), +0.007 at six drills per story under the fair split, and A is the best of the four on story words
  that follow a problem. Only the memorising run under the old split charged 0.021.
- **Computing costs language and delegation gets it back.** B − C = +0.055 and A − B = −0.048 at the recipe where B
  adds 65 % of new sums, +0.066 and −0.055 where it adds 88 %; +0.044 and −0.023 where B only memorises (phase 5); +0.005 and −0.009 where B does nothing (phase 0).
- **The cost is small because the computation is small.** 0.055-0.066 nats at width 128 is a few dozen neurons and
  two or three heads; the residual stream loses no rank and no unit dies. This is a lookup-and-interpolate adder (65-88 % exact,
  96 % within one), not the Fourier circuit of the grokking papers, and ten times the training with weight decay did
  not turn it into one.
- **The one skill the tool really buys is being right**: 99 % on unseen sums with the machine on, against 0 % at the
  base recipe and 65-88 % at the best drill recipes for the self-computing model, and 19-62 % against 0-4 % at
  restating a number in a new sentence.
- **Reading the earlier perplexity gap.** "No-calls 15.33 against tool 15.7" is explained by story tokens per step;
  it was never a cost of the chips.
- **The held-out split.** The repo's addition split held out result values, which made "exact on unseen sums" a
  test of never-seen outputs; every own-weights addition figure measured before 1 Nov 2026 is of that kind.

What would sharpen the number: three seeds for A, C and D at the drill recipes (the 0.048-0.055 is five or six times a seed's
scatter, but that scatter is B's, and the two recipes share init and data order); the width sweep of the plan, to see whether the recovered 0.048 shrinks or grows
with model size (the parameter-equivalent factor); and digit tokens, so that addition is an algorithm over a few
symbols and a real circuit has a chance to form.

## Limitations

- Everything is at one width (128). Phases 1 and 5-8 are at one seed for A, C and D (B has two seeds at each phase 8
  recipe: story loss 2.854 and 2.868 at one cycle, 2.701 and 2.692 at two; 65/78 % and 88/78 % on new sums). Seed spread of story loss is 0.004-0.014 nats, so the
  base-recipe A − B (0.009, all three pairs negative) is at the edge of what three seeds resolve and each phase 8
  A − B (0.048, 0.055) is a point estimate with about ±0.01 of scatter, and the two recipes share init and data
  order, so they are not independent draws.
- At the one-cycle recipe B is below the plan's 80 % gate (65 % at seed 0, 78 % at seed 1); the two-cycle recipe
  passes it at seed 0 (88 %) and misses at seed 1 (78 %). The matched comparison was run at both and agrees, so the
  verdict does not hinge on the gate.
- The skill tests move by up to 0.4 between seeds of the same condition (has/have 0.57 against 0.99; induction
  0.00 against 0.30), so they can rule out large effects only. The fine-tunes are one run per point.
- A's three models were trained by `run_storytool.py` (same code, same seeds, same recipe) and loaded, not
  retrained here; the shared-init claim holds because both runners seed the model the same way.
- The story-loss test set is pure story text; the mixed-stream role losses are the fairer comparison for D and
  make the opposite ranking. The text's distribution, not capacity, drives the D rows.
- Phase 1's non-base recipes change weight decay to 0.1 along with the lever they test; phase 5's `basewd`
  control puts the decay's share at 0.005 nats for the base stream.
- Phase 5's A, C and D are one seed each against phase 1's B; its contrasts are 3-6 times a single seed's
  scatter, not three-seed means.
- Phases 0-6 ran under the repo's original held-out rule, which for addition holds out result values rather than
  pairs (phase 6). Their story losses and contrasts are unaffected (B's story loss is the same under either split),
  but their "exact on unseen sums" figures measure generalisation to never-seen answer tokens; phases 7 and 8 use
  the pair split, now the repo's default.
- The held-out addition set uses trained sentence frames only; the "within ±1" measure counts answered
  problems (no-answer rates are 0-6 %).
- The skill tests are log-probability minimal pairs built from the story lexicon; the gender and number pairs
  are sensitive to which names and nouns the generator draws, hence their 0.03-0.1 seed spreads.
