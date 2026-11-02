# A story GPT that calls the gate computer in writing

> **Note (1 Nov 2026): the held-out split was wrong for addition.** Until today the rule in `cpugpt/data.py`
> (`is_test`) reduced, for addition, to a + b ≡ 2 (mod 5): it held out whole result values (2, 7, 12, ...), never
> pairs, so no model ever saw 17 as the answer to any sum. Every number in this report where a model answers from
> its own weights was measured on that split: the "computer off" figures (2.6-3.5 %, and 8.7-9.9 % for the
> answer-in-loss model; "1-4 %" and "9-10 %" in the text), the `no calls (same problems)` row of the results table,
> and the "unseen sums right, own weights: 0 %" row of the 31 Oct update. On the addition problems those tests
> asked for an answer token never seen as an answer (subtraction and multiplication pairs were split properly, so
> the mixed-operation sets are affected in their addition third). The models that call the computer are
> unaffected: they copy the machine's result, which works for any token. The rule is now a symmetric hash of the
> pair, about 20 % of pairs per operation with every result value in training. The saved models cannot be
> re-scored fairly on it: they were trained on every pair whose sum is not 2 mod 5, which includes most of the new
> held-out pairs. A fair own-weights number needs a retrain (`../results_storyroom/ROOM_REPORT.md`, phases 6-7).

Produced by `python run_storytool.py` (seven training runs, about 75 minutes on a laptop CPU; `--reuse`
trains only runs without a saved model, `--analyse` reuses all of them). Numbers are in `summary.json`,
the figure in `figures/`, models in `models/`, logs in `logs/`. The design follows the research notes in
`../reports/Small GPT in context tool calls.md`. The earlier version, where an attention head calls the
computer and its answer is pushed into the next word, is in `../results_storygpt/`.

## Update (31 Oct 2026): sentence shapes it was never trained on

The weakness found below (1-33% on never-trained sentence shapes) is addressed in a second run,
`../results_storytool_shapes/` (report: `../results_storytool_shapes/SHAPES_REPORT.md`; the same
`run_storytool.py`, which now writes there). The fixed lists of word problems, two-step problems and no-call
sentences are replaced by a compositional generator in `cpugpt/text.py` (openings, starts, changes with 23
verbs, 22 questions, names and pronouns, distractor numbers, no-call sentences with the same verbs and no
question or no numbers); 43 frames and 10 verbs are held out of training for testing, and the three
never-trained wordings below are kept as a test set. Held-out operand pairs, three seeds, same model sizes:

| | word problems, trained shapes | held-out shapes | held-out verbs | the three wordings below | two-step | two-step with a held-out piece | no-call: calls | story ppl (no-calls GPT on the same stream) |
|---|---|---|---|---|---|---|---|---|
| this run's models (12 fixed shapes), on the new sets | 29-35% | 17-20% | 15-16% | 1.5-35% | 8-11% | 2-5% | 0.4-1.4% | 15.28-15.39 (14.70) |
| varied-shape models | 98.4-99.5% | 70-78% | 66-84% | 96.5-98.7% | 98-99% | 50-62% | 0.5-2.0% | 15.66-15.76 (15.33) |

Copy accuracy given a right machine answer stays 100%; the misses on held-out shapes are calls not made
(12-14%, questions with no number cue such as "she looked and saw") and wrong operands (10%, mostly sentences
with the second number first). A decode-time margin on `<calc>` (`cpugpt/tool.py`, `decode(calc_margin=)`;
a slider on the page) trades top-guess calls in story text (75 → 24 → 5 at margins 1 and 2) against 1-3 points
on trained shapes and 3-7 on held-out ones. The models, numbers and figures of this run are unchanged below.

## Update (31 Oct 2026): checking a flaky computer

A third run, `../results_storycheck/` (report: `../results_storycheck/CHECK_REPORT.md`; `run_storycheck.py`,
`cpugpt/check.py`), asks whether the GPT can learn to notice a wrong machine answer and ask again. Trained
against a computer that is wrong 25% of the time, with the call repeated in the text after every wrong
answer, the GPT catches 55-60% of wrong answers from the numbers alone (every timeout, two thirds of
"other operation" answers, one bit flip in four) but asks again after 24-27% of right answers too, and that
habit costs it 10 points with a perfect computer and 40 on two-step problems; against the flaky computer it
ends at 75-77%, the same as the trusting model (75%). When the computer also writes `<unsure>` in the chip
from its own signal margins, the GPT learns "ask again after `<unsure>`" and reaches 83-84% against the flaky
computer (copying a wrong answer in 13.5% of problems instead of 24.7%), 97% with a perfect one, and, on the
real analog CPU under noise (15-18% of answers wrong), 87.8% against 81.6% for the trusting model. Two-step
problems get worse with every kind of checking (the GPT mixes "ask again" with "next step", and three calls
per problem is the decode limit). The models, numbers and figures of this run are unchanged below.

## Update (31 Oct 2026): does the calculator leave room for language?

A third follow-up asks whether handing the sums to the computer leaves these weights more room for
language (`../run_storyroom.py`, report `../results_storyroom/ROOM_REPORT.md`, reading in
`../reports/Does tool use free model capacity.md`). Four models on the same stories, problems and starting
weights, each reading the same 9.82 M story tokens: tool calls (A), answers written by the model (B), the same
problems with the answer left out of the loss (C), stories only (D); three seeds at width 128.

| | A tool | B computes | C masked answer | D stories only | B at equal steps (the no-calls model above) |
|---|---|---|---|---|---|
| story loss, nats (perplexity) | 2.780 (16.12) | 2.789 (16.27) | 2.784 (16.18) | 2.735 (15.40) | 2.752 (15.68) |
| unseen sums right, own weights | 0 % (99 % with the machine) | 0 % | 0 % | 0 % | 0 % |

At equal story text the tool model is the best of the three mixed-stream models, by 0.009 nats over B (all
three seed pairs; the "no-calls is better" gap above is its 12 % more story text per step). But nothing was
freed, because B never learns to add: ten recipes (answer weight 8-16, drill lines, results below 64, three
times the steps, width 256) reach at most 3.2 % on unseen sums while memorising 92-98.5 % of seen ones and,
with drills, landing within ±1 of the truth on 99.5 % of unseen sums without ever being exact. A model that
holds a table and a number line has no adder to give up. D's lead is the training text matching the test
text (on story words with a problem in the context D is the worst). The one transfer: the tool model restates
a number in never-seen sentence shapes 62 % of the time against 0-3 % for the others. The planned width
sweep, probes and transfer tests were gated on B learning addition and did not run. An extra phase priced the
table itself: at the strongest memorising recipe (six drills per story, three times the steps, one seed) the
self-computing model is 0.044 nats worse on new stories than the same stream with the answers masked, and the
tool model gets 0.023 of that back, though at that density the chips cost 0.021 against the masked model. The
grokking attempt (B at ten times the training, weight decay 0.3-1.0, four recipes, 4.2 h each) changed nothing:
0 % exact on unseen sums at all 70 checkpoints of every run, 95-99.5 % on seen sums, within ±1 on 97-99.5 % of
unseen ones. Tracing that ±1 found a flaw in the held-out rule: `cpugpt/data.py`'s `is_test` reduces, for
addition, to a + b ≡ 2 (mod 5), so it holds out whole result values (2, 7, 12, ...), never pairs. Every "unseen
sum" in that run, and the 0 % figures for the no-calls models in this report, asked the model for an answer
token it had never seen as an answer; the tool model's 99 % (copying the machine's result) is unaffected. The
rule was replaced on 1 Nov 2026 by a symmetric hash of the pair (about 20 % of pairs per operation, every result
value in training; the room runner's `--split pair` is now an alias of the default). Re-measured under the fair
split (room report, phase 7), the self-computing model at the one-cycle drill recipe gets 65 % of never-seen
sums exactly right (78 % at a second seed, 88 % with two training cycles) and 96 % within one: it learns
addition partly. The matched comparison at that recipe (phase 8) then gave the answer to the capacity question:
computing the sums costs the self-computing model 0.055 nats of language against the same stream with the answers
masked, the tool model gets 0.048 of it back, and the chips cost 0.007; at the two-cycle recipe where it passes
the 80 % gate the three numbers are 0.066, 0.055 and 0.010 (one seed each; the probes find the answer's exact bits
only in the self-computing model, and fine-tuning finds no dormant adder in the tool model at either recipe).

## Answer first, in plain words

The GPT now uses the computer the way a person uses a calculator. When a sum comes up in a story it
types the question into the text itself, `<calc> 12 + 5 =`; the program pauses, the 8-bit computer built
from trained neuron gates runs the sum and types `17 </calc>` back into the text; the GPT reads that 17
and carries on, "now he has 17 apples". The computer's answer is something the GPT reads from the page,
never something pushed into its mouth. So the GPT can use the answer again: for "she got 7 more and then
4 more" it types `<calc> 20 + 7 =`, reads 27, types `<calc> 27 + 4 =`, reads 31, and writes 31.

On problems with number pairs it never saw in training, it gets **97-99% of written sums, 98-100% of word
problems and 99.3-99.8% of two-step problems right** (three seeds). In every case where the computer gave
the right answer the GPT copied it right (100%); the few errors are calls with the wrong numbers typed in.
Switch the computer off and the GPT fills in its own display value: 1-4% right, so the arithmetic lives in
the computer, not in the GPT. Give it a computer that does the wrong operation and it copies the wrong
answer (1%). On sentences with numbers but nothing to work out ("lily counted 9 shells . that is") it
never calls and restates the number (99.7-100%). In 207,000 words of held-out stories the computer is its
top guess at only 58-73 positions, nearly all right after "had" or "were", the words that end a word
problem; writing 8,000 words freely it called 0-3 times, each time after inventing a sum of its own.

Two things the research predicted came true. Leaving the computer's answer *in* the training loss
(as GSM8K did) makes the GPT start doing sums itself: 92.6% instead of 99% on written sums, 93.7% instead
of 99% on two-step problems, and 9-10% right with the computer off instead of 1-4%. And the decision to
call is learned from the words around the sum: on three sentence shapes it never saw in training it
scores 1-33%, depending on the seed, and a 6-layer model is no better. The old pointer-head version had
the same weakness (0-38%).

Story perplexity is 15.3-15.4 against 14.7 for the same GPT trained without call blocks (14.4-14.5 for
the pointer-head version, whose stream had fewer problems). The call blocks make the training text 12%
longer, so the model reads fewer story words in 7,000 steps; a run trained for as many story words as the
no-calls GPT comes to **14.81**, 0.1 above it, with the same accuracy on the problems.

## What changed from the pointer-head version

| | pointer-head version (`results_storygpt`) | this version |
|---|---|---|
| who picks the numbers | three trained pointers in an attention head | the GPT types them into the call |
| who decides to call | a trained gate, by REINFORCE | the GPT writes `<calc>` or not, by ordinary next-word training |
| where the answer goes | 8 result bits projected into the residual stream (+ a copy bonus) | typed into the text as a token the GPT reads |
| chaining | one call per position, no chaining | a second call can use the first result |
| what the computer's tokens do in training | n/a | in the input, out of the loss |
| visible to the reader | only in the side panel | in the story, as a calculator chip |

**Tokens.** Two tokens, `<calc>` and `</calc>`, appended at the end of the vocabulary (4,281 tokens; every
other id, including the numbers 0-255, is unchanged). A call is `<calc> A op B = R </calc>` with the
existing `+ - * =` tokens; the GPT writes up to `=`, the computer writes `R </calc>`.

**Training text** (`cpugpt/text.py`, `build_tool_stream`). Every problem carries its call block before
the answer, GSM8K style: `what is 14 times 12 ? <calc> 14 * 12 = 168 </calc> 168 .`; `tom had 12 apples .
he got 5 more . now he has <calc> 12 + 5 = 17 </calc> 17 apples .` Per story on average: 2 math templates,
1.5 word problems (the 12 shapes of the previous version), 0.5 two-step problems (6 new shapes, e.g.
`lily had 20 stickers . she got 7 more and then 4 more . now she has <calc> 20 + 7 = 27 </calc> <calc> 27
+ 4 = 31 </calc> 31 stickers .`, operation pairs ++, -+, +-, x-, x+, --) and 0.5 no-call sentences (5
shapes with numbers and nothing to compute, `lily counted 9 shells . that is 9 shells .`). Operand pairs
use the same held-out split as before (`cpugpt.data.is_test`), for both steps of a two-step problem.
The stream is 6.19 M tokens (5.53 M without the call blocks): 42,040 template, 31,514 word, 10,483
two-step and 10,446 no-call problems; 189,040 tokens (3.1%) are written by the computer.

**Loss** (`cpugpt/lm.py`). Ordinary next-word cross-entropy on everything the GPT writes, including
`<calc>`, the operands, the operation, `=` and the copied answer; the targets `R` and `</calc>` are left out
(`ignore` mask). They stay in the input, so later words are trained to read them.

**Decoding** (`cpugpt/tool.py`, mirrored in the page). `</calc>` is never sampled; `<calc>` is not sampled
inside a call or after two calls; inside a call decoding is greedy. On `=` inside a call, if the call reads
exactly NUMBER OP NUMBER the computer's answer and `</calc>` are appended; otherwise (malformed call, or the
computer switched off) the GPT's next token is taken as its own display value and `</calc>` is appended; a
call that runs five tokens without `=` is closed. The computer's answers come from the table computed on
the gate netlist (`run_cpugpt.py`); 300 of the page model's calls were re-run on the fully analog neural
CPU, 0 differences.

**Models** (same sizes as before: 4 layers, d_model 128, 4 heads, MLP 512, 64-token context, 7,000 steps
of 32 windows, AdamW): `story GPT + calc` seeds 0-2; `answer in loss` (the computer's answer is a training
target too); `no calls` (the same problems without call blocks); `6 layers`; `same story words` (7,838
steps, so it reads as many story words as the no-calls run). No CPU head in any of them.

## Results

Every problem sits after held-out story text and is cut at the cue word ("now he has"); the GPT must write
the call(s) and the answer itself. Held-out operand pairs throughout: 3,000 written sums (the templates
of the earlier versions), 3,000 word problems, 1,500 two-step problems, 1,500 problems in three sentence
shapes never trained, 1,000 no-call sentences. Perplexity on the same 256 held-out windows as before.

| model | written sums | word problems | two-step | new shapes | no-call: calls / right | story ppl | `<calc>` top guess in 207k story words | calls in 8,000 free words |
|---|---|---|---|---|---|---|---|---|
| + calc, seed 0 | 99.0% (98.6-99.3) | **99.7%** (99.4-99.8) | **99.3%** (98.7-99.6) | 33.4% | 0% / 99.9% | 15.34 | 58 | 3 |
| + calc, seed 1 | 98.0% | 98.1% | 99.8% | 6.1% | 0% / 100% | 15.39 | 69 | 1 |
| + calc, seed 2 | 96.7% | 98.4% | 99.6% | 1.1% | 0% / 99.7% | 15.28 | 73 | 0 |
| + calc, answer in loss | 92.6% | 99.1% | 93.7% | 23.5% | 0% / 96.6% | 15.61 | 68 | 1 |
| + calc, 6 layers | 99.1% | 99.2% | 99.9% | 32.1% | 0% / 100% | 14.96 | 56 | 0 |
| + calc, same story words (7,838 steps) | 99.1% | 99.3% | 99.7% | 11.1% | 0% / 98.2% | **14.81** | 69 | 3 |
| no calls (same problems) | 4.2% | 4.5% | 0.8% | 0.2% | – / 18.7% | 14.70 | – | – |
| pointer head, seeds 0 / 1 (previous run) | 98.6 / 90.5% | 99.5 / 89.7% | – | 38.1 / 12.8% | – | 14.47 / 14.52 | 0 / 0 | – |

(95% intervals for seed 0; the others are within ±1 point. Per operation, seed 0: sums 99.6 / 98.8 / 98.7%
for + / − / ×, word problems 99.8 / 99.9 / 99.4%.)

### The answers come from the computer

- Every call on these sets was well formed; the GPT called on 100% of the problems and on none of the
  no-call sentences. **Whenever the computer was right, the written answer was right**: copy accuracy
  given a right machine answer is 100% in all seven models and all sets, including answers 192-255 that
  troubled the pointer-head version. All remaining errors are calls with the wrong numbers typed in.
- Computer off (the GPT fills in the display itself): 2.6-3.5% on sums and word problems, 0.9-2.2% on
  two-step problems. Computer doing the wrong operation: 1.0-1.2%. The GPT does not add; it copies.
- Training pairs score the same as held-out pairs (96.9-99.8%), so nothing is memorised.

### Two-step problems: the first answer is read back into the second call

| | call 1 right | second call starts from the first result | call 2 right | final answer right |
|---|---|---|---|---|
| seed 0 / 1 / 2 | 99.5 / 99.8 / 99.6% | **100 / 100 / 100%** | 99.3 / 99.8 / 99.6% | 99.3 / 99.8 / 99.6% |
| answer in loss | 94.7% | 99.7% | 93.7% | 93.7% |

Per operation pair (seed 0): ++ 100%, −+ 98.8%, +− 97.6%, ×− 99.3%, ×+ 100%, −− 100%. The hand-off, which
the research literature never measures, is perfect here: the second call's first operand is what the
computer just wrote, every time. The research report's worry that the second read would be a "most recent
write" lookup with a glitch tail did not materialise at this length (the first result is 2 tokens behind
the second call).

### Leaving the answer out of the loss matters

With the computer's answer as a training target (GSM8K's "they are all just tokens"), the same model:
- learns a little arithmetic itself: 8.7% of sums and 9.9% of word problems right with the computer off
  (masked models: 2.6-3.6%);
- calls less where it thinks it can manage: 96.9% of written sums (92.6% of multiplications) instead of
  100%, and 1.4 calls per new-shape problem;
- is worse with the computer on: 92.6% on sums, 93.7% on two-step problems, 96.6% on no-call sentences,
  and 0.3 perplexity worse (15.61);
- learns more slowly: two-step accuracy 57% at step 3,000 against 33-99% for the masked seeds, 95% at
  7,000 against 99%.

### When not to call

- No-call sentences (numbers, nothing to compute): 0 calls in 1,000 for every model; the restated number
  is right 99.7-100% ("that is 9 shells", "put the 12 apples in the bag").
- Held-out story text, teacher-forced: `<calc>` is the top prediction at 56-73 of 207,168 positions
  (0.03%); for seed 0, 37 of 58 come right after "had", 10 after "were", 9 after "?", the words that end a
  word problem in training ("he had | saved", "they were | twins").
- Free writing from 200 held-out openings (8,000 words): 0-3 calls per model, each after the GPT had
  written a problem of its own ("lily had 13 boxes. each box had 12 blocks. that is `<calc> 13 * 12 =`").
  The page example "they hugged each other and ate more cake ." shows the failure that remains: the GPT
  continues "they had" and, with no numbers in sight, opens a call with made-up numbers.
- The pointer-head version never called in story text (0 of 207k). Its call decision was a separate gate
  trained with a call cost; here the decision is a word like any other, and "had" at the end of a clause
  is a weak trigger. Toolformer-style decode-time control (a logit bias on `<calc>`, or a rule that opens
  a call only when the model's top choice is `<calc>` by a margin) was not needed on the problem sets and
  is left for a follow-up.

### Where the errors are

Seed 0 gets 9 of 3,000 word problems wrong, seed 1 56; every one is a call with wrong operands (never the
operation, never a missing call, never a miscopy). They concentrate in "A friends each brought B", where
the problem starts with its first number (`2 friends each brought 9 blocks. they had` → `<calc> 13 * 9 =`:
seed 0 97.6%, seed 1 80.8% on this shape, 99-100% on the others) and in "ate B of his A", and the wrong
operand is often a number that is not in the text at all (13, 14, 15 for 2, 15, 7). The pointer-head
version had the same two weak shapes.

### Sentence shapes never trained

Three shapes, held-out pairs, never in training (`NEW_WORDINGS`): "picked A . then M gave him B more",
"there were A on the plate . N ate B of them . then there were", "N has A boxes and each box has B . so N has".

| model | + ("gave ... more") | − ("ate B of them") | × ("each box has") | what goes wrong |
|---|---|---|---|---|
| seed 0 | 1.5% | 40.2% | 61.4% | + : every call is `A − B` (reads "gave", ignores "more"); − : 208 of 475 calls use × , 76 wrong operands; × : 121 no call, 68 wrong operands |
| seed 1 | 0.8% | 2.3% | 15.7% | + : `A − B`; − : 375 right calls followed by a second, wrong call ("then there were" looks like a two-step problem); × : 415 no call |
| seed 2 | 0.6% | 1.9% | 1.0% | + : `A − B`; − : mostly ×; × : no call |
| answer in loss | 8.6% | 2.1% | 60.4% | as seed 0 |
| 6 layers | 1.3% | 100% | 0% | + : `A − B`; × : no call |

The calls are triggered by words, and the operation is read from a word: "gave" wins over "more" in every
model; "then there were" pulls in the two-step pattern; "so N has" is not a cue most seeds know. Which
shapes a model gets by chance varies with the seed, and depth does not change the picture (the 6-layer
model gets one shape fully and the others not at all). This is the same limitation as in the pointer-head
version (0-38%) and the main thing left to fix; the research report's prediction (surface triggers, seed
variance) was right.

### Story perplexity

| | no calls | + calc (3 seeds) | + calc, same story words | answer in loss | 6 layers | pointer-head version: plain / + head |
|---|---|---|---|---|---|---|
| 256 held-out windows | 14.70 | 15.28-15.39 | **14.81** | 15.61 | 14.96 | 14.29 / 14.41-14.52 |
| all 207k held-out words | 15.12 | 15.63-15.80 | **15.15** | 15.89 | 15.39 | 14.82 / 14.94-14.98 |

The no-calls GPT is the fair baseline (same stories, same problems, no call blocks). The call blocks make
the stream 12% longer, so in 7,000 steps the calc models read 12% fewer story words. Trained for 7,838 steps
instead, so that it reads as many story words as the no-calls run, the calc GPT comes to 14.81 (15.15 on
all text): 0.1 above the no-calls GPT, with the same accuracy on the problems (99.1 / 99.3 / 99.7% on
sums / word problems / two-step). The 0.6 gap at equal steps is the story-word budget, not the calls. The
previous version's numbers are lower because its stream had fewer problems (3.5 per story against 4.5).
The text loss curves (figure, right) run together, with the no-calls run a little below throughout.

### Learning speed

Word problems pass 90% at step 2,000-4,000 (seed 2 at 2,000, seed 1 at 3,000, seed 0 at 4,000) and
two-step problems in the same window; the pointer-head version reached 85% on word problems at step
2,000 and 90-91% at 3,000 with REINFORCE. Here everything is learned by plain next-word prediction, with
no REINFORCE, no pointer samples and no call cost; a run takes 34-37 minutes on one thread instead of
51 on two.

### Samples

Greedy continuations from the page model (seed 0); the call chips are what the GPT typed, the blue
numbers what the computer typed back:

> lily had 20 stickers. she got 7 more and then 4 more. now she has `<calc> 20 + 7 =` **27** `</calc>`
> `<calc> 27 + 4 =` **31** `</calc>` 31 stickers. she was so happy to have her new friend.
>
> ben had 50 grapes. he ate 12 and then got 30 more. he had `<calc> 50 - 12 =` **38** `</calc>`
> `<calc> 38 + 30 =` **68** `</calc>` 68 grapes.
>
> lily counted 9 shells. that is 9 shells. she was so happy and thanked the man for being so kind.
>
> sam picked 30 grapes. then mia gave him 12 more. sam had `<calc> 175 - 12 =` **163** `</calc>` 163 grapes left.

The last one is a never-trained shape: the GPT reads "gave" as a subtraction and types an operand that
is not in the text. With sampling (temperature 0.8) it invents problems in its stories and solves them
with the computer, as the previous version did: "sara had 22 coins. she got 162 more. now she has `<calc>
22 + 162 =` **184** `</calc>` 184". The no-calls GPT writes 20 apples, 20 left, 36 toys, 55 stickers and 90
grapes for the same openings.

## Try it

`storytool_ui/index.html` (with `model.json`, `cpu.json` and the two weight files) runs the GPT in the
browser: the call chip is typed out, the computer runs the sum on neural gates in the side panel, its
answer appears in the chip in blue, and the GPT's copied answer is marked. Settings: the GPT that calls the
computer or the no-calls GPT; the computer answering, silent (the GPT fills in the display itself: "12 + 5
= 7") or doing the wrong operation (the GPT copies 7); randomness; length.

    python3 -m http.server 8765 --directory storytool_ui      # then open http://localhost:8765

## Limitations

- Calls are triggered by the words the GPT saw in training; new sentence shapes mostly fail, and "had"
  at the end of a clause can open a call with made-up numbers. More shapes and paraphrases in training,
  and a decode-time margin on `<calc>`, are the obvious next steps; a REINFORCE stage with a
  necessity-aware reward (the research report's recommendation) is the other.
- Two-step problems are the only chains trained; the second call follows the first result by two
  tokens. Longer chains and results used several sentences later are untested.
- Perplexity is 0.6 above the no-calls GPT at equal steps and 0.1 above at equal story words.
- One seed for the ablations; three for the main model.
- Story quality is that of a 1.4 M-weight model trained for 35 minutes: simple and repetitive.
