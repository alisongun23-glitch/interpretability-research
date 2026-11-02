# Can the story GPT learn to check the computer?

> **Note (1 Nov 2026): the held-out split was wrong for addition.** Until today the rule in `cpugpt/data.py`
> (`is_test`) reduced, for addition, to a + b ≡ 2 (mod 5): it held out whole result values (2, 7, 12, ...), never
> pairs, so no model ever saw 17 as the correct answer to any sum. The test sets here are held-out pairs, so every
> addition problem in them had a true sum the model had only ever seen as a *wrong* machine answer, if at all. The
> numbers unaffected by this are the ones where the GPT copies the machine: the trusting model, copy accuracy and
> the perfect-computer columns. The ones to read with care are the "checks numbers" models' detection and
> false-alarm rates, which judge an answer from the model's own weights: on addition problems the right answer was
> a token that had never been right in training, which can make it look wrong (more false alarms, and detection
> partly from "that token is never a sum" rather than from arithmetic). No own-weights accuracy is reported in this
> run. The rule is now a symmetric hash of the pair, about 20 % of pairs per operation with every result value in
> training. The saved models cannot be re-scored fairly on it: they were trained on every pair whose sum is not
> 2 mod 5, which includes most of the new held-out pairs; a fair number needs a retrain
> (`../results_storyroom/ROOM_REPORT.md`, phases 6-7).

Produced by `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 run_storycheck.py` (four training runs, the
evaluation of five models, and 2,000 word problems answered by the real analog neural CPU under noise; 68
minutes on a laptop CPU; `--reuse` trains only runs without a saved model, `--analyse` reuses all of them,
`--quick` is a 200-step smoke test). Numbers are in `summary.json`, the figures in `figures/`, the models in
`models/`, the log in `logs/train_log.txt`, the analog CPU's calibration in `analog_calibration.json`. The
baseline is the seed-2 model of the sentence-shapes run (`../results_storytool_shapes/`), never retrained.

## Answer first, in plain words

So far the GPT has trusted the computer completely: whatever number came back inside the `<calc>` chip, it
copied into the story. That is fine while the computer is perfect. The real analog chip is not: with a little
noise on its wires it gets about one answer in five wrong, mostly by timing out (answer 0) or by returning
garbage with several bits flipped.

This run asks whether the GPT can learn to **check** the computer: to notice a wrong answer and ask again.
Two ways were tried.

1. **Checks numbers.** During training the computer is flaky on purpose (25% of answers wrong: a bit flipped,
   another operation's answer, a random byte, or 0). Whenever it is wrong, the training text shows the call
   being made again, until the answer is right. The GPT has to spot a wrong answer from the numbers alone.
2. **Checks numbers + signal.** The same, but the computer also says `<unsure>` inside the chip when its own
   signal margins are poor (in training: on 80% of wrong answers and 10% of right ones). The text asks again
   exactly when the computer said `<unsure>`.

What came out:

- **Checking from the numbers alone does not pay.** The GPT does learn to ask again: it catches 55-60% of
  wrong answers, almost every timeout (0) and two thirds of the "other operation" answers. But it only catches
  one bit flip in four, it also asks again after 24-27% of *right* answers, and the habit of asking again
  damages everything else. With a perfect computer it drops from 99.6% to 89-90% on word problems, from
  98% to 51-61% on two-step problems, and from 90% to 51-70% on sentences that only restate a number. Against
  the flaky computer it ends at 75-77%, the same as the model that never checks (75%).
- **The computer's own doubt is what makes checking work.** With `<unsure>` in the chip, the GPT mostly learns
  the simple rule "when the computer is unsure, ask again": it retries 66% of wrong answers and 52% of right
  ones after an `<unsure>`, but only 21-29% of wrong and 2% of right ones when the computer sounded sure. So
  the signal does the heavy lifting and the GPT's own number sense adds a little on top. Against the flaky
  computer it gets **83-84% of word problems right (up from 75%)** and swallows a wrong answer in 13.5% of
  problems instead of 24.7%. With a perfect computer it keeps 97% on word problems and 82-83% on two-step.
- **On the real analog chip** (noise 0.045, 15-18% of answers wrong, the `<unsure>` flag set from the chip's
  worst signal margin): the signal model answers **87.8%** of word problems right against **81.6%** for the
  model that trusts the chip, and it copies a wrong answer into the story in 8.4% of problems instead of 18%.
  The numbers-only checker gets 81.8%, no better than trusting, while making half again as many calls.
- **When the computer is really bad** (50% wrong) every checker beats trusting (51%): numbers-only 55-58%,
  the signal model 65-66%.
- **Two-step problems get worse with every kind of checking**, even with the signal (47-51% against the
  flaky computer, versus 56% for the trusting model). Two reasons: the GPT confuses "ask the same question
  again" with "ask the next question" (after a wrong first answer it sometimes carries the wrong number into
  the second step, or does the second step twice), and the decoder allows three calls per problem, which two
  steps plus one retry already fill.
- The checkers write stories just as well (perplexity 15.9-16.1 against 15.66) and do not start calling the
  computer in plain text (0.3-0.5 top-guess calls per 1,000 words, the same as before).

The short version: a GPT can be taught to double-check a tool, but teaching it to spot wrong numbers by eye
costs more than it gains. Give it the tool's own confidence and the double-checking pays off, on a simulated
flaky computer and on the real noisy chip. What is still missing is a way to ask again without muddling
multi-step problems, and more training: the learning curves had not flattened at 7,000 steps.

## What was built

- `cpugpt/check.py`: `FlakyMachine`, a calculator that is wrong on purpose. `p_err` of its answers are wrong,
  in four kinds drawn equally: one bit flipped, another operation's table answer, a random byte, or 0 (a
  timeout). With `signal=True` it marks answers `<unsure>` with a hit rate (on wrong answers) and a false-alarm
  rate (on right ones). `blocks()` writes the training text for one problem: up to `max_tries` chips, asking
  again after each wrong answer (policy `oracle`) or after each `<unsure>` (policy `signal`), the last one
  forced right.
- `cpugpt/text.py`: the vocabulary takes extra tokens (`<unsure>`); the stream builder renders a problem's
  chips through the machine object when it has `blocks`.
- `cpugpt/tool.py`: the decoder records for every call whether the machine was wrong, what kind of error it
  made and whether it said `<unsure>`; it writes the `<unsure>` token into the chip when the machine asks.
  Asking again is nothing new at decode time: the GPT just writes another `<calc>` after `</calc>`, up to
  three calls per problem.
- `run_storycheck.py`: the four training runs (two seeds each), the test sets, the evaluation under a perfect
  computer, the flaky one (25% wrong), one twice as flaky (50%), the flaky one with its signal on, and with a
  perfect signal; the live check on the analog CPU; samples; figures.
- `tests/check_all.py`: a new section (15) on the flaky machine, its training blocks and the decoder's
  bookkeeping. ALL PASS, 104 checks.

Training: the same 1.4 M-weight GPT as the sentence-shapes run (context 64, batch 32, 7,000 steps, about 30
minutes for the four runs side by side), on the same stories plus problems from the compositional generator.
The two streams have 6.5 M tokens, 124,000 chips for 84,000 problems; the signal stream has 55,000
`<unsure>` tokens. Machine tokens (the answer, `<unsure>`, `</calc>`) are input only, out of the loss.

Measures. **Detection** is the share of wrong answers the GPT asked again about, counting only answers
after which it still had a call left. **False alarms** is the same share for right answers. **Copied a wrong
answer** is the share of problems whose final number is a wrong machine answer. The test sets use number
pairs never seen in training.

## Table 1. Each model with a perfect computer and with the flaky one (25% of answers wrong)

Word problems (n = 3,000), two-step problems (1,500), sentences that only restate a number (1,000). "No-call
calls" is the share of restating sentences on which the GPT called the computer anyway. For the signal models
the flaky columns are the flaky computer with its `<unsure>` signal on.

| model | word, perfect | 2-step, perfect | restate right | no-call calls | word, flaky | 2-step, flaky | new shapes, flaky | copied a wrong answer | detection | false alarms | calls / problem | perplexity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trusts the computer (shapes run, seed 2) | 99.6 | 98.4 | 90.3 | 0.3 | 75.0 | 56.3 | 57.9 | 24.7 | 0 | 0 | 1.00 | 15.66 |
| checks numbers, seed 0 | 89.0 | 61.1 | 70.0 | 2.9 | 74.8 | 39.3 | 54.3 | 16.6 | 54.6 | 27.5 | 1.52 | 15.92 |
| checks numbers, seed 1 | 90.2 | 51.3 | 51.1 | 5.3 | 76.8 | 32.6 | 57.6 | 14.7 | 59.8 | 24.3 | 1.51 | 15.89 |
| checks numbers + signal, seed 0 | 97.1 | 82.1 | 73.3 | 2.8 | 83.2 | 47.3 | 64.5 | 13.5 | 63.1 | 15.1 | 1.36 | 15.89 |
| checks numbers + signal, seed 1 | 97.0 | 83.3 | 83.4 | 1.4 | 83.9 | 51.0 | 66.9 | 13.5 | 63.2 | 16.2 | 1.38 | 16.06 |

The 95% interval on a word-problem score is about ±1.4 points; on two-step ±2.5.

Where the points go, word problems with the flaky computer (seed 1 of each): the trusting model gets 99.6%
of problems with no wrong answer and 0% of those with one. The numbers-only checker gets 91.5% and 44.5%.
The signal model gets 97.6% and 50.0%. Checking only helps when the answers it saves outnumber the clean
problems it spoils, and the numbers-only checker spoils too many.

## Table 2. Which wrong answers get caught

Share of wrong answers the GPT asked again about, by what the computer did wrong (word problems, flaky
computer, best seed of each; `figures/2_what_is_caught.png`).

| model | another operation | random byte | 0 (timeout) | one bit flipped (any) | bit 0 (±1) | bit 1 (±2) | bit 2 (±4) | bit 3 (±8) | bit 4 (±16) | bit 5 (±32) | bit 6 (±64) | bit 7 (±128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| checks numbers, seed 1 | 63 | 49 | 100 | 29 | 21 | 22 | 27 | 15 | 35 | 21 | 38 | 50 |
| checks numbers + signal, seed 1 (signal on) | 67 | 53 | 97 | 35 | 27 | 52 | 33 | 23 | 35 | 41 | 35 | 40 |
| trusts the computer | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The GPT's own number sense is coarse: a 0 where a sum was expected is always caught, a wildly wrong number
(random byte, the product where a sum was asked) about two times in three, a number off by a few units about
one time in four. Seed 0 of each model gives the same picture (58 / 44 / 100 / 21 and 66 / 51 / 98 / 37).

## Table 3. The computer's own doubt: the signal models with the signal off, on, and perfect

Word problems, the same flaky computer (25% wrong). "After `<unsure>`" and "when sure" split the retries by
what the computer said. The perfect signal marks every wrong answer and no right one.

| model | signal | word right | 2-step right | detection | false alarms | detection after `<unsure>` | detection when sure | false alarms after `<unsure>` | false alarms when sure | calls / problem |
|---|---|---|---|---|---|---|---|---|---|---|
| checks numbers + signal, seed 0 | off | 78.5 | 50.5 | 28.1 | 2.2 | - | 28.1 | - | 2.2 | 1.11 |
| | on (80% hit, 10% false alarm) | 83.2 | 47.3 | 63.1 | 15.1 | 65.7 | 29.0 | 47.0 | 2.4 | 1.36 |
| | perfect | 86.4 | 56.7 | 66.1 | 2.4 | 66.1 | - | - | 2.4 | 1.23 |
| checks numbers + signal, seed 1 | off | 78.1 | 52.3 | 27.4 | 2.0 | - | 27.4 | - | 2.0 | 1.11 |
| | on | 83.9 | 51.0 | 63.2 | 16.2 | 66.3 | 20.9 | 52.1 | 2.0 | 1.38 |
| | perfect | 86.9 | 60.9 | 68.1 | 1.9 | 68.1 | - | - | 1.9 | 1.24 |

Two things to see. The GPT has learned a rule close to "ask again after `<unsure>`", but applies it only about
two thirds of the time (the training text asked again after every `<unsure>`), so even a perfect signal gives
66-68% detection, not 100%: this is the ceiling to push with more training. And with the signal off the
same model is a gentle numbers-only checker (27-28% detection, 2% false alarms) that still beats trusting by
3 points, because it caught the timeouts cheaply.

## Table 4. Twice the errors: a computer wrong half the time

Word problems.

| model | computer | answer right | detection | false alarms | calls / problem | copied a wrong answer |
|---|---|---|---|---|---|---|
| trusts the computer | flaky ×2 | 51.1 | 0 | 0 | 1.00 | 48.7 |
| checks numbers, seed 0 | flaky ×2 | 55.4 | 54.0 | 26.4 | 1.63 | 36.9 |
| checks numbers, seed 1 | flaky ×2 | 57.5 | 58.6 | 24.1 | 1.65 | 34.9 |
| checks numbers + signal, seed 0 | flaky ×2, signal off | 55.8 | 25.3 | 1.7 | 1.17 | 42.3 |
| checks numbers + signal, seed 0 | flaky ×2, signal on | 64.9 | 62.3 | 16.0 | 1.55 | 32.4 |
| checks numbers + signal, seed 1 | flaky ×2, signal off | 54.8 | 23.5 | 2.3 | 1.16 | 43.4 |
| checks numbers + signal, seed 1 | flaky ×2, signal on | 65.5 | 62.5 | 14.9 | 1.55 | 31.8 |

Three calls per problem is the limit here: with half the answers wrong, one problem in eight needs more than
two retries.

## Table 5. The real computer: the analog neural CPU under noise

The CPU is the gate-level machine of the earlier reports run in its analog mode (noise standard deviation
0.045 on every signal, reset and reloaded before each call). Calibration on 300 calls: **19.7% of answers
wrong** (adding 6%, subtracting 7%, multiplying 41%); kinds: several bits wrong 31, one bit 12, timeout
(answer 0) 16. The `<unsure>` flag is set when the worst signal deviation from 0.5 in the run crosses a
threshold chosen on those 300 calls (0.437): it fires on 93% of wrong answers and 29% of right ones, a noisier
signal than the 80% / 10% the model trained on. 500 word problems per row; each row draws its own noise, so
the share of wrong answers differs a little between rows.

| model | signal | answer right (95% interval) | answers wrong | problems with a wrong answer | copied a wrong answer | right when no wrong answer | detection | false alarms | calls / problem |
|---|---|---|---|---|---|---|---|---|---|
| trusts the computer | - | 81.6 (78.0-84.8) | 18.1% (90 / 498) | 18.0 | 18.0 | 99.5 | 0 | 0 | 1.00 |
| checks numbers, seed 1 | - | 81.8 (78.2-84.9) | 16.0% (119 / 742) | 20.4 | 10.2 | 91.2 | 60.6 | 26.4 | 1.48 |
| checks numbers + signal, seed 1 | off | 85.4 (82.0-88.2) | 14.2% (76 / 534) | 14.0 | 12.2 | 97.2 | 19.7 | 3.5 | 1.07 |
| checks numbers + signal, seed 1 | on | **87.8** (84.6-90.4) | 15.4% (97 / 631) | 16.0 | **8.4** | 96.9 | 58.3 | 15.3 | 1.26 |

With the signal on, after an `<unsure>` the GPT asked again about 64% of wrong and 37% of right answers;
when the chip sounded sure, 11% and 4%. The chip's signal being noisier than in training shows up as more
false alarms than on the simulated machine (15% after 37% of right-but-unsure answers).

A fair comparison between rows is the pair "right when no wrong answer" and "copied a wrong answer": the
trusting model keeps every clean problem and loses every spoiled one; the numbers-only checker saves half
the spoiled ones but loses 8 points of clean ones; the signal model saves half the spoiled ones and loses
under 3 points of clean ones.

## Learning curves

`figures/1_learning.png`: word problems with the flaky computer, detection, false alarms, and word problems
with a perfect computer, every 500 steps (300 problems each).

- Detection appears early (about 50% by step 1,000, when the models still get under 10% of problems right)
  and then wanders between 45 and 65% for the rest of training.
- False alarms start at about 50% and fall to 25-30% (numbers only) and 15-20% (signal) by step 4,000.
- Accuracy under the flaky computer climbs until step 4,000 and is still rising slowly at 7,000 (the signal
  models' seed 1 went from 80.7% at step 6,000 to 84.7% at 7,000), so the runs are not converged.
- With a perfect computer the signal models reach 95-98% by step 4,000; the numbers-only models reach
  only 89-95%, their false alarms costing them clean problems throughout.

## Samples

The same openings for the three models, the flaky computer answering with half its answers wrong (seeded,
so the models meet the same answers). Machine text in the chips; `WRONG` marks a wrong machine answer.

Signal model:

- `sam picked 30 grapes. then mia gave him 12 more. sam had <calc> 30 + 12 = 0 <unsure> </calc> <calc> 30 + 12 = 42 <unsure> </calc> 42 grapes.` (0 WRONG and unsure, asked again, 42 right; the second `<unsure>` was a false alarm it rightly ignored)
- `20 stickers. she got 7 more and then 4 more. now she has <calc> 20 + 7 = 27 <unsure> </calc> <calc> 20 + 7 = 140 <unsure> </calc> <calc> 20 + 4 = 0 <unsure> </calc> 0 stickers.` (a false alarm on 27 spent a call, then 140 wrong, then the budget ran out on a wrong 0 in the second step)
- `there were 6 bags with 8 toys each, and then got 6 more. that made <calc> 6 * 8 = 48 </calc> <calc> 48 * 8 = 128 </calc> <calc> 128 + 6 = 249 <unsure> </calc> 249 toys.` (the muddle: it re-asked the first step with the wrong operands, then had no call left)

Numbers-only model:

- `sam picked 30 grapes. then mia gave him 12 more. sam had <calc> 30 + 12 = 0 </calc> <calc> 30 + 12 = 18 </calc> <calc> 30 + 12 = 0 </calc> 0 grapes.` (caught 0, caught 18, three wrong answers in a row and no calls left)
- `there were 6 bags with 8 toys each, and then she got 14 more. she had <calc> 6 * 8 = 48 </calc> <calc> 48 + 14 = 160 </calc> <calc> 160 + 14 = 174 </calc> 174 toys all together.` (after the wrong 160 it carried 160 into a "next step" instead of asking again)
- `tom had 12 apples. he got 5 more. now he has <calc> 12 + 5 = 17 </calc> <calc> 12 + 5 = 17 </calc> 17 apples.` (a false alarm, harmless here)

Trusting model:

- `sam picked 30 grapes. then mia gave him 12 more. sam had <calc> 30 + 12 = 0 </calc> 0 grapes.`
- `20 stickers. she got 7 more and then 4 more. now she has <calc> 20 + 7 = 27 </calc> <calc> 27 + 4 = 5 </calc> 5 stickers.`

Plain story openings ("they hugged each other and ate more cake.", "once upon a time") produce no calls
from any model.

## Limitations and what to try next

- **Three calls per problem at decode time**, while the training text allowed three tries per step (up to six
  chips for a two-step problem). Two-step results for the checkers are partly a budget effect; a run with
  `MAX_CALLS = 6` (or a budget per step) would separate that from the retry / next-step confusion.
- **Retry and next step look the same** (`</calc>` followed by `<calc>`). A distinct token for "ask again"
  (for example `<again>`), or writing the retry as the same operands immediately, would give the GPT a cleaner
  cue; the samples show it mixing the two.
- **The learned rule is incomplete.** Even a perfect `<unsure>` is acted on only two thirds of the time, and
  the curves were still rising at 7,000 steps. Longer training, or a larger share of flaky problems in the
  stream, is the cheap next step.
- **The signal the GPT trained on (80% / 10%) is cleaner than the chip's (93% / 29%).** Training with the
  chip's own calibration statistics, or on the chip itself, would match the false-alarm rate the GPT meets.
- **Checking from the numbers costs more than it gains** at this model size. A different route is to make the
  computer show more of its work (echo the operands, or return a checksum) so the check is a comparison
  rather than a judgement.
- One noise level (0.045) on the analog chip, chosen so that about a fifth of answers are wrong; 500 live
  problems per row; two seeds per condition; the trusting baseline is a single model from the earlier run.
- The browser page in `storytool_ui/` was not updated for this experiment.
