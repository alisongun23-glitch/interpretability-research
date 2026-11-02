# A story-writing GPT that sends math and word problems to the neural-gate computer

Produced by `python run_storygpt.py` (seven training runs, about 1.5 hours on a laptop CPU; `--reuse`
trains only runs without a saved model, `--analyse` reuses all of them). Numbers are in `summary.json`,
the figure in `figures/`, models in `models/`, training logs in `logs/`. The first version (template
math only) is archived in `v1_templates_only/`, with its report, models and numbers.

## Answer first

The story GPT now also sends plain-English word problems to the computer: "tom had 12 apples . he got
5 more . now he has" → it points at 12, 5 and *more*, the gate CPU computes 12 + 5, and it writes 17.
On word problems with number pairs it never saw in training, in the trained sentence shapes, the two
CPU-head seeds score **99.5% / 89.7% on addition, 90.5% / 99.6% on subtraction and 96.7% / 97.5% on
multiplication**. With the copy read-back described below, both seeds score 99–100% on + and −, and
95% / 99% on ×. The plain GPT scores 0%, 2% and 12%, and the first version's models score 0% (they never
call on these sentences). The pointers land on both numbers and on the operation word (*more*, *ate*,
*lost*, *gave*, *each*) in 97–100% of problems for most models and operations. In 207,000 words of held-out
story text, which contain 629 of those operation words, no CPU-head model ever calls the computer.

Story perplexity went up from 13.7–13.9 to 14.4–14.5. Most of that is because the word problems take
10% of the training stream: a plain GPT given as many story words as the first version is back at 13.9.

The addition misreads on large sums are fixed. They were not a bit-pattern problem but a data problem:
the held-out split makes certain answers (227, 237, 247) almost never appear as a training answer. Adding
the result token's embedding to the read-back fixed one seed and half-fixed the other. A copy path, where
a call adds a trained bonus to the result token's score, brought both seeds to 0–1 misreads (from about
110). The copy path also learns about twice as fast.

Word problems in sentence shapes it was never trained on mostly fail (0–38%). The model rarely decides to
call there, even when its pointers land on the right words.

## What changed from the first version

**Word problems in the training stream** (`cpugpt/text.py`). Twelve story-style sentence shapes, four per
operation, with random names, things, food and containers, for example:

    anna had 217 balls . she got 24 more . now she has 241 balls .
    joe ate 75 of his 195 cupcakes . he had 120 cupcakes left .
    amy had 96 shells and gave 30 to jill . she had 66 shells left .
    there were 14 cups with 10 marbles each , 140 marbles in all .
    tom was 5 years old . he had 216 stickers and got 7 more . now he has 223 stickers .

Two shapes put the numbers in reverse order ("ate B of his A") or add a distracting number ("was 5 years
old", "on day 4"). They are dropped in after random full stops, about 1.5 per story, on top of the
unchanged math templates (about 2 per story). Operand pairs use the same split as everything else
(`cpugpt.data.is_test`), so no held-out pair appears in training: 5.16 M training tokens with 41,417
template problems and 31,433 word problems. The vocabulary is unchanged (4,279 tokens), so perplexities
compare directly.

**Operation words in the frozen table.** The bus table that turns the OP pointer's token into an opcode
now also maps *more* → +, *ate*, *lost*, *gave* → −, *each* → × (`cpugpt.text.OPCODE_OF_WORD`). These are
common story words ("more" 4,117 times, "each" 3,137). The table only says what would be sent if the
pointer landed there; whether to call is still the gate's decision.

**Read-back** (`cpugpt/lm.py`). Version 1 added a trained projection of the result's 8 bits to the
residual. Now there are three variants: *bits* (as before); *+ embedding*, which also adds the result's
own token embedding times a trained gain (the default CPU head here); and *+ copy*, which adds the
embedding and also adds a trained bonus to the result token's logit when the head calls.

**Runs** (same sizes as before: 4 layers, d_model 128, 4 heads, MLP 512, 64-token context, 7,000 steps of
32 windows): plain GPT; CPU head (+ embedding) seeds 0 and 1; CPU head with the bits read-back, seed 0
(isolates the read-back change); CPU head with the copy read-back, seeds 0 and 1; and a plain GPT trained
for 7,762 steps so it reads as many story words as the first version did.

## Results

Held-out operand pairs, each problem after 0–40 words of held-out story text: 3,000 template problems
(the same 3,000 as in version 1) and 3,000 word problems (967 +, 1,010 −, 1,023 ×). Perplexity on the same
256 held-out story windows as version 1.

| model | sums + | − | × | word problems + | − | × | story perplexity | calls in story text |
|---|---|---|---|---|---|---|---|---|
| plain story GPT | 0.0% | 1.8% | 10.8% | 0.0% | 2.0% | 11.5% | 14.29 | – |
| + CPU head, seed 0 | 98.6% | 97.8% | 98.9% | **99.5%** (98.8–99.8) | **90.5%** (88.5–92.2) | **96.7%** (95.4–97.6) | 14.47 | 0 |
| + CPU head, seed 1 | 90.5% | 95.4% | 98.3% | **89.7%** (87.6–91.4) | **99.6%** (99.0–99.8) | **97.5%** (96.3–98.3) | 14.52 | 0 |
| + CPU head, copy read-back, seed 0 | 97.4% | 96.4% | 96.7% | 100.0% (99.6–100) | 99.0% (98.2–99.5) | 95.1% (93.6–96.3) | 14.41 | 0 |
| + CPU head, copy read-back, seed 1 | 97.7% | 94.6% | 97.8% | 99.4% (98.7–99.7) | 99.2% (98.4–99.6) | 99.0% (98.2–99.5) | 14.51 | 0 |
| + CPU head, bits read-back, seed 0 | 87.1% | 95.7% | 95.4% | 88.3% | 99.1% | 96.7% | 14.45 | 0 |
| version 1 plain GPT | 0.0% | 2.1% | 17.6% | 0.0% | 0.0% | 0.0% | 13.81 | – |
| version 1 + CPU head, seeds 0 / 1 | 85.7 / 85.1% | 96.8 / 95.6% | 94.7 / 97.7% | 0 / 0% | 0 / 0% | 0 / 0% | 13.88 / 13.68 | 0 |

(95% intervals in brackets. "Calls in story text" counts all 207,168 words of the 1,000 held-out stories.)

- **The answers come from the computer.** Switching the head off drops word problems to 0% / 1–2% / 7–9%.
  Swapping the operation the computer receives drops them to under 3%. Every CPU-head model calls on
  99.9–100% of word-problem answers.
- **Training pairs vs held-out pairs** are the same (CPU head: 92–100% on training pairs), so the model is
  not memorising answers. The plain GPT gets 4–5% on training pairs for + and −: it has not learned to add
  in its head either.
- **The first version cannot do word problems**: it never calls there (0% of answers), and its pointers
  land on the operands 0–9% of the time.

### Where the pointers land

At the answer position of held-out word problems: the share of problems where slot A is on the first
operand, slot B on the second, and slot OP on the operation word, all three at once.

| model | + (*more*) | − (*ate*, *lost*, *gave*) | × (*each*) |
|---|---|---|---|
| CPU head, seed 0 | 100.0% | 90.3% | 96.7% |
| CPU head, seed 1 | 99.7% | 99.6% | 97.3% |
| copy read-back, seed 0 / 1 | 100.0 / 99.4% | 99.0 / 99.4% | 94.6 / 99.0% |
| bits read-back, seed 0 | 99.9% | 99.1% | 96.5% |
| version 1, seed 0 / 1 | 0.0 / 0.0% | 0.0 / 0.0% | 0.1 / 0.4% |

The numbers are never swapped, including in the shape that gives them in reverse order ("ate 75 of his
195"): B lands on 75 and A on 195. The two misses that remain:
- "N ate B of H A F" in seed 0 (61.6% right): the OP pointer lands on another word (*apples*, *who*), so
  the computer gets no operation. Other seeds get this shape 96–100% right.
- "A friends each brought B" (81–96% right in every model): the problem starts with its first number, and
  A then often lands on B, so the computer computes B × B.

The pointers learn word problems faster than the templates: 96–98% of word problems point right after 500
steps, against 79–90% of template problems. The read-back is what takes the remaining steps.

### Calls in plain story text

No CPU-head model calls anywhere in the 207,168 words of held-out stories (0 calls for all five), even
though the text has 629 operation words ("they hugged each other", "she wanted more"). With the head
switched off, perplexity is identical to the last digit. On the page, "they hugged each other and ate more
cake ." produces no call.

### Story perplexity

| | version 1 (math templates only) | now (templates + word problems) |
|---|---|---|
| plain GPT, 7,000 steps | 13.81 | 14.29 |
| plain GPT, as many story words as version 1 (7,762 steps) | – | **13.89** |
| CPU-head GPTs | 13.68–13.88 | 14.41–14.52 |
| same, on all 207 k held-out words: plain / CPU head | 14.23 / 14.17–14.33 | 14.82 (13.89-run: 14.39) / 14.94–14.98 |

The word problems make the stream 11% longer (5.16 M vs 4.65 M tokens), so 7,000 steps read 10% fewer
story words. Training the plain GPT long enough to read as many story words as version 1 brings it back to
13.89, within 0.1 of version 1's 13.81. On all held-out text the gap is 0.16, so word problems may cost a
little beyond the budget. The CPU-head models are 0.12–0.23 above the plain GPT on the same data (0.12–0.16 on all
held-out text), in all five runs; in version 1 the gap was −0.13 to +0.07. There is one plain seed, so this may be seed noise. It does
not come from calls, which never happen in story text.

### The addition misreads on 192–255: cause and fix

In version 1, about 10% of right computer answers to + were written wrong, all of them sums 192–255.
**Cause:** `cpugpt.data.is_test` hashes `a*7919 + b*104729 + op*31337 (mod 5)`. Since 7919 ≡ 104729 ≡ 4
(mod 5), the hash depends only on a + b. So the held-out + pairs are exactly the sums ≡ 2 (mod 5): no
held-out sum is ever the answer of an addition in training. Most of those numbers still appear as answers
of − and × problems, but **227, 247 and 237 are a training answer only 11, 13 and 29 times** in 5.2 M
tokens (other numbers 192–255: median 371). Those three account for almost every misread, in version 1
and now (e.g. 227 written as 226, 163 or 243). The model rarely had to write those tokens, so it could not.

| read-back | model | sums: misread / answers 192–255 | right on 192–255 | word problems: misread / answers 192–255 | right on 192–255 |
|---|---|---|---|---|---|
| bits (version 1) | v1 seed 0 / 1 | 104 / 107 of 603 | 79.1 / 77.9% | – | – |
| bits | seed 0 | 112 of 603 | 79.4% | 112 of 554 | 79.8% |
| + token embedding | seed 0 | 2 | 98.7% | 5 | 99.1% |
| + token embedding | seed 1 | 66 | 86.2% | 97 | 82.1% |
| + embedding + copy | seed 0 | **0** | 97.3% | **0** | 100.0% |
| + embedding + copy | seed 1 | **1** | 97.2% | **0** | 99.3% |

(A misread = the head called, the computer was right, the written answer was wrong. The copy runs' small
remaining errors on 192–255 are pointer errors.)

- Adding the result's token embedding (trained gain about 3.4) fixed seed 0 but only partly seed 1.
- The copy path (gain about 3.0 on the result token's logit) removes the misreads in both seeds. It does not
  depend on how well the rare token was trained. It also makes the whole head learn faster: held-out
  template math averages 71–72% at step 1,000, against 37–42% without it (figure, left and middle; green
  curves).
- "More steps" was not tried. Version 1's addition accuracy had levelled off by step 7,000 (85–86% from
  step 6,000 on), and more steps add few training answers for these three tokens, because the split
  withholds them.

The page (`storygpt_ui/`) uses the copy read-back, seed 1, the best on held-out math and word problems.

### Sentence shapes never trained

Three extra shapes, held-out pairs, never in training (1,500 problems):

    molly picked 145 candies . then sam gave her 44 more . molly had 189 candies .
    there were 230 cookies on the plate . jill ate 155 of them . then there were 75 cookies left .
    max has 20 boxes and each box has 8 flowers . so max has 160 flowers .

| model | + | − | × | calls at the answer (+ / − / ×) | all three pointers right (+ / − / ×) |
|---|---|---|---|---|---|
| CPU head, seed 0 | 38.1% | 0.0% | 0.0% | 100 / 0 / 0.2% | 38 / 59 / 80% |
| CPU head, seed 1 | 12.8% | 0.8% | 0.0% | 13 / 0 / 0% | 99 / 44 / 5% |
| copy read-back, seed 0 | 11.6% | 0.0% | 1.2% | 30 / 0 / 1% | 26 / 4 / 38% |
| copy read-back, seed 1 | 4.1% | 2.1% | 0.4% | 20 / 0.2 / 0.6% | 12 / 98 / 35% |
| bits read-back, seed 0 | 9.8% | 0.0% | 0.0% | 88 / 0 / 0% | 12 / 99 / 32% |

This is the main limitation. **The decision to call does not transfer to new sentence shapes**: for − and ×
the gate almost never fires, even when all three pointers are right (97–99% for − in two models). The pointers
transfer partly and unevenly across seeds. In the + shape, *gave* (a minus word) sits next to *more*. The
OP pointer picks *more* in 44–100% of cases, so the model has partly learned which word names the
operation. With twelve training shapes, the gate has learned the shapes rather than "this is a word problem".

## Samples

Greedy continuations from the page's model (bold = written from the computer's result):

> tom had 12 apples. he got 5 more. now he has **17** apples. he was so happy and he couldn't wait to go to the park.
>
> mia had 40 cookies and ate 13. she had **27** left. she was so happy and thanked the man for his help.
>
> there were 6 bags with 8 toys each, **48** toys in all. the little girl was so happy and thanked the little girl for helping her.
>
> once upon a time, ben found 23 shells and then 9 more. he had **32** shells in all.

The plain GPT writes 75 apples, 20 left, 36 toys and 36 shells for the same openings. In a shape it never
saw ("sam picked 30 grapes. then mia gave him 12 more. sam had"), the CPU-head model does not call. It
continues with a problem in a trained shape and solves that one: "sam had 21 cups. each cup had 6 balls.
that is **126** balls in all." 300 of the page model's calls (150 template, 150 word problem) were re-run
on the fully analog neural CPU: 0 differences from the table.

## Try it

`storygpt_ui/index.html` (with `model.json`, `cpu.json` and the two weight files) runs the copy-read-back
GPT and the plain GPT in the browser. The examples include word problems. The side panel shows where the
three pointers landed (A, B and OP on 12, 5 and *more*), and the computer runs on neural gates in the panel.

    python3 -m http.server 8765 --directory storygpt_ui      # then open http://localhost:8765

## Other changes

- `StoryGPT` now copies its frozen bus tables instead of sharing memory with the caller's arrays. Before,
  loading a checkpoint wrote that checkpoint's operator table into the vocabulary's array and into every
  other model built from it. It did not affect version 1's numbers, because all its checkpoints had the
  same table. Here it would have given version 1's table to the new models during analysis; it was caught
  and fixed before the final analysis. `tests/check_all.py` checks it now, along with the word problems
  (answers, operand positions, operation words, split) and the new read-backs.
- The page declares its character set and a phone viewport.

## Limitations

- New sentence shapes mostly fail (above). The next step is many more shapes, generated more freely, with
  some shapes held out to measure transfer.
- The held-out split for + is "sums ≡ 2 (mod 5)" rather than random pairs (see the misread section). This
  is a stricter test, since no held-out sum is ever an addition answer in training. The same split
  restricts which − and × results appear in training, through a + b.
- Single-step problems only: one call per answer, two operands, results 0–255.
- One plain seed; two seeds for each CPU-head variant (one for the bits read-back).
- Story quality is that of a 1.4 M-weight model trained for under an hour: simple and repetitive.
