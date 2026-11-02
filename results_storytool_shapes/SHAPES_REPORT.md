# The story GPT that calls the computer: sentence shapes it was never trained on

> **Note (1 Nov 2026): the held-out split was wrong for addition.** Until today the rule in `cpugpt/data.py`
> (`is_test`) reduced, for addition, to a + b ≡ 2 (mod 5): it held out whole result values (2, 7, 12, ...), never
> pairs, so no model ever saw 17 as the answer to any sum. The numbers here where a model answers from its own
> weights were measured on that split: the `no calls (same stream)` row of the results table (4.0 / 4.1 / 3.3 ... %)
> and the "computer off" figures (0.9-3.5 % on sums and word problems, 0.7-1.8 % on two-step problems). On the
> addition problems those tests asked for an answer token never seen as an answer (subtraction and multiplication
> pairs were split properly, so the mixed-operation sets are affected in their addition third). Everything the
> models score with the computer on is unaffected: they copy the machine's result, which works for any token. The
> rule is now a symmetric hash of the pair, about 20 % of pairs per operation with every result value in training.
> The saved models cannot be re-scored fairly on it: they were trained on every pair whose sum is not 2 mod 5,
> which includes most of the new held-out pairs. A fair own-weights number needs a retrain
> (`../results_storyroom/ROOM_REPORT.md`, phases 6-7).

Produced by `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python3 run_storytool.py` (four training runs and the
evaluation of seven models, 52 minutes on a laptop CPU; `--reuse` trains only runs without a saved model,
`--analyse` reuses all of them, `--quick` is a 200-step smoke test). Numbers are in `summary.json`, the figures
in `figures/`, the models in `models/`, the log in `logs/`. The first run (12 fixed sentence shapes) is left as it
was in `../results_storytool/` (its report: `../results_storytool/STORYTOOL_REPORT.md`); its three models are
re-evaluated here on the new test sets.

## Answer first, in plain words

The first version of this GPT learned to pull out the calculator only for the twelve sentence shapes it had
seen in training. On three shapes it had never seen it scored 1-33%: "gave" meant minus even when "more"
followed, "then there were" made it start a two-step problem, and "so mia has" was not a cue at all.

This run trains the same GPT on problems written by a small grammar instead of a list: openings, many ways to
state the first number, 23 verbs for the changes ("got / found / won / was given 5 more", "ate / lost / broke /
sold 5", "gave 5 to mia", "mia gave him 5 more", "5 balloons flew away"), 22 ways to ask for the total, names,
pronouns, "mom", "the kids", distractor numbers, and no-call sentences that use the same verbs without asking
anything, or without numbers at all. 43 frames and 10 verbs are kept out of training entirely to test on.

On number pairs never seen in training the GPT now gets **98-99.5% of word problems in trained shapes,
70-78% in held-out shapes, 66-84% with held-out verbs, 96.5-98.7% on the first run's three never-trained
wordings** (up from 1-33%), and 98-99% of two-step problems (hand-off of the first answer 99.6-99.9%). The
first run's models score 29-35% on the same trained-shape set, 17-20% on the held-out shapes, 15-16% on the
held-out verbs and 8-11% on the two-step set, so the comparison is on the same sentences. Whenever the
computer answered right, the GPT copied the answer right (100%, all three seeds, every set); the misses on new
shapes are calls not made (12-14% of held-out-shape problems) and calls with the wrong numbers or operation
(10%). Two-step problems with a held-out piece are harder: 50-62%.

When not to call: the GPT calls on 0.5-2% of no-call sentences with trained pieces and 1.6-11% with held-out
pieces (seed dependent), restating the right number 89-91% / 61-69% of the time. In 207,000 words of held-out
stories `<calc>` is the top guess at 70-98 positions (58-73 in the first run), mostly after "had"; in 8,000 words
of free writing it called 5-7 times, every time after a problem it had written itself ("what is 9 times 11 ?",
"there were 13 piles . leo put 14 pebbles in each pile . he had"). The prompt that made the first run invent a
sum with made-up numbers ("they hugged each other and ate more cake . they had `<calc> 0 * 10 =`") now continues
"they were happy". A decode-time margin on `<calc>` (the call must beat the best word by that much) cuts the
top-guess calls in story text from 75 to 24 (margin 1) or 5 (margin 2) at a cost of 1 and 3 points on trained
shapes and 3 and 7 points on held-out shapes; it does not change the free-writing calls, because those come
after self-written problems where `<calc>` wins by a wide margin. The page has a slider for it; the default is 0.

Story perplexity is 15.66-15.76 against 15.33 for the same GPT trained on the same stream without call blocks
(a gap of 0.3-0.4; the first run's gap was 0.6-0.7: 15.28-15.39 against 14.70). Both the baseline and the calc
models are about 0.6 worse than their counterparts in the first run: the richer problem text (10,749 distinct
word-problem shapes instead of 12, longer sentences) takes more of a 1.4 M-weight model than twelve fixed
sentences did, at the same number of training steps.

## What changed: sentences from a generator instead of fixed lists

The first run trained on 12 fixed word-problem shapes, 6 two-step shapes and 5 no-call shapes. This run
replaces the lists with a compositional generator (`cpugpt/text.py`: `gen_word`, `gen_two`, `gen_nocall`;
`build_tool_stream(varied=True)`, the default), keeping the interface of the first version (tokens, machine
flags, answer positions, info) and the operand split `cpugpt.data.is_test`. The old lists stay in the file
(`build_tool_stream(varied=False)`, the pointer-head version and the test sets below still use them).

A word problem is assembled from frames, each a sentence pattern with placeholders:

- an **opening** (35% of problems; 10 trained): "one sunny day ,", "N woke up early .", "N was D years old ."
  (D a distractor number), "N had E Q ." (a distractor count of another kind of thing);
- a **start** that gives the first number (15 trained): "N had A T .", "there were A T on the table .",
  "N counted A T .", "N had A T and E Q .", "N put A T in the k ." ...;
- a **change** that gives the second number with a verb slot (8 + and 8 − frames trained, 23 trained verbs):
  "P V B more ." with got / found / picked / bought / won / made, "M V O B ." with gave / brought / handed
  (gave *to* the subject is plus), "P WAS given B more .", "P V B ." with ate / lost / broke / sold / spent,
  "P V B to M .", "P V M B .", "M V B ." with took / ate, "then B T flew / ran / rolled away .", "then B T popped .",
  "P threw B away .", "P shared B with M ."; a connector ("then", "later", "after that") 40% of the time; or a
  **combined** frame holding both numbers, some with the second number first ("N ate B of H A T .",
  "N had A T but lost B .");
- for ×, a **container frame** (9 trained): "there were A K with B T each .", "N had A K . each k had B T .",
  "A friends each brought B T .", "N saw A K . there were B T in each k .", "N filled A K with B T each ." ...;
- a **question** (22 trained): "now P has R T .", "so P had R T .", "P had R T .", "that is R T .", "that made R T .",
  "how many T did P have ? R .", "P ended up with R T .", "R T in all .", "P had R T left .", "there were R T left ." ...

Subjects are boys' and girls' names, mom, dad, grandma, grandpa, the teacher, or two children ("tom and mia",
"the kids", with they / their / them / have / were); the things counted come from classes the verb needs (food
for ate, breakable things for broke, balloons for popped, animals for ran away, coins for spent). Two-step
problems chain two change frames (the second with a connector and the third number), or a one-sentence double
("P got B more and then C more ."), or a container frame and a change; the operation pairs are those of the
first run (++, −+, +−, ×−, ×+, −−). No-call sentences come in three kinds: fixed frames restating a number
("N counted A T . that is R T .", "how old was P ? P was R ."), the same starts and changes as the problems with
an **ending instead of a question** ("he put the B new T in the box .", "she was sad about the B T .", "he had a
lot of fun ."), and the verbs with **no numbers at all** ("N ate some F . then P went out to play .", "N got more
T for H birthday . P was so happy ."). The last two kinds are the new negatives: a change verb, or "had" at the
end of a clause, is no longer a sufficient reason to call.

**Held out of training.** 43 frames (marked `*` in the lists) and 10 verbs (grabbed, collected, caught, sent,
passed, dropped, used, stole, carried, every) never appear in the training stream. The held-out frames cover
every slot: openings ("when N was D years old ,"), starts ("N started with A T .", "N had only A T .", "in the k
there were A T ."), changes ("M gave O B more .", the first run's failure; "P V B of them ."; "P V away B .";
"M came with B more ."; "P let M have B ."; "then B of the T were gone ."), combined frames ("N added B more to
H A T ."), container frames ("N put B T in each of H A K .", "each of the A K had B T .", "every k had B T . there
were A K .", "N has A K and each k has B T ."), questions ("so N has R T .", "then there were R T left .", "only R T
were left .", "that left R T .", "all together P had R T .", "P counted them . there were R T ."), doubles, no-call
frames and endings. The three never-trained wordings of the first run (`NEW_WORDINGS`) are kept as a test set;
their pieces ("M gave O B more", "ate B of them", "N has A K and each k has B T", "so N has") are all among the
held-out frames, so they remain untrained. The held-out verbs are chosen so that the operation is still
recoverable from the frame ("caught B more", "passed B to M", "every box had B") but the word itself was never
seen in a problem; it only occurs in ordinary story text.

**Test sets** (held-out operand pairs throughout, each problem after held-out story text, cut at the cue word,
so the GPT must write the call and the answer):

| set | what | n |
|---|---|---|
| sums | the math templates of the earlier versions ("what is 14 times 12 ?") | 2,000 |
| word, trained shapes | the generator in training mode: trained frames and verbs, new number pairs | 3,000 |
| held-out shapes | exactly one held-out frame per problem, trained verbs | 2,000 |
| held-out verbs | trained frames, exactly one held-out verb | 2,000 |
| 3 old new wordings | the first run's `NEW_WORDINGS` | 1,500 |
| 12 old fixed shapes | the first run's `WORD_PROBLEMS` (its own training shapes) | 1,500 |
| two-step | the generator's two-step problems, trained pieces | 1,500 |
| two-step, held-out | one held-out frame or verb | 1,500 |
| no-call | fixed frames and setups with an ending that restates a number | 1,000 |
| no-call, held-out | the same with one held-out frame or verb | 1,000 |

**Training stream** (`summary.json`, `data`): 6.34 M tokens (5.67 M without the call blocks, 11.7% longer),
189,236 (3.0%) written by the computer; per story 2.0 math templates, 1.49 word problems, 0.50 two-step
problems and 0.51 no-call sentences, the same rates as the first run. The 31,351 word problems come in 10,749
distinct frame combinations (the first run: 12), the 10,595 two-step problems in 7,591 (6), the 10,623 no-call
sentences in 1,285 (5): 4,240 setups with an ending, 3,192 fixed frames, 3,191 numberless. Verbs in the
problems: each 6,110, gave 4,266, got 3,546, handed 3,276, ate 2,358, found 2,158, sold 2,106, brought 1,844,
broke 1,359, bought 1,274, lost 1,249, threw 1,127, took 1,019, given 935, had 560, popped 498, made 338, picked
329, rolled 324, won 318, ran 307, flew 297, spent 177.

**Models.** The same sizes and recipe as the first run (4 layers, d_model 128, 4 heads, MLP 512, 64-token
context, 7,000 steps of 32 windows, AdamW, the computer's tokens in the input and out of the loss): `story GPT +
calc` seeds 0-2 and `story GPT, no calls` on the same stream without call blocks. The three seeds of the first
run (`../results_storytool/models/`, 12 fixed shapes) are evaluated on the new sets as the comparison, without
retraining. The page model is the seed with the best mean over trained shapes, held-out shapes, held-out verbs
and two-step problems (seed 2).

**Decode-time margin** (`cpugpt/tool.py`, `decode(calc_margin=)`): the margin is taken off the `<calc>` logit
before the choice, so a call opens only when `<calc>` beats the best word by that much (a negative margin is a
bias toward calling, ToolkenGPT's `logits_bias`). The same rule is on the page as a slider. It is measured on
1,000 problems of each set, the no-call sets, the top guess at every position of the 207k held-out story words,
and 8,000 words of free writing.

## Results

| model | sums | word, trained shapes | held-out shapes | held-out verbs | 3 old new wordings | 12 old fixed shapes | two-step | two-step, held-out | no-call: calls / right | no-call held-out: calls / right | story ppl | `<calc>` top guess in 207k words | calls in 8,000 free words |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| + calc, seed 0 | 98.7% | 98.8% | **77.6%** | 75.0% | 98.3% | 88.5% | 98.2% | **62.2%** | 0.6% / 91.1% | 4.4% / 65.7% | 15.76 | 98 | 5 |
| + calc, seed 1 | 98.6% | 98.4% | 69.7% | 66.1% | 98.7% | 83.7% | 97.8% | 49.5% | 2.0% / 88.6% | 11.4% / 60.9% | 15.69 | 70 | 7 |
| + calc, seed 2 (page) | **99.7%** | **99.5%** | 76.4% | **84.0%** | 96.5% | **92.9%** | **99.0%** | 61.3% | 0.5% / 91.2% | 1.6% / 68.7% | 15.66 | 75 | 7 |
| no calls (same stream) | 4.0% | 4.1% | 3.3% | 1.7% | 3.3% | 3.0% | 1.2% | 0.9% | – / 3.9% | – / 2.1% | **15.33** | – | – |
| first run, seed 0 (12 fixed shapes) | 98.9% | 35.3% | 20.1% | 16.4% | 35.4% | **99.9%** | 7.8% | 2.3% | 1.4% / 43.5% | 1.3% / 10.1% | 15.34 | 58 | 3 |
| first run, seed 1 | 97.9% | 31.5% | 18.9% | 14.6% | 5.3% | 98.3% | 10.7% | 4.8% | 0.5% / 46.3% | 0.7% / 14.1% | 15.39 | 69 | 1 |
| first run, seed 2 | 96.2% | 29.2% | 17.2% | 16.3% | 1.5% | 98.4% | 7.5% | 3.4% | 0.4% / 44.0% | 0.2% / 8.8% | 15.28 | 73 | 0 |

(95% intervals: ±0.3-0.5 points on the 98-99% cells, ±2 points on the 2,000-problem held-out sets, ±2.5 on the
1,500-problem sets. Per operation, seed 2: trained shapes 99.3 / 99.8 / 99.4% for + / − / ×, held-out shapes
73.2 / 80.1 / 76.1%, held-out verbs 82.6 / 86.8 / 82.5%; seeds 0 and 1 are within 5 points except seed 1's
held-out verbs for − at 56.6%.)

### The answers come from the computer, as before

- Copy accuracy given a right machine answer is **100%** for all three seeds on every set, including the held-out
  ones (the first run's models: 99.7-100%). Every miss is a call not made or a call with the wrong numbers or operation.
- Computer off (the GPT fills in the display itself): 0.9-3.5% on sums and word problems, 0.7-1.8% on two-step
  problems. Computer doing the wrong operation: 1.0-1.5%. Training pairs score the same as held-out pairs
  (98.7-99.1% word, 98.0-99.4% two-step), so nothing is memorised.
- 300 of the page model's calls re-run on the fully analog neural CPU: 0 differences from the table.

### Trained shapes, held-out shapes, held-out verbs

| seed 2 | called | call right | answer right | what the misses are |
|---|---|---|---|---|
| word, trained shapes | 99.8% | 99.5% | 99.5% | 0.5% wrong operands (the weakest trained verb is "had" in "A friends each had B", 92.7%) |
| held-out shapes | 86.4% | 77.0% | 76.4% | 13.6% no call, 9.4% wrong call |
| held-out verbs | 94.8% | 83.7% | 84.0% | 5.2% no call, 11.1% wrong call |
| 3 old new wordings | 100% | 97.2% | 96.5% | "picked A . then M gave him B more": 87.8% (seeds 0 / 1: 97.7 / 95.7%); the other two 100% |
| 12 old fixed shapes | 95.0% | 92.9% | 92.9% | "there were A bags with B toys each **,** R in all": 50.5% (seeds 0 / 1: 4.5 / 9.0%), the only shape whose call comes after a comma; "A in the box , N put in B more": 90% (seed 1: 16%); the other ten 93-100% |

The learning curves (figure 1) say where the limit is: trained shapes pass 90% at step 2,000 and go on to 99%,
but held-out shapes reach 73-75% at step 2,000 and stay there to step 7,000 (73-78%). More training on the same
frames does not transfer further; more frames would.

### Which held-out pieces work and which do not (figure 2)

Answer right on the held-out-shape and held-out-verb sets, per piece (call rate in brackets for seed 2):

| held-out piece | seed 2 | seed 0 | seed 1 | first run, seed 0 |
|---|---|---|---|---|
| question "so N has R T ." (the first run's failure) | 98% (100) | 99% | 98% | 32% |
| question "then there were R T left ." (first run's failure) | 95% (97) | 72% | 100% | 21% |
| change "M gave O B more ." (first run's failure) | 91% (96) | 96% | 91% | 9% |
| start "N has A K and each k has B T ." | 99% (99) | 100% | 97% | 38% |
| start "when N was D years old ," | 99% (100) | 99% | 100% | 34% |
| opening "N had E Q and loved them ." | 98% (100) | 99% | 96% | 33% |
| start "N started with A T ." | 98% (100) | 98% | 100% | 9% |
| change "P V away B ." | 97% (100) | 100% | 97% | 34% |
| question "only R T were left ." | 97% (100) | 89% | 14% | 6% |
| change "P V B more and put them in the k ." | 97% (97) | 84% | 52% | 6% |
| question "R T all together ." | 96% (98) | 97% | 96% | 50% |
| question "P counted them . there were R T ." | 96% (99) | 99% | 93% | 14% |
| start "in the k there were A T ." | 96% (98) | 96% | 100% | 14% |
| start "N had only A T ." | 94% (100) | 98% | 98% | 23% |
| change "P V B of them ." | 93% (97) | 100% | 97% | 41% |
| change "P picked up B more ." | 88% (100) | 94% | 94% | 3% |
| change "M came with B more ." | 83% (97) | 90% | 47% | 20% |
| change "P let M have B ." | 80% (100) | 67% | 0% | 17% |
| combined "N had A T . B of them V away ." | 79% (88) | 66% | 75% | 1% |
| start "N kept A T in a k ." | 78% (100) | 62% | 82% | 4% |
| combined "N had A T and M gave O B more ." | 76% (94) | 98% | 94% | 12% |
| container "each of the A K had B T ." | 71% (96) | 96% | 87% | 37% |
| change "P added B more to the k ." | 70% (82) | 95% | 65% | 5% |
| question "all together P had R T ." | 63% (63) | 85% | 8% | 26% |
| question "that left R T ." | 58% (58) | 47% | 12% | 9% |
| change "then B of the T were gone ." | 57% (57) | 37% | 30% | 0% |
| start "N took A T to school ." | 51% (100) | 57% | 72% | 3% |
| container "N put B T in each of H A K ." (B before A) | 49% (100) | 55% | 45% | 7% |
| container "every k had B T . there were A K ." (B before A) | 17% (67) | 8% | 4% | 8% |
| combined "N added B more to H A T ." (B before A) | 10% (73) | 12% | 0% | 2% |
| question "P looked and saw R T ." | 0% (0) | 2% | 0% | 0% |
| verb collected ("P collected B more") | 99% (100) | 96% | 67% | 31% |
| verb sent ("M sent O B more", "P was sent B more") | 95% (98) | 93% | 79% | 8% |
| verb dropped ("P dropped B") | 92% (98) | 84% | 65% | 18% |
| verb stole ("M stole B") | 85% (100) | 77% | 62% | 13% |
| verb every ("every k had B T") | 83% (88) | 80% | 64% | 13% |
| verb carried ("A friends each carried B T") | 82% (92) | 94% | 92% | 37% |
| verb used ("P used B") | 81% (95) | 19% | 59% | 15% |
| verb caught ("P caught B more") | 79% (98) | 55% | 74% | 2% |
| verb passed ("P passed B to M", "M passed O B") | 77% (96) | 66% | 54% | 5% |
| verb grabbed ("P grabbed B more") | 66% (100) | 43% | 67% | 47% |

Three kinds of held-out piece fail, and they say what the GPT has and has not learned:

- **A question with no number cue at all** ("she looked and saw R", 0%; "that left R", 58%; "all together she
  had R", 63%) does not open a call: the GPT has learned that a total is asked by "has / had / is / were / made /
  in all", not that a story with two numbers and a change wants a total. "P counted them . there were R" (96%)
  works because "there were" is a trained cue.
- **The second number before the first** in a single sentence ("tom added 5 more to his 12 apples", 10%;
  "every box had 5 toys . there were 12 boxes", 17%; "she put 5 toys in each of her 12 bags", 49%) makes the
  GPT type the numbers in text order or pick the wrong pair; the trained reversed frames ("ate 5 of his 12
  apples", "lost 5 of them") did not generalise to new reversed frames. Everything else is order-preserving.
- **A verb whose story meaning points the other way**: "grabbed 5 more" (66%) and "took 12 apples to school"
  (51%) are read as taking away (grabbed / took are "take" verbs in the stories), "used 5" (81%, seed 0: 19%)
  and "passed 5 to mia" (77%) are weaker than "collected" (99%), "sent" (95%) and "dropped" (92%). The
  operation of a held-out verb is inferred from the frame plus whatever the stories taught about the word.

Seed variance is large on the weak pieces (e.g. "P let M have B": 80 / 67 / 0%, "only R T were left": 97 / 89 /
14%) and small on the strong ones, as the first run found for its three wordings.

### Two-step problems

| | call 1 right | hand-off (call 2 starts from the first result) | call 2 right | final answer right | per pair (++, −+, +−, ×−, ×+, −−) |
|---|---|---|---|---|---|
| trained pieces, seed 0 / 1 / 2 | 98.5 / 98.1 / 99.5% | **99.7 / 99.9 / 99.6%** | 98.1 / 97.8 / 99.0% | 98.2 / 97.8 / 99.0% | seed 2: 98.5 / 98.3 / 98.8 / 99.6 / 99.2 / 99.6% |
| one held-out piece, seed 0 / 1 / 2 | 74.9 / 68.1 / 75.5% | 78.0 / 68.5 / 76.9% | 61.4 / 48.1 / 60.5% | 62.2 / 49.5 / 61.3% | seed 2: 79.5 / 46.4 / 53.5 / 62.6 / 64.3 / 61.6% |
| first run, seed 0 | 13.2% | 16.6% | 6.9% | 7.8% | 3-13% |

With trained pieces the chain is as good as in the first run (99.3-99.8% there, on its six fixed shapes). With
a held-out piece the first call is wrong a quarter of the time and the second call then goes wrong as often
again (hand-off 77-78%: when the first call is wrong the GPT often does not continue from its result): a
held-out piece in a two-step problem costs more than in a one-step problem because both calls must survive it.
By piece (seed 2): the questions "so N has", "that left R", "only R were left" and the change "P V away B" are
100%, "sent" 81%, "every" 23%, "stole" 25%, "passed" 35%, "then B of the T were gone" 8%, "P looked and saw"
0% (never calls).

### When not to call

- No-call sentences with trained pieces: calls on 0.5 / 2.0 / 0.6% (seeds 2 / 1 / 0), the restated number
  right 91.2 / 88.6 / 91.1% (fixed frames 84.5-84.9%, setups with an ending 95.6-99.4%). The first run's models
  restate 44-46% (their five frames) and the no-calls GPT 3.9%: the copy of a number from the sentence is
  something the call mechanism teaches.
- No-call sentences with a held-out piece: calls on 1.6 / 11.4 / 4.4%, restated right 68.7 / 60.9 / 65.7%.
  Seed 1 calls on 25% of × setups with a held-out ending (73% of "she picked the R toys she liked best") and
  on 45% of "carried" setups; seed 2 never calls on more than 11% of any piece. The held-out fixed frame
  "N had A T and E Q . the R Q were H favorite" is restated right 0-4% (the GPT writes the count of the first
  thing) but almost never called (0-9%).
- Held-out story text, teacher-forced: `<calc>` is the top prediction at 98 / 70 / 75 of 207,168 positions
  (0.03-0.05%; the first run 58-73). For seed 2: 33 after "had", 11 after "is", 10 after "were", 9 after "?",
  5 after "made"; seed 0 has 29 more after "with" ("ended up with" is a trained question).
- Free writing from 200 held-out openings (8,000 words): 5-7 calls per model (first run 0-3), all after a
  problem the GPT wrote itself: "what is 9 times 11 ? `<calc> 9 * 11 =`", "there were 13 piles . leo put 14
  pebbles in each pile . he had `<calc> 13 * 14 =`". The first run's failure ("they hugged each other and ate
  more cake . they had `<calc> 0 * 10 = 0`") no longer happens: the continuation is "they were happy" (greedy)
  and "sara and ben were happy . they liked the cake together" (sampled), with no call.

### The decode-time margin

Seed 2, 1,000 problems per set (answer right, call rate in brackets), 1,000 no-call sentences per set:

| margin | word, trained | held-out shapes | held-out verbs | two-step | no-call: calls | no-call held-out: calls | `<calc>` top guess in 207k words | calls in 8,000 free words |
|---|---|---|---|---|---|---|---|---|
| −2 | 99.4% (100) | 78.1% (90) | 85.6% (97) | 99.3% | 0.5% | 2.0% | 132 | 7 |
| −1 | 99.5% (100) | 77.4% (88) | 85.1% (96) | 99.1% | 0.5% | 1.8% | 121 | 7 |
| 0 | 99.4% (100) | 75.6% (86) | 82.6% (94) | 99.0% | 0.5% | 1.6% | 75 | 7 |
| 1 | 98.9% (99) | 72.7% (82) | 79.9% (91) | 98.3% | 0.5% | 1.2% | 24 | 7 |
| 2 | 96.1% (96) | 68.6% (78) | 74.1% (85) | 97.4% | 0.5% | 0.8% | 5 | 6 |
| 3 | 90.0% (90) | 64.2% (73) | 67.0% (77) | 95.2% | 0.5% | 0.5% | 2 | 6 |
| 4 | 84.7% (85) | 59.7% (68) | 62.8% (72) | 86.4% | 0.4% | 0.4% | 1 | 3 |

A margin of 1-2 removes most of the top-guess calls in story text (75 → 24 → 5) for 0.5-3 points on trained
shapes and 3-7 on held-out ones; a negative margin buys 2-3 points on held-out shapes and verbs (the calls it
adds are mostly right) for 46-57 more top-guess positions in 207k words. The calls in free writing do not move
until margin 4, because they follow problems the GPT wrote itself, where `<calc>` leads by a wide margin. The
first run's seed 0 behaves the same way (65% → 54% calls on trained shapes from margin 0 to 4, 58 → 0 text
calls). No margin is the default; the slider is on the page.

### Story perplexity

| | no calls, same stream | + calc, seeds 0 / 1 / 2 | first run: no calls / + calc |
|---|---|---|---|
| 256 held-out windows | **15.33** | 15.76 / 15.69 / 15.66 | 14.70 / 15.28-15.39 |
| all 207k held-out words | 15.68 | 16.12 / 16.19 / 16.05 | 15.12 / 15.63-15.80 |

The gap between the calc models and the no-calls GPT on the same stream is 0.33-0.43 (the first run: 0.6-0.7),
but both are about 0.6 above their counterparts in the first run. The stream's share of story words is nearly
the same (the problems take 1.4 M of 5.7 M tokens against 1.2 M of 5.5 M); what changed is the problems
themselves: 10,749 shapes with names, places, containers and distractor numbers are a much richer text than
twelve fixed sentences, and the model spends more of its 1.4 M weights on them. The text-loss curves (figure 1,
right) run together, the no-calls run a little below throughout. A run trained for as many story words as the
no-calls GPT was not repeated here; in the first run that closed the gap to 0.1.

### Samples

Greedy continuations of the page model (seed 2) on held-out pieces; the chips are what the GPT typed, the
bold numbers what the computer typed back:

> sam picked 30 grapes. then mia gave him 12 more. sam had `<calc> 30 + 12 =` **42** `</calc>` 42 grapes. he
> was very happy. he had a big smile on his face.   (first run: `<calc> 175 - 12 =` 163 grapes left)
>
> there were 25 cookies on the plate. tom ate 9 of them. then there were `<calc> 25 - 9 =` **16** `</calc>`
> 16 cookies.   (first run: `<calc> 25 * 9 =` 225 cookies in all)
>
> mia has 6 boxes and each box has 7 toys. so mia has `<calc> 6 * 7 =` **42** `</calc>` 42 toys.
>
> ben found 40 shells. he caught 15 more. now he has `<calc> 40 + 15 =` **55** `</calc>` 55 shells.
> (first run: `<calc> 7 * 15 =` 105)
>
> tom had 5 bags and every bag had 9 stickers. that is `<calc> 5 * 9 =` **45** `</calc>` 45 stickers.
>
> tom had 12 apples. he got 5 more. he put the 5 new apples in the cup. he was so happy.   (no call)
>
> lily had 30 marbles. she dropped 12. she had `<calc> 12 - 12 =` **0** `</calc>` 0 marbles left.

The last one is a miss of the kind that remains: a held-out verb, the call is made, the first operand is wrong.
The no-calls GPT writes 16 apples, 44 stickers, 70 toys and 60 shells for the same openings.

## Try it

`storytool_ui/index.html` (with `model.json`, `cpu.json` and the two weight files, all rewritten by this run)
runs the seed-2 GPT and the new no-calls GPT in the browser. New: example prompts in held-out shapes and with
held-out verbs, the results table with held-out shapes and verbs and the first run's model as a third row, the
list of held-out verbs, and the call-margin slider.

    python3 -m http.server 8765 --directory storytool_ui      # then open http://localhost:8765

## Limitations

- Held-out shapes stop at 70-78% and do not improve after step 2,000; the misses are questions without a
  number cue (never a call), sentences with the second number first (wrong operands) and verbs whose story
  meaning points the wrong way. More frames of those kinds in training is the direct fix; whether a 4-layer
  model can learn "a story with two numbers and a change wants a total" without a cue word is the open question.
- A held-out piece in a two-step problem costs twice (50-62%).
- Seed variance on the weak pieces is large (0-80%); three seeds, one no-calls baseline, no answer-in-loss or
  6-layer ablation this time.
- The richer problems cost about 0.6 perplexity for the no-calls GPT and the calc GPTs alike; the "same story
  words" run of the first run was not repeated.
- Spurious calls in free writing went from 0-3 to 5-7 per 8,000 words, all after self-written problems; the
  margin does not touch those.
- The generator's sentences are TinyStories-plain and sometimes odd ("tim saw 0 cups", "dad was 2 years old");
  operands run 0-255 as the 8-bit computer requires.
