> Archived first version (math templates only). The current report is `../STORYGPT_REPORT.md`; paths below refer
> to the project as it was then (this folder holds that version's `summary.json`, models, figure and page data).

# A story-writing GPT that sends math to the neural-gate computer

Produced by `python run_storygpt.py` (about 35 minutes of training on a laptop CPU; `--analyse` reuses
the saved models). Numbers are in `summary.json`, the figure in `figures/`, models in `models/`.

## Answer first

The GPT now writes children's stories, and when a math problem comes up, in a story or on its own,
it sends it to the computer head and writes the computer's answer. On math with numbers it never saw
in training, placed after held-out story text, it scores 85–86% on addition, 96–97% on subtraction
and 95–98% on multiplication. The same GPT without the computer scores 0%, 2% and 18%. Its stories are
as good as the plain GPT's (perplexity 13.7–13.9 against 13.8). In held-out story text it **never** calls
the computer (0 calls in 16,000 words), and it calls on every math answer (100%).

## Setup

**Text.** TinyStories (`data/TinyStories-valid.txt`, 21,990 short stories, downloaded from
huggingface.co/datasets/roneneldan/TinyStories; the last 1,000 stories are held out). Words are
lower-cased and split into words, numbers and punctuation. The vocabulary is 4,279 tokens: specials,
every number 0–255 as one token, the math words, and the 4,000 most common story words (1% of held-out
words are unknown). About two math problems from the earlier templates (`a + b =`,
`what is a times b ?`, `subtract b from a =`, `c ; a - b =`, …) are dropped into each training story
after a random full stop, with their answers, using training operand pairs only: 4.65 M training tokens,
41,709 problems.

**Model.** 4 layers, d_model 128, 4 heads of 32, MLP 512, 64-token context, tied input/output
embeddings, PyTorch. Plain GPT: 4 ordinary heads in every layer. CPU-head GPT: the last layer has 3
ordinary heads and the CPU head (`cpugpt/lm.py`), which now runs at **every** position. At each position
it points three pointers (A, B, operator) at earlier tokens, decides whether to call, sends the picked
tokens through the frozen bus tables to the computer (the calculator program on the gate CPU), and, if
it called, adds a trained projection of the 8 result bits before the next word is predicted.

**Learning.** Next-word cross-entropy on everything. The pointers and the call decision are discrete
and get no gradient through the computer, so they learn by REINFORCE:

| part | reward | why |
|---|---|---|
| pointers | +1 when the computer's result is the next token, 64 cheap tries per position (a table lookup each) | in a 64-token context three random pointers almost never land; the pilot with 2 tries per position never learned |
| call decision | log p(next token) + 1 if the call returned the next token − 0.05 per call; at least 5% of calls explored | the earlier experiment showed that rewarding only the model's own answer stops calls on memorized problems |

The pointers also get a learned relative-position bias, started with a slight lean toward recent
tokens. The computer's answers come from the table computed on the gate netlist in `run_cpugpt.py`
(all 262,144 calls); 300 test calls were re-run on the fully analog neural CPU, 0 differences.

## Results

Math: 3,000 problems with held-out operand pairs, each after 0–40 words of held-out story text.
Text: 256 windows of 64 tokens of held-out stories without math.

| model | + (95% CI) | − | × | same on training pairs (+ / − / ×) | story perplexity | calls in story text | calls at math answers | pointers right |
|---|---|---|---|---|---|---|---|---|
| plain story GPT | 0.0% (0–0.4) | 2.1% | 17.6% | 6% / 5% / 72% | 13.81 | – | – | – |
| + CPU head, seed 0 | 85.7% (83.5–87.7) | 96.8% | 94.7% | 96% / 97% / 95% | 13.89 | 0 | 100% | 95.9% |
| + CPU head, seed 1 | 85.1% (82.8–87.2) | 95.6% | 97.7% | 96% / 96% / 99% | 13.68 | 0 | 100% | 96.1% |

- **The plain GPT cannot do the math**, not even on training pairs for + and − (6%, 5%). One answer per
  pair is too little signal among 4.6 M story tokens; it only learned some small products.
- **Switching the head off** drops the CPU-head models to 0–8%; swapping the operation the computer
  receives drops them to under 2%. The answers come from the computer.
- **Story writing is unchanged**: the text loss curves of all three runs lie on top of each other
  (figure, right), and with the head off the story perplexity is identical, because the head never calls
  on story text.

### Where the remaining errors are (seed 1)

| | answer right | computer right | computer right, answer misread | computer wrong (pointers) |
|---|---|---|---|---|
| + | 85.1% | 95.5% | 107 of 1,030 | 46 |
| − | 95.6% | 95.7% | 1 of 1,144 | 49 |
| × | 97.7% | 97.7% | 0 of 826 | 19 |

Two kinds. About 2–5% of calls point at the wrong tokens. And for addition only, the model sometimes
misreads a right answer: every misread is a large number, 192–255 (answers below 192 are read back
100% right), usually one bit off (247 read as 245 or 231). Big sums are the only results above 191, so
the read-back projection has seen those bit patterns least.

### It uses the computer for problems it makes up

With sampling (temperature 0.8) the model sometimes writes a math problem into a story by itself, since
it saw them there in training, and then calls the computer to answer it:

> lily went to the park with her mom. what is 14 times 12? **168**. they found a pretty flower on it and
> decided to make it look beautiful. but then, lily's mom told her what it was. what is 19 minus 3? **16**.

(bold = written from the computer's result; both calls correct). The plain GPT, given the same openings,
writes 14 × 12 = 144 or 84, 60 + 117 = 180, and "subtract 54 from 120 =" 30 or 48.

## Try it

`storygpt_ui/index.html` (with `model.json`, `cpu.json` and the two weight files) runs both GPTs in
the browser. Type the start of a story or a question. Words that came from the computer are marked
with the call (for example 168 with "14×12"), the side panel shows where the three pointers landed in
the context, and the computer runs on neural gates in the panel. Settings: with/without the CPU head,
head on/off, randomness, length.

    python3 -m http.server 8765 --directory storygpt_ui      # then open http://localhost:8765

## Limitations

- Math is recognised in the trained wordings only; "Tom had 3 apples and got 4 more" is not a call.
  Making that work needs training text where such sentences are paired with answers.
- Story quality is that of a 4-layer, 1.4 M-weight model trained for 35 minutes on 4.6 M words: simple,
  repetitive, sometimes confused.
- One call per position and a fixed calculator program; numbers are 0–255.
- Two seeds of the CPU-head model, one of the plain model.
