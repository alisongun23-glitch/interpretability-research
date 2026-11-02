"""Tokens, problem templates and the train/test split for the GPT with a CPU head.

Every number 0-255 is one token. Math templates move the operands around (reversed order,
a distractor number) so the model has to learn where to point; non-math templates are tasks
the computer cannot help with.
"""
import numpy as np

WORDS = ["<pad>", "<bos>", "+", "-", "*", "plus", "minus", "times", "sum", "what", "is", "?", "=",
         "subtract", "from", ";", "copy", "first", "last", "of", "and"]
VOCAB = [str(i) for i in range(256)] + WORDS
TOK = {w: i for i, w in enumerate(VOCAB)}
PAD, BOS = TOK["<pad>"], TOK["<bos>"]
OPCODE_OF_WORD = {"+": 1, "plus": 1, "sum": 1, "-": 2, "minus": 2, "subtract": 2, "*": 3, "times": 3}

# frozen value tables: what the CPU head sends to the computer for each token (its bus format)
NUMVAL = np.array([i if i < 256 else 0 for i in range(len(VOCAB))], np.int64)
OPVAL = np.array([OPCODE_OF_WORD.get(w, 0) for w in VOCAB], np.int64)

ADD, SUB, MUL = 1, 2, 3
OPS = (ADD, SUB, MUL)       # the operations the generators draw from (run_storyroom.py narrows it to addition)
MAX_RESULT = 256            # operands and results stay below this; 256 = every number token, smaller = more exposures per fact
# (name, words with placeholders A B C, op or None, answer)
TEMPLATES = [
    ("a + b", "A + B =", ADD), ("a - b", "A - B =", SUB), ("a * b", "A * B =", MUL),
    ("what is a plus b", "what is A plus B ?", ADD), ("what is a minus b", "what is A minus B ?", SUB),
    ("what is a times b", "what is A times B ?", MUL), ("sum of a and b", "sum of A and B =", ADD),
    ("subtract b from a", "subtract B from A =", SUB),
    ("c ; a + b", "C ; A + B =", ADD), ("c ; a - b", "C ; A - B =", SUB), ("c ; a * b", "C ; A * B =", MUL),
    ("copy a", "copy A =", "copy"), ("first of a and b", "first of A and B =", "first"),
    ("last of a and b", "last of A and B =", "last"),
]
MATH = [t for t in TEMPLATES if isinstance(t[2], int)]
NONMATH = [t for t in TEMPLATES if not isinstance(t[2], int)]
T_MAX = 1 + max(len(t[1].split()) for t in TEMPLATES)


def answer(kind, a, b):
    return {ADD: a + b, SUB: a - b, MUL: a * b, "copy": a, "first": a, "last": b}[kind]


def is_test(kind, a, b):
    """Held-out operand pairs (about 20 %), the same rule for every template of an operation.

    A symmetric hash of the unordered pair (min, max) and the kind: a + b and b + a fall on the same side, each
    kind gets its own 20 %, and every result value keeps training pairs (addition: every sum below 256 has
    training pairs; all sums from 19 up have held-out pairs too). Until 1 Nov 2026 the rule was
    (a * 7919 + b * 104729 + k * 31337) % 5 == 0; both multipliers are 4 mod 5, so for addition it reduced to
    a + b == 2 (mod 5) and held out whole RESULT values (2, 7, 12, ...) instead of pairs. Models saved before
    that date were trained under the old split (results_storyroom/ROOM_REPORT.md, phase 6). For addition this
    is run_storyroom.py's former --split pair rule, so its "_pairsplit" jobs are on the default split.
    """
    k = {ADD: 1, SUB: 2, MUL: 3}.get(kind, 4)
    lo, hi = int(min(a, b)), int(max(a, b))
    h = (lo * 2654435761 + hi * 40503 + 7 * k) & 0xffffffff
    h ^= h >> 16; h = (h * 0x45d9f3b) & 0xffffffff; h ^= h >> 16
    return h % 5 == 0


def sample_pair(kind, r):
    M = MAX_RESULT
    if kind == ADD:
        a = int(r.integers(0, M)); b = int(r.integers(0, M - a))
    elif kind == SUB:
        a = int(r.integers(0, M)); b = int(r.integers(0, a + 1))
    elif kind == MUL:
        b = int(r.integers(0, min(16, M))); a = int(r.integers(0, M // max(b, 1)))
    else:
        a, b = int(r.integers(0, M)), int(r.integers(0, M))
    return a, b


def encode(words, a, b, c):
    out = [BOS]
    for w in words.split():
        out.append(TOK[str({"A": a, "B": b, "C": c}[w])] if w in "ABC" else TOK[w])
    return out


def batch(r, n, split="train", math_frac=0.8, templates=None):
    """Returns dict of numpy arrays: tokens (n, T_MAX) right-padded, last (index of the last token),
    target, kind (0 non-math, 1-3 op), tid (template index), a, b."""
    toks = np.full((n, T_MAX), PAD, np.int64)
    out = {k: np.zeros(n, np.int64) for k in ("last", "target", "kind", "tid", "a", "b")}
    pool = templates if templates is not None else None
    for i in range(n):
        if pool is not None:
            tid = pool[int(r.integers(len(pool)))]
        else:
            group = MATH if r.random() < math_frac else NONMATH
            tid = TEMPLATES.index(group[int(r.integers(len(group)))])
        name, words, kind = TEMPLATES[tid]
        while True:
            a, b = sample_pair(kind, r)
            if is_test(kind, a, b) == (split == "test"):
                break
        c = int(r.integers(0, 256))
        seq = encode(words, a, b, c)
        toks[i, :len(seq)] = seq
        out["last"][i] = len(seq) - 1; out["target"][i] = answer(kind, a, b)
        out["kind"][i] = kind if isinstance(kind, int) else 0; out["tid"][i] = tid
        out["a"][i], out["b"][i] = a, b
    out["tokens"] = toks
    return out


def tokenize(text):
    """Text typed by a person -> token ids (raises ValueError on unknown words)."""
    ids = [BOS]
    for w in text.replace("?", " ? ").replace("=", " = ").replace(";", " ; ").split():
        if w not in TOK or (w.isdigit() and int(w) > 255):
            raise ValueError(f"unknown token '{w}'")
        ids.append(TOK[w])
    return ids
