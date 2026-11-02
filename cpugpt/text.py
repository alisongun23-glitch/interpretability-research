"""Text for the story GPT: TinyStories with math problems dropped in.

Words are lower-cased and split into words, numbers and punctuation. The vocabulary is the
special tokens, every number 0-255 as one token (so the CPU head's bus format still works),
the math words of cpugpt.data, and the most common story words. A training stream is the
stories with math problems (cpugpt.data templates, train operand pairs only) inserted after
random sentence ends, about two per story, and optionally story-style word problems
("tom had 12 apples . he got 5 more . now he has 17 apples .") with the same operand split.

For the GPT that calls the computer in writing (cpugpt.tool), build_tool_stream renders every problem
with its call block before the answer, "now he has <calc> 12 + 5 = 17 </calc> 17 apples .": the model
writes "<calc> 12 + 5 =", the machine writes "17 </calc>" (flagged so it is left out of the loss), and
the model copies 17. Two-step problems chain a second call on the first result; no-call sentences
have numbers but nothing to compute. The sentences come from a compositional generator (frames with many verbs,
openings, distractors, several questions; some frames and verbs held out for testing, see gen_word) or, with
varied=False, from the fixed lists of the first version.
"""
import re
from collections import Counter
import numpy as np

from . import data as D

WORD_RE = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)
NUM0 = len(SPECIAL)                                  # token id of the number 0; number v has id NUM0 + v
MATH_WORDS = [w for w in D.WORDS if not w.startswith("<")]
# the frozen operator table, extended with the story words that name an operation in the word problems
OPCODE_OF_WORD = {**D.OPCODE_OF_WORD, "more": D.ADD, "ate": D.SUB, "lost": D.SUB, "gave": D.SUB, "each": D.MUL}

# Word problems: (name, words, op, operation word). A, B operands, R the answer, D a distracting number,
# N a name (P he/she, H his/her, O him/her), M another name, T things, F food, K/k container plural/singular.
WORD_PROBLEMS = [
    ("had A, got B more", "N had A T . P got B more . now P has R T .", D.ADD, "more"),
    ("found A and then B more", "N found A T and then B more . P had R T in all .", D.ADD, "more"),
    ("A in the box, put in B more", "there were A T in the k . N put in B more . now there were R T .", D.ADD, "more"),
    ("D years old, had A, got B more", "N was D years old . P had A T and got B more . now P has R T .", D.ADD, "more"),
    ("had A and ate B", "N had A F and ate B . P had R left .", D.SUB, "ate"),
    ("ate B of A", "N ate B of H A F . P had R F left .", D.SUB, "ate"),
    ("had A but lost B", "N had A T but lost B . now P has R T .", D.SUB, "lost"),
    ("had A and gave B to M", "N had A T and gave B to M . P had R T left .", D.SUB, "gave"),
    ("A bags with B each", "there were A K with B T each , R T in all .", D.MUL, "each"),
    ("A bags, each had B", "N had A K . each k had B T . that is R T in all .", D.MUL, "each"),
    ("A friends each brought B", "A friends each brought B T . they had R T in all .", D.MUL, "each"),
    ("day D, A bags with B each", "on day D , N saw A K with B T each . that was R T .", D.MUL, "each"),
]
# never in training: sentence shapes the model has not seen (the first also has 'gave', a minus word, next to 'more')
NEW_WORDINGS = [
    ("picked A, M gave B more", "N picked A F . then M gave O B more . N had R F .", D.ADD, "more"),
    ("A on the plate, ate B of them", "there were A F on the plate . N ate B of them . then there were R F left .", D.SUB, "ate"),
    ("has A bags and each has B", "N has A K and each k has B T . so N has R T .", D.MUL, "each"),
]
# Two-step problems: (name, words, op1, op2). The first call is A op1 B = R1, the second R1 op2 C = R; only R
# appears in the text, R1 only inside the call block. The second step is never x (R1 is often too big).
TWO_STEP = [
    ("had A, got B more and then C more", "N had A T . P got B more and then C more . now P has R T .", D.ADD, D.ADD),
    ("had A, ate B and then got C more", "N had A F . P ate B and then got C more . P had R F .", D.SUB, D.ADD),
    ("had A and got B more, then lost C", "N had A T and got B more . then P lost C . now P has R T .", D.ADD, D.SUB),
    ("A bags with B each, took C", "there were A K with B T each . N took C . there were R T left .", D.MUL, D.SUB),
    ("A bags with B each and found C more", "N had A K with B T each and found C more . now P has R T .", D.MUL, D.ADD),
    ("had A, ate B and then ate C more", "N had A F and ate B . then P ate C more . P had R F left .", D.SUB, D.SUB),
]
# Sentences with numbers and nothing to compute: (name, words, which number R restates). No call is the right move.
NOCALL = [
    ("counted A, that is A", "N counted A T . that is R T .", "A"),
    ("had A and B, put the A in the box", "N had A T and B F . P put the R T in the k .", "A"),
    ("D years old, had A, loved the A", "N was D years old . P had A T . P loved H R T .", "A"),
    ("A in the box and B on the plate, took the A", "there were A T in the k and B F on the plate . N took the R T .", "A"),
    ("saw A and B, liked the B", "N saw A T and B F . P liked the R F .", "B"),
]
TOOL_TOKENS = ["<calc>", "</calc>"]
CHECK_TOKENS = ["<unsure>"]       # the machine's own doubt about an answer (cpugpt.check, run_storycheck.py)
OPSYM = {D.ADD: "+", D.SUB: "-", D.MUL: "*"}
BOYS = "tom sam ben max tim jack bob joe leo".split()
GIRLS = "lily mia sue anna lucy sara sally amy jill emma molly".split()
THINGS = "apples toys balls flowers rocks books cars stickers shells blocks crayons coins marbles leaves stars buttons".split()
FOOD = "apples cookies cakes candies grapes berries carrots cupcakes bananas strawberries sweets treats".split()
BOXES = [("bags", "bag"), ("boxes", "box"), ("cups", "cup")]


def words(text):
    return WORD_RE.findall(text.lower())


def load_stories(path):
    raw = open(path, encoding="utf-8").read()
    return [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]


class Vocab:
    def __init__(self, stories, n_words=4000, tool_tokens=False, extra=()):
        cnt = Counter(w for s in stories for w in words(s))
        self.itos = SPECIAL + [str(i) for i in range(256)] + MATH_WORDS
        have = set(self.itos)
        self.itos += [w for w, _ in cnt.most_common() if w not in have and not w.isdigit()][:n_words]
        if tool_tokens:                                  # at the end, so every other id stays as without them
            self.itos += TOOL_TOKENS
        self.itos += list(extra)                         # after them (CHECK_TOKENS for the GPT that checks the machine)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.count = cnt                                 # story word counts (frequency bins in cpugpt.probes)
        self.numval = np.array([i - NUM0 if NUM0 <= i < NUM0 + 256 else 0 for i in range(len(self))], np.int64)
        self.opval = np.array([OPCODE_OF_WORD.get(w, 0) for w in self.itos], np.int64)
        self.CALC, self.END = self.stoi.get("<calc>"), self.stoi.get("</calc>")
        self.UNSURE = self.stoi.get("<unsure>")

    def __len__(self):
        return len(self.itos)

    def encode(self, ws):
        return [self.stoi.get(w, UNK) for w in ws]

    def decode(self, ids):
        """Token ids -> readable text (spacing rules for punctuation, quotes and apostrophes)."""
        out, quote_open, glue = "", False, False
        for i in ids:
            w = self.itos[i]
            if w in ("<pad>", "<bos>"):
                continue
            if w == "<eos>":
                out += "\n\n"; quote_open = glue = False; continue
            if w == "'":
                out += w; glue = True; continue
            if w == '"':
                if quote_open:
                    out += w
                else:
                    out += ("" if not out or out.endswith(("\n", " ")) else " ") + w; glue = True
                quote_open = not quote_open; continue
            space = not (glue or not out or out.endswith(("\n", " ")) or w in ".,!?;:)")
            if w in "+-*=;":
                space = bool(out) and not out.endswith(("\n", " "))
            out += (" " if space else "") + w
            glue = False
        return out


def trace(v, a, op, b, r):
    """The call block "<calc> a op b = r </calc>" as ids, with flags marking what the machine writes (r and
    </calc>), which the loss leaves out."""
    ids = [v.CALC] + v.encode([str(a), OPSYM[op], str(b), "="]) + [NUM0 + r, v.END]
    return ids, [False] * 5 + [True, True]


def _render(v, ws, fill, calls, call):
    """Words with placeholders -> (ids, answer_offset, env flags, cue offset). The call blocks go right before R;
    answer_offset is the token before R (the one that must predict the answer), cue the word before the first block.
    call: False (no blocks), True (the machine answers right), or a machine object whose blocks(v, a, op, b, r)
    writes the block(s) of one call (cpugpt.check.FlakyMachine: a wrong answer is followed by the model's retry)."""
    k = ws.index("R")
    ids = v.encode([fill.get(w, w) for w in ws[:k]]); env = [False] * len(ids); cue = len(ids) - 1
    if call:
        for a, op, b, res in calls:
            t, e = call.blocks(v, a, op, b, res) if hasattr(call, "blocks") else trace(v, a, op, b, res); ids += t; env += e
    off = len(ids) - 1
    post = v.encode([fill.get(w, w) for w in ws[k:]])
    return ids + post, off, env + [False] * len(post), cue


def snippet(v, r, split, tid=None, pair=None, call=False):
    """One math problem with its answer and a full stop. Returns (ids, answer_offset, info): the token at
    answer_offset must predict the answer. With call=True the call block precedes the answer; info["calls"]
    lists the calls (a, op, b, result), info["env"] flags the machine-written tokens, info["cue"] is the
    offset of the last question word."""
    tid = _pick(r, [i for i, t in enumerate(D.MATH) if t[2] in D.OPS]) if tid is None else tid
    name, wds, kind = D.MATH[tid]
    while True:
        a, b = pair if pair is not None else D.sample_pair(kind, r)
        if pair is not None or D.is_test(kind, a, b) == (split == "test"):
            break
    c = int(r.integers(0, D.MAX_RESULT))
    ans = D.answer(kind, a, b)
    calls = [(a, kind, b, ans)]
    ids, off, env, cue = _render(v, wds.split() + ["R", "."], {"A": str(a), "B": str(b), "C": str(c), "R": str(ans)}, calls, call)
    return ids, off, {"kind": kind, "a": a, "b": b, "target": ans, "tid": tid, "calls": calls, "env": env, "cue": cue}


def _people(r):
    girl = r.random() < 0.5
    names = GIRLS if girl else BOYS
    n1 = names[int(r.integers(len(names)))]
    n2 = [w for w in BOYS + GIRLS if w != n1][int(r.integers(len(BOYS) + len(GIRLS) - 1))]
    box = BOXES[int(r.integers(len(BOXES)))]
    return {"D": str(int(r.integers(2, 13))), "N": n1, "M": n2, "P": "she" if girl else "he", "H": "her" if girl else "his",
            "O": "her" if girl else "him", "T": THINGS[int(r.integers(len(THINGS)))], "F": FOOD[int(r.integers(len(FOOD)))],
            "K": box[0], "k": box[1]}


def word_snippet(v, r, split, tid=None, pair=None, templates=WORD_PROBLEMS, call=False):
    """One word problem with its answer. Returns (ids, answer_offset, info) like snippet(); info["at"]
    holds the offsets of the two operands and of the operation word (where the pointers should land)."""
    tid = _pick(r, [i for i, t in enumerate(templates) if t[2] in D.OPS]) if tid is None else tid
    name, wds, kind, opword = templates[tid]
    while True:
        a, b = pair if pair is not None else D.sample_pair(kind, r)
        if pair is not None or D.is_test(kind, a, b) == (split == "test"):
            break
    ans = D.answer(kind, a, b)
    fill = {**_people(r), "A": str(a), "B": str(b), "R": str(ans)}
    ws = wds.split()
    calls = [(a, kind, b, ans)]
    ids, off, env, cue = _render(v, ws, fill, calls, call)
    at = (ws.index("A"), ws.index("B"), ws.index(opword))
    return ids, off, {"kind": kind, "a": a, "b": b, "target": ans, "tid": tid, "at": at, "calls": calls, "env": env, "cue": cue}


def two_step_snippet(v, r, split, tid=None, call=True):
    """A two-step word problem: A op1 B = R1, then R1 op2 C = R. Train problems use training pairs for both
    steps, test problems held-out pairs for both (is_test depends on a + b only, so for chains starting
    with + this pins C to one residue class mod 5)."""
    tid = _pick(r, [i for i, t in enumerate(TWO_STEP) if t[2] in D.OPS and t[3] in D.OPS]) if tid is None else tid
    name, wds, k1, k2 = TWO_STEP[tid]
    while True:
        a, b = D.sample_pair(k1, r)
        r1 = D.answer(k1, a, b)
        c = int(r.integers(0, D.MAX_RESULT - r1)) if k2 == D.ADD else int(r.integers(0, r1 + 1))
        if D.is_test(k1, a, b) == D.is_test(k2, r1, c) == (split == "test"):
            break
    ans = D.answer(k2, r1, c)
    fill = {**_people(r), "A": str(a), "B": str(b), "C": str(c), "R": str(ans)}
    calls = [(a, k1, b, r1), (r1, k2, c, ans)]
    ids, off, env, cue = _render(v, wds.split(), fill, calls, call)
    return ids, off, {"kind": k1, "kind2": k2, "a": a, "b": b, "c": c, "r1": r1, "target": ans, "tid": tid,
                      "calls": calls, "env": env, "cue": cue}


def nocall_snippet(v, r, split, tid=None):
    """A sentence with numbers and nothing to compute; R restates one of them. The operand split is kept
    so held-out tests use unseen (A, B) pairs here too."""
    tid = int(r.integers(len(NOCALL))) if tid is None else tid
    name, wds, which = NOCALL[tid]
    while True:
        a, b = D.sample_pair("nocall", r)
        if D.is_test("nocall", a, b) == (split == "test"):
            break
    fill = {**_people(r), "A": str(a), "B": str(b)}
    fill["R"] = fill[which]
    ids, off, env, cue = _render(v, wds.split(), fill, [], False)
    return ids, off, {"kind": 0, "a": a, "b": b, "target": int(fill["R"]), "tid": tid, "calls": [], "env": env, "cue": cue}


def build_stream(v, stories, r, split, per_story=2.0, word_per_story=0.0):
    """Token stream of stories (separated by <eos>) with math problems (per_story of them on average,
    cpugpt.data templates) and word problems (word_per_story) inserted after random sentence ends.
    Returns (tokens, answer_positions, answer_info); info["word"] tells which kind of problem."""
    toks, ans_pos, ans_info = [], [], []
    period = v.stoi["."]
    p_word = word_per_story / (per_story + word_per_story) if word_per_story else 0.0
    for s in stories:
        ids = v.encode(words(s))
        ends = [i for i, t in enumerate(ids) if t == period]
        k = min(len(ends), int(r.poisson(per_story + word_per_story)))
        cut = set(r.choice(ends, size=k, replace=False).tolist()) if k else set()
        for i, t in enumerate(ids):
            toks.append(t)
            if i in cut:
                word = bool(p_word) and r.random() < p_word
                sn, off, info = word_snippet(v, r, split) if word else snippet(v, r, split)
                ans_pos.append(len(toks) + off); ans_info.append({**info, "word": word})
                toks.extend(sn)
        toks.append(EOS)
    return np.array(toks, np.int64), np.array(ans_pos, np.int64), ans_info


def build_tool_stream(v, stories, r, split, per_story=2.0, word_per_story=1.5, two_per_story=0.5, nocall_per_story=0.5, calls=True, varied=True):
    """Token stream for the GPT that calls the computer in writing: math templates, word problems, two-step
    problems and no-call sentences (so many of each per story on average) after random sentence ends, each
    problem with its call block(s) before the answer. calls=False renders the same problems (same random
    draws) without call blocks. Returns (tokens, env, answer_positions, answer_info): env flags the tokens the
    machine wrote; info["family"] is 0 math, 1 word, 2 two-step, 3 no-call. varied=True draws the word, two-step and
    no-call sentences from the compositional generator (gen_word, gen_two, gen_nocall: trained frames and verbs only);
    varied=False uses the fixed lists of the first version (WORD_PROBLEMS, TWO_STEP, NOCALL). A no-call sentence
    without a number to restate has target -1 and its answer position on its last token."""
    toks, env, ans_pos, ans_info = [], [], [], []
    period = v.stoi["."]
    rates = np.array([per_story, word_per_story, two_per_story, nocall_per_story], float)
    if varied:
        makers = (lambda: snippet(v, r, split, call=calls), lambda: gen_word(v, r, split, call=calls),
                  lambda: gen_two(v, r, split, call=calls), lambda: gen_nocall(v, r, split))
    else:
        makers = (lambda: snippet(v, r, split, call=calls), lambda: word_snippet(v, r, split, call=calls),
                  lambda: two_step_snippet(v, r, split, call=calls), lambda: nocall_snippet(v, r, split))
    for s in stories:
        ids = v.encode(words(s))
        ends = [i for i, t in enumerate(ids) if t == period]
        k = min(len(ends), int(r.poisson(rates.sum())))
        cut = set(r.choice(ends, size=k, replace=False).tolist()) if k else set()
        for i, t in enumerate(ids):
            toks.append(t); env.append(False)
            if i in cut:
                fam = int(r.choice(4, p=rates / rates.sum()))
                sn, off, info = makers[fam]()
                ans_pos.append(len(toks) + off); ans_info.append({**info, "family": fam, "start": len(toks), "len": len(sn)})
                toks.extend(sn); env.extend(info["env"])
        toks.append(EOS); env.append(False)
    return np.array(toks, np.int64), np.array(env, bool), np.array(ans_pos, np.int64), ans_info


def windows(stream, r, n, T, env=None, *extra):
    """n random training windows: inputs (n, T) and next-token targets (n, T); with env, also the flags of
    the targets (n, T) that the machine wrote (to leave out of the loss); every further per-token array in
    extra (roles, weights) comes back the same way, aligned with the targets."""
    start = r.integers(0, len(stream) - T - 1, n)
    idx = start[:, None] + np.arange(T + 1)[None]
    w = stream[idx]
    if env is None and not extra:
        return w[:, :-1], w[:, 1:]
    out = [w[:, :-1], w[:, 1:]]
    if env is not None:
        out.append(env[idx][:, 1:])
    out += [x[idx][:, 1:] for x in extra]
    return tuple(out)


ROLE_STORY, ROLE_PROBLEM, ROLE_ANSWER, ROLE_MACHINE, ROLE_CALL, ROLE_RESTATE = 0, 1, 2, 3, 4, 5


def roles(tokens, env, ans_pos, ans_info, calc=None):
    """What each token of a tool stream is (int8): 0 story text, 1 problem text, 2 the answer of a math, word or
    two-step problem (the token the model computes or copies), 3 what the machine wrote, 4 the call syntax the
    model writes ("<calc> a op b =", five tokens before each machine result; calc is the <calc> id), 5 the
    restated number of a no-call sentence. Needs the "start" / "len" that build_tool_stream records."""
    ro = np.zeros(len(tokens), np.int8)
    for p, info in zip(ans_pos, ans_info):
        s = info["start"]
        ro[s:s + info["len"]] = ROLE_PROBLEM
        if info["family"] == 3:
            if info["target"] >= 0:
                ro[p + 1] = ROLE_RESTATE
        else:
            ro[p + 1] = ROLE_ANSWER
    if calc is not None:
        for c in np.nonzero(tokens == calc)[0]:
            ro[c:c + 5] = ROLE_CALL
    ro[np.asarray(env, bool)] = ROLE_MACHINE
    return ro


# ---------------------------------------------------------------------------------------------------------
# Varied sentence shapes: a compositional generator. A word problem is put together from frames: an optional
# opening, a start that gives the first number, a change that gives the second with one of many verbs, and a
# question that asks for the total (or a combined frame holding both numbers; or, for x, a frame with containers).
# Frames whose text starts with "*" and verbs written "*word" are held out of training and are only drawn by the
# held-out test sets: gen_word(held="shape") uses exactly one held-out frame, gen_word(held="verb") exactly one
# held-out verb. No-call sentences reuse the same starts, changes and verbs without a question, restate a number
# that is already there, or use the verbs with no numbers at all.
# Placeholders: N the subject (a name, mom, the kids, tom and mia), P/H/O its pronouns, HAS/WAS has-have/was-were,
# M another person, T the things counted, Q another kind of thing (a distractor), F a food, K/k a container
# plural/singular, G/G1 a group of people plural/singular, V a verb slot ("word:class" names the kind of thing the
# verb needs), EACH each (every in the held-out verb test), A B C the operands, D a small distractor number (an
# age, a day), E a distractor count, R the answer.
# ---------------------------------------------------------------------------------------------------------
PLACEHOLDERS = set("N M P H O HAS WAS T Q F K k G G1 V EACH A B C D E R".split())
OBJECTS = {
    "things": "toys balls blocks books cars stickers shells rocks crayons marbles leaves stars buttons flowers hats cards pencils seeds sticks pebbles coins".split(),
    "food": "apples cookies cakes candies grapes berries carrots cupcakes bananas strawberries sweets treats eggs nuts cherries pies muffins peaches acorns".split(),
    "fly": "balloons birds butterflies kites bees".split(),
    "pet": "dogs cats ducks bunnies frogs".split(),
    "break": "toys cups plates eggs crayons pencils".split(),
    "use": "crayons stickers seeds eggs cards sticks".split(),
    "money": ["coins"],
    "roll": "balls marbles".split(),
    "pop": ["balloons"],
}
OBJECTS["any"] = OBJECTS["things"] + OBJECTS["food"]
CONTAINERS = [("bags", "bag"), ("boxes", "box"), ("cups", "cup"), ("buckets", "bucket"), ("piles", "pile"), ("pockets", "pocket"),
              ("plates", "plate"), ("trees", "tree"), ("pots", "pot")]
GROUPS = [("friends", "friend"), ("kids", "kid"), ("boys", "boy"), ("girls", "girl")]
OTHERS = ["mom", "dad", "grandma", "grandpa", "H friend", "H sister", "H brother", "the teacher"]
SUBJECTS = [("mom", True), ("grandma", True), ("the girl", True), ("the teacher", True), ("dad", False), ("grandpa", False), ("the boy", False)]
CONNECTORS = ["then", "then", "later", "after that"]


def _frame(text, verbs=None, cls="any"):
    """A sentence pattern. verbs: "word word:class *word" (a class names the things the verb needs; * = held out)."""
    held = text.startswith("*"); text = text.lstrip("* ")
    vs = []
    for w in (verbs or "").split():
        h = w.startswith("*"); w = w.lstrip("*"); verb, _, c = w.partition(":")
        vs.append((verb, c or cls, h))
    toks = text.split()
    slot = "V" if "V" in toks else "EACH" if "EACH" in toks else None
    if slot == "EACH" and not vs:
        vs = [("each", cls, False), ("every", cls, True)]
    return {"text": text, "held": held, "verbs": vs, "slot": slot, "cls": cls, "person": any(w in ("N", "P", "H", "O") for w in toks),
            "verbs_held": any(h for _, _, h in vs), "verbs_train": any(not h for _, _, h in vs)}


def _frames(spec):
    return [_frame(*(s if isinstance(s, tuple) else (s,))) for s in spec]


OPENINGS = _frames(["one day ,", "one sunny day ,", "it was a nice day .", "N woke up early .", "N went to the park .", "N WAS D years old .",
                    "on day D ,", "after school ,", "N had E Q .", "N WAS at the beach .", "*when N WAS D years old ,", "*N had E Q and loved them ."])
STARTS = _frames(["N had A T .", "N HAS A T .", "N found A T .", "N counted A T .", "N saw A T .", "N got A T .", "N had A T in H k .",
                  "N brought A T .", "at first N had A T .", "N picked A T .", "N had A T and E Q .", "N put A T in the k .", "N made A T .",
                  "there were A T in the k .", "there were A T on the table .",
                  "*N took A T to school .", "*N kept A T in a k .", "*N had only A T .", "*in the k there were A T .", "*N started with A T ."])
ADD_CHANGES = _frames([("P V B more .", "got found picked bought won made *grabbed *collected *caught"),
                       ("M V O B more .", "brought handed *sent"), ("M V O B .", "gave brought handed *passed"),
                       ("P WAS V B more .", "given handed *sent"), ("P V B more from M .", "got bought"),
                       "P got B more for H birthday .", "P found B more in the k .", ("then P V B .", "found got"),
                       "*M gave O B more .", ("*P V B more and put them in the k .", "got found"), "*M came with B more .",
                       "*P added B more to the k .", "*P picked up B more ."])
SUB_CHANGES = _frames([("P V B .", "ate:food lost broke:break sold spent:money *dropped *used:use"), ("P V B to M .", "gave sold *passed"),
                       ("P V M B .", "gave handed"), ("M V B .", "took ate:food *stole"), ("then B T V away .", "flew:fly ran:pet rolled:roll"),
                       ("P V B away .", "threw gave"), ("then B T V .", "popped:pop broke:break"), "P shared B with M .",
                       ("*P V B of them .", "ate:food lost gave"), ("*P V away B .", "threw gave"), "*P let M have B .", "*then B of the T were gone ."])
ADD_COMBINED = _frames([("N had A T and V B more .", "got found *grabbed *collected"), "N found A T and then B more .",
                        "*N added B more to H A T .", "*N had A T and M gave O B more ."])
SUB_COMBINED = _frames([("N V B of H A T .", "ate:food lost broke:break *dropped"), ("N had A T but V B .", "lost ate:food broke:break *dropped *used:use"),
                        ("N had A T and V B to M .", "gave sold *passed"), ("*N had A T . B of them V away .", "flew:fly ran:pet")])
MUL_FRAMES = _frames(["there were A K with B T each .", "N had A K . EACH k had B T .", "N had A K with B T each .",
                      ("A G each V B T .", "brought had got *carried"), "on day D , N saw A K with B T each .",
                      "N saw A K . there were B T in EACH k .", "N filled A K with B T each .", "there were A K . N put B T in EACH k .",
                      "N had A K and EACH k had B T .",
                      "*N put B T in each of H A K .", "*N HAS A K and each k has B T .", "*every k had B T . there were A K .",
                      "*there were A K , B T in each .", "*each of the A K had B T ."])
ASKS_ALL = _frames(["now P HAS R T .", "so P had R T .", "then P had R T .", "P had R T .", "N had R T .", "that is R T .", "that made R T .",
                    "P now had R T .", "P HAS R T now .", "now there were R T .", "so now P HAS R T .", "how many T did P have ? R .",
                    "P ended up with R T .", "*so N HAS R T .", "*P counted them . there were R T .", "*P looked and saw R T ."])
ASKS_MORE = _frames(["P had R T in all .", "that is R T in all .", "R T in all .", "P had R T all together .", "that was R T in all .",
                     "*all together P had R T .", "*R T all together ."])
ASKS_LESS = _frames(["P had R T left .", "now P HAS R T left .", "there were R T left .", "P had only R T left .",
                     "*then there were R T left .", "*only R T were left .", "*that left R T ."])
CHANGES = {D.ADD: ADD_CHANGES, D.SUB: SUB_CHANGES}
COMBINED = {D.ADD: ADD_COMBINED, D.SUB: SUB_COMBINED}
ASKS = {D.ADD: ASKS_ALL + ASKS_MORE, D.MUL: ASKS_ALL + ASKS_MORE, D.SUB: ASKS_ALL + ASKS_LESS}
TWO_STEP_PAIRS = [(D.ADD, D.ADD), (D.SUB, D.ADD), (D.ADD, D.SUB), (D.MUL, D.SUB), (D.MUL, D.ADD), (D.SUB, D.SUB)]
DOUBLES = {(D.ADD, D.ADD): _frames([("P V B more and then C more .", "got found *grabbed")]),
           (D.SUB, D.ADD): _frames([("P ate B and then got C more .", None, "food"), ("*P V B and then found C more .", "lost ate:food")]),
           (D.ADD, D.SUB): _frames(["P got B more but then lost C .", "*P got B more and then gave C to M ."]),
           (D.SUB, D.SUB): _frames([("P ate B and then ate C more .", None, "food")])}
# no-call sentences: fixed frames restating a number (which one), setups with an ending instead of a question, and verbs without numbers
NOCALL_FRAMES = [(f, w) for f, w in zip(_frames(["N counted A T . that is R T .", "N had A T and E Q . P put the R T in the k .",
                                                  "N WAS D years old . P had A T . P loved H R T .",
                                                  "there were A T in the k and E Q on the plate . N took the R T .",
                                                  "N saw A T and E Q . P liked the R Q .", "N had A T . how old WAS P ? P WAS R .",
                                                  "*N had A T and E Q . the R Q were H favorite .",
                                                  "*N looked in the k . there were A T . P took the R T out ."]),
                                         ["A", "A", "A", "A", "E", "D", "E", "A"])]
ENDINGS = {D.ADD: [(f, w) for f, w in zip(_frames(["P put the R new T in the k .", "P WAS happy about the R new T .", "the R new T were nice .",
                                                    "P WAS so happy .", "P had a lot of fun .", "P had a big smile .", "P WAS happy with them .",
                                                    "that was fun .", "*P liked the R new T best .", "*P had a great day ."]),
                                           ["B", "B", "B", None, None, None, None, None, "B", None])],
           D.SUB: [(f, w) for f, w in zip(_frames(["P WAS sad about the R T .", "P missed the R T .", "the R T were gone .", "P WAS sad .",
                                                    "P had to go home .", "P had fun anyway .", "that was sad .",
                                                    "*P WAS sad about the R T but then P smiled .", "*P had a little cry ."]),
                                           ["B", "B", "B", None, None, None, None, "B", None])],
           D.MUL: [(f, w) for f, w in zip(_frames(["P liked the R T in the k .", "P took the K home .", "P WAS happy .", "P had fun with them .",
                                                    "the R T in each k were nice .", "what a lot of T !", "*P picked the R T P liked best ."]),
                                           ["B", None, None, None, "B", None, "B"])]}
NUMBERLESS = _frames(["N had some T . P got a few more . P WAS happy .", "N ate some F . then P went out to play .", "N gave M one of H T . M said thank you .",
                      "there were many T in the k . N took one .", "N had lots of T and M had lots of Q too .", "N found more T . now P had a lot .",
                      "N lost one of H T . P WAS sad .", "N had a few T . P gave some to M . they had fun .", "N and M ate more F . they had a good time .",
                      "N got more T for H birthday . P WAS so happy .", "N picked some T . then P picked more . P had a big pile .",
                      "N had a few T and M had a few too . they had fun .", "*N counted the T . there were a lot .", "*N broke one of the T . P had to say sorry ."])
HELD_OUT_VERBS = sorted({w for fr in (ADD_CHANGES, SUB_CHANGES, ADD_COMBINED, SUB_COMBINED, MUL_FRAMES, *DOUBLES.values()) for f in fr
                         for w, _, h in f["verbs"] if h})
TRAINED_VERBS = sorted({w for fr in (ADD_CHANGES, SUB_CHANGES, ADD_COMBINED, SUB_COMBINED, MUL_FRAMES, *DOUBLES.values()) for f in fr
                        for w, _, h in f["verbs"] if not h})
assert not set(HELD_OUT_VERBS) & set(TRAINED_VERBS), set(HELD_OUT_VERBS) & set(TRAINED_VERBS)


def _all_frames():
    for fr in (OPENINGS, STARTS, ADD_CHANGES, SUB_CHANGES, ADD_COMBINED, SUB_COMBINED, MUL_FRAMES, ASKS_ALL, ASKS_MORE, ASKS_LESS, NUMBERLESS,
               [f for f, _ in NOCALL_FRAMES], *DOUBLES.values(), *[[f for f, _ in e] for e in ENDINGS.values()]):
        yield from fr


GENERATOR_WORDS = sorted({w for f in _all_frames() for w in f["text"].split() + [x[0] for x in f["verbs"]] if w not in PLACEHOLDERS}
                         | {w for lst in OBJECTS.values() for w in lst} | {w for pair in CONTAINERS + GROUPS for w in pair}
                         | {w for o in OTHERS + CONNECTORS + [s for s, _ in SUBJECTS] for w in o.split() if w not in PLACEHOLDERS} | set(BOYS + GIRLS)
                         | set("he she his her him they their them has have was were the kids and".split()))


def _pick(r, items):
    return items[int(r.integers(len(items)))]


def _cast(r, cls):
    """The people and things of one problem."""
    u = r.random()
    if u < 0.88:
        girl = u < 0.44
        n = _pick(r, GIRLS if girl else BOYS)
        used = {n}
    elif u < 0.94:
        n, girl = _pick(r, SUBJECTS); used = set()
    else:
        n1, n2 = _pick(r, BOYS), _pick(r, GIRLS)
        n, girl, used = (f"{n1} and {n2}" if r.random() < 0.7 else "the kids"), None, {n1, n2}
    they = girl is None
    h = "their" if they else "her" if girl else "his"
    m = _pick(r, [x for x in BOYS + GIRLS + OTHERS * 2 if x not in used and x != n]).replace("H", h)
    t = _pick(r, OBJECTS[cls]); q = _pick(r, [x for x in OBJECTS["any"] if x != t]); f = _pick(r, OBJECTS["food"])
    K, k = _pick(r, CONTAINERS); G, G1 = _pick(r, GROUPS)
    return {"N": n, "M": m, "P": "they" if they else "she" if girl else "he", "H": h, "O": "them" if they else "her" if girl else "him",
            "HAS": "have" if they else "has", "WAS": "were" if they else "was", "T": t, "Q": q, "F": f, "K": K, "k": k, "G": G, "G1": G1,
            "D": str(int(r.integers(2, 13))), "E": str(int(r.integers(1, 31)))}


def _expand(ws, fill):
    out = []
    for w in ws:
        out += fill[w].split() if w in fill else [w]
    return out


def _render_frames(v, texts, fill, ans, calls, call):
    """Frame texts (placeholders filled except R) -> (ids, answer_offset, env, cue), as _render; without R the
    sentence has no answer: the offset is the last token."""
    ws = _expand(" ".join(texts).split(), fill)
    if "R" not in ws:
        ids = v.encode(ws)
        return ids, len(ids) - 1, [False] * len(ids), len(ids) - 1
    return _render(v, ws, {"R": str(ans)}, calls, call)


def _fits(pool, held_frame=False, held_verb=False, classes=None, person=None, first=None):
    """The frames of pool (with their usable verbs) that fit: held_frame -> a held-out frame; held_verb -> a frame
    with a held-out verb; classes -> only verbs of those object classes; person False -> only frames that do not
    refer to the subject (or, with first="P", frames starting with P, whose P then becomes N)."""
    out = []
    for f in pool:
        if f["held"] != held_frame or (person is False and f["person"] and not (first == "P" and f["text"].split()[0] == "P")):
            continue
        vs = [x for x in f["verbs"] if x[2] == held_verb and (classes is None or x[1] in classes)]
        if (f["verbs"] and not vs) or (held_verb and not f["verbs"]):
            continue
        out.append((f, vs))
    return out


def _choose(r, pool, held_frame=False, held_verb=False, classes=None, person=None, first=None):
    """One fitting frame from pool with its verb filled in -> (tokens, frame, verb, object class). When no frame
    fits the person constraint the subject is simply introduced by name (its first P becomes N)."""
    cands = _fits(pool, held_frame, held_verb, classes, person, first)
    if not cands:
        cands = _fits(pool, held_frame, held_verb, classes); first = "P"
    assert cands, (held_frame, held_verb, classes, person, first)
    f, vs = _pick(r, cands)
    verb, cls = (None, f["cls"])
    toks = f["text"].split()
    if vs:
        verb, cls, _ = _pick(r, vs)
        toks = [verb if w == f["slot"] else w for w in toks]
    if person is False and first == "P" and "P" in toks:
        toks[toks.index("P")] = "N"
    return toks, f, verb, cls


def gen_word(v, r, split, held=None, call=True, kind=None, pair=None):
    """A one-step word problem from the compositional generator. held: None (trained frames and verbs only),
    "shape" (exactly one held-out frame), "verb" (exactly one held-out verb). Returns (ids, answer_offset, info)
    like word_snippet; info["shape"] names the frames, info["verb"] the verb, info["held"] the held-out piece."""
    kind = _pick(r, D.OPS) if kind is None else kind
    use_open = r.random() < 0.35
    if kind == D.MUL:
        slots = ["frame", "ask"]
    else:
        slots = ["combined", "ask"] if r.random() < 0.3 else ["start", "change", "ask"]
    if use_open:
        slots = ["opening"] + slots
    pools = {"opening": OPENINGS, "start": STARTS, "change": CHANGES.get(kind), "combined": COMBINED.get(kind), "frame": MUL_FRAMES, "ask": ASKS[kind]}
    ho_slot = _pick(r, [s for s in slots if _fits(pools[s], held_frame=True)]) if held == "shape" else None
    verb_slot = _pick(r, [s for s in slots if _fits(pools[s], held_verb=True)]) if held == "verb" else None
    texts, shape, verbs, heldname, person, cls = [], [], [], "", False, "any"
    for s in slots:
        toks, f, verb, vcls = _choose(r, pools[s], held_frame=(s == ho_slot), held_verb=(s == verb_slot),
                                      person=None if person or s in ("opening", "start", "frame", "combined") else False,
                                      first="P" if s == "change" else None)
        if s in ("start", "frame", "combined") and person and toks[0] == "N":
            toks[0] = "P"
        if s == "change" and toks[0] != "then" and r.random() < 0.4:
            toks = _pick(r, CONNECTORS).split() + toks
        if "N" in toks:
            person = True
        if verb is not None:
            verbs.append(verb)
        if vcls != "any":
            cls = vcls
        if s == ho_slot:
            heldname = f["text"]
        if s == verb_slot:
            heldname = verb
        texts.append(" ".join(toks)); shape.append(f["text"])
    while True:
        a, b = pair if pair is not None else D.sample_pair(kind, r)
        if pair is not None or D.is_test(kind, a, b) == (split == "test"):
            break
    ans = D.answer(kind, a, b)
    fill = {**_cast(r, cls), "A": str(a), "B": str(b)}
    calls = [(a, kind, b, ans)]
    ids, off, env, cue = _render_frames(v, texts, fill, ans, calls, call)
    return ids, off, {"kind": kind, "a": a, "b": b, "target": ans, "tid": -1, "calls": calls, "env": env, "cue": cue,
                      "shape": " | ".join(shape), "verb": " ".join(verbs), "held": heldname}


def gen_two(v, r, split, held=None, call=True):
    """A two-step word problem (A op1 B = R1, then R1 op2 C = R) from the compositional generator: a start, two
    changes (the second with a connector) or one double change, and a question; x problems start with a container
    frame. Verbs are limited to things and food so both changes fit the same things."""
    k1, k2 = _pick(r, [p for p in TWO_STEP_PAIRS if p[0] in D.OPS and p[1] in D.OPS])
    if k1 == D.MUL:
        slots = ["frame", "change2", "ask"]
    elif (k1, k2) in DOUBLES and held != "verb" and r.random() < 0.25:
        slots = ["start", "double", "ask"]
    else:
        slots = ["start", "change", "change2", "ask"]
    pools = {"start": STARTS, "change": CHANGES.get(k1), "change2": CHANGES[k2], "double": DOUBLES.get((k1, k2)), "frame": MUL_FRAMES, "ask": ASKS[k2]}
    ho_slot = _pick(r, [s for s in slots if _fits(pools[s], held_frame=True, classes=("any", "food"))]) if held == "shape" else None
    verb_slot = _pick(r, [s for s in slots if _fits(pools[s], held_verb=True, classes=("any", "food"))]) if held == "verb" else None
    texts, shape, verbs, heldname, person, cls = [], [], [], "", False, "any"
    for s in slots:
        toks, f, verb, vcls = _choose(r, pools[s], held_frame=(s == ho_slot), held_verb=(s == verb_slot), classes=("any", "food"),
                                      person=None if person or s in ("start", "frame") else False, first="P" if s in ("change", "change2", "double") else None)
        if s == "change2":
            toks = ["C" if w == "B" else w for w in toks]
            toks = (["and"] if toks[0] == "then" and r.random() < 0.5 else [] if toks[0] == "then" else _pick(r, CONNECTORS + ["and then"]).split()) + toks
        if "N" in toks:
            person = True
        if verb is not None:
            verbs.append(verb)
        if vcls != "any":
            cls = vcls
        if s == ho_slot:
            heldname = f["text"]
        if s == verb_slot:
            heldname = verb
        texts.append(" ".join(toks)); shape.append(f["text"])
    while True:
        a, b = D.sample_pair(k1, r)
        r1 = D.answer(k1, a, b)
        c = int(r.integers(0, D.MAX_RESULT - r1)) if k2 == D.ADD else int(r.integers(0, r1 + 1))
        if D.is_test(k1, a, b) == D.is_test(k2, r1, c) == (split == "test"):
            break
    ans = D.answer(k2, r1, c)
    fill = {**_cast(r, cls), "A": str(a), "B": str(b), "C": str(c)}
    calls = [(a, k1, b, r1), (r1, k2, c, ans)]
    ids, off, env, cue = _render_frames(v, texts, fill, ans, calls, call)
    return ids, off, {"kind": k1, "kind2": k2, "a": a, "b": b, "c": c, "r1": r1, "target": ans, "tid": -1, "calls": calls, "env": env,
                      "cue": cue, "shape": " | ".join(shape), "verb": " ".join(verbs), "held": heldname}


def gen_nocall(v, r, split, held=None, restate_only=False):
    """A sentence with nothing to compute, from the compositional generator: a fixed frame restating one of its
    numbers, a problem-like setup (start + change, or a container frame) with an ending instead of a question
    (restating the second number, or no number at all), or the verbs with no numbers at all. info["target"] is the
    restated number, or -1 when there is none (such sentences are for training only; restate_only leaves them out).
    held: None, "shape" (one held-out frame) or "verb" (one held-out verb, setups only)."""
    u = r.random()
    typ = "fixed" if u < (0.5 if restate_only else 0.3) else "setup" if (restate_only or u < 0.7) else "numberless"
    if held == "verb":
        typ = "setup"
    texts, shape, heldname, kind, person, verbs, cls = [], [], "", 0, False, [], "any"
    if typ == "fixed":
        f, which = _pick(r, [fw for fw in NOCALL_FRAMES if fw[0]["held"] == (held == "shape")])
        texts, shape = [f["text"]], [f["text"]]
        heldname = f["text"] if held == "shape" else ""
        while True:
            a, b = D.sample_pair("nocall", r)
            if D.is_test("nocall", a, b) == (split == "test"):
                break
        fill = {**_cast(r, "any"), "A": str(a), "B": str(b)}
        fill["E"] = str(b)
        target = int(fill[which])
    elif typ == "numberless":
        f = _pick(r, [x for x in NUMBERLESS if x["held"] == (held == "shape")])
        texts, shape = [f["text"]], [f["text"]]
        heldname = f["text"] if held == "shape" else ""
        a = b = 0; target = -1
        fill = _cast(r, "any")
    else:
        kind = _pick(r, D.OPS)
        slots = ["frame", "ending"] if kind == D.MUL else ["combined", "ending"] if r.random() < 0.3 else ["start", "change", "ending"]
        pools = {"start": STARTS, "change": CHANGES.get(kind), "combined": COMBINED.get(kind), "frame": MUL_FRAMES,
                 "ending": [f for f, w in ENDINGS[kind] if w is not None or not restate_only]}
        ho_slot = _pick(r, [s for s in slots if _fits(pools[s], held_frame=True)]) if held == "shape" else None
        verb_slot = _pick(r, [s for s in slots if _fits(pools[s], held_verb=True)]) if held == "verb" else None
        which = None
        for s in slots:
            toks, f, verb, vcls = _choose(r, pools[s], held_frame=(s == ho_slot), held_verb=(s == verb_slot),
                                          person=None if person or s in ("start", "frame", "combined") else False, first="P" if s == "change" else None)
            if s == "change" and toks[0] != "then" and r.random() < 0.4:
                toks = _pick(r, CONNECTORS).split() + toks
            if "N" in toks:
                person = True
            if verb is not None:
                verbs.append(verb)
            if vcls != "any":
                cls = vcls
            if s == ho_slot:
                heldname = f["text"]
            if s == verb_slot:
                heldname = verb
            if s == "ending":
                which = dict((x["text"], w) for x, w in ENDINGS[kind])[f["text"]]
            texts.append(" ".join(toks)); shape.append(f["text"])
        while True:
            a, b = D.sample_pair(kind, r)
            if D.is_test(kind, a, b) == (split == "test"):
                break
        fill = {**_cast(r, cls), "A": str(a), "B": str(b)}
        target = int(fill[which]) if which else -1
    ids, off, env, cue = _render_frames(v, texts, fill, target, [], False)
    return ids, off, {"kind": kind, "a": a, "b": b, "target": target, "tid": -1, "calls": [], "env": env, "cue": cue,
                      "shape": " | ".join(shape), "verb": " ".join(verbs), "held": heldname, "nocall_type": typ}
