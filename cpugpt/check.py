"""A computer that can be wrong, for the story GPT that has to check it (run_storycheck.py).

FlakyMachine wraps the gate computer's answer table (cpugpt.tool.Calculator). With probability p_err a call
gets a wrong answer, of one of four kinds drawn equally: the right answer with one bit flipped ("bit"), the
answer of another operation ("op", a control slip), a random byte ("random", the garbage of a slipped jump) or 0
("zero", the timeout of a run that never prints), the last two being what the analog neural CPU mostly does
under noise. With signal=True the machine also says "<unsure>" after an answer it doubts: with
probability `hit` after a wrong one and `false_alarm` after a right one (the rates of the fully analog neural
CPU's worst-signal margin under noise, measured in run_storycheck.py).

The same object writes the call blocks of the training stream (blocks(), used by cpugpt.text._render) and
answers the model's calls at decoding time (__call__, used by cpugpt.tool.decode, which reads .last). In the
training text a wrong answer is followed by the model's retry, "<calc> a op b =" again, up to max_tries blocks,
the last one right: that is the behaviour the model is trained to copy. With policy="signal" the text retries
after a doubted answer instead (whether or not it was wrong), the rule a model that only reads the signal
would follow.
"""
import numpy as np

from . import data as D
from .text import NUM0, OPSYM
from .tool import Calculator

ERRS = ("bit", "op", "random", "zero")


class FlakyMachine(Calculator):
    def __init__(self, table, v, p_err=0.25, signal=False, hit=0.8, false_alarm=0.1, errs=ERRS, max_tries=3, seed=0, policy="oracle"):
        super().__init__(table, v, "on")
        self.p_err, self.signal, self.hit, self.false_alarm, self.errs = p_err, signal, hit, false_alarm, tuple(errs)
        self.max_tries, self.policy = max_tries, policy
        self.r = np.random.default_rng(seed)
        self.last = None                           # the last answer: {r, right, wrong, err, bit, unsure}
        self.n_calls = self.n_wrong = self.n_unsure = 0

    def draw(self, a, op, b, force_right=False):
        """One answer of the machine for a op b: a dict with r (what it says), right (what the computer would
        say), wrong, err ("bit" or "op" or None), bit (the flipped bit) and unsure (its doubt)."""
        right = int(self.table[op, a, b])
        r, err, bit = right, None, None
        if not force_right and self.r.random() < self.p_err:
            err = self.errs[int(self.r.integers(len(self.errs)))]
            if err == "op":
                other = int(self.r.choice([o for o in (D.ADD, D.SUB, D.MUL) if o != op]))
                r = int(self.table[other, a, b])
            elif err == "random":
                r = int(self.r.integers(256))
            elif err == "zero":
                r = 0
            if err == "bit" or r == right:             # the other kinds can agree with the right answer (2 + 2 = 2 x 2, 0 - 0 = 0): flip a bit instead
                err, bit = "bit", int(self.r.integers(8)); r = right ^ (1 << bit)
        wrong = r != right
        unsure = bool(self.signal and self.r.random() < (self.hit if wrong else self.false_alarm))
        return {"r": r, "right": right, "wrong": wrong, "err": err, "bit": bit, "unsure": unsure}

    def __call__(self, a, op, b):
        """The machine's answer to the model's call at decoding time (cpugpt.tool.decode reads .last)."""
        self.last = d = self.draw(a, op, b)
        self.n_calls += 1; self.n_wrong += d["wrong"]; self.n_unsure += d["unsure"]
        return d["r"]

    def block(self, v, a, op, b, d):
        """One call block "<calc> a op b = r [<unsure>] </calc>" as ids, with the machine's tokens flagged."""
        ids = [v.CALC] + v.encode([str(a), OPSYM[op], str(b), "="]) + [NUM0 + d["r"]] + ([v.UNSURE] if d["unsure"] else []) + [v.END]
        return ids, [False] * 5 + [True] * (len(ids) - 5)

    def blocks(self, v, a, op, b, right):
        """The block(s) of one call in the training text: a retry follows a wrong (policy "oracle") or a doubted
        (policy "signal") answer; the last of max_tries blocks is right. Returns (ids, env flags)."""
        ids, env = [], []
        for t in range(self.max_tries):
            d = self.draw(a, op, b, force_right=t == self.max_tries - 1)
            assert d["right"] == right, (a, op, b, right, d)
            i, e = self.block(v, a, op, b, d); ids += i; env += e
            if not (d["wrong"] if self.policy == "oracle" else d["unsure"]):
                break
        return ids, env
