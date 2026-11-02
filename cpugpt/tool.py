"""The gate computer as a tool the story GPT uses in writing.

The model writes "<calc> 12 + 5 =" into the story; decoding pauses, the computer answers "17 </calc>"
into the text, decoding resumes and the model copies 17 into "now he has 17 apples". The same
state machine runs on the page (storytool_ui/index.html).

Decoding rules (GSM8K's calculator shape): "</calc>" is never sampled, "<calc>" is not sampled while
a call is open or after max_calls; on "=" inside an open call, if the span is exactly NUMBER OP
NUMBER and the machine answers, the result and "</calc>" are appended; otherwise the model's next
token is taken as its own result (the machine is off, or the call was malformed) and "</calc>" is
appended; a call that runs 5 tokens without "=" is closed. Inside a call decoding is greedy. A row
stops at its first number written outside a call (its answer). An optional margin on "<calc>" (calc_margin)
is taken off its logit before the choice: a call is opened only when "<calc>" beats the best word by that much
(a negative margin is a bias toward calling, as in ToolkenGPT).
"""
import numpy as np
import torch

from . import data as D
from .text import NUM0

OPMAP_WRONG = np.array([0, 2, 3, 1])     # the ablation: every opcode is swapped for another


class Calculator:
    """Answers a call from the table of the gate computer's results (cpu_table.npz). mode: "on", "off"
    (never answers: the model writes its own result), "wrong" (opcodes swapped)."""

    def __init__(self, table, v, mode="on"):
        self.table, self.mode, self.num0 = np.asarray(table), mode, NUM0
        self.sym = {v.stoi[s]: op for op, s in ((D.ADD, "+"), (D.SUB, "-"), (D.MUL, "*"))}
        self.eq, self.calc, self.end = v.stoi["="], v.CALC, v.END
        self.unsure = getattr(v, "UNSURE", None)
        self.overflow = 0

    def parse(self, span):
        """Token ids between <calc> and = -> (a, opcode, b), or None unless exactly NUMBER OP NUMBER."""
        if len(span) != 3:
            return None
        a, o, b = (int(t) for t in span)
        if not (self.num0 <= a < self.num0 + 256 and self.num0 <= b < self.num0 + 256) or o not in self.sym:
            return None
        return a - self.num0, self.sym[o], b - self.num0

    def __call__(self, a, op, b):
        """The machine's answer (0-255, wrapping like the 8-bit computer), or None when it is off."""
        if self.mode == "off":
            return None
        if not 0 <= D.answer(op, a, b) <= 255:
            self.overflow += 1
        return int(self.table[OPMAP_WRONG[op] if self.mode == "wrong" else op, a, b])


def decode(model, rows, ptr, calc, n_new=12, max_calls=2, temperature=0.0, seed=0, chunk=250, stop_at_number=True, calc_margin=0.0):
    """rows (N, T): left-aligned token windows with room to grow; ptr (N,): index of each row's last prompt
    token. Writes up to n_new model tokens per row (the machine's tokens do not count), with the machine
    answering calls; calc_margin is taken off the "<calc>" logit (see above). Returns one dict per row: answer
    (the first number written outside a call, or None),
    calls [{a, op, b, r, ok (well-formed), by ("machine" or "model"), wrong, unsure, err}], written (every appended
    token id), n_model, malformed (calls that did not parse or ran past 5 tokens). A machine that can be wrong
    (cpugpt.check.FlakyMachine) reports each answer in calc.last: wrong (the answer is not the computer's), err (what
    went wrong) and unsure (its doubt, written into the text as "<unsure>" before "</calc>"); a model's retry is
    simply another call."""
    g = torch.Generator().manual_seed(seed)
    rows, ptr = rows.clone(), ptr.clone()
    N, T = rows.shape
    num0 = calc.num0
    out = [{"answer": None, "calls": [], "written": [], "n_model": 0, "malformed": 0} for _ in range(N)]
    st = [{"open": False, "span": [], "pending": False, "calls": 0, "done": False} for _ in range(N)]

    def put(i, tok):
        if ptr[i] + 1 >= T:
            st[i]["done"] = True; return False
        ptr[i] += 1; rows[i, ptr[i]] = tok; out[i]["written"].append(int(tok)); return True

    model.eval()
    with torch.no_grad():
        for _ in range(n_new):
            active = [i for i in range(N) if not st[i]["done"]]
            if not active:
                break
            for s in range(0, len(active), chunk):
                idx = active[s:s + chunk]
                lg = model(rows[idx])["logits"][torch.arange(len(idx)), ptr[idx], 0].clone()      # (n, V)
                lg[:, calc.end] = float("-inf")
                opn = torch.tensor([st[i]["open"] or st[i]["calls"] >= max_calls for i in idx])
                lg[opn, calc.calc] = float("-inf")
                if calc_margin:
                    lg[:, calc.calc] -= calc_margin
                greedy = lg.argmax(-1)
                if temperature > 0:
                    top = torch.topk(lg, 20)
                    pick = top.indices.gather(1, torch.multinomial(torch.softmax(top.values / temperature, -1), 1, generator=g))[:, 0]
                    inside = torch.tensor([st[i]["open"] for i in idx])
                    nxt = torch.where(inside, greedy, pick)
                else:
                    nxt = greedy
                for j, i in enumerate(idx):
                    t, S, o = int(nxt[j]), st[i], out[i]
                    if not put(i, t):
                        continue
                    o["n_model"] += 1
                    if S["open"]:
                        if S["pending"]:                 # the model wrote its own result; the machine closes
                            c = o["calls"][-1]
                            c["r"] = t - num0 if num0 <= t < num0 + 256 else None
                            S["pending"] = False; S["open"] = False; S["span"] = []
                            put(i, calc.end)
                        else:
                            S["span"].append(t)
                            if t == calc.eq:
                                p = calc.parse(S["span"][:-1])
                                r = calc(*p) if p else None
                                last = getattr(calc, "last", None) if r is not None else None
                                c = {"a": p[0] if p else None, "op": p[1] if p else None, "b": p[2] if p else None,
                                     "r": r, "ok": p is not None, "by": "machine" if r is not None else "model",
                                     "wrong": bool(last and last["wrong"]), "unsure": bool(last and last["unsure"]), "err": last["err"] if last else None,
                                     "bit": last.get("bit") if last else None}
                                o["calls"].append(c); S["calls"] += 1
                                if p is None:
                                    o["malformed"] += 1
                                if r is not None:
                                    put(i, num0 + r)
                                    if c["unsure"]:
                                        put(i, calc.unsure)
                                    put(i, calc.end); S["open"] = False; S["span"] = []
                                else:
                                    S["pending"] = True
                            elif len(S["span"]) >= 5:    # no "=": close the call
                                o["calls"].append({"a": None, "op": None, "b": None, "r": None, "ok": False, "by": "none", "wrong": False, "unsure": False, "err": None, "bit": None})
                                o["malformed"] += 1; S["calls"] += 1; S["open"] = False; S["span"] = []
                                put(i, calc.end)
                    elif t == calc.calc:
                        S["open"] = True; S["span"] = []
                    elif num0 <= t < num0 + 256 and stop_at_number:
                        o["answer"] = t - num0; S["done"] = True
                    if o["n_model"] >= n_new:
                        S["done"] = True
    model.train()
    return out


def call_right(c, a, op, b):
    """The call sent a op b to the machine (operands may be swapped for + and x)."""
    if not c["ok"] or c["op"] != op:
        return False
    return (c["a"], c["b"]) == (a, b) or (op != D.SUB and (c["a"], c["b"]) == (b, a))
