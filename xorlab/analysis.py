"""Interpretability and causal tests for trained XOR networks.

Definitions used throughout (all over the 4 truth-table inputs, uniform weight):
  activation pattern  h_i(x)          hidden unit i on each input
  contribution        c_i(x) = w2_i * h_i(x)    what unit i adds to the output logit
  circuit C           a set of hidden units
  mean ablation       replace h_i(x) by its mean over the 4 inputs (keeps the unit's
                      average effect, removes its input-dependence)
  sufficient circuit  keeping C and mean-ablating every other unit still gives XOR
  minimal circuit     sufficient, and no proper subset is sufficient
"""
from itertools import combinations
import numpy as np
from .model import X, Y, hidden_acts, act_fn

# The 14 non-constant Boolean functions of (A, B), as values on inputs 00, 01, 10, 11.
BOOL = {
    "A": [0, 0, 1, 1], "B": [0, 1, 0, 1], "NOT A": [1, 1, 0, 0], "NOT B": [1, 0, 1, 0],
    "AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "NAND": [1, 1, 1, 0], "NOR": [1, 0, 0, 0],
    "A AND NOT B": [0, 0, 1, 0], "B AND NOT A": [0, 1, 0, 0],
    "A OR NOT B": [1, 0, 1, 1], "B OR NOT A": [1, 1, 0, 1],
    "XOR": [0, 1, 1, 0], "XNOR": [1, 0, 0, 1],
}
BOOL = {k: np.array(v, float) for k, v in BOOL.items()}


def bool_label(vec, threshold=0.9):
    """Best-matching Boolean function by correlation (positive direction only).

    Returns (label, correlation). Constant vectors are 'constant'. Matches below the
    threshold are reported as 'mixed(best)' so no gate reading is forced.
    """
    vec = np.asarray(vec, float)
    if np.std(vec) < 1e-9 * (1 + np.abs(vec).max()):
        return "constant", 1.0
    best, bc = None, -2.0
    for k, pat in BOOL.items():
        c = np.corrcoef(vec, pat)[0, 1]
        if c > bc:
            best, bc = k, c
    return (best if bc >= threshold else f"mixed({best})"), float(bc)


# ---------------- forward passes with interventions ----------------
def forward(p, act, ablate=(), mode="mean", patch=None):
    """Logits on the 4 inputs with optional interventions.

    ablate: units to ablate ("mean" -> mean over inputs, "zero" -> 0)
    patch:  (units, source_index) - replace these units' activations on every input
            with their activation on the source input
    """
    H = hidden_acts(p, act).copy()
    ablate = list(ablate)
    if ablate:
        H[:, ablate] = H[:, ablate].mean(0) if mode == "mean" else 0.0
    if patch is not None:
        units, src = patch
        units = list(units)
        H[:, units] = H[src, units]
    return H @ p["w2"] + p["b2"]


def correct(lg):
    return np.where(Y == 1, lg, -lg) > 0


def contribution(p, act, units):
    H = hidden_acts(p, act)
    u = list(units)
    return H[:, u] @ p["w2"][u]


def rest(p, act, units):
    """Recipient side of a transplant: output bias + every unit outside the circuit."""
    H = hidden_acts(p, act)
    out = [i for i in range(H.shape[1]) if i not in set(units)]
    return p["b2"] + (H[:, out] @ p["w2"][out] if out else 0.0)


# ---------------- circuit search ----------------
def minimal_circuits(p, act, mode="mean"):
    """Exhaustively enumerate minimal sufficient circuits (2^H subsets)."""
    Hn = p["w2"].shape[0]
    units = range(Hn)
    sufficient = {}
    for k in range(Hn + 1):
        for C in combinations(units, k):
            others = [i for i in units if i not in C]
            lg = forward(p, act, ablate=others, mode=mode)
            sufficient[C] = (bool(np.all(correct(lg))), float(np.min(np.where(Y == 1, lg, -lg))))
    minimal = []
    for C, (ok, mg) in sufficient.items():
        if not ok:
            continue
        if any(sufficient[S][0] for r in range(len(C)) for S in combinations(C, r)):
            continue
        minimal.append({"units": list(C), "margin": mg})
    minimal.sort(key=lambda c: (len(c["units"]), -c["margin"]))
    return minimal


def describe_units(p, act, units, threshold):
    H = hidden_acts(p, act)
    out = []
    for i in units:
        c = H[:, i] * p["w2"][i]
        lab_h, cor_h = bool_label(H[:, i], threshold)
        lab_c, cor_c = bool_label(c, threshold)
        out.append({"unit": int(i), "activation": H[:, i].tolist(), "activation_gate": lab_h,
                    "activation_purity": cor_h, "w_out": float(p["w2"][i]),
                    "contribution": c.tolist(), "contribution_gate": lab_c,
                    "contribution_purity": cor_c})
    return out


def signature(desc):
    """Circuit type = sorted list of what each unit pushes the output toward.

    Using contributions (not raw activations) merges equivalent readings, e.g. a
    unit computing AND with a negative output weight contributes the NAND pattern.
    """
    return " + ".join(sorted(d["contribution_gate"] for d in desc)) if desc else "(empty)"


# ---------------- causal tests ----------------
def causal_tests(p, act, C, mode="mean"):
    Hn = p["w2"].shape[0]
    allu = list(range(Hn))
    other = [i for i in allu if i not in C]
    res = {}
    res["intact_acc"] = float(np.mean(correct(forward(p, act))))
    lg_abl = forward(p, act, ablate=C, mode=mode)
    res["ablate_circuit_acc"] = float(np.mean(correct(lg_abl)))
    res["ablate_circuit_zero_acc"] = float(np.mean(correct(forward(p, act, ablate=C, mode="zero"))))
    res["ablate_circuit_logits"] = lg_abl.tolist()
    # size-matched random controls: every other subset of the same size
    ctrl = [list(S) for S in combinations(allu, len(C)) if set(S) != set(C)]
    res["ablate_random_acc"] = (float(np.mean([np.mean(correct(forward(p, act, ablate=S, mode=mode)))
                                               for S in ctrl])) if ctrl else None)
    res["ablate_complement_acc"] = float(np.mean(correct(forward(p, act, ablate=other, mode=mode))))
    res["single_unit_breaks"] = [int(i) for i in allu
                                 if not np.all(correct(forward(p, act, ablate=[i], mode=mode)))]
    # activation patching: put the circuit's activations from a source input into a
    # clean input with the opposite XOR label; does the output follow the source?
    pairs = [(c, s) for c in range(4) for s in range(4) if Y[c] != Y[s]]

    def flip_rate(units):
        if not units:
            return None
        hits = 0
        for c, s in pairs:
            lg = forward(p, act, patch=(units, s))
            hits += int((lg[c] > 0) == (Y[s] == 1))
        return hits / len(pairs)
    res["patch_circuit_flip"] = flip_rate(C)
    res["patch_complement_flip"] = flip_rate(other)
    res["patch_random_flip"] = (float(np.mean([flip_rate(S) for S in ctrl])) if ctrl else None)
    return res


# ---------------- transplants ----------------
def moment_match(c_donor, c_recipient):
    """Rescale donor contribution to the recipient circuit's mean and spread.
    Uses only the recipient circuit's own activity statistics, never the labels."""
    sd = np.std(c_donor)
    if sd < 1e-12:
        return np.full_like(c_donor, np.mean(c_recipient))
    return (c_donor - np.mean(c_donor)) / sd * np.std(c_recipient) + np.mean(c_recipient)


def transplant(rec, rec_act, rec_C, donor_contrib):
    """Recipient logits when its circuit is replaced by a donor contribution."""
    c_r = contribution(rec, rec_act, rec_C)
    base = rest(rec, rec_act, rec_C)
    direct = base + donor_contrib
    aligned = base + moment_match(donor_contrib, c_r)
    native = base + c_r
    ablated = base + np.mean(c_r)
    return {"native": native, "direct": direct, "aligned": aligned, "ablated": ablated}


def reference_contribution(act, scale):
    """Analytically constructed XOR circuits (NOT learned):
       tanh: OR unit and AND unit, output = OR - AND
       relu: relu(A+B) - 2*relu(A+B-1)
    Both give 0,1,1,0 up to saturation; multiplied by `scale` logits."""
    s = X.sum(1)
    if act == "tanh":
        h_or, h_and = np.tanh(10 * (s - 0.5)), np.tanh(10 * (s - 1.5))
        return scale * (h_or - h_and) / 2.0
    return scale * (np.maximum(s, 0) - 2 * np.maximum(s - 1, 0))


# ---------------- representational similarity ----------------
def linear_cka(A, B):
    """Linear CKA between two representations of the same 4 inputs.
    Invariant to neuron order, rotation and overall scale. With only 4 inputs the
    centred representations live in at most 3 dimensions, so CKA is coarse and
    random networks already score fairly high: always compare with the baseline."""
    A = A - A.mean(0); B = B - B.mean(0)
    num = np.linalg.norm(A.T @ B) ** 2
    den = np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)
    return float(num / den) if den > 1e-15 else float("nan")


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
