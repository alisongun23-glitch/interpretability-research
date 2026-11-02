"""3Blue1Brown-style animations of the 2 -> 2 -> 1 XOR network in xor_by_hand.py.

    python xor_by_hand/xor_by_hand.py     # first: trains and writes history.npz
    python xor_by_hand/animate.py         # writes 1_forward_pass.mp4, 2_backprop_step.mp4, 3_training.mp4

Matplotlib + ffmpeg (manim is not installed here); colours, easing and type follow 3b1b's look.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
BG, WHITE, GREY, DARK = "#0d0d0d", "#ece6e2", "#7d7d7d", "#2b2b2b"
BLUE, RED, YELLOW, GREEN, TEAL, PINK = "#58C4DD", "#FC6255", "#FFFF00", "#83C167", "#5CD0B3", "#D147BD"
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
                     "text.color": WHITE, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
FPS, SIZE, DPI = 30, (12.8, 7.2), 100
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
Y = np.array([0, 1, 1, 0], float)
H = np.load(os.path.join(HERE, "history.npz"))
sig = lambda z: 1 / (1 + np.exp(-z))
smooth = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))          # 3b1b's "smooth" rate function
ramp = lambda t, a, b: smooth((t - a) / (b - a))                             # 0 before a, 1 after b, eased


def params(step):
    """Weights at a (possibly fractional) training step, interpolated for smooth motion."""
    s0 = int(np.floor(step)); s1 = min(s0 + 1, len(H["loss"]) - 1); f = step - s0
    g = lambda k: (1 - f) * H[k][s0] + f * H[k][s1]
    return g("W"), g("b"), g("v"), float(g("c"))


def run(W, b, v, c, x):
    z = x @ W + b; h = sig(z); zo = h @ v + c
    return z, h, zo, sig(zo)


# ---------------- network diagram ----------------
POS = {"x1": (0.14, 0.72), "x2": (0.14, 0.38), "h1": (0.47, 0.76), "h2": (0.47, 0.34), "o": (0.80, 0.55)}
R = 0.036


def edge_color(w):
    return BLUE if w >= 0 else RED


def draw_net(ax, W, b, v, c, act=None, alpha_act=1.0, labels=True, label_alpha=1.0, hi=None, grads=None, grad_alpha=0.0):
    """act: dict node -> value in [0,1] (node fill). hi: set of edges to highlight. grads: dict edge -> value."""
    edges = [("x1", "h1", W[0, 0]), ("x2", "h1", W[1, 0]), ("x1", "h2", W[0, 1]), ("x2", "h2", W[1, 1]),
             ("h1", "o", v[0]), ("h2", "o", v[1])]
    names = {("x1", "h1"): "w_{11}", ("x2", "h1"): "w_{21}", ("x1", "h2"): "w_{12}", ("x2", "h2"): "w_{22}",
             ("h1", "o"): "v_1", ("h2", "o"): "v_2"}
    for a, bb, w in edges:
        (x0, y0), (x1, y1) = POS[a], POS[bb]
        lw = 0.6 + 1.6 * min(abs(w), 8) ** 0.8
        on = hi is None or (a, bb) in hi
        ax.plot([x0, x1], [y0, y1], color=edge_color(w), lw=lw, alpha=0.9 if on else 0.18, solid_capstyle="round", zorder=1)
        if labels:
            t = 0.22 if bb != "o" else 0.5
            mx, my = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            off = 0.035 if y1 >= y0 else -0.035
            ax.text(mx, my + off, rf"${names[(a, bb)]}={w:.2f}$", color=WHITE, fontsize=12, ha="center", va="center",
                    alpha=label_alpha * (1 if on else 0.35), zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.75 * label_alpha))
        if grads is not None and grad_alpha > 0:
            gv = grads[(a, bb)]
            t = 0.76 if bb != "o" else 0.30
            mx, my = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            ax.annotate("", xy=(mx - 0.04 * (x1 - x0), my - 0.04 * (y1 - y0)), xytext=(mx + 0.04 * (x1 - x0), my + 0.04 * (y1 - y0)),
                        arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2, alpha=grad_alpha), zorder=6)
            ax.text(mx, my - 0.045 if y1 >= y0 else my + 0.045, rf"$\partial L/\partial {names[(a, bb)]}={gv:+.3f}$",
                    color=YELLOW, fontsize=11, ha="center", va="center", alpha=grad_alpha, zorder=6)
    bb_ = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ry = R * (bb_.width * fw) / (bb_.height * fh)                # radius in y data units, so nodes stay round
    for node, (x, y) in POS.items():
        val = (act or {}).get(node)
        fc = DARK if val is None else plt.matplotlib.colors.to_hex(np.array(plt.matplotlib.colors.to_rgb(DARK)) * (1 - val * alpha_act)
                                                                 + np.array(plt.matplotlib.colors.to_rgb(WHITE)) * val * alpha_act)
        ax.add_patch(matplotlib.patches.Ellipse((x, y), 2 * R, 2 * ry, fc=fc, ec=WHITE, lw=2, zorder=3))
        lab = {"x1": "x_1", "x2": "x_2", "h1": "h_1", "h2": "h_2", "o": "p"}[node]
        ax.text(x, y + ry + 0.035, rf"${lab}$", color=WHITE, fontsize=17, ha="center", va="center", zorder=4)
        if val is not None and alpha_act > 0.05:
            ax.text(x, y, f"{val:.2f}", color=BG if val > 0.55 else WHITE, fontsize=12, ha="center", va="center",
                    zorder=4, alpha=alpha_act)
    for node, bias, nm in (("h1", b[0], "b_1"), ("h2", b[1], "b_2"), ("o", c, "c")):
        x, y = POS[node]
        ax.text(x, y - ry - 0.035, rf"${nm}={bias:.2f}$", color=GREY, fontsize=11, ha="center", va="center", zorder=4,
                alpha=label_alpha if labels else 0)


def blank(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.clear(); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")


def save(anim, name):
    path = os.path.join(HERE, name)
    anim.save(path, writer=FFMpegWriter(fps=FPS, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"]), dpi=DPI)
    print("wrote", path)


def lx(terms, bias):
    """LaTeX 'a\\cdot x + b\\cdot y - c' with proper signs."""
    out = ""
    for i, (w, x) in enumerate(list(terms) + [(bias, None)]):
        mag = f"{abs(w):.2f}" + (rf"\cdot {x}" if x is not None else "")
        out += (("-" if w < 0 else "") if i == 0 else (" - " if w < 0 else " + ")) + mag
    return out


# ---------------- 1. forward pass ----------------
def forward_scene():
    W, b, v, c = params(0)
    fig = plt.figure(figsize=SIZE); ax = fig.add_axes([0, 0, 1, 1])
    INTRO, PER, OUTRO = 3.0, 6.0, 3.0
    total = INTRO + 4 * PER + OUTRO
    preds = run(W, b, v, c, X)[3]

    def frame(i):
        t = i / FPS
        blank(ax)
        ax.text(0.5, 0.94, "Forward pass: what the random network says", ha="center", fontsize=24, color=WHITE,
                alpha=ramp(t, 0, 0.8))
        draw_net(ax, W, b, v, c, label_alpha=ramp(t, 0.6, 2.0))
        # truth table on the right
        tx, ty = 0.905, 0.80
        ax.text(tx, ty, r"$x_1\;x_2\;\to\;y\quad p$", fontsize=16, ha="center", alpha=ramp(t, 1.0, 2.0))
        done = int((t - INTRO) // PER) if t > INTRO else -1
        for r in range(4):
            shown = r < done or (r == done and (t - INTRO - r * PER) > 4.6)
            txt = rf"${int(X[r,0])}\;\;\;{int(X[r,1])}\;\;\to\;{int(Y[r])}\quad$" + (f" {preds[r]:.2f}" if shown else "  ·")
            ax.text(tx, ty - 0.07 * (r + 1), txt, fontsize=16, ha="center", alpha=ramp(t, 1.0, 2.0),
                    color=YELLOW if r == done else WHITE)
        if INTRO <= t < INTRO + 4 * PER:
            r = int((t - INTRO) // PER); u = t - INTRO - r * PER
            x = X[r]; z, h, zo, p = run(W, b, v, c, x[None])
            z, h, zo, p = z[0], h[0], zo[0], p[0]
            act = {"x1": x[0], "x2": x[1]}
            a1, a2, a3 = ramp(u, 0.0, 0.6), ramp(u, 1.4, 2.2), ramp(u, 3.0, 3.8)
            if u > 1.4: act.update(h1=h[0], h2=h[1])
            if u > 3.0: act.update(o=p)
            hi = {("x1", "h1"), ("x2", "h1"), ("x1", "h2"), ("x2", "h2")} if u < 3.0 else {("h1", "o"), ("h2", "o")}
            blank(ax)
            ax.text(0.5, 0.94, "Forward pass: what the random network says", ha="center", fontsize=24, color=WHITE)
            draw_net(ax, W, b, v, c, act=act, alpha_act=1.0, hi=hi)
            ax.text(tx, ty, r"$x_1\;x_2\;\to\;y\quad p$", fontsize=16, ha="center")
            for rr in range(4):
                shown = rr < r or (rr == r and u > 4.6)
                txt = rf"${int(X[rr,0])}\;\;\;{int(X[rr,1])}\;\;\to\;{int(Y[rr])}\quad$" + (f" {preds[rr]:.2f}" if shown else "  ·")
                ax.text(tx, ty - 0.07 * (rr + 1), txt, fontsize=16, ha="center", color=YELLOW if rr == r else WHITE)
            eq_y = 0.13
            ax.text(0.47, eq_y + 0.03, rf"$z_1 = {lx([(W[0,0], int(x[0])), (W[1,0], int(x[1]))], b[0])} = {z[0]:.2f}"
                    rf"\quad h_1=\sigma(z_1)={h[0]:.2f}$", fontsize=16, ha="center", alpha=a2)
            ax.text(0.47, eq_y - 0.03, rf"$z_2 = {lx([(W[0,1], int(x[0])), (W[1,1], int(x[1]))], b[1])} = {z[1]:.2f}"
                    rf"\quad h_2=\sigma(z_2)={h[1]:.2f}$", fontsize=16, ha="center", alpha=a2)
            ax.text(0.47, eq_y - 0.09, rf"$p = \sigma({lx([(v[0], f'{h[0]:.2f}'), (v[1], f'{h[1]:.2f}')], c)}) = {p:.2f}$",
                    fontsize=16, ha="center", alpha=a3, color=YELLOW)
            ax.text(0.14, 0.13, rf"input $({int(x[0])},{int(x[1])})$, target $y={int(Y[r])}$", fontsize=16, ha="center", alpha=a1)
        if t >= INTRO + 4 * PER:
            ax.text(0.47, 0.10, "Nearly the same answer for every input: it knows nothing yet.", fontsize=18,
                    ha="center", color=YELLOW, alpha=ramp(t, INTRO + 4 * PER, INTRO + 4 * PER + 0.8))
    anim = FuncAnimation(fig, frame, frames=int(total * FPS))
    save(anim, "1_forward_pass.mp4"); plt.close(fig)


# ---------------- 2. one backprop step ----------------
def backprop_scene():
    W, b, v, c = params(0); W1, b1, v1, c1 = params(1)
    z, h, zo, p = run(W, b, v, c, X)
    d_o = (p - Y) / 4
    d_h = np.outer(d_o, v) * h * (1 - h)
    gW = X.T @ d_h; gv = h.T @ d_o
    grads = {("x1", "h1"): gW[0, 0], ("x2", "h1"): gW[1, 0], ("x1", "h2"): gW[0, 1], ("x2", "h2"): gW[1, 1],
             ("h1", "o"): gv[0], ("h2", "o"): gv[1]}
    fig = plt.figure(figsize=SIZE); ax = fig.add_axes([0, 0, 1, 1])
    total = 16.0

    def frame(i):
        t = i / FPS
        blank(ax)
        ax.text(0.5, 0.94, "Backward pass: which way should each weight move?", ha="center", fontsize=24, alpha=ramp(t, 0, 0.8))
        s = ramp(t, 10.0, 13.0)                                   # tween old -> new weights
        Wt, bt, vt, ct = (1 - s) * W + s * W1, (1 - s) * b + s * b1, (1 - s) * v + s * v1, (1 - s) * c + s * c1
        ga = ramp(t, 3.0, 4.0) * (1 - ramp(t, 9.0, 10.0))
        hi = {("h1", "o"), ("h2", "o")} if 3.0 <= t < 5.5 else None
        draw_net(ax, Wt, bt, vt, ct, grads=grads, grad_alpha=ga, hi=hi)
        ax.text(0.5, 0.17, r"output error signal  $\delta_o = (p - y)/4$  for each input:", fontsize=17, ha="center",
                alpha=ramp(t, 0.8, 1.6) * (1 - ramp(t, 9.0, 10.0)))
        ax.text(0.5, 0.11, "   ".join(f"{d:+.3f}" for d in d_o), fontsize=17, ha="center", color=YELLOW,
                alpha=ramp(t, 1.2, 2.0) * (1 - ramp(t, 9.0, 10.0)))
        ax.text(0.5, 0.05, r"$\partial L/\partial w = \sum_{\mathrm{inputs}} (\mathrm{error\ signal\ at\ the\ end\ of\ } w)\times(\mathrm{value\ at\ its\ start})$",
                fontsize=15, ha="center", color=WHITE, alpha=ramp(t, 4.5, 5.5) * (1 - ramp(t, 9.0, 10.0)))
        ax.text(0.5, 0.11, r"update:  $w \leftarrow w - \eta\,\partial L/\partial w$,  with $\eta = 1$", fontsize=19, ha="center",
                alpha=ramp(t, 10.0, 10.8))
        L0, L1 = H["loss"][0], H["loss"][1]
        ax.text(0.5, 0.05, rf"loss  ${L0:.4f} \;\to\; {L1:.4f}$", fontsize=19, ha="center", color=YELLOW, alpha=ramp(t, 13.0, 13.8))
    anim = FuncAnimation(fig, frame, frames=int(total * FPS))
    save(anim, "2_backprop_step.mp4"); plt.close(fig)


# ---------------- 3. training: the space bends ----------------
def training_scene():
    T = len(H["loss"]) - 1
    TRAIN, HOLD = 26.0, 9.0
    total = TRAIN + HOLD
    fig = plt.figure(figsize=SIZE)
    axI = fig.add_axes([0.04, 0.30, 0.34, 0.56]); axH = fig.add_axes([0.45, 0.30, 0.34, 0.56])
    axL = fig.add_axes([0.10, 0.07, 0.69, 0.14]); axN = fig.add_axes([0.80, 0.30, 0.20, 0.56])
    g = np.linspace(-0.25, 1.25, 140); GX, GY = np.meshgrid(g, g); GP = np.stack([GX.ravel(), GY.ravel()], 1)
    cmap = LinearSegmentedColormap.from_list("b3", [BG, "#1d3d4a", BLUE, "#d6f2fa"])
    ptcol = [BLUE if y == 0 else YELLOW for y in Y]

    def step_of(t):
        u = np.clip(t / TRAIN, 0, 1)
        return T * (u ** 1.6)                                     # slow at the start, where most change happens

    def frame(i):
        t = i / FPS; s = step_of(t)
        W, b, v, c = params(s)
        _, hX, _, pX = run(W, b, v, c, X)
        _, _, _, pG = run(W, b, v, c, GP)
        # input space
        axI.clear(); axI.set_xlim(-0.25, 1.25); axI.set_ylim(-0.25, 1.25); axI.set_aspect("equal")
        axI.imshow(pG.reshape(GX.shape), extent=(-0.25, 1.25, -0.25, 1.25), origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.9)
        for j, col in ((0, TEAL), (1, PINK)):                     # where each hidden neuron switches: w·x + b = 0
            w1, w2, bb = W[0, j], W[1, j], b[j]
            if abs(w2) > 1e-6:
                xs = np.array([-0.25, 1.25]); axI.plot(xs, -(w1 * xs + bb) / w2, color=col, lw=2.2)
        axI.scatter(X[:, 0], X[:, 1], s=170, c=ptcol, edgecolors=BG, linewidths=2, zorder=5)
        for k in range(4):
            axI.text(X[k, 0], X[k, 1] - 0.13, f"{pX[k]:.2f}", color=WHITE, fontsize=12, ha="center")
        axI.set_title(r"input space $(x_1, x_2)$", color=WHITE, fontsize=17)
        for sp in axI.spines.values(): sp.set_color(GREY)
        axI.tick_params(colors=GREY, labelsize=9)
        # hidden space
        axH.clear(); axH.set_xlim(-0.05, 1.05); axH.set_ylim(-0.05, 1.05); axH.set_aspect("equal")
        hg = np.linspace(-0.05, 1.05, 120); HX, HY = np.meshgrid(hg, hg)
        axH.imshow(sig(v[0] * HX + v[1] * HY + c), extent=(-0.05, 1.05, -0.05, 1.05), origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=0.9)
        if abs(v[1]) > 1e-6:
            xs = np.array([-0.05, 1.05]); axH.plot(xs, -(v[0] * xs + c) / v[1], color=WHITE, lw=2.2, ls=(0, (4, 3)))
        axH.scatter(hX[:, 0], hX[:, 1], s=170, c=ptcol, edgecolors=BG, linewidths=2, zorder=5)
        for k in range(4):
            axH.text(hX[k, 0] + 0.03, hX[k, 1] + 0.05 + (0.06 if k == 2 else 0), f"({int(X[k,0])},{int(X[k,1])})", color=WHITE, fontsize=11)
        axH.set_title(r"hidden space $(h_1, h_2)$: the output draws one line", color=WHITE, fontsize=17)
        axH.set_xlabel(r"$h_1$", color=TEAL, fontsize=16); axH.set_ylabel(r"$h_2$", color=PINK, fontsize=16)
        for sp in axH.spines.values(): sp.set_color(GREY)
        axH.tick_params(colors=GREY, labelsize=9)
        # loss
        axL.clear(); axL.set_facecolor(BG)
        st = np.arange(0, int(s) + 1)
        axL.plot(st, H["loss"][st], color=YELLOW, lw=2.2); axL.set_xlim(0, T); axL.set_ylim(0, 1.0)
        axL.set_ylabel("loss", color=GREY, fontsize=13); axL.tick_params(colors=GREY, labelsize=9)
        for sp in ("top", "right"): axL.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): axL.spines[sp].set_color(GREY)
        axL.text(T, 0.85, f"step {int(s)}   loss {H['loss'][int(s)]:.3f}", color=WHITE, fontsize=14, ha="right")
        # mini network
        blank(axN); draw_net(axN, W, b, v, c, act={"x1": None, "x2": None}, labels=False)
        a = ramp(t, TRAIN + 0.5, TRAIN + 2.0)
        if a > 0:
            axI.text(0.5, 1.13, r"teal: $h_1$ = AND      pink: $h_2$ = NOR", color=WHITE, fontsize=14, ha="center", alpha=a)
            axH.text(0.5, 1.0, r"$p$ = NOR$(h_1, h_2)$", color=YELLOW, fontsize=16, ha="center", alpha=a)
            fig.texts.clear()
            fig.text(0.5, 0.95, r"XOR = NOR( AND, NOR ):  one or the other, not both, not neither", ha="center",
                     fontsize=20, color=YELLOW, alpha=ramp(t, TRAIN + 2.5, TRAIN + 3.5))
        else:
            fig.texts.clear()
            fig.text(0.5, 0.95, "Training: the hidden layer bends the space until one line can separate XOR",
                     ha="center", fontsize=20, color=WHITE)
    anim = FuncAnimation(fig, frame, frames=int(total * FPS))
    save(anim, "3_training.mp4"); plt.close(fig)


if __name__ == "__main__":
    forward_scene()
    backprop_scene()
    training_scene()
