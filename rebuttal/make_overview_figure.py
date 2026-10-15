"""
Overview figure: the items Infogain, CMA-ES, and CMA-ES-IG actually show a user.

    python3 make_overview_figure.py

Each panel is a real simulated query, not a sketch.  The algorithms run exactly as in the
user study (K = 3 items per query, Luce-Shepard rationality 10, the plain 10,000-sample
belief, CMA-ES population 3) in a d = 3 feature space against one simulated user with true
preference omega*.  Each panel shows that algorithm's 2nd query (after 1 ranking): its
items, its own estimate omega-hat, the ring of directions orthogonal to omega-hat, and for
the CMA-ES methods 500 samples of its current search distribution.  Items and samples live
in the box [-1, 1]^3; they are drawn as directions on the unit sphere, since query quality
is the cosine between an item and omega*.

Examples (40 simulated users, one seed each):
  (a) Infogain   -- the user with the clearest contrast: widest item spread, lowest quality
  (b), (c)       -- the user closest to the median spread and quality for that method
The selection and the medians are printed when the script runs.

Writes updated_text_and_figures/figures/figure1_overview.{pdf,png,svg}.  The photo panels
are placeholders (dashed boxes) to be replaced with the real-world photos.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
OUT = os.path.join(HERE, 'updated_text_and_figures', 'figures')
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'preference-learning-from-selection'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from irlpreference.input_models import LuceShepardChoice
from irlpreference.query_generation import InfoGainQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from cmaes_generators import CMAESGenerator, CMAESIGGenerator

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

C_TRUE = '#2b59c3'      # user's true preference
C_EST = '#c8382d'       # robot's estimate (and the ring orthogonal to it)
C_CLOUD = '#8f7fc8'     # CMA-ES search distribution
C_XI = ['#2fa4e7', '#f0892a', '#3a9d5d']   # xi_1..xi_3
C_GOOD, C_BAD = '#1f6fb2', '#b8322a'
INK, MUTED = '#222222', '#6b6b6b'

D, K, BETA, SHOWN_QUERY, N_USERS = 3, 3, 10, 2, 40

# ------------------------------------------------------------------ projection
EL, AZ = np.deg2rad(18), np.deg2rad(-58)
E_U = np.array([-np.sin(AZ), np.cos(AZ), 0.0])
E_V = np.array([-np.sin(EL) * np.cos(AZ), -np.sin(EL) * np.sin(AZ), np.cos(EL)])
E_D = np.cross(E_U, E_V)                       # towards the viewer


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def proj(p):
    p = np.atleast_2d(p)
    return p @ E_U, p @ E_V, p @ E_D


def from_screen(u, v):
    """Unit vector on the visible hemisphere that projects to screen position (u, v)."""
    return u * E_U + v * E_V + np.sqrt(max(0.0, 1 - u * u - v * v)) * E_D


OMEGA = from_screen(-0.12, 0.80)               # the simulated user's true preference


# ------------------------------------------------------------------ simulation
def simulate(method, seed):
    """Run `method` for SHOWN_QUERY - 1 rankings, then return its next query and state."""
    np.random.seed(seed)                       # the library draws its samples from np.random
    rng = np.random.default_rng(seed)
    cm = LuceShepardChoice(rationality=BETA)
    belief = MonteCarloLinearReward(D, number_samples=10_000)
    limits = [(-1, 1)] * D
    gen = {'ig': lambda: InfoGainQueryGenerator(limits),
           'cmaes': lambda: CMAESGenerator(D, limits, K),
           'cmaesig': lambda: CMAESIGGenerator(D, limits, K)}[method]()
    for t in range(SHOWN_QUERY):
        query = np.array(gen.get_query(K, belief, cm))
        if t == SHOWN_QUERY - 1:
            break
        working, ranked = query.copy(), []
        for _ in range(K - 1):
            p = cm.get_choice_probabilities(working, OMEGA[None]).flatten()
            c = rng.choice(len(working), p=p)
            cm.tell_input(c, working)
            belief.update(cm.get_probability_of_input)
            ranked.append(working[c])
            working = np.delete(working, c, axis=0)
        ranked.append(working[0])
        if method != 'ig':
            gen.tell(ranked, list(range(K))[::-1])
    cloud = None
    if method != 'ig':
        cloud = np.array([np.clip(gen.optimizer.ask(), -1, 1) for _ in range(500)])
    items = query
    directions = unit(items)
    omega_hat = unit(belief.get_expectation())
    spread = np.mean([np.linalg.norm(items[i] - items[j]) for i in range(K) for j in range(i + 1, K)])
    return dict(items=directions, omega_hat=omega_hat, cloud=cloud,
                quality=float((directions @ OMEGA).mean()), spread=float(spread),
                mi=float(mutual_information(items, belief)),
                error=float(np.degrees(np.arccos(np.clip(omega_hat @ OMEGA, -1, 1)))))


def mutual_information(query, belief):
    """Bits of information the user's first choice carries about omega under the belief."""
    lw = belief.hypothesis_log_probabilities
    pi = np.exp(lw - lw.max())
    pi /= pi.sum()
    r = BETA * (np.asarray(query) @ belief.hypothesis_samples.T)
    p = np.exp(r - r.max(0))
    p /= p.sum(0)
    entropy = lambda q: -(q * np.log2(np.clip(q, 1e-300, None))).sum(0)
    return entropy(p @ pi) - entropy(p) @ pi


def choose_examples():
    """Among users whose estimate is clearly off (error at or above the 60th percentile, so
    omega* and omega-hat are visibly apart), take the clearest Infogain example and the most
    typical CMA-ES / CMA-ES-IG examples (closest to the medians of information and quality)."""
    states = {m: [simulate(m, s) for s in range(N_USERS)] for m in ('ig', 'cmaes', 'cmaesig')}
    chosen = {}
    for m, runs in states.items():
        q, mi, sp, err = (np.array([r[k] for r in runs]) for k in ('quality', 'mi', 'spread', 'error'))
        ok = err >= np.percentile(err, 60)
        if m == 'ig':
            score = sp / sp.std() - q / q.std()                 # widest spread, lowest quality
            k = int(np.argmax(np.where(ok, score, -np.inf)))
        else:
            score = np.abs(mi - np.median(mi)) / mi.std() + np.abs(q - np.median(q)) / q.std()
            k = int(np.argmin(np.where(ok, score, np.inf)))
        chosen[m] = runs[k]
        print(f'{m:8} seed {k:2}: information {mi[k]:.2f} bits (median {np.median(mi):.2f}), '
              f'quality {q[k]:+.2f} (median {np.median(q):+.2f}), '
              f'estimate error {err[k]:.0f} deg (median {np.median(err):.0f})')
    return chosen


# ------------------------------------------------------------------ drawing
def tangent_basis(n):
    a = unit(np.cross(n, E_D))
    return a, unit(np.cross(n, a))


def draw_sphere(ax):
    n = 500
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    rr = u ** 2 + v ** 2
    w = np.sqrt(np.clip(1 - rr, 0, None))
    light = unit([-0.45, 0.55, 0.70])
    shade = np.clip(u * light[0] + v * light[1] + w * light[2], 0, 1)
    lo, hi = np.array([0.80, 0.82, 0.86]), np.array([0.985, 0.988, 0.995])
    img = lo + (hi - lo) * shade[..., None] ** 0.8
    img = np.concatenate([img, (rr <= 1)[..., None].astype(float)], axis=-1)
    ax.imshow(img, extent=(-1, 1, -1, 1), origin='lower', zorder=0, interpolation='bilinear')
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor='#a9afb8', linewidth=0.9, zorder=1))


def draw_ring(ax, normal):
    a, b = tangent_basis(normal)
    t = np.linspace(0, 2 * np.pi, 720)
    pts = np.outer(np.cos(t), a) + np.outer(np.sin(t), b)
    x, y, d = proj(pts)
    front = d >= 0
    for mask, style, alpha in ((~front, (0, (1, 2.2)), 0.45), (front, (0, (4, 2.5)), 0.95)):
        ax.plot(np.where(mask, x, np.nan), np.where(mask, y, np.nan), color=C_EST,
                linestyle=style, linewidth=1.1, alpha=alpha, zorder=2)


def draw_cloud(ax, cloud):
    x, y, d = proj(unit(cloud))
    ax.scatter(x[d > 0], y[d > 0], s=5, color=C_CLOUD, alpha=0.17, linewidths=0, zorder=3)


def draw_arrow(ax, vec, color, label, offset):
    x, y, _ = proj(vec)
    ax.add_patch(FancyArrowPatch((0, 0), (x[0], y[0]), arrowstyle='-|>', mutation_scale=11,
                                 color=color, linewidth=1.8, zorder=5, shrinkA=0, shrinkB=0))
    ax.text(x[0] + offset[0], y[0] + offset[1], label, color=color, fontsize=11,
            ha='center', va='center', zorder=6)


def draw_items(ax, items):
    """Items on the visible side are filled; ones on the far side are drawn hollow."""
    x, y, d = proj(items)
    placed = []
    for k in range(len(items)):
        front = d[k] >= 0
        ax.scatter(x[k], y[k], s=70, zorder=7, linewidths=1.4,
                   color=C_XI[k] if front else 'white',
                   edgecolors='white' if front else C_XI[k])
        # label just outside the point, pushed away from the centre and from earlier labels
        direction = np.array([x[k], y[k]])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        pos = np.array([x[k], y[k]]) + 0.15 * direction
        for other in placed:
            if np.linalg.norm(pos - other) < 0.17:
                pos = pos + 0.17 * np.array([-direction[1], direction[0]])
        placed.append(pos)
        ax.text(*pos, rf'$\xi_{k + 1}$', color=C_XI[k], fontsize=11, ha='center',
                va='center', zorder=7)


def draw_axes_triad(ax):
    origin = np.array([-1.04, -1.0])
    for vec, name in ((np.eye(3)[0], r'$\Phi_1$'), (np.eye(3)[1], r'$\Phi_2$'),
                      (np.eye(3)[2], r'$\Phi_3$')):
        x, y, _ = proj(0.26 * vec)
        tip = origin + [x[0], y[0]]
        ax.add_patch(FancyArrowPatch(tuple(origin), tuple(tip), arrowstyle='-|>', mutation_scale=6,
                                     color=MUTED, linewidth=0.8, zorder=4))
        ax.text(*(origin + 1.55 * np.array([x[0], y[0]])), name, color=MUTED, fontsize=7,
                ha='center', va='center')


def sphere_panel(ax, state):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    draw_sphere(ax)
    draw_ring(ax, state['omega_hat'])
    if state['cloud'] is not None:
        draw_cloud(ax, state['cloud'])
    draw_arrow(ax, OMEGA, C_TRUE, r'$\omega^\ast$', (-0.15, 0.03))
    # after a few rankings the estimate is usually close to omega*; keep the labels apart
    tx, ty, _ = proj(OMEGA)
    ux, uy, _ = proj(state['omega_hat'])
    if np.hypot(ux[0] - tx[0], uy[0] - ty[0]) < 0.3:
        offset = (tx[0] - ux[0] + 0.17, ty[0] - uy[0] + 0.03)   # mirror omega*'s label
    else:
        offset = (0.14 * (np.sign(ux[0]) or 1.0), -0.05)
    draw_arrow(ax, state['omega_hat'], C_EST, r'$\hat{\omega}$', offset)
    draw_items(ax, state['items'])
    draw_axes_triad(ax)


def photo_placeholder(ax, name):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle='round,pad=0,rounding_size=0.04',
                                facecolor='#f3f4f6', edgecolor='#b7bcc4', linewidth=1.0,
                                linestyle=(0, (4, 3))))
    ax.text(0.5, 0.56, 'photo', fontsize=10, color='#9aa0a8', ha='center', va='center',
            style='italic')
    ax.text(0.5, 0.42, f'{name} query on the robot', fontsize=7.5, color='#9aa0a8',
            ha='center', va='center')


def tag(ax, y, text, good):
    """good=True/False draws a check/cross badge; good=None a neutral grey label."""
    color = MUTED if good is None else C_GOOD if good else C_BAD
    icon = '' if good is None else (r'$\checkmark$  ' if good else r'$\times$  ')
    ax.text(0.5, y, f'{icon}{text}', fontsize=8.5, color=color, ha='center', va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.8', facecolor=color + '14',
                      edgecolor=color, linewidth=0.8))


def main():
    states = choose_examples()
    # Neutral tags: the shown query's measured information and quality.  At the 2nd query
    # CMA-ES and CMA-ES-IG are equally informative (~0.74 bits each over 40 users, three
    # repeats), so the tags state numbers rather than a comparison.
    num = lambda m: [(f"Information: {states[m]['mi']:.2f} bits", None),
                     (f"Quality: {states[m]['quality']:+.2f}", None)]
    panels = [('ig', '(a) Infogain', 'Infogain', num('ig')),
              ('cmaes', '(b) CMA-ES', 'CMA-ES', num('cmaes')),
              ('cmaesig', '(c) CMA-ES-IG', 'CMA-ES-IG', num('cmaesig'))]

    fig = plt.figure(figsize=(7.16, 5.45))          # IEEE two-column \textwidth
    gs = fig.add_gridspec(3, 3, height_ratios=[2.25, 1.52, 0.55], hspace=0.06, wspace=0.06,
                          left=0.01, right=0.99, top=0.86, bottom=0.005)
    for c, (kind, title, name, tags) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        sphere_panel(ax, states[kind])
        ax.set_title(title, fontsize=12, fontweight='bold', pad=2, color=INK)
        photo_placeholder(fig.add_subplot(gs[1, c]), name)
        tax = fig.add_subplot(gs[2, c])
        tax.axis('off')
        tax.set_xlim(0, 1)
        tax.set_ylim(0, 1)
        for (text, good), y in zip(tags, (0.72, 0.24) if len(tags) == 2 else (0.5,)):
            tag(tax, y, text, good)

    handles = [Line2D([], [], color=C_TRUE, linewidth=1.8, marker='>', markersize=5,
                      label="user's true preference $\\omega^\\ast$"),
               Line2D([], [], color=C_EST, linewidth=1.8, marker='>', markersize=5,
                      label="robot's estimate $\\hat{\\omega}$"),
               Line2D([], [], color=C_EST, linewidth=1.1, linestyle=(0, (4, 2.5)),
                      label='directions orthogonal to $\\hat{\\omega}$'),
               Patch(facecolor=C_CLOUD, alpha=0.35, edgecolor='none',
                     label='CMA-ES search distribution'),
               Line2D([], [], color='none', marker='o', markerfacecolor=C_XI[0],
                      markeredgecolor='white', markersize=7,
                      label='query items $\\xi_1, \\xi_2, \\xi_3$'),
               Line2D([], [], color='none', marker='o', markerfacecolor='white',
                      markeredgecolor=C_XI[0], markersize=7, label='item on far side')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.998), handlelength=1.8, columnspacing=1.6)

    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT, f'figure1_overview.{ext}'), dpi=300)
    print(f'wrote {OUT}/figure1_overview.{{pdf,png,svg}}')


if __name__ == '__main__':
    main()
