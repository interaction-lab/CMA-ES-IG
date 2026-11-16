"""Generate the four CMA-ES-IG step glyphs (Sample, Diversify, Rank, Adapt) as inline SVG.

All four panels share one 2D feature space: the same Gaussian samples, the same
K-means centroids, the same reward direction, and a CMA-style mean/covariance update,
so the panels read as one continuous story.

usage: python figures/make_step_svgs.py > steps_svg.html
Then paste each of the four printed <svg> lines into the matching <li> of the
Method steps in website/index.html (they go right before <span>1</span> etc.).
"""
import math
import numpy as np

W, H = 220, 140
rng = np.random.default_rng(7)

# --- search distribution N(mu, C) in SVG coordinates (y points down) ----------------
mu = np.array([98.0, 72.0])
ang = math.radians(-28)                       # major axis tilted up-right
R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
C = R @ np.diag([33.0 ** 2, 13.0 ** 2]) @ R.T

D = 40
X = rng.multivariate_normal(mu, C, size=D)
X = X[np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(C), X - mu) < 2.4 ** 2]  # keep it tidy

# --- K-means (K = 3), k-means++ init -------------------------------------------------
K = 3
cent = [X[rng.integers(len(X))]]
for _ in range(K - 1):
    d2 = np.min([((X - c) ** 2).sum(1) for c in cent], axis=0)
    cent.append(X[rng.choice(len(X), p=d2 / d2.sum())])
cent = np.array(cent)
for _ in range(50):
    lab = np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), axis=1)
    cent = np.array([X[lab == k].mean(0) for k in range(K)])
# order centroids left-to-right so A/B/C read naturally
order = np.argsort(cent[:, 0])
cent = cent[order]
lab = np.argsort(order)[lab]

# --- user's reward: omega points up-right; rank centroids by omega . x ----------------
omega = np.array([math.cos(math.radians(-38)), math.sin(math.radians(-38))])
rank = np.argsort(-(cent @ omega))            # rank[0] = favourite
place = np.empty(K, int); place[rank] = np.arange(K)

# --- CMA-style update from the ranking -------------------------------------------------
w = np.array([0.6, 0.3, 0.1])                 # recombination weights by rank
target = (w[:, None] * cent[rank]).sum(0)
mu_new = mu + 1.6 * (target - mu)
step = mu_new - mu
p = step / np.linalg.norm(step)
C_new = 0.6 * C + 0.4 * (np.linalg.norm(step) * 0.6) ** 2 * np.outer(p, p)  # shrink + rank-one stretch along the step

LANE = ["#0072b2", "#d55e00", "#009e73"]      # demo lane colours A, B, C
NAMES = ["A", "B", "C"]
INK, MUTED, ACCENT = "#1d1f23", "#5d636d", "#e8771e"
BLOB = "#4a4f57"


def ellipse_params(Cm, k=1.0):
    vals, vecs = np.linalg.eigh(Cm)
    a, b = k * math.sqrt(vals[1]), k * math.sqrt(vals[0])
    rot = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    return a, b, rot


def blob(uid, m, Cm, color=BLOB, k=2.3, opacity=0.55):
    a, b, rot = ellipse_params(Cm, k)
    return (f'<defs><radialGradient id="{uid}"><stop offset="0" stop-color="{color}" stop-opacity="{opacity}"/>'
            f'<stop offset="0.45" stop-color="{color}" stop-opacity="{opacity * 0.45:.3f}"/>'
            f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient></defs>'
            f'<ellipse cx="{m[0]:.1f}" cy="{m[1]:.1f}" rx="{a:.1f}" ry="{b:.1f}" '
            f'transform="rotate({rot:.1f} {m[0]:.1f} {m[1]:.1f})" fill="url(#{uid})"/>')


def contour(m, Cm, k=2.0, **attrs):
    a, b, rot = ellipse_params(Cm, k)
    extra = " ".join(f'{key.replace("_", "-")}="{v}"' for key, v in attrs.items())
    return (f'<ellipse cx="{m[0]:.1f}" cy="{m[1]:.1f}" rx="{a:.1f}" ry="{b:.1f}" '
            f'transform="rotate({rot:.1f} {m[0]:.1f} {m[1]:.1f})" fill="none" {extra}/>')


def arrow_marker(uid, color):
    return (f'<defs><marker id="{uid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker></defs>')


def math_text(x, y, body, size=13, anchor="middle", color=INK, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="EB Garamond, Georgia, serif" font-style="italic" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{color}" {extra}>{body}</text>')


def sub(s):
    return f'<tspan font-size="0.72em" dy="3">{s}</tspan><tspan dy="-3">\u200a</tspan>'


def CAP(body, extra=''):
    return math_text(W / 2, 131, body, 13, extra=extra)


def svg(label, body):
    return (f'<svg class="step-fig" viewBox="0 0 {W} {H}" role="img" aria-label="{label}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def dots(points, r=2.1, fill=INK, opacity=0.85):
    return "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"/>' for x, y in points)


def centroid(i, x, y, r=5.2):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{LANE[i]}" stroke="#fff" stroke-width="1.6"/>')


# 1. Sample ---------------------------------------------------------------------------
s1 = (blob("g1", mu, C)
      + dots(X)
      + CAP(f'Φ(ξ{sub("i")}) ~ 𝒩(μ, C),  i = 1…D'))
svg1 = svg(f"{len(X)} samples drawn from a Gaussian search distribution N(mu, C)", s1)

# 2. Diversify ------------------------------------------------------------------------
lines = "".join(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{cent[k][0]:.1f}" y2="{cent[k][1]:.1f}" '
                f'stroke="{LANE[k]}" stroke-opacity="0.35" stroke-width="0.8"/>' for (x, y), k in zip(X, lab))
pts = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.9" fill="{LANE[k]}" fill-opacity="0.55"/>' for (x, y), k in zip(X, lab))
cents = "".join(centroid(i, *c) for i, c in enumerate(cent))
labels = "".join(
    f'<text x="{c[0] - 8:.1f}" y="{c[1] - 7:.1f}" font-family="Atkinson Hyperlegible, sans-serif" font-weight="700" '
    f'font-size="10" text-anchor="end" fill="{LANE[i]}" stroke="#f6f4ef" stroke-width="3" paint-order="stroke">{NAMES[i]}</text>' for i, c in enumerate(cent))
s2 = (blob("g2", mu, C, opacity=0.22) + lines + pts + cents + labels
      + CAP('K-means, <tspan font-style="italic">K</tspan> = 3', extra='font-style="normal"'))
svg2 = svg("K-means groups the samples into 3 clusters; the centroids A, B, C are shown to the user", s2)

# 3. Rank -----------------------------------------------------------------------------
# iso-reward lines (perpendicular to omega), reward increasing toward the upper right
perp = np.array([-omega[1], omega[0]])
iso = ""
for t in np.linspace(-60, 60, 7):
    c0 = mu + t * omega
    a0, b0 = c0 - 140 * perp, c0 + 140 * perp
    iso += (f'<line x1="{a0[0]:.1f}" y1="{a0[1]:.1f}" x2="{b0[0]:.1f}" y2="{b0[1]:.1f}" '
            f'stroke="{MUTED}" stroke-opacity="0.22" stroke-width="0.8" stroke-dasharray="3 3"/>')
o0 = np.array([16.0, 108.0]); o1 = o0 + 30 * omega
om = (arrow_marker("m3", INK)
      + f'<line x1="{o0[0]}" y1="{o0[1]}" x2="{o1[0]:.1f}" y2="{o1[1]:.1f}" stroke="{INK}" stroke-width="1.4" marker-end="url(#m3)"/>'
      + math_text(o1[0] - 4, o1[1] - 4, "ω", 14, anchor="end"))
ranks = "".join(centroid(i, *c) + f'<text x="{c[0] + 9:.1f}" y="{c[1] + 4:.1f}" font-family="Atkinson Hyperlegible, sans-serif" '
                f'font-weight="700" font-size="10" fill="{LANE[i]}">{place[i] + 1}</text>' for i, c in enumerate(cent))
pref = " ≻ ".join(f'ξ{sub(NAMES[i])}' for i in rank)
s3 = ('<clipPath id="c3"><rect width="220" height="112"/></clipPath>'
      f'<g clip-path="url(#c3)">{iso}</g>' + om + ranks
      + CAP(pref))
svg3 = svg("The user ranks the three centroids; the ranking reveals the reward direction omega", s3)

# 4. Adapt ----------------------------------------------------------------------------
a1 = mu + 0.12 * step; a2 = mu_new - 0.12 * step
nrm = np.array([-p[1], p[0]])  # unit normal to the step, for label placement
if nrm[1] < 0: nrm = -nrm
s4 = (contour(mu, C, 2.0, stroke=INK, stroke_width="1.1", stroke_dasharray="3.5 3", stroke_opacity="0.7")
      + blob("g4", mu_new, C_new, color=ACCENT, k=2.1, opacity=0.6)
      + "".join(f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="3.4" fill="{LANE[i]}" fill-opacity="0.45"/>' for i, c in enumerate(cent))
      + arrow_marker("m4", ACCENT)
      + f'<line x1="{a1[0]:.1f}" y1="{a1[1]:.1f}" x2="{a2[0]:.1f}" y2="{a2[1]:.1f}" stroke="{ACCENT}" stroke-width="1.8" marker-end="url(#m4)"/>'
      + f'<circle cx="{mu[0]:.1f}" cy="{mu[1]:.1f}" r="2.2" fill="{INK}"/>'
      + f'<circle cx="{mu_new[0]:.1f}" cy="{mu_new[1]:.1f}" r="2.4" fill="{ACCENT}"/>'
      + math_text(*(mu + 12 * nrm + np.array([0, 4])), "μ", 13)
      + math_text(*(mu_new + 13 * nrm + np.array([0, 4])), "μ′", 13)
      + CAP("𝒩(μ, C)  →  𝒩(μ′, C′)"))
svg4 = svg("The ranking moves the search distribution: the mean shifts toward the favourite and the covariance stretches along that direction", s4)

if __name__ == "__main__":
    for s in (svg1, svg2, svg3, svg4):
        print(s)
