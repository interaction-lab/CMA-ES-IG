"""
User-study statistics (H5-H7) from rebuttal/user_study_data/study results - Data.csv.

    python3 user_study_analysis.py                 # reproduce the paper's numbers
    python3 user_study_analysis.py --corrected     # participant grouping fixed from session logs
    python3 user_study_analysis.py --tables        # corrected tables + task / mixed-model analysis
    python3 user_study_analysis.py --bars          # bar charts by task (graphs/inspect/user_study_bars.png)
    python3 user_study_analysis.py --boxes         # the same as box plots (graphs/inspect/user_study_boxes.png)
    python3 user_study_analysis.py --mixed         # box plots for EOU/BA, stacked rank shares (user_study_mixed.png)

Reproduces the paper exactly with: pilots (SUBJECT 1, 2) dropped; scores averaged
over the two tasks per SUBJECT; Wilcoxon signed-rank with the normal
approximation and Pratt handling of zero differences; independent-groups Hedges'
g; Cronbach's alpha over all rated sessions *including* the pilots.

Two participants sometimes ran at once on different PIDs, and the session
timestamps in user_study/data/ranking*.csv show two labelling errors:
  * SUBJECT 12 holds four sessions from two people (PIDs 32+12 at 18:41-19:21 and
    PIDs 13+34 at 21:10-21:52 on 07-30), so the analysis has 13 units although
    14 participants were recruited;
  * SUBJECTs 7 and 8 pair concurrent sessions (PID 7 with 27, PID 8 with 28).
    Each participant did one handover and one blossom session back to back, so
    the only consistent pairing is 7+28 and 27+8.
--corrected applies both fixes.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'graphs')
DATA = os.path.join(HERE, 'user_study_data', 'study results - Data.csv')
PILOTS = (1, 2)
EOU = [f'EOU{i}' for i in range(1, 5)]
BA = [f'BA{i}' for i in range(1, 5)]
TESTS = [('BA', 'CMAESIG', 'IG'), ('BA', 'CMAESIG', 'CMAES'), ('EOU', 'CMAESIG', 'CMAES'),
         ('EOU', 'CMAESIG', 'IG'), ('RANK', 'CMAESIG', 'IG'), ('RANK', 'CMAESIG', 'CMAES')]


def cronbach_alpha(items):
    x = items.dropna()
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def hedges_g_independent(a, b):
    """The formula the paper used: pooled SD of the two conditions."""
    n = len(a)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd * (1 - 3 / (8 * n - 9))


def hedges_g_paired(a, b):
    """Standardised mean difference of the paired differences (g_z)."""
    diff = a - b
    return diff.mean() / diff.std(ddof=1) * (1 - 3 / (4 * (len(diff) - 1) - 1))


LABEL = {'IG': 'Infogain', 'CMAES': 'CMA-ES', 'CMAESIG': 'CMA-ES-IG'}
ORDER = ['IG', 'CMAES', 'CMAESIG']
SCALES = [('EOU', 'EOU'), ('BA', 'BA'), ('RANK', 'Rank')]


def load(corrected=True):
    d = pd.read_csv(DATA).dropna(subset=['EOU']).copy()
    d['participant'] = d.SUBJECT
    if corrected:
        d.loc[d.PID.isin([13, 34]), 'participant'] = 100
        d.loc[d.PID.isin([7, 28]), 'participant'] = 7
        d.loc[d.PID.isin([27, 8]), 'participant'] = 8
    d = d[~d.participant.isin(PILOTS)].reset_index(drop=True)
    d['task'] = np.where(d.PID < 20, 'handover', 'blossom')   # PIDs 1-16 handover, 21-36 blossom
    return d


def wilcoxon(x, y):
    """Signed-rank test as in the paper: normal approximation, Pratt zero handling."""
    return (stats.wilcoxon(x, y).statistic,
            stats.wilcoxon(x, y, zero_method='pratt', method='approx').pvalue)


def bh(ps):
    """Benjamini-Hochberg adjusted p-values."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)[::-1]
    adj, running = np.empty(len(ps)), 1.0
    for rank, k in zip(range(len(ps), 0, -1), order):
        running = min(running, ps[k] * len(ps) / rank)
        adj[k] = running
    return adj


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        running = max(running, (len(ps) - i) * ps[k])
        adj[k] = min(1.0, running)
    return adj


# Sum-coded method x task design for the mixed models.
TERMS = {'method': ['m1', 'm2'], 'task': ['t'], 'method:task': ['mt1', 'mt2']}


def _design(d):
    d = d.assign(m1=d.METHOD.map({'IG': 1, 'CMAES': 0, 'CMAESIG': -1}),
                 m2=d.METHOD.map({'IG': 0, 'CMAES': 1, 'CMAESIG': -1}),
                 t=np.where(d.task == 'handover', 1, -1))
    return d.assign(mt1=d.m1 * d.t, mt2=d.m2 * d.t)


def _lmm(y, cols, d):
    """Random intercepts for participant and for session (PID) within participant."""
    model = smf.mixedlm(f'{y} ~ ' + (' + '.join(cols) or '1'), d, groups='participant',
                        re_formula='1', vc_formula={'session': '0 + C(PID)'})
    fits = []
    for method in ('lbfgs', 'bfgs', 'powell', 'nm'):
        try:
            fits.append(model.fit(reml=False, method=method, maxiter=5000))
        except Exception:
            pass
    return max(fits, key=lambda r: r.llf)


def lrt(y, d, effect):
    full_cols = sum(TERMS.values(), [])
    full = _lmm(y, full_cols, d)
    reduced = _lmm(y, [c for c in full_cols if c not in TERMS[effect]], d)
    return stats.chi2.sf(2 * (full.llf - reduced.llf), len(TERMS[effect]))


def art(d, y, effect):
    """Aligned rank transform (Wobbrock et al., 2011): align for one effect, then rank."""
    g = d[y].mean()
    cell = d.groupby(['METHOD', 'task'])[y].transform('mean')
    a = d.groupby('METHOD')[y].transform('mean')
    b = d.groupby('task')[y].transform('mean')
    resid = d[y] - cell
    aligned = {'method': resid + a - g, 'task': resid + b - g,
               'method:task': resid + cell - a - b + g}[effect]
    return d.assign(r=stats.rankdata(aligned))


def ms(x):
    return f'{x.mean():.2f} ({x.std(ddof=1):.2f})'


def tables():
    warnings.filterwarnings('ignore')
    d = _design(load())
    agg = d.groupby(['participant', 'METHOD'])[['EOU', 'BA', 'RANK']].mean().unstack('METHOD')
    n = len(agg)

    # ---- Table A: corrected results, participants averaged over tasks
    tests = []
    for scale, _ in SCALES:
        for b in ('IG', 'CMAES'):
            x, y = agg[(scale, 'CMAESIG')], agg[(scale, b)]
            w, pv = wilcoxon(x, y)
            tests.append(dict(scale=scale, vs=b, W=w, p=pv, g=hedges_g_paired(x, y)))
    # FDR (Benjamini-Hochberg) over the two comparisons within each scale
    for scale, _ in SCALES:
        ts = [t for t in tests if t['scale'] == scale]
        for t, pa in zip(ts, bh([t['p'] for t in ts])):
            t['p_bh'] = pa

    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{User study results ($n=' + str(n) + r'$ participants, each score averaged '
             r'over the physical and social tasks). Mean (SD) of ease of use (EOU), behavioral '
             r'adaptation (BA), and preference rank (0 = least, 2 = most preferred). Bottom: '
             r'Wilcoxon signed-rank tests of CMA-ES-IG against each baseline, with paired '
             r'Hedges\textquotesingle{} $g$ and FDR-corrected (Benjamini--Hochberg) $p$ over the '
             r'two comparisons within each scale.}',
             r'\label{tab:RQ3_results}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{l ccc}', r'\toprule',
             r'\textbf{Algorithm} & \textbf{EOU} & \textbf{BA} & \textbf{Rank} \\', r'\midrule']
    for m in ORDER:
        lines.append(LABEL[m] + ' & ' + ' & '.join(ms(agg[(s, m)]) for s, _ in SCALES) + r' \\')
    lines += [r'\midrule', r'\multicolumn{4}{l}{\textit{CMA-ES-IG vs.}} \\']
    for b in ('IG', 'CMAES'):
        cells = []
        for scale, _ in SCALES:
            t = next(t for t in tests if t['scale'] == scale and t['vs'] == b)
            cells.append(f"$W={t['W']:g}$, $p={t['p']:.3f}$ ($p_{{BH}}={t['p_bh']:.3f}$), $g={t['g']:.2f}$")
        lines.append(r'\quad ' + LABEL[b] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table}']
    open(f'{OUT}/table_user_study.tex', 'w').write('\n'.join(lines) + '\n')

    # ---- Table B: task marginal means, task effect within participants
    pt = d.groupby(['participant', 'task'])[['EOU', 'BA', 'RANK']].mean().unstack('task')
    both = pt.dropna()
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Marginal means (SD) by task and algorithm over sessions. Task $p$: '
             r'Wilcoxon signed-rank test of each participant\textquotesingle{}s handover vs.\ '
             r'blossom mean (averaged over algorithms; $n=' + str(len(both)) + r'$ participants '
             r'with both tasks). Interaction $p$: likelihood-ratio test of algorithm $\times$ task '
             r'in an aligned-rank-transform mixed model with participant and session random '
             r'intercepts. Rank sums to the same value in every session, so it has no task effect.}',
             r'\label{tab:user_study_tasks}', r'\begin{tabular}{ll ccc}', r'\toprule',
             r'\textbf{Task} & \textbf{Algorithm} & \textbf{EOU} & \textbf{BA} & \textbf{Rank} \\',
             r'\midrule']
    for task, name in (('handover', 'Physical (handover)'), ('blossom', 'Social (Blossom)')):
        sub = d[d.task == task]
        for i, m in enumerate(ORDER):
            lines.append((name + f' ($n={sub.PID.nunique()}$)' if i == 0 else '') + ' & ' + LABEL[m]
                         + ' & ' + ' & '.join(ms(sub[sub.METHOD == m][s]) for s, _ in SCALES) + r' \\')
        lines.append(r' & \textit{all algorithms} & ' + ' & '.join(ms(sub[s]) for s, _ in SCALES[:2])
                     + r' & -- \\')
        lines.append(r'\midrule')
    tp = [wilcoxon(both[(s, 'handover')], both[(s, 'blossom')])[1] for s in ('EOU', 'BA')]
    ip = [lrt('r', art(d, s, 'method:task'), 'method:task') for s, _ in SCALES]
    lines += [r'\multicolumn{2}{l}{Task $p$} & ' + ' & '.join(f'{v:.3f}' for v in tp) + r' & -- \\',
              r'\multicolumn{2}{l}{Algorithm $\times$ task $p$} & ' + ' & '.join(f'{v:.3f}' for v in ip)
              + r' \\', r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/table_user_study_tasks.tex', 'w').write('\n'.join(lines) + '\n')

    # ---- console report
    print(f'{n} participants ({len(both)} with both tasks), {d.PID.nunique()} sessions\n')
    print('Table A (participant means, averaged over tasks)')
    for m in ORDER:
        print(f'  {LABEL[m]:10} ' + '  '.join(f'{lab} {ms(agg[(s, m)])}' for s, lab in SCALES))
    for t in tests:
        print(f"  {t['scale']:4} CMA-ES-IG vs {LABEL[t['vs']]:8}: W={t['W']:5g} p={t['p']:.3f} "
              f"p_BH={t['p_bh']:.3f} g_paired={t['g']:.2f}")
    print('\nTable B (sessions by task)')
    for task in ('handover', 'blossom'):
        sub = d[d.task == task]
        for m in ORDER:
            print(f'  {task:8} {LABEL[m]:10} ' + '  '.join(
                f'{lab} {ms(sub[sub.METHOD == m][s])}' for s, lab in SCALES))
        print(f'  {task:8} {"all":10} ' + '  '.join(f'{lab} {ms(sub[s])}' for s, lab in SCALES[:2]))
    print(f'  task effect (Wilcoxon, n={len(both)}): EOU p={tp[0]:.3f}  BA p={tp[1]:.3f}')
    print('\nRepeated-measures models (sessions nested in participants), LRT p-values')
    for s, lab in SCALES:
        para = [lrt(s, d, e) for e in TERMS]
        artp = [lrt('r', art(d, s, e), e) for e in TERMS]
        print(f'  {lab:4} LMM method {para[0]:.3f} task {para[1]:.3f} method x task {para[2]:.3f}   '
              f'ART method {artp[0]:.3f} task {artp[1]:.3f} method x task {artp[2]:.3f}')
    print(f'\nwrote {OUT}/table_user_study.tex and {OUT}/table_user_study_tasks.tex')


# Same method colours as the simulation figures (make_rebuttal_figures.py).
COLOR = {'IG': '#ababab', 'CMAES': '#0770bb', 'CMAESIG': '#f49249'}


def _stars(p):
    return ('***' if p < .001 else '**' if p < .01 else '*' if p < .05
            else '\u2020' if p < .1 else None)


def bars(kind='bar'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

    d = load()
    # Combined = each participant averaged over their tasks, as in the main table
    # (paired by participant); per-task bars are over that task's sessions
    # (paired by session).
    combined = (d.groupby(['participant', 'METHOD'])[['EOU', 'BA', 'RANK']].mean()
                .reset_index().rename(columns={'participant': 'unit'}))
    per_task = lambda t: d[d.task == t].rename(columns={'PID': 'unit'})
    groups = [('Combined', combined), ('Handover', per_task('handover')),
              ('Gestures', per_task('blossom'))]
    # BA first: the paper discusses H5 (BA) before H6 (EOU)
    panels = [('BA', 'Behavioral adaptation ($\\uparrow$)', 8), ('EOU', 'Ease of use ($\\uparrow$)', 8),
              ('RANK', 'Preference rank ($\\uparrow$)', 2)]
    pairs = [('CMAESIG', 'IG'), ('CMAESIG', 'CMAES')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    width = 0.8 / len(ORDER)
    xpos = lambda i, m: i - 0.4 + width * (ORDER.index(m) + 0.5)
    rank_share = {'Combined': d, 'Handover': d[d.task == 'handover'], 'Gestures': d[d.task == 'blossom']}
    # 1st place in the method colour; 2nd hatched and 3rd near-white so neither can be
    # confused with Infogain's grey.
    SEG = [(2, None, None, '1st'), (1, 'white', '////', '2nd'), (0, '#f2f2f2', None, '3rd')]
    limits = []      # (ax, top, highest bracket, kind); shared headroom set after the loop
    for ax, (col, title, top) in zip(axes, panels):
        pk = ('stack' if col == 'RANK' else 'box') if kind == 'mixed' else kind
        if pk == 'stack':
            top, title = 1, 'Preference rank'
        tops = {}
        for m in ORDER:
            if pk == 'stack':
                # share of sessions ranking this method 1st / 2nd / 3rd (all sessions pooled)
                light = tuple(0.55 * c + 0.45 for c in matplotlib.colors.to_rgb(COLOR[m]))
                for i, (name, _) in enumerate(groups):
                    r = rank_share[name]
                    r = r[r.METHOD == m].RANK
                    bottom, outlines = 0, []
                    for value, face, hatch, _ in SEG:
                        share = (r == value).mean()
                        color = face or (COLOR[m] if i == 0 else light)
                        ax.bar(xpos(i, m), share, width * 0.88, bottom=bottom, color=color, hatch=hatch,
                               edgecolor='#a8a8a8' if hatch else color, linewidth=0)
                        if share > 0:
                            # 1st place: a darker shade of its fill; 2nd/3rd: greys
                            edge = (tuple(0.7 * c for c in matplotlib.colors.to_rgb(color))
                                    if face is None else '#a8a8a8' if hatch else '#c9c9c9')
                            outlines.append((value, bottom, share, edge))
                        if value == 2 and share >= 0.08:
                            ax.text(xpos(i, m), bottom + share / 2, f'{share:.0%}', ha='center',
                                    va='center', fontsize=8, color='white' if i == 0 else '#333333')
                        bottom += share
                    # outlines drawn after the fills, 1st place last, so shared edges stay crisp
                    for value, b0, share, edge in sorted(outlines, key=lambda o: o[0]):
                        ax.bar(xpos(i, m), share, width * 0.88, bottom=b0, fill=False, edgecolor=edge,
                               linewidth=1.0 if value == 2 else 0.8, zorder=3 + value)
                    tops[(i, m)] = 1.0
                continue
            xs, ys, es, vals = [], [], [], []
            for i, (_, g) in enumerate(groups):
                v = g[g.METHOD == m][col].dropna()
                xs.append(xpos(i, m)); ys.append(v.mean()); es.append(v.std(ddof=1) / np.sqrt(len(v)))
                vals.append(v.to_numpy())
                tops[(i, m)] = v.max() if pk == 'box' else ys[-1] + es[-1]
            if pk == 'box':
                light = tuple(0.55 * c + 0.45 for c in matplotlib.colors.to_rgb(COLOR[m]))
                rng = np.random.default_rng(0)
                for k in range(len(groups)):
                    face, edge = (COLOR[m], '#333333') if k == 0 else (light, '#777777')
                    ax.boxplot([vals[k]], positions=[xs[k]], widths=width * 0.85, patch_artist=True,
                               showfliers=False, whis=1.5,
                               boxprops=dict(facecolor=face, edgecolor=edge, linewidth=1),
                               medianprops=dict(color=edge, linewidth=1.5),
                               whiskerprops=dict(color=edge, linewidth=1),
                               capprops=dict(color=edge, linewidth=1))
                    jitter = rng.uniform(-width * 0.25, width * 0.25, len(vals[k]))
                    ax.scatter(xs[k] + jitter, vals[k], s=9, color='#222222' if k == 0 else '#666666',
                               alpha=0.4, zorder=3, linewidths=0)
                continue
            # Combined (group 0) is the headline result; per-task bars use an opaque
            # tint of the same colour (mixed 45% toward white).
            light = tuple(0.55 * c + 0.45 for c in matplotlib.colors.to_rgb(COLOR[m]))
            for sel, color, ecolor, label in (([0], COLOR[m], '#333333', LABEL[m]),
                                              (range(1, len(groups)), light, '#777777', None)):
                ax.bar([xs[k] for k in sel], [ys[k] for k in sel], width,
                       yerr=[es[k] for k in sel], color=color, label=label,
                       edgecolor='white', linewidth=1, capsize=3,
                       error_kw={'elinewidth': 1, 'ecolor': ecolor})
        # significance brackets, stacked above the tallest bar of each group
        step = top * 0.07
        highest = top
        for i, (_, g) in enumerate(groups):
            wide = g.pivot_table(index='unit', columns='METHOD', values=col).dropna()
            level = max(tops[(i, m)] for m in ORDER) + step * 0.5
            # markers use uncorrected p-values (the table reports the BH-corrected ones)
            p_bh = {(a, b): wilcoxon(wide[a], wide[b])[1] for a, b in pairs}
            for a, b in sorted(pairs, key=lambda ab: abs(ORDER.index(ab[0]) - ORDER.index(ab[1]))):
                star = _stars(p_bh[(a, b)])
                if star is None:
                    continue
                x1, x2 = sorted((xpos(i, a), xpos(i, b)))
                ax.plot([x1, x1, x2, x2], [level, level + step * 0.3, level + step * 0.3, level],
                        color='#333333', linewidth=1, clip_on=False)
                # asterisks sit high in their text box, the dagger low, so offset them differently
                ax.text((x1 + x2) / 2, level + step * (0.38 if star == '\u2020' else 0.12), star,
                        ha='center', va='bottom', fontsize=12, clip_on=False)
                highest = max(highest, level + step * 0.3)
                level += step
        ax.vlines(0.5, 0, top, color='#666666', linestyle=':', linewidth=1.5)   # stop at the scale max
        ax.set_xticks(range(len(groups)))
        if pk == 'stack':
            ax.set_xticklabels([f'{name}\n({rank_share[name].PID.nunique()} sessions)'
                                for name, _ in groups])
        else:
            ax.set_xticklabels([f'{name}\n(n={g.unit.nunique()})' for name, g in groups])
        limits.append((ax, top, highest + step * 0.8, pk))
        ax.spines['left'].set_bounds(0, top)     # axis line and ticks end at the scale maximum
        if pk in ('box', 'stack'):
            ax.set_xlim(-0.55, len(groups) - 0.45)
        ax.set_yticks(np.linspace(0, top, 5))
        if pk == 'stack':
            ax.set_yticklabels([f'{v:.0%}' for v in np.linspace(0, top, 5)])
            ax.set_ylabel('Share of sessions', labelpad=-2, fontsize=14)
        ax.set_title(title, fontsize=16)   # matches the simulation figures
        ax.grid(True, axis='y', color='lightgrey', linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # same headroom (as a fraction of the scale) in every panel so the scales line up
    head = max(1.03, max(h / t for _, t, h, _ in limits))
    for ax, t, _, pk in limits:
        ax.set_ylim(-0.03 * t if pk in ('box', 'stack') else 0, t * head)
    axes[0].set_ylabel('Score' if kind in ('box', 'mixed') else 'Mean $\\pm$ SE', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if kind in ('box', 'mixed'):
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=COLOR[m], edgecolor='#333333') for m in ORDER]
        labels = [LABEL[m] for m in ORDER]
    if kind == 'mixed':
        handles += [Patch(facecolor=f, hatch=h, edgecolor='#a8a8a8') for _, f, h, _ in SEG[1:]]
        labels += ['Ranked 2nd', 'Ranked 3rd']
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.008, '\u2020 p < .1     * p < .05     ** p < .01     *** p < .001     '
             '(Wilcoxon signed-rank vs. CMA-ES-IG, uncorrected)',
             ha='center', va='bottom', fontsize=11)
    os.makedirs(f'{OUT}/inspect', exist_ok=True)
    name = {'box': 'user_study_boxes', 'mixed': 'user_study_mixed'}.get(kind, 'user_study_bars')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/inspect/{name}.{ext}', bbox_inches='tight', dpi=200)
    print(f'wrote {OUT}/inspect/{name}.{{png,pdf}}')
    print('Wilcoxon (CMA-ES-IG vs IG / vs CMA-ES): plotted uncorrected p (BH over 2 in brackets)')
    for name, g in groups:
        for col, _, _ in panels:
            wide = g.pivot_table(index='unit', columns='METHOD', values=col).dropna()
            raw = [wilcoxon(wide[a], wide[b])[1] for a, b in pairs]
            print(f'  {name:9} {col:4} n={len(wide):2}  ' + '  '.join(
                f'vs {LABEL[b]} p={r:.3f} {_stars(r) or "":3} (BH {q:.3f})'
                for (a, b), r, q in zip(pairs, raw, bh(raw))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrected', action='store_true',
                   help='split SUBJECT 12 into two people and re-pair PIDs 7/27/8/28')
    p.add_argument('--bars', action='store_true', help='bar charts by task')
    p.add_argument('--boxes', action='store_true', help='box plots by task')
    p.add_argument('--mixed', action='store_true', help='box plots for EOU/BA, stacked rank shares')
    p.add_argument('--tables', action='store_true',
                   help='write the corrected tables and run the task / mixed-model analysis')
    args = p.parse_args()
    if args.tables:
        return tables()
    if args.bars:
        return bars()
    if args.boxes:
        return bars('box')
    if args.mixed:
        return bars('mixed')

    df = pd.read_csv(DATA)
    rated = df.dropna(subset=EOU + BA, how='all')
    print(f"Cronbach's alpha, all rated sessions incl. pilots (as reported): "
          f'BA={cronbach_alpha(rated[BA]):.3f}  EOU={cronbach_alpha(rated[EOU]):.3f}')

    d = df.dropna(subset=['EOU']).copy()
    d['participant'] = d.SUBJECT
    if args.corrected:
        d.loc[d.PID.isin([13, 34]), 'participant'] = 100
        d.loc[d.PID.isin([7, 28]), 'participant'] = 7
        d.loc[d.PID.isin([27, 8]), 'participant'] = 8
    d = d[~d.participant.isin(PILOTS)]
    print(f"Cronbach's alpha, pilots excluded:                         "
          f'BA={cronbach_alpha(d[BA]):.3f}  EOU={cronbach_alpha(d[EOU]):.3f}')

    agg = d.groupby(['participant', 'METHOD'])[['EOU', 'BA', 'RANK']].mean().unstack('METHOD')
    print(f'\n{len(agg)} participants; mean score per method:')
    print(agg.mean().unstack('METHOD').round(2).to_string())
    print(f'\n{"test":24} {"W":>5} {"p":>6} {"g (paper)":>10} {"g paired":>9}')
    for scale, a, b in TESTS:
        x, y = agg[(scale, a)], agg[(scale, b)]
        w = stats.wilcoxon(x, y).statistic
        pval = stats.wilcoxon(x, y, zero_method='pratt', method='approx').pvalue
        print(f'{scale + ": " + a + " vs " + b:24} {w:5g} {pval:6.3f} '
              f'{hedges_g_independent(x, y):10.3f} {hedges_g_paired(x, y):9.3f}')


if __name__ == '__main__':
    main()
