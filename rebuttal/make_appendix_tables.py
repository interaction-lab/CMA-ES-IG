"""
Appendix tables reporting every statistical test in the revised paper.

    python3 make_appendix_tables.py

Writes updated_text_and_figures/text/appendix_statistics.tex.  Simulation tests use the
resampling-belief results (graphs/data_rs, simulated_experiments/*/data/results_rs); the user
study uses rebuttal/user_study_data with the corrected participant grouping.  Needs the
LaTeX packages booktabs and graphicx.
"""

import importlib.util
import os
import warnings

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'updated_text_and_figures', 'text', 'appendix_statistics.tex')
warnings.filterwarnings('ignore')


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load('figures', 'make_rebuttal_figures.py')
F.HYPER = os.path.join(HERE, 'graphs', 'data_rs')
F.HYPER_FALLBACK = {}
F.DOMAIN_RESULTS = 'results_rs'
F.ITEMS = 5          # RQ1 uses five-item queries; the ablation (four items) sets its own
U = _load('user_study', 'user_study_analysis.py')

MARGIN = 0.05
BASELINES = ['IG', 'Habibian-IR-log', 'CMA-ES']
HYPER_KEY = {'alignment': 'alignment', 'regret': 'regret', 'quality': 'per_query_alignment'}
DOMAIN_KEY = {'alignment': 'alignment', 'regret': 'regret', 'quality': 'quality_avg'}
RQ1 = ([(d, f'$d={d}$') for d in F.DIMS], lambda k, m, x: F.metric_values(m, HYPER_KEY[x], k))
RQ2 = (F.DOMAINS, lambda k, m, x: F.domain_values(k, m, DOMAIN_KEY[x]))


# ------------------------------------------------------------------ formatting
def fp(p):
    """APA-style p value."""
    if p < .001:
        return '$<.001$'
    return f'${p:.3f}$'.replace('0.', '.', 1)


def fnum(x, prec=3):
    return f'${x:+.{prec}f}$'


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean(), se, df


def cohen_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def table(label, caption, colspec, header, rows, wide=True):
    env = 'table*' if wide else 'table'
    out = [rf'\begin{{{env}}}[ht]', r'\centering', r'\footnotesize',
           rf'\caption{{{caption}}}', rf'\label{{{label}}}', r'\resizebox{\textwidth}{!}{%',
           rf'\begin{{tabular}}{{{colspec}}}', r'\toprule', header, r'\midrule']
    out += rows
    out += [r'\bottomrule', r'\end{tabular}', r'}', rf'\end{{{env}}}', '']
    return '\n'.join(out)


def grouped(rows_by_group):
    """Rows grouped by baseline with a rule between groups."""
    out = []
    for i, rows in enumerate(rows_by_group):
        if i:
            out.append(r'\midrule')
        out += rows
    return out


# ------------------------------------------------------------------ simulation tables
def noninferiority(setting, unit):
    cols, get = setting
    groups = []
    for b in BASELINES:
        rows = []
        for i, (key, title) in enumerate(cols):
            x = {m: F.auc(get(key, 'CMA-ES-IG', m)) for m in ('alignment', 'regret')}
            y = {m: F.auc(get(key, b, m)) for m in ('alignment', 'regret')}
            cells = []
            for metric in ('alignment', 'regret'):
                d, se, df = welch(x[metric], y[metric])
                crit = stats.t.ppf(.975, df)
                t = (d + MARGIN) / se if metric == 'alignment' else (MARGIN - d) / se
                p = min(1.0, 3 * stats.t.sf(t, df))
                cells += [fnum(d), f'[{fnum(d - crit * se)}, {fnum(d + crit * se)}]',
                          f'${t:.2f}$ (${df:.0f}$)', fp(p)]
            rows.append((F.LABEL[b] if i == 0 else '') + f' & {title} & ' + ' & '.join(cells) + r' \\')
        groups.append(rows)
    header = (r'& & \multicolumn{4}{c}{\textbf{AUC alignment}} & \multicolumn{4}{c}{\textbf{AUC normalized regret}} \\'
              '\n' r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}' '\n'
              rf'\textbf{{Baseline}} & \textbf{{{unit}}} & Diff. & 95\% CI & $t$ ($df$) & $p$ & Diff. & 95\% CI & $t$ ($df$) & $p$ \\')
    return 'll cccc cccc', header, grouped(groups)


def superiority(setting, unit, metrics):
    """Two-sided Welch tests of CMA-ES-IG vs each baseline (Bonferroni over 3 baselines)."""
    cols, get = setting
    groups = []
    for b in BASELINES:
        rows = []
        for i, (key, title) in enumerate(cols):
            cells = []
            for metric in metrics:
                x, y = F.auc(get(key, 'CMA-ES-IG', metric)), F.auc(get(key, b, metric))
                d, se, df = welch(x, y)
                t = d / se
                p = min(1.0, 3 * 2 * stats.t.sf(abs(t), df))
                cells += [f'${x.mean():.3f}$', f'${y.mean():.3f}$', f'${t:.2f}$ (${df:.0f}$)', fp(p),
                          f'${cohen_d(x, y):+.2f}$']
            rows.append((F.LABEL[b] if i == 0 else '') + f' & {title} & ' + ' & '.join(cells) + r' \\')
        groups.append(rows)
    names = {'quality': 'AUC quality', 'alignment': 'AUC alignment', 'regret': 'AUC normalized regret'}
    head = ' & '.join(rf'\multicolumn{{5}}{{c}}{{\textbf{{{names[m]}}}}}' for m in metrics)
    rules = ' '.join(rf'\cmidrule(lr){{{3 + 5 * j}-{7 + 5 * j}}}' for j in range(len(metrics)))
    sub = ' & '.join(['CMA-ES-IG & Baseline & $t$ ($df$) & $p$ & $d$'] * len(metrics))
    header = f'& & {head} \\\\\n{rules}\n\\textbf{{Baseline}} & \\textbf{{{unit}}} & {sub} \\\\'
    return 'll ' + ' '.join(['ccccc'] * len(metrics)), header, grouped(groups)


def ablation():
    F.ITEMS = 4          # the ablation was run with four-item queries
    groups = []
    for m in F.ABLATION[1:]:
        rows = []
        for i, d in enumerate(F.DIMS):
            cells = []
            for metric in ('per_query_alignment', 'regret'):
                x = F.auc(F.metric_values(m, metric, d))
                base = F.auc(F.metric_values('IGapprox-kmeans100', metric, d))
                diff, se, df = welch(x, base)
                p = min(1.0, (len(F.ABLATION) - 1) * 2 * stats.t.sf(abs(diff / se), df))
                cells += [fnum(diff), f'${diff / se:.2f}$ (${df:.0f}$)', fp(p)]
            rows.append((F.ABL_LABEL[m] if i == 0 else '') + f' & $d={d}$ & ' + ' & '.join(cells) + r' \\')
        groups.append(rows)
    F.ITEMS = 5
    header = (r'& & \multicolumn{3}{c}{\textbf{AUC quality}} & \multicolumn{3}{c}{\textbf{AUC normalized regret}} \\'
              '\n' r'\cmidrule(lr){3-5} \cmidrule(lr){6-8}' '\n'
              r'\textbf{Variant} & & Diff. & $t$ ($df$) & $p$ & Diff. & $t$ ($df$) & $p$ \\')
    return 'll ccc ccc', header, grouped(groups)


def robustness(items=(3, 4, 5)):
    """H1 and H2 in the parameter-estimation task for each number of items per query."""
    rows = []
    for k_items in items:
        F.ITEMS = k_items
        if F.metric_values('CMA-ES-IG', 'alignment', F.DIMS[0]) is None:
            continue
        for i, d in enumerate(F.DIMS):
            cells, n_ok = [], 0
            for metric, sign in (('alignment', 1), ('regret', -1)):
                worst = None
                x = F.auc(F.metric_values('CMA-ES-IG', metric, d))
                for b in BASELINES:
                    diff, se, df = welch(x, F.auc(F.metric_values(b, metric, d)))
                    crit = stats.t.ppf(.975, df)
                    t = (diff + MARGIN) / se if sign == 1 else (MARGIN - diff) / se
                    n_ok += min(1.0, 3 * stats.t.sf(t, df)) < .05
                    bound = diff - crit * se if sign == 1 else diff + crit * se
                    worst = bound if worst is None else (min(worst, bound) if sign == 1 else max(worst, bound))
                cells.append(fnum(worst))
            q = {m: F.auc(F.metric_values(m, 'per_query_alignment', d)) for m in ['CMA-ES-IG'] + BASELINES}
            diff, se, df = welch(q['CMA-ES-IG'], q['CMA-ES'])
            p = min(1.0, 3 * 2 * stats.t.sf(abs(diff / se), df))
            q_all = all(q['CMA-ES-IG'].mean() > q[b].mean() for b in BASELINES)
            rows.append((f'$K={k_items}$' if i == 0 else '') + f' & $d={d}$ & ${n_ok}/6$ & ' + ' & '.join(cells)
                        + f' & {fnum(diff)} & ${diff / se:.2f}$ (${df:.0f}$) & {fp(p)} & '
                        + (r'$\checkmark$' if q_all else r'$\times$') + r' \\')
        rows.append(r'\midrule')
    F.ITEMS = 5
    if rows and rows[-1] == r'\midrule':
        rows.pop()
    header = (r'& & \multicolumn{3}{c}{\textbf{H1: non-inferiority}} & \multicolumn{4}{c}{\textbf{H2: quality vs.\ CMA-ES}} \\'
              '\n' r'\cmidrule(lr){3-5} \cmidrule(lr){6-9}' '\n'
              r'\textbf{Items} & \textbf{Dim.} & Passed & Worst align.\ bound & Worst regret bound & '
              r'Diff. & $t$ ($df$) & $p$ & Best overall \\')
    return 'll ccc cccc', header, rows


# ------------------------------------------------------------------ user study tables
def user_study_main():
    d = U.load()
    agg = d.groupby(['participant', 'METHOD'])[['EOU', 'BA', 'RANK']].mean().unstack('METHOD')
    rows = []
    for scale, name in U.SCALES:
        tests = []
        for b in ('IG', 'CMAES'):
            x, y = agg[(scale, 'CMAESIG')], agg[(scale, b)]
            w, p = U.wilcoxon(x, y)
            tests.append((b, x, y, w, p, U.hedges_g_paired(x, y)))
        adj = U.bh([t[4] for t in tests])
        for j, ((b, x, y, w, p, g), pa) in enumerate(zip(tests, adj)):
            rows.append((name if j == 0 else '') + f' & {U.LABEL[b]} & ${x.mean():.2f}$ (${x.std(ddof=1):.2f}$) & '
                        f'${y.mean():.2f}$ (${y.std(ddof=1):.2f}$) & ${w:g}$ & {fp(p)} & {fp(pa)} & ${g:.2f}$ \\\\')
        if scale != 'RANK':
            rows.append(r'\midrule')
    header = (r'\textbf{Measure} & \textbf{CMA-ES-IG vs.} & CMA-ES-IG $M$ ($SD$) & Baseline $M$ ($SD$) & '
              r'$W$ & $p$ & $p_{\mathrm{BH}}$ & $g_z$ \\')
    return 'll cccccc', header, rows, len(agg)


def user_study_tasks():
    d = U.load()
    rows = []
    for task, name in (('handover', 'Handover'), ('blossom', 'Gestures')):
        sub = d[d.task == task]
        for j, (scale, sname) in enumerate(U.SCALES):
            wide = sub.pivot_table(index='PID', columns='METHOD', values=scale).dropna()
            cells = []
            for b in ('IG', 'CMAES'):
                w, p = U.wilcoxon(wide['CMAESIG'], wide[b])
                cells += [f'${w:g}$', fp(p)]
            rows.append((f'{name} ($n={len(wide)}$)' if j == 0 else '') + f' & {sname} & ' + ' & '.join(cells) + r' \\')
        if task == 'handover':
            rows.append(r'\midrule')
    header = (r'& & \multicolumn{2}{c}{\textbf{vs. Infogain}} & \multicolumn{2}{c}{\textbf{vs. CMA-ES}} \\'
              '\n' r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}' '\n'
              r'\textbf{Task} & \textbf{Measure} & $W$ & $p$ & $W$ & $p$ \\')
    return 'll cc cc', header, rows


def user_study_models():
    d = U._design(U.load())
    pt = d.groupby(['participant', 'task'])[['EOU', 'BA']].mean().unstack('task').dropna()
    rows = []
    for scale, name in U.SCALES:
        task_p = (fp(U.wilcoxon(pt[(scale, 'handover')], pt[(scale, 'blossom')])[1])
                  if scale != 'RANK' else '--')
        lmm = [U.lrt(scale, d, e) for e in U.TERMS]
        art = [U.lrt('r', U.art(d, scale, e), e) for e in U.TERMS]
        if scale == 'RANK':            # ranks sum to the same value in every session
            lmm[1] = art[1] = None
        cells = [task_p] + [fp(p) if p is not None else '--' for p in lmm + art]
        rows.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    header = (r'& \textbf{Task} & \multicolumn{3}{c}{\textbf{Linear mixed model}} & \multicolumn{3}{c}{\textbf{Aligned rank transform}} \\'
              '\n' r'\cmidrule(lr){3-5} \cmidrule(lr){6-8}' '\n'
              r'\textbf{Measure} & Wilcoxon $p$ & Algorithm & Task & Alg.\ $\times$ task & Algorithm & Task & Alg.\ $\times$ task \\')
    rated = U.load().copy()
    alpha = (U.cronbach_alpha(rated[U.BA]), U.cronbach_alpha(rated[U.EOU]))
    return 'l c ccc ccc', header, rows, len(pt), alpha


# ------------------------------------------------------------------ assemble
def main():
    parts = [
        '% Appendix: statistical tests (generated by rebuttal/make_appendix_tables.py).',
        '% Requires \\usepackage{booktabs} and \\usepackage{graphicx}.', '',
        r'\section*{Appendix: Statistical Tests}', '',
        r'All simulation comparisons use the per-user area under the curve (AUC) over the 30 '
        r'queries (trapezoidal rule divided by the number of intervals), with 100 simulated users '
        r'per algorithm, and Welch\textquotesingle{}s $t$-tests. Regret is normalized by each '
        r'user\textquotesingle{}s range of true rewards. Non-inferiority tests are one-sided with a '
        rf'margin of $\Delta={MARGIN}$; all other tests are two-sided. Unless noted, $p$ values are '
        r'Bonferroni-corrected across the three baselines within each dimension or task. Diff.\ is '
        r'CMA-ES-IG minus the baseline.', '',
    ]
    def text(*paragraphs):
        for para in paragraphs:
            parts.append(para)
            parts.append('')

    # ---------------- RQ1
    text(r'\subsection*{RQ1: Parameter estimation}',
         r'% TODO: align this summary of RQ1 with its wording in the paper.',
         r'RQ1 asks whether CMA-ES-IG can learn a user\textquotesingle{}s preference as accurately as '
         r'existing query-generation methods while showing the user higher-quality options. We study '
         r'it in a parameter-estimation task: simulated users with linear reward functions rank '
         r'queries of five items drawn from feature spaces of $d \in \{4, 8, 16, 32\}$ dimensions. '
         r'\textbf{H1} predicts that CMA-ES-IG learns at least as well as the baselines '
         r'(non-inferior alignment and normalized regret), and \textbf{H2} predicts that it '
         r'generates higher-quality queries.')
    colspec, header, rows = noninferiority(RQ1, 'Dim.')
    parts.append(table('tab:app_h1', r'Non-inferiority tests for \textbf{H1} (parameter estimation). '
                       rf'CMA-ES-IG is non-inferior when $t$ is significant: $H_0$ is a loss of at least '
                       rf'$\Delta={MARGIN}$ (lower alignment or higher normalized regret).', colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_h1}.} Each row compares CMA-ES-IG with one '
         r'baseline at one dimension. Diff.\ is CMA-ES-IG minus the baseline, so a negative '
         r'alignment difference or a positive regret difference means CMA-ES-IG did slightly worse. '
         r'The 95\% CI shows how large that shortfall could plausibly be. Each $t$-test asks whether '
         rf'the shortfall is smaller than the margin $\Delta={MARGIN}$: a significant $p$ means '
         rf'CMA-ES-IG is non-inferior, which is equivalent to the CI staying above $-{MARGIN}$ for '
         rf'alignment and below ${MARGIN}$ for normalized regret. Every comparison is significant, so '
         r'CMA-ES-IG learned as well as each baseline at every dimension.')
    colspec, header, rows = superiority(RQ1, 'Dim.', ['quality'])
    parts.append(table('tab:app_h2', r'Two-sided tests of query quality for \textbf{H2} (parameter '
                       r'estimation), with Cohen\textquotesingle{}s $d$.', colspec, header, rows, wide=False))
    text(r'\noindent\textit{Reading \autoref{tab:app_h2}.} Query quality is the average alignment '
         r'between the items shown to the user and their true preference, so higher values mean the '
         r'user sees options closer to what they want. The table gives each algorithm\textquotesingle{}s '
         r'mean AUC quality; positive $t$ and $d$ favor CMA-ES-IG. Cohen\textquotesingle{}s $d$ expresses '
         r'the difference in standard deviations (0.2, 0.5, and 0.8 are conventionally small, medium, '
         r'and large). CMA-ES-IG had higher quality than every baseline at every dimension, by very '
         r'large margins over Infogain and Informative+Revealing, and by a margin over CMA-ES that '
         r'grows with dimension.')
    colspec, header, rows = superiority(RQ1, 'Dim.', ['alignment', 'regret'])
    parts.append(table('tab:app_learning_rq1', r'Supplementary two-sided tests of learning in the parameter '
                       r'estimation task. These describe the direction of the (within-margin) differences '
                       r'reported alongside \textbf{H1}; they are not tests of a hypothesis.',
                       colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_learning_rq1}.} These conventional two-sided tests '
         r'show which algorithm learned faster, regardless of the margin. They complement '
         r'\autoref{tab:app_h1}: a difference can be statistically significant here and still be too '
         r'small to matter under the non-inferiority margin. The baselines achieved slightly better '
         r'learning in some higher-dimensional settings (Infogain in alignment at $d=8$ and $16$; '
         r'Informative+Revealing in alignment and normalized regret at $d=16$ and $32$; CMA-ES in '
         r'alignment at $d=32$), but all of these differences are within '
         rf'$\Delta={MARGIN}$.')

    colspec, header, rows = robustness()
    parts.append(table('tab:app_items', r'Robustness of \textbf{H1} and \textbf{H2} to the number of items '
                       r'per query $K$ in the parameter-estimation task (the main results use $K=5$). Passed: '
                       rf'non-inferiority tests ($\Delta={MARGIN}$) passed out of six (three baselines $\times$ '
                       r'two metrics). Worst bounds: the 95\% CI bound closest to the margin across baselines '
                       r'(lower bound for alignment, upper bound for normalized regret). Quality: two-sided '
                       r'Welch test against CMA-ES, the closest baseline; Best overall marks whether CMA-ES-IG had '
                       r'the highest mean quality of all four algorithms.', colspec, header, rows))
    parts.append(r'\begin{figure*}[ht]' '\n' r'\centering' '\n'
                 r'\includegraphics[width=\textwidth]{figures/figure_items_per_query.pdf}' '\n'
                 r'\caption{AUC of alignment (top), normalized regret (middle), and query quality (bottom) '
                 r'against the number of items per query $K$ in the parameter-estimation task, for each '
                 r'feature dimension $d$. Mean $\pm$ SD over 100 simulated users; methods are offset '
                 r'horizontally for readability.}' '\n'
                 r'\label{fig:app_items}' '\n' r'\end{figure*}' '\n')
    text(r'\noindent\textit{Reading \autoref{tab:app_items}.} Each row repeats the tests of \textbf{H1} and '
         r'\textbf{H2} with a different number of items per query. A row passes \textbf{H1} when all six '
         rf'non-inferiority tests pass, which is equivalent to the worst alignment bound staying above '
         rf'$-{MARGIN}$ and the worst regret bound staying below ${MARGIN}$. The quality columns compare '
         r'CMA-ES-IG with CMA-ES, its closest competitor in quality. \autoref{fig:app_items} shows the '
         r'same comparison graphically: every algorithm learns faster with more items per query, '
         r'while the ordering of the algorithms in query quality is unchanged.')

    # ---------------- RQ2
    text(r'\subsection*{RQ2: Simulated robot tasks}',
         r'% TODO: align this summary of RQ2 with its wording in the paper.',
         r'RQ2 asks whether the results of RQ1 carry over to realistic robot tasks, where the items are '
         r'real trajectories rather than arbitrary feature vectors. We use four tasks (Lunar Lander, '
         r'Driving, Robot Face Design, and Robot Voice Design), each with a learned four-dimensional '
         r'trajectory embedding, and simulated users who rank queries of five trajectories. '
         r'\textbf{H3} and \textbf{H4} mirror \textbf{H1} and \textbf{H2}: non-inferior learning and '
         r'higher query quality.')
    colspec, header, rows = noninferiority(RQ2, 'Task')
    parts.append(table('tab:app_h3', r'Non-inferiority tests for \textbf{H3} (simulated robot tasks), '
                       rf'with margin $\Delta={MARGIN}$.', colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_h3}.} The table has the same structure as '
         r'\autoref{tab:app_h1}, with one row per baseline and task. CMA-ES-IG was non-inferior to every '
         r'baseline in every task; its largest shortfalls were against Infogain (alignment in Robot Voice '
         r'Design, normalized regret in Lunar Lander), and both were well inside the margin.')
    colspec, header, rows = superiority(RQ2, 'Task', ['quality'])
    parts.append(table('tab:app_h4', r'Two-sided tests of query quality for \textbf{H4} (simulated '
                       r'robot tasks), with Cohen\textquotesingle{}s $d$.', colspec, header, rows, wide=False))
    text(r'\noindent\textit{Reading \autoref{tab:app_h4}.} As in \autoref{tab:app_h2}, positive $t$ and '
         r'$d$ favor CMA-ES-IG. It had higher quality than every baseline in every task. The advantage '
         r'over CMA-ES is smaller than in RQ1 (effect sizes around 0.4), consistent with these tasks\textquotesingle{} '
         r'low-dimensional feature spaces, where the two methods\textquotesingle{} search distributions behave '
         r'more similarly.')
    colspec, header, rows = superiority(RQ2, 'Task', ['alignment', 'regret'])
    parts.append(table('tab:app_learning_rq2', r'Supplementary two-sided tests of learning in the simulated '
                       r'robot tasks, reported alongside \textbf{H3}.', colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_learning_rq2}.} As in RQ1, Infogain learned slightly '
         r'faster than CMA-ES-IG in most tasks, and these differences are within the non-inferiority '
         r'margin. Differences from Informative+Revealing and CMA-ES were small in both directions.')

    # ---------------- ablation
    text(r'\subsection*{Ablation: approximating information gain}',
         r'CMA-ES-IG chooses which of its CMA-ES samples to show by clustering them with $k$-means, as a '
         r'cheap approximation to maximizing information gain. This ablation keeps the CMA-ES search '
         r'fixed and varies only that selection rule, in the parameter-estimation task with '
         r'four-item queries.')
    colspec, header, rows = ablation()
    parts.append(table('tab:app_ablation', r'Two-sided tests of each information-gain approximation against '
                       r'the deployed rule ($k$-means, pool 100), Bonferroni-corrected across the six '
                       r'variants within each dimension.', colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_ablation}.} Diff.\ is each variant minus the deployed '
         r'$k$-means rule. A negative quality difference means the variant showed lower-quality items; a '
         r'negative regret difference means it learned better. Selecting items by exact information gain '
         r'learned better but showed much lower-quality items, while the $k$-means and $k$-medoids '
         r'variants traded smaller amounts of quality for little change in learning: choosing items '
         r'for information and choosing them for quality pull in different directions.')

    # ---------------- RQ3
    text(r'\subsection*{RQ3: User study}',
         r'% TODO: align this summary of RQ3 with its wording in the paper.',
         r'RQ3 asks how real users experience the three algorithms. Participants taught a robot their '
         r'preferences in a physical handover task and a social gesture task, ranking queries of three '
         r'trajectories with each algorithm, and rated each algorithm\textquotesingle{}s ease of use (EOU) '
         r'and behavioral adaptation (BA) on 0--8 scales before ranking the three algorithms. '
         r'\textbf{H5} predicts higher BA than Infogain, \textbf{H6} higher EOU than CMA-ES, and '
         r'\textbf{H7} a preference for CMA-ES-IG over both baselines.')
    colspec, header, rows, n = user_study_main()
    parts.append(table('tab:app_user_study', rf'User study ($n={n}$ participants, each averaged over the two '
                       r'tasks): Wilcoxon signed-rank tests of CMA-ES-IG against each baseline (normal '
                       r'approximation, Pratt handling of zero differences), with Benjamini--Hochberg '
                       r'correction over the two comparisons within each measure and paired Hedges\textquotesingle{} $g_z$. '
                       r'Rank ranges from 0 (least) to 2 (most preferred).', colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_user_study}.} Each participant\textquotesingle{}s scores '
         r'are averaged over the two tasks, and each test compares CMA-ES-IG with one baseline within '
         r'participants. $W$ is the signed-rank statistic: smaller values mean participants more '
         r'consistently favored one algorithm. $p_{\mathrm{BH}}$ corrects for the two comparisons per '
         r'measure, and $g_z$ is the standardized within-participant difference. Participants rated '
         r'CMA-ES-IG as easier to use than CMA-ES and ranked it above both baselines; its higher BA than '
         r'Infogain was significant before correction and a trend after it.')
    colspec, header, rows = user_study_tasks()
    parts.append(table('tab:app_user_study_tasks', r'Exploratory per-task Wilcoxon signed-rank tests of '
                       r'CMA-ES-IG against each baseline (paired by session; uncorrected).',
                       colspec, header, rows, wide=False))
    text(r'\noindent\textit{Reading \autoref{tab:app_user_study_tasks}.} The same comparisons within each '
         r'task separately. With only 13 to 14 sessions per task these tests have little power and are '
         r'uncorrected, so they indicate where effects appear rather than confirm them.')
    colspec, header, rows, n_both, (a_ba, a_eou) = user_study_models()
    parts.append(table('tab:app_user_study_models', rf'Task effects and repeated-measures models. Task: '
                       rf'Wilcoxon signed-rank test of each participant\textquotesingle{{}}s handover vs.\ gesture '
                       rf'mean ($n={n_both}$ participants with both tasks). Models: likelihood-ratio $p$ values '
                       r'for each effect in linear mixed models and aligned-rank-transform models with '
                       r'random intercepts for participant and session. Rank sums to the same value in '
                       r'every session, so it has no task effect. Internal consistency (pilots excluded): '
                       rf'Cronbach\textquotesingle{{}}s $\alpha={a_ba:.2f}$ (BA) and ${a_eou:.2f}$ (EOU).',
                       colspec, header, rows))
    text(r'\noindent\textit{Reading \autoref{tab:app_user_study_models}.} These checks support the analysis '
         r'in \autoref{tab:app_user_study}. The task column shows that ratings did not differ between the '
         r'handover and gesture tasks, and the interaction columns show that the algorithms did not '
         r'compare differently across tasks, which justifies averaging each participant over the two '
         r'tasks. The models also test the overall effect of algorithm while treating each session as a '
         r'repeated measure: it is significant for rank, but not for EOU or BA, so the EOU and BA results '
         r'rest on the specific pairwise comparisons in \autoref{tab:app_user_study}.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(parts))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
