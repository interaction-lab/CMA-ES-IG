"""
Regenerate Figures 2 and 4 and Tables 1-3 of the IJRR submission with the
Habibian et al. (THRI 2022) informative+revealing baseline added.

Outputs everything to rebuttal/graphs/.

AUC convention matches the submission: trapezoid(y) / 30, i.e. normalised by the
number of queries.  Verified by reproducing the published Infogain row of
Table 1 (.849 / .606 / .374 alignment, .332 / 1.243 / 2.115 regret) to three
decimals.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
HYPER = os.path.join(HERE, 'graphs', 'data')
OUT = os.path.join(HERE, 'graphs')
SIM = os.path.join(ROOT, 'simulated_experiments')

os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

DIMS = [4, 8, 16, 32]
ITEMS = 4          # items per query in the parameter-estimation task (--items)
NQ = 30

METHODS = ['IG', 'Habibian-IR-log', 'CMA-ES', 'CMA-ES-IG']
LABEL = {'IG': 'Infogain', 'CMA-ES': 'CMA-ES', 'CMA-ES-IG': 'CMA-ES-IG',
         'Habibian-IR-log': 'Informative+Revealing'}
COLOR = {'IG': '#ababab', 'CMA-ES': '#0770bb', 'CMA-ES-IG': '#f49249',
         'Habibian-IR-log': '#4d4d4d'}

DOMAINS = [('lunar_lander', 'Lunar Lander'), ('driving', 'Driving'),
           ('pylips_appearance', 'Robot Face Design'), ('blossom_voice', 'Robot Voice Design')]

# Ablation over how CMA-ES-IG approximates information gain.  kmeans@100 is the
# deployed rule; the three @20 variants share a pool size so the selection rule
# is isolated from the pool size.
ABLATION = ['IGapprox-kmeans100', 'IGapprox-kmeans20', 'IGapprox-kmedoids20',
            'IGapprox-exact20', 'IGapprox-ellipsoid', 'IGapprox-ellipgrad',
            'IGapprox-ellipgradfast']
# Ellipsoid confidence-mass sweep, run with the analytic-gradient optimiser so
# every point gets the same iteration budget at every dimension.
MASS_SWEEP = [1, 5, 10, 25, 50, 75, 90, 99]
ABL_LABEL = {
    'IGapprox-kmeans100':  r'$k$-means, pool 100 (deployed)',
    'IGapprox-kmeans20':   r'$k$-means, pool 20',
    'IGapprox-kmedoids20': r'$k$-medoids, pool 20',
    'IGapprox-exact20':    r'exact (exhaustive), pool 20',
    'IGapprox-ellipsoid':  r'ellipsoid 90\%, fin. diff.',
    'IGapprox-ellipgrad':  r'ellipsoid 90\%, analytic grad.',
    'IGapprox-ellipgradfast': r'ellipsoid 90\%, grad. + subsample',
}
ABL_PLAIN = {k: v.replace('$', '').replace(r'\%', '%').replace('\\', '')
             for k, v in ABL_LABEL.items()}
ABL_COLOR = {'IGapprox-kmeans100': '#f49249', 'IGapprox-kmeans20': '#b35806',
             'IGapprox-kmedoids20': '#2166ac', 'IGapprox-exact20': '#1b7837',
             'IGapprox-ellipsoid': '#c2a5cf', 'IGapprox-ellipgrad': '#762a83',
             'IGapprox-ellipgradfast': '#9970ab'}


def auc(arr):
    """Area under each row's curve (trapezoidal rule) divided by the number of intervals,
    i.e. the curve's average height.  Alignment has 31 points (it starts with the
    pre-query value), the other per-query metrics 30, so they are divided by 30 and 29."""
    return np.trapezoid(arr, axis=1) / (arr.shape[1] - 1)


# Infogain was not re-run at every dimension under the corrected choice
# simulation (it is by far the slowest method); where the new run is missing,
# fall back to the earlier-protocol result.
HYPER_FALLBACK = {'IG': os.path.join(HERE, 'graphs', 'data_oldprotocol')}


def load_hyper(method, metric, dim):
    for d in (HYPER, HYPER_FALLBACK.get(method)):
        p = f'{d}/{method}_{metric}_{ITEMS}items_dim{dim}.npy'
        if d and os.path.exists(p):
            return np.load(p)
    return None


def legend_entries(axes):
    """Handles from every panel, deduplicated, so a series missing from the
    first panel still gets a legend entry."""
    seen = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    return list(seen.values()), list(seen.keys())


# Robot-domain CSVs live in data/results/; the resampling-belief runs
# (run_domains_rs.py) write to data/results_rs/.
DOMAIN_RESULTS = 'results'


def load_domain(domain, method):
    p = f'{SIM}/{domain}/data/{DOMAIN_RESULTS}/{method}_dim4_items5_nodiscretization.csv'
    return pd.read_csv(p) if os.path.exists(p) else None


def domain_curves(domain, method, metric):
    df = load_domain(domain, method)
    if df is None:
        return None
    piv = df.pivot_table(index='trial', columns='query_num', values=metric)
    return piv.to_numpy()


# ---- normalized regret: regret / (R_best - R_worst) over each simulated user's evaluation
# trajectories, so it lies in [0, 1] at every dimension and in every domain.  The per-user
# ranges are reconstructed from the trial seeds (see run_domains_rs.reward_range and the
# *_regret_range_* files next to the data).
def hyper_regret_range(method, dim):
    p = f'{HYPER}/{method}_regret_range_{ITEMS}items_dim{dim}.npy'
    return np.load(p) if os.path.exists(p) else None


def hyper_regret_norm(method, dim):
    a, r = load_hyper(method, 'regret', dim), hyper_regret_range(method, dim)
    return None if a is None or r is None else a / r[:, None]


def metric_values(method, metric, dim):
    """Per-user curves for the tables and tests; regret is normalized by each user's range."""
    return hyper_regret_norm(method, dim) if metric == 'regret' else load_hyper(method, metric, dim)


def domain_values(domain, method, metric):
    return domain_regret_norm(domain, method) if metric == 'regret' else domain_curves(domain, method, metric)


def domain_regret_norm(domain, method):
    a = domain_curves(domain, method, 'regret')
    p = f'{SIM}/{domain}/data/{DOMAIN_RESULTS}/{method}_regret_range_dim4_items5_nodiscretization.npy'
    if a is None or not os.path.exists(p):
        return None
    return a / np.load(p)[:, None]          # trials are in sorted order in both


def fmt(mean, se, bold=False, prec=3):
    body = f'${mean:.{prec}f} \\pm {{\\scriptstyle {se:.{prec}f}}}$'
    if bold:
        body = f'$\\bm{{{mean:.{prec}f}}} \\pm {{\\scriptstyle {se:.{prec}f}}}$'
    return body


# ---------------------------------------------------------------- Figure 2
# Rows shared by every per-iteration figure: (metric, label, draw a zero line).
# Rotated y-labels would turn up/down arrows sideways, so the direction is spelled out.
def _row_label(name, better):
    return r'$\mathbf{' + name.replace(' ', r'\ ') + '}$' + f'\n({better} is better)'


CURVE_ROWS = [('alignment', _row_label('Alignment', 'higher'), False),
              ('regret', _row_label('Normalized regret', 'lower'), False),
              ('quality', _row_label('Quality', 'higher'), True)]
# Row labels: large and bold so each metric is identifiable at a glance (Mean +/- SE is in
# the captions).
YLABEL = dict(fontsize=15, labelpad=8, linespacing=1.4)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'   # bold row names in the figure's serif face
TICKS = dict(labelsize=11)


def two_decimal_ticks(ax, max_ticks=7):
    """Ticks on a 0.01 / 0.02 / 0.05 / 0.1 grid, labelled with two decimals.  The spacing is
    the finest that gives at most `max_ticks` over the panel's range, so labels never
    repeat (a 0.025 or 0.005 spacing would round to duplicates at two decimals)."""
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    lo, hi = ax.get_ylim()
    step = next((s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if (hi - lo) / s <= max_ticks), 1.0)
    # widen the axis to whole steps so even a narrow panel shows at least three labels;
    # regret is never negative, so the axis never goes below 0
    lo2 = max(0.0, np.floor(lo / step + 1e-9) * step)
    hi2 = np.ceil(hi / step - 1e-9) * step
    while round((hi2 - lo2) / step) < 2:
        hi2 += step
    ax.set_ylim(lo2, hi2)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def _metric_grid(columns, curve, series, color, label, name, alpha=0.3, ncol=4):
    """One row per metric (alignment, regret, quality), one column per dimension or
    domain.  curve(column, series, metric) returns a trials x iterations array or None."""
    fig, axes = plt.subplots(len(CURVE_ROWS), len(columns), figsize=(18, 11))
    for r, (metric, mlabel, zero_line) in enumerate(CURVE_ROWS):
        for c, (key, title) in enumerate(columns):
            ax = axes[r, c]
            for m in series:
                a = curve(key, m, metric)
                if a is None:
                    continue
                mu = np.nanmean(a, axis=0)
                se = np.nanstd(a, axis=0) / np.sqrt(len(a))
                ax.fill_between(range(len(mu)), mu - se, mu + se, alpha=alpha, color=color[m])
                ax.plot(mu, color=color[m], linewidth=2, label=label[m])
            if r == 0:
                ax.set_title(title, fontsize=20, pad=10)
            if r == len(CURVE_ROWS) - 1:
                ax.set_xlabel('Iteration')
            if c == 0:
                ax.set_ylabel(mlabel, **YLABEL)
            ax.tick_params(**TICKS)
            if zero_line:
                ax.axhline(y=0, color='black', linewidth=1)
            if metric == 'regret':
                two_decimal_ticks(ax)
            if metric == 'alignment':
                # ticks every 0.1 with one decimal in every panel (narrow panels would
                # otherwise tick every 0.05 and print duplicate one-decimal labels)
                from matplotlib.ticker import FormatStrFormatter, MultipleLocator
                ax.yaxis.set_major_locator(MultipleLocator(0.1))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(True, color='lightgrey', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    handles, labels = legend_entries(axes)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote {name}.{{pdf,png}}')


HYPER_METRIC = {'alignment': 'alignment', 'regret': 'regret', 'quality': 'per_query_alignment'}


def hyper_curve(method, metric, dim):
    """Per-query curve.  The stored alignment starts with the pre-query value (0), so drop
    it to line up with regret and quality, which are recorded after each query."""
    if metric == 'regret':
        return hyper_regret_norm(method, dim)
    a = load_hyper(method, HYPER_METRIC[metric], dim)
    if a is not None and metric == 'alignment' and a.shape[1] == NQ + 1:
        a = a[:, 1:]
    return a
DOMAIN_METRIC = {'alignment': 'alignment', 'regret': 'regret', 'quality': 'quality_avg'}


def figure2():
    _metric_grid([(d, f'$d = {d}$') for d in DIMS],
                 lambda d, m, metric: hyper_curve(m, metric, d),
                 METHODS, COLOR, LABEL, 'figure2_quality_hypersphere')


# ---------------------------------------------------------------- Figure 4
def figure4():
    _metric_grid(DOMAINS, lambda dom, m, metric: (domain_regret_norm(dom, m) if metric == 'regret'
                                                  else domain_curves(dom, m, DOMAIN_METRIC[metric])),
                 METHODS, COLOR, LABEL, 'figure4_quality_domains')


# ---------------------------------------------------------------- Table 1
def table1():
    cells, raw = {}, {}
    for m in METHODS:
        for metric in ('alignment', 'regret'):
            for d in DIMS:
                a = metric_values(m, metric, d)
                raw[(m, metric, d)] = auc(a) if a is not None else None

    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Marginal means for accuracy metrics in the parameter estimation task, '
             r'with the informative+revealing baseline of Habibian et al. added. '
             r'Mean ($\pm$ standard error) of the AUC of alignment and normalized regret over 100 simulated '
             r'users. Bolded values denote the highest-performing method.}',
             r'\label{tab:quantitative_results}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l cccc cccc}', r'\toprule',
             r'\multirow{2}{*}{\textbf{Algorithm}} & \multicolumn{4}{c}{\textbf{AUC Alignment ($\uparrow$)}} & \multicolumn{4}{c}{\textbf{AUC Normalized Regret ($\downarrow$)}} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
             r'& $d=4$ & $d=8$ & $d=16$ & $d=32$ & $d=4$ & $d=8$ & $d=16$ & $d=32$ \\',
             r'\midrule']

    best = {}
    for metric, better in (('alignment', max), ('regret', min)):
        for d in DIMS:
            vals = {m: raw[(m, metric, d)].mean() for m in METHODS if raw[(m, metric, d)] is not None}
            best[(metric, d)] = better(vals, key=vals.get) if vals else None

    for m in METHODS:
        row = [LABEL[m]]
        for metric in ('alignment', 'regret'):
            for d in DIMS:
                v = raw[(m, metric, d)]
                if v is None:
                    row.append('--')
                else:
                    row.append(fmt(v.mean(), v.std() / np.sqrt(len(v)),
                                   bold=(best[(metric, d)] == m)))
        lines.append(' & '.join(row) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{OUT}/table1_accuracy_hypersphere.tex', 'w').write('\n'.join(lines) + '\n')
    print('wrote table1_accuracy_hypersphere.tex')
    return raw


# ---------------------------------------------------------------- Table 3
def table3():
    raw = {}
    for dom, _ in DOMAINS:
        for m in METHODS:
            for metric in ('alignment', 'regret'):
                a = domain_values(dom, m, metric)
                raw[(dom, m, metric)] = auc(a) if a is not None else None

    best = {}
    for dom, _ in DOMAINS:
        for metric, better in (('alignment', max), ('regret', min)):
            vals = {m: raw[(dom, m, metric)].mean() for m in METHODS
                    if raw[(dom, m, metric)] is not None}
            best[(dom, metric)] = better(vals, key=vals.get) if vals else None

    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Mean accuracy metrics across simulated robot tasks, with the '
             r'informative+revealing baseline of Habibian et al. added. AUC of alignment ($\uparrow$) '
             r'and normalized regret ($\downarrow$), 100 simulated users per domain.}',
             r'\label{tab:simulated_results}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l cc cc cc cc}', r'\toprule',
             r'\multirow{2}{*}{\textbf{Algorithm}} & ' +
             ' & '.join(rf'\multicolumn{{2}}{{c}}{{\textbf{{{t}}}}}' for _, t in DOMAINS) + r' \\',
             ' '.join(rf'\cmidrule(lr){{{2+2*i}-{3+2*i}}}' for i in range(len(DOMAINS))),
             r'& ' + ' & '.join([r'Alignment $\uparrow$ & Norm.\ Regret $\downarrow$'] * len(DOMAINS)) + r' \\',
             r'\midrule']

    for m in METHODS:
        row = [LABEL[m]]
        for dom, _ in DOMAINS:
            for metric in ('alignment', 'regret'):
                v = raw[(dom, m, metric)]
                row.append('--' if v is None else
                           fmt(v.mean(), v.std() / np.sqrt(len(v)),
                               bold=(best[(dom, metric)] == m)))
        lines.append(' & '.join(row) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{OUT}/table3_domains.tex', 'w').write('\n'.join(lines) + '\n')
    print('wrote table3_domains.tex')
    return raw


# ---------------------------------------------------------------- Table 2
def table2(repeats=(3, 20)):
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'preference-learning-from-selection'))
    sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    sys.path.insert(0, HERE)
    import time
    from irlpreference.input_models import LuceShepardChoice
    from irlpreference.reward_parameterizations import MonteCarloLinearReward
    from irlpreference.query_generation import InfoGainQueryGenerator
    from cmaes_generators import CMAESGenerator, CMAESIGGenerator
    from habiban import InformativeRevealingQueryGenerator

    makers = {
        'CMA-ES-IG': lambda d: CMAESIGGenerator(d, [(-1, 1)] * d, ITEMS),
        'CMA-ES': lambda d: CMAESGenerator(d, [(-1, 1)] * d, ITEMS),
        'Habibian-IR-log': lambda d: InformativeRevealingQueryGenerator(
            d, [(-1, 1)] * d, lam=1.0, score_log_belief=True),
        'IG': lambda d: InfoGainQueryGenerator([(-1, 1)] * d),
    }
    order = ['CMA-ES-IG', 'CMA-ES', 'IG', 'Habibian-IR-log']

    timings = {}
    for m in order:
        for d in DIMS:
            est = MonteCarloLinearReward(d, number_samples=10_000)
            cm = LuceShepardChoice(rationality=10)
            g = makers[m](d)
            n = repeats[0] if m == 'IG' else repeats[1]
            g.get_query(ITEMS, est, cm)                      # warm up
            t0 = time.perf_counter()
            for _ in range(n):
                g.get_query(ITEMS, est, cm)
            timings[(m, d)] = (time.perf_counter() - t0) / n * 1000

    def cell(ms):
        return f'{ms:.2f} ms' if ms < 1 else f'{ms:.1f} ms' if ms < 100 else f'{ms:.0f} ms'

    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Mean runtime (in milliseconds) for query generation methods, '
             r'measured on identical hardware. Values averaged across repeated calls at each '
             r'representation dimension ($d$).}',
             r'\label{tab:runtime}', r'\begin{tabular}{l cccc}', r'\toprule',
             r'\textbf{Algorithm} & $d=4$ & $d=8$ & $d=16$ & $d=32$ \\', r'\midrule']
    for m in order:
        lines.append(LABEL[m] + ' & ' + ' & '.join(cell(timings[(m, d)]) for d in DIMS) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/table2_runtime.tex', 'w').write('\n'.join(lines) + '\n')
    print('wrote table2_runtime.tex')
    return timings


def ablation_timings(repeats=(3, 30)):
    """Re-measure each approximation's query-generation cost on this machine."""
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'preference-learning-from-selection'))
    sys.path.insert(0, HERE)
    import time
    from irlpreference.input_models import LuceShepardChoice
    from irlpreference.reward_parameterizations import MonteCarloLinearReward
    from cmaes_ig_ablation import CMAESIGAblation

    cfg = {'IGapprox-kmeans100': ('kmeans', 100), 'IGapprox-kmeans20': ('kmeans', 20),
           'IGapprox-kmedoids20': ('kmedoids', 20), 'IGapprox-exact20': ('exact', 20),
           'IGapprox-ellipsoid': ('ellipsoid', 20)}
    out = {}
    for m, (meth, pool) in cfg.items():
        for d in DIMS:
            est = MonteCarloLinearReward(d, number_samples=10_000)
            cm = LuceShepardChoice(rationality=10)
            g = CMAESIGAblation(d, [(-1, 1)] * d, population_size=4,
                                method=meth, pool_size=pool)
            q = g.get_query(4, est, cm)
            cm.tell_input(1, q)
            est.update(cm.get_probability_of_input)
            n = repeats[0] if meth in ('exact', 'ellipsoid') else repeats[1]
            t0 = time.perf_counter()
            for _ in range(n):
                g.get_query(4, est, cm)
            out[(m, d)] = (time.perf_counter() - t0) / n * 1000
    return out


# ---------------------------------------------------------------- stats
def stats_report(t1, t3, timings):
    out = ['# Regenerated results with the Habibian et al. baseline', '',
           'AUC convention: trapezoid(y) / (number of intervals), i.e. the curve average;',
           'regret is normalized by each user\'s reward range. Welch t-tests. 100 simulated users, 30 queries, 4 items (hypersphere)',
           'and 5 items (robot domains).', '',
           '## Pairwise t-tests, CMA-ES-IG vs each baseline', '']

    out.append('### Hypersphere (Table 1 / Figure 2)')
    out.append('')
    out.append('| metric | d | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |')
    out.append('|---|---|---|---|---|---|---|---|')
    for metric in ('alignment', 'regret', 'per_query_alignment'):
        for d in DIMS:
            a = metric_values('CMA-ES-IG', metric, d)
            if a is None:
                continue
            av = auc(a)
            for m in METHODS:
                if m == 'CMA-ES-IG':
                    continue
                b = metric_values(m, metric, d)
                if b is None:
                    continue
                bv = auc(b)
                t, p = stats.ttest_ind(av, bv, equal_var=False)
                sd = np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2)
                dd = (av.mean() - bv.mean()) / sd if sd else np.nan
                out.append(f'| {metric} | {d} | {LABEL[m]} | {av.mean():.3f} | {bv.mean():.3f} '
                           f'| {t:.2f} | {p:.2e} | {dd:+.2f} |')

    out += ['', '### Robot domains (Table 3 / Figure 4)', '',
            '| domain | metric | baseline | CMA-ES-IG | baseline mean | t | p | Cohen d |',
            '|---|---|---|---|---|---|---|---|']
    for dom, title in DOMAINS:
        for metric in ('alignment', 'regret', 'quality_avg'):
            a = domain_values(dom, 'CMA-ES-IG', metric)
            if a is None:
                continue
            av = auc(a)
            for m in METHODS:
                if m == 'CMA-ES-IG':
                    continue
                b = domain_values(dom, m, metric)
                if b is None:
                    continue
                bv = auc(b)
                t, p = stats.ttest_ind(av, bv, equal_var=False)
                sd = np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2)
                dd = (av.mean() - bv.mean()) / sd if sd else np.nan
                out.append(f'| {title} | {metric} | {LABEL[m]} | {av.mean():.3f} '
                           f'| {bv.mean():.3f} | {t:.2f} | {p:.2e} | {dd:+.2f} |')

    out += ['', '## Runtime (Table 2), measured on this machine', '',
            '| algorithm | ' + ' | '.join(f'd={d}' for d in DIMS) + ' |',
            '|---|' + '---|' * len(DIMS)]
    for m in ['CMA-ES-IG', 'CMA-ES', 'IG', 'Habibian-IR-log']:
        out.append(f'| {LABEL[m]} | ' + ' | '.join(f'{timings[(m,d)]:.1f} ms' for d in DIMS) + ' |')

    open(f'{OUT}/README.md', 'w').write('\n'.join(out) + '\n')
    print('wrote README.md')




# ---------------------------------------------------------------- ablation
def figure5():
    _metric_grid([(d, f'$d = {d}$') for d in DIMS],
                 lambda d, m, metric: hyper_curve(m, metric, d),
                 ABLATION, ABL_COLOR, {m: ABL_LABEL[m].replace(r'\%', '%') for m in ABLATION},
                 'figure5_ig_ablation', alpha=0.25)


def table4(timings=None):
    rows = {}
    for m in ABLATION:
        for metric in ('alignment', 'regret', 'per_query_alignment'):
            for d in DIMS:
                a = metric_values(m, metric, d)
                rows[(m, metric, d)] = auc(a) if a is not None else None

    best = {}
    for metric, better in (('alignment', max), ('regret', min),
                           ('per_query_alignment', max)):
        for d in DIMS:
            vals = {m: rows[(m, metric, d)].mean() for m in ABLATION
                    if rows[(m, metric, d)] is not None}
            best[(metric, d)] = better(vals, key=vals.get) if vals else None

    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Ablation over how CMA-ES-IG approximates information gain when '
             r'selecting which sampled candidates to show. All variants share identical '
             r'CMA-ES machinery and differ only in the selection rule. Exhaustive search '
             r'over the deployed pool of 100 samples is intractable '
             r'($\binom{100}{4} = 3{,}921{,}225$ subsets per query), so the exact variant '
             r'uses a pool of 20; $k$-means is reported at both pool sizes as a control. '
             r'100 simulated users.}',
             r'\label{tab:ig_ablation}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l cccc cccc}', r'\toprule',
             r'\multirow{2}{*}{\textbf{IG approximation}} & '
             r'\multicolumn{4}{c}{\textbf{AUC Quality ($\uparrow$)}} & '
             r'\multicolumn{4}{c}{\textbf{AUC Normalized Regret ($\downarrow$)}} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
             r'& $d=4$ & $d=8$ & $d=16$ & $d=32$ & $d=4$ & $d=8$ & $d=16$ & $d=32$ \\',
             r'\midrule']
    for m in ABLATION:
        row = [ABL_LABEL[m]]
        for metric in ('per_query_alignment', 'regret'):
            for d in DIMS:
                v = rows[(m, metric, d)]
                row.append('--' if v is None else
                           fmt(v.mean(), v.std() / np.sqrt(len(v)),
                               bold=(best[(metric, d)] == m)))
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{OUT}/table4_ig_ablation.tex', 'w').write('\n'.join(lines) + '\n')
    print('wrote table4_ig_ablation.tex')
    return rows


# Non-inferiority margins per metric (same for RQ1 and RQ2), on each metric's 0-1 scale.
NI_MARGIN = {'alignment': 0.05, 'regret': 0.05}


def _welch_ci(a, b, level=0.95):
    """Difference in means a - b with a Welch confidence interval."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = a.mean() - b.mean()
    return d, d - t * se, d + t * se


def noninferiority_tables(margins=NI_MARGIN):
    """CMA-ES-IG minus each baseline on AUC alignment (higher is better) and AUC normalized
    regret (lower is better), with Welch 95% CIs.  CMA-ES-IG is non-inferior when the CI
    excludes a loss larger than `margin`: lower bound > -margin for alignment, upper bound
    < margin for regret (one-sided alpha = 0.025)."""
    baselines = [m for m in METHODS if m != 'CMA-ES-IG']
    settings = {'rq1': ([(d, f'$d={d}$') for d in DIMS],
                        lambda key, m, metric: metric_values(m, metric, key)),
                'rq2': (DOMAINS, lambda key, m, metric: domain_values(key, m, metric))}
    captions = {'rq1': 'parameter estimation task (RQ1)', 'rq2': 'simulated robot domains (RQ2)'}
    summary = {}
    for name, (cols, get) in settings.items():
        lines = [r'\begin{table}[t]', r'\centering',
                 r'\caption{Non-inferiority of CMA-ES-IG in the ' + captions[name] + r'. '
                 r'Difference in mean AUC (CMA-ES-IG minus baseline) with Welch 95\% '
                 r'confidence intervals over 100 simulated users. CMA-ES-IG is non-inferior '
                 r'($\checkmark$) when the interval excludes a loss larger than '
                 rf'$\Delta={margins["alignment"]}$ for alignment and $\Delta={margins["regret"]}$ for '
                 r'normalized regret.}',
                 r'\label{tab:noninferiority_' + name + '}',
                 r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{ll cc}', r'\toprule',
                 r'\textbf{Baseline} & & \textbf{Alignment ($\uparrow$)} & '
                 r'\textbf{Norm.\ regret ($\downarrow$)} \\', r'\midrule']
        passed = total = 0
        for b in baselines:
            for i, (key, title) in enumerate(cols):
                cells = []
                for metric, sign in (('alignment', 1), ('regret', -1)):
                    x, y = get(key, 'CMA-ES-IG', metric), get(key, b, metric)
                    if x is None or y is None:
                        cells.append('--')
                        continue
                    d, lo, hi = _welch_ci(auc(x), auc(y))
                    margin = margins[metric]
                    ok = lo > -margin if sign == 1 else hi < margin
                    passed += ok
                    total += 1
                    cells.append(f'${d:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]' + (r' $\checkmark$' if ok else ''))
                lines.append((LABEL[b] if i == 0 else '') + f' & {title} & ' + ' & '.join(cells) + r' \\')
            lines.append(r'\midrule' if b != baselines[-1] else r'\bottomrule')
        lines += [r'\end{tabular}', r'}', r'\end{table}']
        open(f'{OUT}/table_noninferiority_{name}.tex', 'w').write('\n'.join(lines) + '\n')
        summary[name] = (passed, total)
        print(f'wrote table_noninferiority_{name}.tex: {passed}/{total} non-inferior at margins {margins}')
    return summary


def ablation_report(rows, timings):
    out = ['', '## CMA-ES-IG information-gain approximation ablation', '',
           'All variants share identical CMA-ES machinery; only the rule for choosing',
           'which sampled candidates to show differs. Exhaustive search over the deployed',
           '100-sample pool is intractable (3,921,225 subsets per query), so `exact` uses a',
           '20-sample pool and `k-means, pool 20` is included as the control that separates',
           'the selection rule from the pool size.', '',
           '| variant | metric | ' + ' | '.join(f'd={d}' for d in DIMS) + ' |',
           '|---|---|' + '---|' * len(DIMS)]
    for m in ABLATION:
        for metric, lab in (('per_query_alignment', 'quality'), ('regret', 'regret'),
                            ('alignment', 'alignment')):
            cells = []
            for d in DIMS:
                v = rows[(m, metric, d)]
                cells.append('--' if v is None else f'{v.mean():.3f}')
            out.append(f'| {ABL_PLAIN[m]} | {lab} | ' + ' | '.join(cells) + ' |')

    out += ['', '### Paired comparison against the deployed rule (k-means, pool 100)', '',
            '| variant | metric | d | delta | t | p |', '|---|---|---|---|---|---|']
    for metric, lab in (('per_query_alignment', 'quality'), ('regret', 'regret')):
        for d in DIMS:
            base = rows[('IGapprox-kmeans100', metric, d)]
            if base is None:
                continue
            for m in ABLATION[1:]:
                v = rows[(m, metric, d)]
                if v is None:
                    continue
                t, p = stats.ttest_ind(v, base, equal_var=False)
                out.append(f'| {ABL_PLAIN[m]} | {lab} | {d} | {v.mean()-base.mean():+.3f} '
                           f'| {t:.2f} | {p:.2e} |')

    if timings:
        out += ['', '### Query-generation cost of each approximation (ms, idle machine)', '',
                '| variant | ' + ' | '.join(f'd={d}' for d in DIMS) + ' |',
                '|---|' + '---|' * len(DIMS)]
        for m in ABLATION:
            if (m, DIMS[0]) in timings:
                out.append(f'| {ABL_PLAIN[m]} | ' +
                           ' | '.join(f'{timings[(m,d)]:.1f}' for d in DIMS) + ' |')
    with open(f'{OUT}/README.md', 'a') as f:
        f.write('\n'.join(out) + '\n')
    print('appended ablation section to README.md')


def _mass_auc(mass, metric, dim):
    a = metric_values(f'IGapprox-egrad{mass:02d}', metric, dim)
    return auc(a) if a is not None else None


def figure6():
    """AUC against ellipsoid confidence mass, versus the deployed k-means rule."""
    fig, axes = plt.subplots(2, len(DIMS), figsize=(18, 7.5))
    rows = [('per_query_alignment', _row_label('AUC Quality', 'higher')),
            ('regret', _row_label('AUC norm. regret', 'lower'))]
    for c, dim in enumerate(DIMS):
        for r, (metric, lab) in enumerate(rows):
            ax = axes[r, c]
            xs, ys, es = [], [], []
            for mass in MASS_SWEEP:
                v = (auc(hyper_regret_norm(f'IGapprox-egrad{mass:02d}', dim)) if metric == 'regret'
                     else _mass_auc(mass, metric, dim))
                if v is None:
                    continue
                xs.append(MASS_SWEEP.index(mass))      # evenly spaced; labelled with the mass
                ys.append(v.mean())
                es.append(v.std() / np.sqrt(len(v)))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker='o', color='#762a83',
                            linewidth=2, capsize=3,
                            label='ellipsoid (analytic gradient)')
            base = (hyper_regret_norm('IGapprox-kmeans100', dim) if metric == 'regret'
                    else load_hyper('IGapprox-kmeans100', metric, dim))
            if base is not None:
                b = auc(base)
                se = b.std() / np.sqrt(len(b))
                ax.axhline(b.mean(), color='#f49249', linestyle='--', linewidth=2,
                           label='$k$-means, pool 100 (deployed)')
                ax.axhspan(b.mean() - se, b.mean() + se, color='#f49249', alpha=0.25)
            # categorical axis: on a log scale 75/90/99 overlap
            ax.set_xticks(range(len(MASS_SWEEP)))
            ax.set_xticklabels([str(v) for v in MASS_SWEEP])
            if r == 0:
                ax.set_title(f'$d = {dim}$')
            if r == len(rows) - 1:
                ax.set_xlabel('Ellipsoid confidence mass (%)')
            if c == 0:
                ax.set_ylabel(lab, **YLABEL)
            ax.tick_params(**TICKS)
            if metric == 'regret':
                two_decimal_ticks(ax)
            ax.grid(True, color='lightgrey', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    handles, labels = legend_entries(axes)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/figure6_ellipsoid_mass.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote figure6_ellipsoid_mass.{pdf,png}')


def table5():
    """Ellipsoid confidence-mass sweep."""
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Ellipsoid confidence-mass sweep for the information-gain '
             r'approximation, run with the analytic-gradient optimiser so that every '
             r'mass receives the same iteration budget at every dimension. The deployed '
             r'$k$-means rule is shown for reference. 100 simulated users.}',
             r'\label{tab:ellipsoid_mass}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{l cccc cccc}', r'\toprule',
             r'\multirow{2}{*}{\textbf{Confidence mass}} & '
             r'\multicolumn{4}{c}{\textbf{AUC Quality ($\uparrow$)}} & '
             r'\multicolumn{4}{c}{\textbf{AUC Normalized Regret ($\downarrow$)}} \\',
             r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
             r'& $d=4$ & $d=8$ & $d=16$ & $d=32$ & $d=4$ & $d=8$ & $d=16$ & $d=32$ \\',
             r'\midrule']
    # every row (masses + the k-means reference), so the best value per column can be bolded
    rows = [(str(mass) + r'\%', lambda metric, d, m=mass: _mass_auc(m, metric, d)) for mass in MASS_SWEEP]
    rows.append((r'$k$-means, pool 100', lambda metric, d: (
        lambda a: auc(a) if a is not None else None)(metric_values('IGapprox-kmeans100', metric, d))))
    vals = {(name, metric, d): get(metric, d) for name, get in rows
            for metric in ('per_query_alignment', 'regret') for d in DIMS}
    best = {}
    for metric, better in (('per_query_alignment', max), ('regret', min)):
        for d in DIMS:
            means = {name: v.mean() for (name, m, dd), v in vals.items()
                     if m == metric and dd == d and v is not None}
            best[(metric, d)] = better(means, key=means.get) if means else None
    for k, (name, _) in enumerate(rows):
        if k == len(rows) - 1:
            lines.append(r'\midrule')
        row = [name]
        for metric in ('per_query_alignment', 'regret'):
            for d in DIMS:
                v = vals[(name, metric, d)]
                row.append('--' if v is None else
                           fmt(v.mean(), v.std() / np.sqrt(len(v)), bold=(best[(metric, d)] == name)))
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{OUT}/table5_ellipsoid_mass.tex', 'w').write('\n'.join(lines) + '\n')
    print('wrote table5_ellipsoid_mass.tex')


# ---------------------------------------------------------------- items-per-query figure
def figure_items(items=(3, 4, 5), out_dir=None):
    """AUC of alignment, normalized regret, and quality against the number of items per query
    K, for each dimension (parameter-estimation task).  Mean +/- SD over simulated users;
    item counts without results are skipped."""
    global ITEMS
    keep = ITEMS
    rows = [('alignment', _row_label('AUC alignment', 'higher')),
            ('regret', _row_label('AUC norm. regret', 'lower')),
            ('per_query_alignment', _row_label('AUC quality', 'higher'))]
    stats_ = {}
    for k in items:
        ITEMS = k
        for d in DIMS:
            for m in METHODS:
                for metric, _ in rows:
                    a = metric_values(m, metric, d)
                    if a is not None:
                        v = auc(a)
                        stats_[(k, d, m, metric)] = (v.mean(), v.std(ddof=1))      # error bars: SD
    ITEMS = keep
    present = [k for k in items if any(key[0] == k for key in stats_)]
    fig, axes = plt.subplots(len(rows), len(DIMS), figsize=(18, 11))
    offsets = dict(zip(METHODS, np.linspace(-0.12, 0.12, len(METHODS))))
    for r, (metric, label) in enumerate(rows):
        for c, d in enumerate(DIMS):
            ax = axes[r, c]
            for m in METHODS:
                pts = [(k, *stats_[(k, d, m, metric)]) for k in present if (k, d, m, metric) in stats_]
                if not pts:
                    continue
                ks, mu, se = map(np.array, zip(*pts))
                ax.errorbar(ks + offsets[m], mu, yerr=se, color=COLOR[m], marker='o', markersize=6,
                            linewidth=2, capsize=3, label=LABEL[m])
            if metric == 'per_query_alignment':
                ax.axhline(0, color='black', linewidth=1)
            ax.set_xticks(present)
            ax.set_xlim(min(present) - 0.4, max(present) + 0.4)
            if r == 0:
                ax.set_title(f'$d = {d}$', fontsize=20, pad=10)
            if r == len(rows) - 1:
                ax.set_xlabel('Items per query $K$')
            if c == 0:
                ax.set_ylabel(label, **YLABEL)
            ax.tick_params(**TICKS)
            from matplotlib.ticker import FormatStrFormatter
            if metric == 'regret':
                two_decimal_ticks(ax)
            elif metric == 'alignment':
                # AUC alignment spans as little as 0.02 at d=4, so it needs a third decimal
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(True, color='lightgrey', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    handles, labels = legend_entries(axes)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    out_dir = out_dir or OUT
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_dir}/figure_items_per_query.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'wrote figure_items_per_query.{{pdf,png}} (K = {present})')


# ---------------------------------------------------------------- table bar charts
# Quick-inspection views of Tables 1-5 as grouped bars (mean +/- SE).  Not for
# the paper; written to graphs/inspect/.
def _bars(ax, groups, series, get, color, label):
    width = 0.8 / len(series)
    for j, s in enumerate(series):
        xs, ys, es = [], [], []
        for i, g in enumerate(groups):
            v = get(s, g)
            if v is None:
                continue
            v = np.atleast_1d(v)
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(v.mean())
            es.append(v.std() / np.sqrt(len(v)) if len(v) > 1 else 0)
        if not xs:
            continue
        ax.bar(xs, ys, width, yerr=es, color=color[s], label=label[s],
               edgecolor='white', linewidth=1, capsize=2,
               error_kw={'elinewidth': 1, 'ecolor': '#333333'})
    ax.set_xticks(range(len(groups)))
    ax.grid(True, axis='y', color='lightgrey', linestyle='--')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_bars(fig, axes, name, ncol):
    handles, labels = legend_entries(axes)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(handles, labels, loc='lower center', ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)
    os.makedirs(f'{OUT}/inspect', exist_ok=True)
    fig.savefig(f'{OUT}/inspect/{name}.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'wrote inspect/{name}.png')


def table_bars(t1, timings, t3, t4):
    dlab = [f'$d={d}$' for d in DIMS]
    two = [('alignment', 'AUC Alignment ($\\uparrow$)'), ('regret', 'AUC Normalized Regret ($\\downarrow$)')]
    abl = [('per_query_alignment', 'AUC Quality ($\\uparrow$)'), ('regret', 'AUC Normalized Regret ($\\downarrow$)')]

    # Table 1: methods x dimension
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (metric, lab) in zip(axes, two):
        _bars(ax, DIMS, METHODS, lambda m, d: t1[(m, metric, d)], COLOR, LABEL)
        ax.set_xticklabels(dlab)
        ax.set_title(lab)
    fig.suptitle('Table 1: accuracy, hypersphere')
    _save_bars(fig, axes, 'table1_bars', len(METHODS))

    # Table 2: runtime, log scale since Infogain is orders of magnitude slower
    rt = ['CMA-ES-IG', 'CMA-ES', 'IG', 'Habibian-IR-log']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bars(ax, DIMS, rt, lambda m, d: timings.get((m, d)), COLOR, LABEL)
    ax.set_xticklabels(dlab)
    ax.set_yscale('log')
    ax.set_ylabel('ms per query (log scale)')
    ax.set_title('Table 2: query-generation runtime')
    _save_bars(fig, [ax], 'table2_bars', len(rt))

    # Table 3: methods x domain
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (metric, lab) in zip(axes, two):
        _bars(ax, [d for d, _ in DOMAINS], METHODS,
              lambda m, dom: t3[(dom, m, metric)], COLOR, LABEL)
        ax.set_xticklabels([t for _, t in DOMAINS])
        ax.set_title(lab)
    fig.suptitle('Table 3: accuracy, robot domains')
    _save_bars(fig, axes, 'table3_bars', len(METHODS))

    # Table 4: IG-approximation ablation
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, (metric, lab) in zip(axes, abl):
        _bars(ax, DIMS, ABLATION, lambda m, d: t4[(m, metric, d)], ABL_COLOR, ABL_PLAIN)
        ax.set_xticklabels(dlab)
        ax.set_title(lab)
    fig.suptitle('Table 4: IG-approximation ablation')
    _save_bars(fig, axes, 'table4_bars', 4)

    # Table 5: confidence mass is ordered, so one sequential hue light->dark;
    # the deployed k-means reference keeps its orange from the other figures.
    keys = [f'm{m:02d}' for m in MASS_SWEEP] + ['kmeans']
    ramp = plt.get_cmap('Purples')(np.linspace(0.3, 0.95, len(MASS_SWEEP)))
    col = dict(zip(keys, list(ramp) + ['#f49249']))
    lab5 = {**{f'm{m:02d}': f'{m}%' for m in MASS_SWEEP}, 'kmeans': 'k-means, pool 100'}

    def get5(k, d, metric):
        if k == 'kmeans':
            a = metric_values('IGapprox-kmeans100', metric, d)
            return auc(a) if a is not None else None
        return _mass_auc(int(k[1:]), metric, d)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, (metric, lab) in zip(axes, abl):
        _bars(ax, DIMS, keys, lambda k, d: get5(k, d, metric), col, lab5)
        ax.set_xticklabels(dlab)
        ax.set_title(lab)
    fig.suptitle('Table 5: ellipsoid confidence-mass sweep')
    _save_bars(fig, axes, 'table5_bars', len(keys))


def merge_shards(folder):
    """Combine finished *_shardNNN.npy files so partially complete runs can be plotted."""
    import glob, re, collections
    groups = collections.defaultdict(list)
    for f in glob.glob(f'{folder}/*_shard*.npy'):
        groups[re.sub(r'_shard\d+\.npy$', '.npy', f)].append(f)
    for out, fs in groups.items():
        fs.sort(key=lambda s: int(re.search(r'_shard(\d+)', s).group(1)))
        np.save(out, np.concatenate([np.load(f) for f in fs]))
    print(f'merged {len(groups)} shard groups in {os.path.basename(folder)}')


def runtime_from_table2():
    """Read the timings back from the last table2_runtime.tex instead of re-benchmarking."""
    names = {v: k for k, v in LABEL.items()}
    tm = {}
    for line in open(os.path.join(HERE, 'graphs', 'table2_runtime.tex')):
        if ' ms' in line:
            parts = [c.strip() for c in line.rstrip(' \\\n').split('&')]
            for d, c in zip(DIMS, parts[1:]):
                tm[(names[parts[0]], d)] = float(c.split()[0])
    return tm


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', choices=['current', 'rs'], default='current',
                    help="'rs': results with the resampling belief, written to graphs/rs/")
    ap.add_argument('--items', type=int, default=4,
                    help='items per query for the parameter-estimation task (output to graphs/rs_k<items>/)')
    ap.add_argument('--no-bench', action='store_true',
                    help='reuse Table 2 timings and skip the ablation timing benchmark')
    args = ap.parse_args()
    if args.data == 'rs':
        HYPER = os.path.join(HERE, 'graphs', 'data_rs')
        ITEMS = args.items
        OUT = os.path.join(HERE, 'graphs', 'rs' if ITEMS == 4 else f'rs_k{ITEMS}')
        HYPER_FALLBACK = {}          # Infogain is fully re-run under this belief
        DOMAIN_RESULTS = 'results_rs'
        os.makedirs(OUT, exist_ok=True)
        merge_shards(HYPER)
        import run_domains_rs
        run_domains_rs.merge()

    def step(fn, *a):
        # In preview mode a partially finished grid can leave a table short of
        # data; report and carry on rather than abort the whole rebuild.
        try:
            return fn(*a)
        except Exception as e:
            if args.data != 'rs':
                raise
            print(f'skipped {fn.__name__}: {type(e).__name__}: {e}')

    t1 = step(table1)
    t3 = step(table3)
    step(figure2)
    step(figure4)
    tm = runtime_from_table2() if args.no_bench else step(table2)
    step(stats_report, t1, t3, tm)
    t4 = step(table4)
    step(figure5)
    if not args.no_bench:
        step(ablation_report, t4, ablation_timings())
    step(figure6)
    step(table5)
    step(table_bars, t1, tm, t3, t4)
    if args.data == 'rs':
        step(noninferiority_tables)
