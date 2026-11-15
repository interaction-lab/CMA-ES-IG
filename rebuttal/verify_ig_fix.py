"""
Verify the mutual-information objective and the resampling belief before any
full re-run.

    python3 verify_ig_fix.py --checks                       # unit checks, ~1 min
    python3 verify_ig_fix.py --config exact-mi-rs --dim 8 --shard 0 --nshards 4
    python3 verify_ig_fix.py --summary

Configs pair a selection rule with a belief:
    exact-legacy     exact@20, library objective, plain belief  (current results)
    exact-mi         exact@20, mutual information, plain belief
    exact-legacy-rs  exact@20, library objective, resampling belief
    exact-mi-rs      exact@20, mutual information, resampling belief
    kmeans-rs        k-means@100, resampling belief (k-means ignores the belief,
                     so this isolates the belief's effect on the metrics)
    kmeans           k-means@100, plain belief
Every config uses the same per-trial seeds, so comparisons are paired.
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'experiments', 'preference-learning-from-selection'))
sys.path.insert(0, HERE)

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from cmaes_ig_ablation import CMAESIGAblation
from resampling_belief import ResamplingMonteCarloLinearReward

OUT = os.path.join(HERE, 'graphs', 'data_verify')
CONFIGS = {
    'exact-legacy':    dict(method='exact', pool_size=20, ig_objective='legacy', rs=False),
    'exact-mi':        dict(method='exact', pool_size=20, ig_objective='mi', rs=False),
    'exact-legacy-rs': dict(method='exact', pool_size=20, ig_objective='legacy', rs=True),
    'exact-mi-rs':     dict(method='exact', pool_size=20, ig_objective='mi', rs=True),
    'kmeans':          dict(method='kmeans', pool_size=100, ig_objective='legacy', rs=False),
    'kmeans-rs':       dict(method='kmeans', pool_size=100, ig_objective='legacy', rs=True),
    # Analytic-gradient ellipsoid search.  Its gradient code implements the
    # library objective only; with a resampled belief that objective matches
    # mutual information to within noise (see the exact-* pairs).
    'ellip90':         dict(method='ellipsoid_grad', pool_size=20, ig_objective='legacy',
                            ellipsoid_mass=0.90, grad_maxiter=40, rs=False),
    'ellip90-rs':      dict(method='ellipsoid_grad', pool_size=20, ig_objective='legacy',
                            ellipsoid_mass=0.90, grad_maxiter=40, rs=True),
    'ellip01-rs':      dict(method='ellipsoid_grad', pool_size=20, ig_objective='legacy',
                            ellipsoid_mass=0.01, grad_maxiter=40, rs=True),
}
RATIONALITY = 10
ITEMS = 4


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def true_mi(query, belief, beta):
    """Brute-force float64 mutual information of the first choice."""
    lw = belief.hypothesis_log_probabilities
    pi = np.exp(lw - lw.max()); pi /= pi.sum()
    r = beta * (np.asarray(query) @ belief.hypothesis_samples.T)       # (N, S)
    p = np.exp(r - r.max(0)); p /= p.sum(0)
    h = lambda q: -np.sum(q * np.log2(np.clip(q, 1e-300, None)), axis=0)
    return h(p @ pi) - h(p) @ pi


def ess(belief):
    lw = belief.hypothesis_log_probabilities
    pi = np.exp(lw - lw.max()); pi /= pi.sum()
    return 1.0 / np.sum(pi ** 2)


def run_trial(config, dim, queries, seed):
    c = CONFIGS[config]
    np.random.seed(seed)                        # the library draws samples from np.random
    rng = np.random.default_rng(seed)
    true_w = rng.uniform(-1, 1, dim)
    trajectories = rng.standard_normal((5000, dim))
    cm = LuceShepardChoice(rationality=RATIONALITY)
    belief = (ResamplingMonteCarloLinearReward(dim, 10_000, rng=np.random.default_rng(seed + 1))
              if c['rs'] else MonteCarloLinearReward(dim, number_samples=10_000))
    gen = CMAESIGAblation(dim, [(-1, 1)] * dim, population_size=ITEMS,
                          rng=np.random.default_rng(seed + 2),
                          **{k: v for k, v in c.items() if k != 'rs'})
    rec = {k: [] for k in ('alignment', 'regret', 'quality', 'ess', 'mi')}
    for _ in range(queries):
        query = gen.get_query(ITEMS, belief, cm)
        rec['mi'].append(true_mi(query, belief, RATIONALITY))
        rec['quality'].append(np.mean([cos(q, true_w) for q in query]))
        working, ranked = np.array(query), []
        for _ in range(ITEMS - 1):
            p = cm.get_choice_probabilities(working, np.array([true_w])).flatten()
            choice = rng.choice(len(working), p=p)
            cm.tell_input(choice, working)
            belief.update(cm.get_probability_of_input)
            ranked.append(working[choice])
            working = np.delete(working, choice, axis=0)
        ranked.append(working[0])
        gen.tell(ranked, list(range(ITEMS))[::-1])
        est = belief.get_expectation()
        rec['ess'].append(ess(belief))
        rec['alignment'].append(cos(est, true_w))
        best = trajectories[np.argmax(trajectories @ est)]
        rec['regret'].append(np.max(trajectories @ true_w) - best @ true_w)
    return rec


# ---------------------------------------------------------------- unit checks
def checks():
    rng = np.random.default_rng(0)
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= passed
        print(f'[{"PASS" if passed else "FAIL"}] {name}: {detail}')

    # 1. batched MI matches brute force on a non-uniform belief
    d = 6
    np.random.seed(0)
    belief = MonteCarloLinearReward(d, number_samples=4000)
    belief.hypothesis_log_probabilities = rng.normal(0, 3, 4000)
    gen = CMAESIGAblation(d, [(-1, 1)] * d, population_size=4, method='exact',
                          pool_size=10, ig_objective='mi', rng=rng)
    cm = LuceShepardChoice(rationality=RATIONALITY)
    pool = rng.uniform(-1, 1, (10, d))
    subs = np.array(list(itertools.combinations(range(10), 4)))
    fast = gen._batch_info_gain(pool, subs, belief, cm)
    slow = np.array([true_mi(pool[s], belief, RATIONALITY) for s in subs])
    err = np.max(np.abs(fast - slow))
    report('batched MI == brute-force MI', err < 1e-3, f'max abs err {err:.2e} over {len(subs)} subsets')
    report('0 <= MI <= log2(4)', fast.min() > -1e-4 and fast.max() <= 2 + 1e-4,
           f'range [{fast.min():.3f}, {fast.max():.3f}]')

    # 2. legacy path unchanged
    gen_l = CMAESIGAblation(d, [(-1, 1)] * d, population_size=4, method='exact',
                            pool_size=10, rng=rng)
    leg = gen_l._batch_info_gain(pool, subs[:20], belief, cm)
    lib = np.array([gen_l._info_gain(belief, cm, pool[s]) for s in subs[:20]])
    err = np.max(np.abs(leg - lib))
    report('legacy objective still matches library', err < 1e-3, f'max abs err {err:.2e}')

    # 3. resampling belief tracks a large-sample reference posterior.  d=4 with
    #    1M plain samples keeps a usable ESS for the whole check.
    d, n_obs = 4, 30
    true_w = rng.uniform(-1, 1, d)
    np.random.seed(1)
    ref = MonteCarloLinearReward(d, number_samples=1_000_000)
    np.random.seed(2)
    rs = ResamplingMonteCarloLinearReward(d, 10_000, rng=np.random.default_rng(3))
    cm = LuceShepardChoice(rationality=RATIONALITY)
    for _ in range(n_obs):
        q = rng.uniform(-1, 1, (4, d))
        p = cm.get_choice_probabilities(q, np.array([true_w])).flatten()
        cm.tell_input(rng.choice(4, p=p), q)
        ref.update(cm.get_probability_of_input)
        rs.update(cm.get_probability_of_input)
    a = cos(rs.get_expectation(), ref.get_expectation())
    report('resampled posterior mean ~ 1M-sample reference', a > 0.98,
           f'cosine {a:.4f}; reference ESS {ess(ref):.0f}, resampling ESS {rs.ess():.0f}, '
           f'{rs.n_resamples} resamples, mean MH acceptance {np.mean(rs.acceptance):.2f}')
    report('resampling belief keeps ESS above threshold', rs.ess() >= 0.5 * rs.N * 0.5,
           f'ESS {rs.ess():.0f} of {rs.N}')
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    return ok


# ---------------------------------------------------------------- summary
def summary():
    rows = {}
    for f in sorted(glob.glob(f'{OUT}/*_regret_*.npy')):
        name = os.path.basename(f)
        config, rest = name.split('_regret_')
        dim = int(rest.split('dim')[1].split('_')[0])
        rows.setdefault((config, dim), []).append(f)
    print(f'{"config":16} {"d":>3} {"n":>3} {"align":>7} {"regret":>8} {"quality":>8} '
          f'{"MI/query":>9} {"ESS@10":>8} {"ESS@30":>8}')
    for (config, dim) in sorted(rows, key=lambda k: (k[1], list(CONFIGS).index(k[0]))):
        load = lambda m: np.concatenate([np.load(f.replace('_regret_', f'_{m}_'))
                                         for f in rows[(config, dim)]])
        A, R, Q, E, M = (load(m) for m in ('alignment', 'regret', 'quality', 'ess', 'mi'))
        auc = lambda x: (np.trapezoid(x, axis=1) / x.shape[1]).mean()
        print(f'{config:16} {dim:3} {len(A):3} {auc(A):7.3f} {auc(R):8.3f} {auc(Q):8.3f} '
              f'{M.mean():9.3f} {np.median(E[:, 9]):8.0f} {np.median(E[:, -1]):8.0f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checks', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--config', choices=list(CONFIGS))
    p.add_argument('--dim', type=int)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--queries', type=int, default=30)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    args = p.parse_args()
    if args.checks:
        sys.exit(0 if checks() else 1)
    if args.summary:
        return summary()

    os.makedirs(OUT, exist_ok=True)
    trials = np.array_split(np.arange(args.trials), args.nshards)[args.shard]
    recs = [run_trial(args.config, args.dim, args.queries, 1000 * args.dim + int(t))
            for t in trials]
    for k in recs[0]:
        np.save(f'{OUT}/{args.config}_{k}_dim{args.dim}_shard{args.shard:03d}.npy',
                np.array([r[k] for r in recs]))
    print(f'done {args.config} d={args.dim} shard {args.shard} ({len(trials)} trials)')


if __name__ == '__main__':
    main()
