"""
Regenerate the hypersphere (parameter-estimation) experiments for the rebuttal,
with the Habibian et al. baseline added alongside CMA-ES-IG, CMA-ES and
Infogain.

Follows the protocol in simluate_preferences.py and the IJRR submission: 4 items
per query, 30 queries, 100 simulated users, d in {4, 8, 16, 32}.  Unlike the
original driver every trial is seeded, so runs are reproducible and can be
sharded across processes (Infogain at d=32 costs ~5 s per query).

    python3 run_rebuttal_experiments.py --method IG --dim 32 --shard 0 --nshards 12
    python3 run_rebuttal_experiments.py --merge
"""

import argparse
import glob
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'preference-learning-from-selection'))
sys.path.insert(0, os.path.join(HERE, '..', 'rebuttal'))

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from irlpreference.query_generation import InfoGainQueryGenerator

OUTDIR = os.path.join(HERE, '..', 'rebuttal', 'graphs', 'data')
# Output for runs with the resample-move belief (see rebuttal/resampling_belief.py).
OUTDIR_RS = os.path.join(HERE, '..', 'rebuttal', 'graphs', 'data_rs')
METRICS = ['alignment', 'regret', 'per_query_alignment']
DIMS = [4, 8, 16, 32]
METHODS = ['CMA-ES-IG', 'CMA-ES', 'IG', 'Habibian-IR-log', 'Habibian-Reveal',
           'IGapprox-kmeans100', 'IGapprox-kmeans20', 'IGapprox-kmedoids20',
           'IGapprox-exact20', 'IGapprox-ellipsoid']

# Ablation over how CMA-ES-IG approximates information gain (see
# cmaes_ig_ablation.py).  kmeans100 is the deployed rule; the three @20 variants
# share a pool size so that the selection rule is isolated from the pool size.
IG_ABLATION = {
    'IGapprox-kmeans100':  dict(method='kmeans', pool_size=100),
    'IGapprox-kmeans20':   dict(method='kmeans', pool_size=20),
    'IGapprox-kmedoids20': dict(method='kmedoids', pool_size=20),
    'IGapprox-exact20':    dict(method='exact', pool_size=20),
    'IGapprox-ellipsoid':  dict(method='ellipsoid', pool_size=20),
}

# Sweep over the confidence mass of the CMA-ES ellipsoid the query is optimised
# over.  A tighter ellipsoid stays closer to the CMA-ES mean (the high-reward
# region) but offers less spread for the information-gain term to exploit.
# Analytic-gradient ellipsoid search.  'ellipgrad' is the same objective and
# the same feasible region as 'ellipsoid' -- only the optimiser changes, so the
# pair isolates the cost of finite differencing.  'ellipgradfast' additionally
# subsamples the hypothesis set and caps iterations, trading estimator noise for
# speed to see whether the approach can undercut k-means.
IG_ABLATION['IGapprox-ellipgrad'] = dict(
    method='ellipsoid_grad', pool_size=20, ellipsoid_mass=0.90, grad_maxiter=40)
IG_ABLATION['IGapprox-ellipgradfast'] = dict(
    method='ellipsoid_grad', pool_size=20, ellipsoid_mass=0.90, grad_maxiter=10,
    omega_subsample=500)
METHODS += ['IGapprox-ellipgrad', 'IGapprox-ellipgradfast']

# Mass sweep re-run with the analytic-gradient optimiser, which is ~72x cheaper
# and (unlike the finite-difference version) gets the same iteration budget at
# every dimension, so the sweep isolates the geometry rather than confounding it
# with optimiser effort.
GRAD_MASSES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
for _m in GRAD_MASSES:
    IG_ABLATION[f'IGapprox-egrad{int(_m * 100):02d}'] = dict(
        method='ellipsoid_grad', pool_size=20, ellipsoid_mass=_m, grad_maxiter=40)
METHODS += [f'IGapprox-egrad{int(m * 100):02d}' for m in GRAD_MASSES]

ELLIPSOID_MASSES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
for _mass in ELLIPSOID_MASSES:
    IG_ABLATION[f'IGapprox-ellip{int(_mass * 100):02d}'] = dict(
        method='ellipsoid', pool_size=20, ellipsoid_mass=_mass)
METHODS += [f'IGapprox-ellip{int(m * 100):02d}' for m in ELLIPSOID_MASSES]


def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))


def build(method, dim, items, rng):
    limits = [(-1, 1)] * dim
    if method == 'IG':
        return InfoGainQueryGenerator(limits)
    if method in ('CMA-ES', 'CMA-ES-IG'):
        from cmaes_generators import CMAESGenerator, CMAESIGGenerator
        cls = CMAESGenerator if method == 'CMA-ES' else CMAESIGGenerator
        return cls(dim, limits, items)
    if method in IG_ABLATION:
        from cmaes_ig_ablation import CMAESIGAblation
        return CMAESIGAblation(dim, limits, population_size=items, rng=rng,
                               **IG_ABLATION[method])
    from habiban import InformativeRevealingQueryGenerator, RevealingQueryGenerator
    if method == 'Habibian-Reveal':
        return RevealingQueryGenerator(dim, limits, rng=rng)
    return InformativeRevealingQueryGenerator(dim, limits, lam=1.0,
                                              score_log_belief=True, rng=rng)


def run_trial(method, dim, items, queries, seed, rationality=10, belief='plain'):
    rng = np.random.default_rng(seed)
    # The belief draws its hypothesis samples from the global RNG.
    np.random.seed(seed)

    user_choice_model = LuceShepardChoice(rationality=rationality)
    if belief == 'resample':
        from resampling_belief import ResamplingMonteCarloLinearReward
        user_estimate = ResamplingMonteCarloLinearReward(
            dim, number_samples=10_000, rng=np.random.default_rng(seed + 1))
    else:
        user_estimate = MonteCarloLinearReward(dim, number_samples=10_000)
    generator = build(method, dim, items, rng)

    all_trajectories = rng.standard_normal((5000, dim))
    true_preference = rng.uniform(low=-1, high=1, size=dim)

    alignment = [0.0]
    regret = []
    per_query_alignment = []

    for _ in range(queries):
        query = generator.get_query(items, user_estimate, user_choice_model)

        # Simulate a full ranking the way simulated_experiments/run_experiments.py
        # does: repeatedly sample the top remaining item from the Luce-Shepard
        # choice distribution.  The earlier hypersphere driver instead handed
        # CMA-ES `probabilities.argsort()`, which is a monotone function of true
        # reward and therefore a noiseless oracle ranking -- the CMA-ES family
        # was immune to user noise by construction while the belief-based
        # methods were not.
        working = np.array(query)
        sorted_query = []
        for _ in range(items - 1):
            probabilities = user_choice_model.get_choice_probabilities(
                working, np.array([true_preference])).flatten()
            choice = rng.choice(np.arange(len(working)), p=probabilities)
            user_choice_model.tell_input(choice, working)
            user_estimate.update(user_choice_model.get_probability_of_input)
            sorted_query.append(working[choice])
            working = np.delete(working, choice, axis=0)
        sorted_query.append(working[0])

        if method in ('CMA-ES', 'CMA-ES-IG') or method in IG_ABLATION:
            generator.tell(sorted_query, list(range(items))[::-1])

        estimated_omega = user_estimate.get_expectation()

        alignment.append(alignment_metric(estimated_omega, true_preference))
        best = all_trajectories[np.argmax(all_trajectories @ estimated_omega)]
        regret.append(np.max(all_trajectories @ true_preference) - best @ true_preference)
        per_query_alignment.append(
            np.average([alignment_metric(q, true_preference) for q in query]))

    return alignment, regret, per_query_alignment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method')
    p.add_argument('--dim', type=int)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--queries', type=int, default=30)
    p.add_argument('--items', type=int, default=4)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--rationality', type=float, default=10.0)
    p.add_argument('--sweep-tag', dest='sweep_tag', action='store_true',
                   help='always tag output with the rationality, even at the default')
    p.add_argument('--merge', action='store_true')
    p.add_argument('--belief', choices=['plain', 'resample'], default='plain')
    args = p.parse_args()
    global OUTDIR
    if args.belief == 'resample':
        OUTDIR = OUTDIR_RS

    os.makedirs(OUTDIR, exist_ok=True)

    if args.merge:
        for method in METHODS:
            for dim in DIMS:
                for metric in METRICS:
                    shards = sorted(glob.glob(
                        f'{OUTDIR}/{method}_{metric}_{args.items}items_dim{dim}_shard*.npy'))
                    if not shards:
                        continue
                    idx, rows = [], []
                    for s in shards:
                        a = np.load(s)
                        base = int(s.rsplit('shard', 1)[1].split('_')[0].replace('.npy', ''))
                        rows.append(a)
                        idx.append(base)
                    merged = np.concatenate([rows[i] for i in np.argsort(idx)], axis=0)
                    out = f'{OUTDIR}/{method}_{metric}_{args.items}items_dim{dim}.npy'
                    np.save(out, merged)
                    print(f'merged {len(shards):3} shards -> {os.path.basename(out)} {merged.shape}')
        return

    trials = np.array_split(np.arange(args.trials), args.nshards)[args.shard]
    A, R, Q = [], [], []
    for t in trials:
        # crc32 rather than hash(): str hashing is salted per process, which
        # made the earlier seeds irreproducible.
        seed = zlib.crc32(f'{args.method}|{args.dim}|{int(t)}'.encode())
        a, r, q = run_trial(args.method, args.dim, args.items, args.queries, seed,
                            rationality=args.rationality, belief=args.belief)
        A.append(a); R.append(r); Q.append(q)

    # Always tag sweep output.  Leaving beta=10 untagged would overwrite the
    # main ablation files, which were produced under the earlier protocol.
    tag = f'_rat{args.rationality:g}' if args.sweep_tag else (
        '' if args.rationality == 10.0 else f'_rat{args.rationality:g}')
    stem = (f'{OUTDIR}/{args.method}{tag}_%s_{args.items}items_dim{args.dim}'
            f'_shard{args.shard:03d}.npy')
    np.save(stem % 'alignment', np.array(A))
    np.save(stem % 'regret', np.array(R))
    np.save(stem % 'per_query_alignment', np.array(Q))
    print(f'done {args.method} d={args.dim} shard {args.shard} ({len(trials)} trials)')


if __name__ == '__main__':
    main()
