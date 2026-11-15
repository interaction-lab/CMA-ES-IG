"""
Robot-domain experiments (Table 3 / Figure 4) with the resample-move belief.

Mirrors the trial loop of simulated_experiments/run_experiments.py and
run_habibian_domains.py (Luce-Shepard rationality 5, 5 items per query,
d=4, robust-scaled features, full simulated ranking, CMA-ES told the user's
ranking) but swaps MonteCarloLinearReward for ResamplingMonteCarloLinearReward,
seeds every trial deterministically, and can be sharded.

    python3 run_domains_rs.py --domain driving --method IG --shard 0 --nshards 5
    python3 run_domains_rs.py --merge
"""

import argparse
import glob
import os
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SIM = os.path.join(ROOT, 'simulated_experiments')
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'preference-learning-from-selection'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
sys.path.insert(0, HERE)

from irlpreference.input_models import LuceShepardChoice
from irlpreference.query_generation import InfoGainQueryGenerator
from resampling_belief import ResamplingMonteCarloLinearReward

DOMAINS = ['lunar_lander', 'driving', 'pylips_appearance', 'blossom_voice']
METHODS = ['IG', 'Habibian-IR-log', 'CMA-ES', 'CMA-ES-IG']
DIM, ITEMS, QUERIES, RATIONALITY = 4, 5, 30, 5
SUFFIX = f'_dim{DIM}_items{ITEMS}_nodiscretization'


def results_dir(domain):
    return os.path.join(SIM, domain, 'data', 'results_rs')


def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))


def build(method, rng):
    limits = [(-1, 1)] * DIM
    if method == 'IG':
        return InfoGainQueryGenerator(limits)
    if method in ('CMA-ES', 'CMA-ES-IG'):
        from cmaes_generators import CMAESGenerator, CMAESIGGenerator
        cls = CMAESGenerator if method == 'CMA-ES' else CMAESIGGenerator
        return cls(DIM, limits, population_size=ITEMS, sigma=1.3)
    from habiban import InformativeRevealingQueryGenerator
    return InformativeRevealingQueryGenerator(DIM, limits, rng=rng, lam=1.0,
                                              score_log_belief=True)


def run_trial(domain, method, trial, get_features):
    seed = zlib.crc32(f'{domain}|{method}|{trial}'.encode())
    np.random.seed(seed)                       # library samples, features, user choices
    rng = np.random.default_rng(seed)
    user_choice_model = LuceShepardChoice(rationality=RATIONALITY)
    user_estimate = ResamplingMonteCarloLinearReward(
        DIM, number_samples=10_000, rng=np.random.default_rng(seed + 1))
    generator = build(method, rng)

    all_trajectories, _ = get_features(
        DIM, emb_dir=os.path.join(SIM, domain, 'data', 'embeddings') + os.sep)
    all_trajectories = robust_scale(all_trajectories, quantile_range=(1.0, 99.0))
    true_preference = np.random.uniform(low=-1, high=1, size=DIM)

    rows = []
    for q in range(QUERIES):
        query = generator.get_query(ITEMS, user_estimate, user_choice_model)
        quality = [alignment_metric(item, true_preference) for item in query]

        working, sorted_query = np.array(query), []
        for _ in range(ITEMS - 1):
            probabilities = user_choice_model.get_choice_probabilities(
                working, np.array([true_preference])).flatten()
            choice = np.random.choice(np.arange(len(working)), p=probabilities)
            user_choice_model.tell_input(choice, working)
            user_estimate.update(user_choice_model.get_probability_of_input)
            sorted_query.append(working[choice])
            working = np.delete(working, choice, axis=0)
        sorted_query.append(working[0])

        if method in ('CMA-ES', 'CMA-ES-IG'):
            generator.tell(sorted_query, list(range(ITEMS))[::-1])

        estimated_omega = user_estimate.get_expectation()
        rows.append({
            'query_num': q, 'trial': trial, 'method': method,
            'alignment': alignment_metric(estimated_omega, true_preference),
            'regret': np.max(all_trajectories @ true_preference)
                      - all_trajectories[np.argmax(all_trajectories @ estimated_omega)] @ true_preference,
            'quality_avg': np.average(quality), 'quality_max': np.max(quality),
            'quality_min': np.min(quality), 'quality_median': np.median(quality),
        })
    return rows


def reward_range(domain, method, trial, get_features):
    """R_best - R_worst over the domain's trajectories for one simulated user.  Repeats the
    setup of run_trial in the same order (it shares the global RNG with the belief), so the
    trajectories and omega* are exactly the ones that trial used."""
    seed = zlib.crc32(f'{domain}|{method}|{trial}'.encode())
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ResamplingMonteCarloLinearReward(DIM, number_samples=10_000, rng=np.random.default_rng(seed + 1))
    build(method, rng)
    all_trajectories, _ = get_features(
        DIM, emb_dir=os.path.join(SIM, domain, 'data', 'embeddings') + os.sep)
    all_trajectories = robust_scale(all_trajectories, quantile_range=(1.0, 99.0))
    rewards = all_trajectories @ np.random.uniform(low=-1, high=1, size=DIM)
    return rewards.max() - rewards.min()


def merge():
    for domain in DOMAINS:
        for method in METHODS:
            shards = sorted(glob.glob(f'{results_dir(domain)}/{method}{SUFFIX}_shard*.csv'))
            if not shards:
                continue
            df = pd.concat([pd.read_csv(s) for s in shards]).sort_values(['trial', 'query_num'])
            out = f'{results_dir(domain)}/{method}{SUFFIX}.csv'
            df.to_csv(out, index=False)
            print(f'merged {len(shards)} shards -> {domain}/{os.path.basename(out)} '
                  f'({df.trial.nunique()} trials)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=DOMAINS)
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--merge', action='store_true')
    args = p.parse_args()
    if args.merge:
        return merge()

    sys.path.insert(0, os.path.join(SIM, args.domain))
    from utils import get_features

    os.makedirs(results_dir(args.domain), exist_ok=True)
    trials = np.array_split(np.arange(args.trials), args.nshards)[args.shard]
    rows = [r for t in trials for r in run_trial(args.domain, args.method, int(t), get_features)]
    out = f'{results_dir(args.domain)}/{args.method}{SUFFIX}_shard{args.shard:03d}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'done {args.domain} {args.method} shard {args.shard} ({len(trials)} trials)')


if __name__ == '__main__':
    main()
