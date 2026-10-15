"""
Run the Habibian et al. baseline on the four simulated robot domains, matching
run_experiments.py exactly so the output CSVs sit alongside the existing
CMA-ES / CMA-ES-IG / Infogain / Random results and feed Figure 4 and Table 3.

    python3 run_habibian_domains.py --domain driving [--trials 100]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'experiments', 'preference-learning-from-selection'))
sys.path.append(os.path.join(HERE, '..', 'rebuttal'))

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from habiban import InformativeRevealingQueryGenerator, RevealingQueryGenerator


def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))


CONFIGS = {
    'Habibian-IR-log': dict(lam=1.0, score_log_belief=True),
    'Habibian-IR': dict(lam=1.0),
    'Habibian-Reveal': None,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', required=True,
                   choices=['lunar_lander', 'pylips_appearance', 'blossom_voice', 'driving'])
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--queries', type=int, default=30)
    p.add_argument('--items', type=int, default=5)
    p.add_argument('--dim', type=int, default=4)
    p.add_argument('--configs', default='Habibian-IR-log')
    p.add_argument('--suffix', default='_nodiscretization')
    args = p.parse_args()

    sys.path.append(os.path.join(HERE, args.domain))
    from utils import get_features

    outdir = os.path.join(HERE, args.domain, 'data', 'results')
    os.makedirs(outdir, exist_ok=True)

    # Matches run_experiments.py.
    user_choice_model = LuceShepardChoice(rationality=5)
    user_estimate = MonteCarloLinearReward(args.dim, number_samples=10_000)
    limits = [(-1, 1)] * args.dim

    for name in args.configs.split(','):
        data = []
        for trial_num in tqdm(range(args.trials), desc=f'{args.domain} {name}'):
            rng = np.random.default_rng(hash((args.domain, name, trial_num)) % (2 ** 32))
            generator = (RevealingQueryGenerator(args.dim, limits, rng=rng)
                         if CONFIGS[name] is None
                         else InformativeRevealingQueryGenerator(args.dim, limits, rng=rng,
                                                                 **CONFIGS[name]))
            user_estimate.reset()

            all_trajectories, _ = get_features(
                args.dim, emb_dir=os.path.join(HERE, args.domain, 'data', 'embeddings') + os.sep)
            all_trajectories = robust_scale(all_trajectories, quantile_range=(1.0, 99.0))

            true_preference = np.random.uniform(low=-1, high=1, size=args.dim)

            for q in range(args.queries):
                query = generator.get_query(args.items, user_estimate, user_choice_model)
                quality = [alignment_metric(item, true_preference) for item in query]

                # Simulate the full ranking by repeatedly picking the top item.
                working = np.array(query)
                for _ in range(args.items - 1):
                    probabilities = user_choice_model.get_choice_probabilities(
                        working, np.array([true_preference])).flatten()
                    choice = np.random.choice(np.arange(len(working)), p=probabilities)
                    user_choice_model.tell_input(choice, working)
                    user_estimate.update(user_choice_model.get_probability_of_input)
                    working = np.delete(working, choice, axis=0)

                estimated_omega = user_estimate.get_expectation()

                data.append({
                    'query_num': q,
                    'trial': trial_num,
                    'method': name,
                    'alignment': alignment_metric(estimated_omega, true_preference),
                    'regret': np.max(all_trajectories @ true_preference)
                              - all_trajectories[np.argmax(all_trajectories @ estimated_omega)] @ true_preference,
                    'quality_avg': np.average(quality),
                    'quality_max': np.max(quality),
                    'quality_min': np.min(quality),
                    'quality_median': np.median(quality),
                })

        out = os.path.join(outdir, f'{name}_dim{args.dim}_items{args.items}{args.suffix}.csv')
        pd.DataFrame(data).to_csv(out, index=False)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
