"""
Run the Habibian et al. (THRI 2022) informative+revealing generators on the
hypersphere task, using the identical protocol as simluate_preferences.py so the
output drops into ./results/ alongside the CMA-ES, CMA-ES-IG and IG baselines
that plot_comparisons.py reads.

    python3 run_habibian_comparison.py [--dims 8,16,32] [--trials 30]
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'preference-learning-from-selection'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'rebuttal'))

from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

from habiban import InformativeRevealingQueryGenerator, RevealingQueryGenerator


def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))


# Named exactly as they will appear in ./results/<name>_<metric>_<n>items_dim<d>.npy
CONFIGS = {
    'Habibian-Info':      dict(lam=0.0),
    'Habibian-IR':        dict(lam=1.0),                             # literal Eq. 15
    'Habibian-IR-log':    dict(lam=1.0, score_log_belief=True),      # scale-stable Eq. 15
    'Habibian-Reveal':    None,                                      # Eq. 13
}


def build(name, dim, seed):
    limits = [(-1, 1)] * dim
    rng = np.random.default_rng(seed)
    if CONFIGS[name] is None:
        return RevealingQueryGenerator(dim, limits, rng=rng)
    return InformativeRevealingQueryGenerator(dim, limits, rng=rng, **CONFIGS[name])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', default='8,16,32')
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--queries', type=int, default=30)
    parser.add_argument('--items', type=int, default=4)
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--outdir', default='./results')
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(',')]
    names = args.configs.split(',')
    os.makedirs(args.outdir, exist_ok=True)

    for dim_embedding in dims:
        # Matches simluate_preferences.py.
        user_choice_model = LuceShepardChoice(rationality=10)
        user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

        for name in names:
            cumulative_values = []
            cumulative_regret = []
            cumulative_per_query_alignment = []

            desc = f'dim{dim_embedding} {name}'
            for trial in tqdm(range(args.trials), desc=desc):
                generator = build(name, dim_embedding, seed=1000 * dim_embedding + trial)
                user_estimate.reset()

                all_trajectories = np.random.randn(5000, dim_embedding)
                true_preference = np.random.uniform(low=-1, high=1, size=dim_embedding)
                alignment = [0]
                regret = []
                per_query_alignment = []

                for _ in range(args.queries):
                    query = generator.get_query(args.items, user_estimate, user_choice_model)

                    probabilities = user_choice_model.get_choice_probabilities(
                        query, np.array([true_preference])).flatten()
                    choice = np.random.choice(np.arange(args.items), p=probabilities)
                    user_choice_model.tell_input(choice, query)

                    user_estimate.update(user_choice_model.get_probability_of_input)

                    estimated_omega = user_estimate.get_expectation()

                    alignment.append(alignment_metric(estimated_omega, true_preference))
                    best_est_trajectory = all_trajectories[np.argmax(all_trajectories @ estimated_omega)]
                    regret.append(np.max(all_trajectories @ true_preference)
                                  - best_est_trajectory @ true_preference)
                    per_query_alignment.append(
                        np.average([alignment_metric(q, true_preference) for q in query]))

                cumulative_values += [alignment]
                cumulative_regret += [regret]
                cumulative_per_query_alignment += [per_query_alignment]

            stem = f'{args.outdir}/{name}'
            suffix = f'{args.items}items_dim{dim_embedding}.npy'
            np.save(f'{stem}_alignment_{suffix}', cumulative_values)
            np.save(f'{stem}_regret_{suffix}', cumulative_regret)
            np.save(f'{stem}_per_query_alignment_{suffix}', cumulative_per_query_alignment)

            print(f'  {desc}: final regret {np.mean(cumulative_regret, axis=0)[-1]:.4f}, '
                  f'final quality {np.mean(cumulative_per_query_alignment, axis=0)[-1]:.4f}')


if __name__ == '__main__':
    main()
