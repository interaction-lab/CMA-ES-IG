"""
Compare the Habibian et al. generators against the CMA-ES / CMA-ES-IG / IG /
Random baselines already saved in ./results/ for the hypersphere task.

Reports the two metrics plot_comparisons.py plots -- regret and per-query
alignment ("quality") -- as final values and as AUC over the 30 queries.
"""

import os
import numpy as np

RESULTS = './results'
DIMS = [8, 16, 32]
ITEMS = 4

BASELINES = ['IG', 'Random', 'CMA-ES', 'CMA-ES-IG']
OURS = ['Habibian-Info', 'Habibian-IR', 'Habibian-IR-log', 'Habibian-Reveal']

LABEL = {
    'IG': 'IG (Infogain)',
    'Random': 'Random',
    'CMA-ES': 'CMA-ES',
    'CMA-ES-IG': 'CMA-ES-IG  (yours)',
    'Habibian-Info': 'Habibian Informative',
    'Habibian-IR': 'Habibian I+R (Eq.15, l=1)',
    'Habibian-IR-log': 'Habibian I+R (log, l=1)',
    'Habibian-Reveal': 'Habibian Revealing',
}


def load(method, metric, dim):
    path = f'{RESULTS}/{method}_{metric}_{ITEMS}items_dim{dim}.npy'
    return np.load(path) if os.path.exists(path) else None


def stats(arr):
    """Final-query mean +- SE, and normalised AUC over the query sequence."""
    n = len(arr)
    final = arr[:, -1]
    auc = np.trapezoid(arr, axis=1) / arr.shape[1]
    return final.mean(), final.std() / np.sqrt(n), auc.mean(), auc.std() / np.sqrt(n)


def table(metric, title, note):
    print(f'\n{"=" * 92}\n{title}\n{note}\n{"=" * 92}')
    header = f'{"method":28}' + ''.join(f'{"dim " + str(d):>21}' for d in DIMS)
    print(header)
    print(f'{"":28}' + ''.join(f'{"final / AUC":>21}' for _ in DIMS))
    for method in BASELINES + OURS:
        cells = []
        for dim in DIMS:
            arr = load(method, metric, dim)
            if arr is None:
                cells.append(f'{"--":>21}')
                continue
            fm, fse, am, ase = stats(arr)
            cells.append(f'{fm:8.3f} /{am:8.3f}   ')
        if method == OURS[0]:
            print('-' * 92)
        print(f'{LABEL[method]:28}' + ''.join(cells))


def se_table(metric, title):
    print(f'\n{title} -- final-query mean +- SE')
    for dim in DIMS:
        print(f'  dim {dim}:')
        for method in BASELINES + OURS:
            arr = load(method, metric, dim)
            if arr is None:
                continue
            fm, fse, _, _ = stats(arr)
            print(f'    {LABEL[method]:28}{fm:8.3f} +- {fse:.3f}')


if __name__ == '__main__':
    table('regret', 'ROBOT REGRET  (lower is better)',
          'how well the robot learned the preference')
    table('per_query_alignment', 'QUERY QUALITY  (higher is better)',
          'mean alignment of the trajectories actually shown to the user')
    se_table('regret', 'REGRET')
    se_table('per_query_alignment', 'QUALITY')
