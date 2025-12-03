import numpy as np
import matplotlib.pyplot as plt

# Example data: replace with your actual data
dimensions = [8,16,32]


# Simulate data for three methods
metric3_cma_es_ig = {dim: np.load(f'./results/CMA-ES-IG_regret_4items_dim{dim}.npy') for dim in dimensions}
metric3_cma_es = {dim: np.load(f'./results/CMA-ES_regret_4items_dim{dim}.npy') for dim in dimensions}
metric3_rand = {dim: np.load(f'./results/Random_regret_4items_dim{dim}.npy') for dim in dimensions}
metric3_ig = {dim: np.load(f'./final/IG_regret_4items_dim{dim}.npy') for dim in dimensions}



for row in metric3_cma_es[64]:
    print(row)
    plt.plot(row, color='blue', alpha=0.1)

for row in metric3_cma_es_ig[64]:
    print(row)
    plt.plot(row, color='orange', alpha=0.1)

for row in metric3_rand[64]:
    print(row)
    plt.plot(row, color='grey', alpha=0.1)

plt.show()