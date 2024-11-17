import pandas as pd
import numpy as np

data = []

for technique in ['CMA-ES', 'CMA-ES-IG', 'IG']:
    for dim in [8, 16, 32]:
        for metric in ['alignment', 'per_query_alignment', 'regret']:
            metric_data = np.load(f'../results/{technique}_{metric}_4items_dim{dim}.npy')

            for trial in range(30):
                data.append({
                    'technique': technique,
                    'dimension': dim,
                    'trial': trial,
                    'metric': metric,
                    'value': np.trapz(metric_data[trial], dx=1/30),
                })

df = pd.DataFrame(data)
print(df.groupby(['technique', 'metric', 'dimension'])['value'].mean())