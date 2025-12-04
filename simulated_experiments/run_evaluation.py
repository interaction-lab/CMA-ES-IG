import pandas as pd
import numpy as np

EXPERIMENTAL_DOMAIN = 'pylips_appearance'  # 'lunar_lander', 'pylips_appearance', 'blossom_voice', 'driving'
NUM_TRIALS = 100
MAX_QUERIES = 30
ITEMS_PER_QUERY = 5
DIM_EMBEDDING = 4  # 2, 4, 6, 8
METRICS = ['alignment', 'regret', 'quality_avg', 'quality_min', 'quality_max']

if __name__ == "__main__":
    all_data = []

    for name in ['CMA-ES-IG', 'CMA-ES', 'Random', 'IG']:
        try:
            results_df = pd.read_csv(f'./{EXPERIMENTAL_DOMAIN}/data/results/{name}_dim{DIM_EMBEDDING}_items{ITEMS_PER_QUERY}.csv')
            results_df['method'] = name
            all_data.append(results_df)
        except FileNotFoundError:
            print(f"File for {name} not found. Skipping.")

    results_df = pd.concat(all_data, ignore_index=True)

    print(results_df.head())

    # print the AUC of regret and quality_avg for each method
    for method in results_df['method'].unique():
        method_df = results_df[results_df['method'] == method]
        aucs = {}
        for metric in METRICS:
            aucs[metric] = []

            for trial in method_df['trial'].unique():
                trial_df = method_df[method_df['trial'] == trial]
                x = trial_df['query_num']
                y = trial_df[metric]
                # print(x)
                auc = np.trapezoid(y, x) / (x.max() - x.min())
                aucs[metric].append(auc)
                
        for metric in METRICS:
            mean_auc = np.mean(aucs[metric])
            std_auc = np.std(aucs[metric])
            print(f"Method: {method}, Mean AUC of {metric.capitalize()}: {mean_auc:.4f}, sterr AUC of {metric.capitalize()}: {std_auc/10:.4f}")
        print('\n')