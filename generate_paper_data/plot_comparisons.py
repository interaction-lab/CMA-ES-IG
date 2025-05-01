import numpy as np
import matplotlib.pyplot as plt
import argparse

# Set the style for the plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] 
plt.rcParams['axes.titlesize'] = 16  # Title font size
plt.rcParams['axes.labelsize'] = 12   # Axis labels font size
plt.rcParams['legend.fontsize'] = 16
dimensions = [8, 16, 32]

def plot_metric_subplots(dimensions, metric_data1, metric_data2, metric_name1, metric_name2, use_regret):
    fig, axes = plt.subplots(2, len(dimensions), figsize=(15, 8))
    colors = ['#ababab','#0770bb','#f49249']
    methods = ['IG', 'CMA-ES', 'CMA-ES-IG']
    
    for i, dim in enumerate(dimensions):
        # Plot Metric 1
        for j, (method, color) in enumerate(zip(methods, colors)):
            cumulative_values = metric_data1[method][dim]
            m = np.mean(np.array(cumulative_values), axis=0) 
            print(f'computed ALIGNMENT AUC for {dim}, {method}: {np.trapz(m)/len(m)}')
            std = np.std(np.array(cumulative_values), axis=0) / np.sqrt(30)
            axes[0,i].fill_between(range(len(m)), m-std, m+std, alpha=0.3, color=color)
            axes[0,i].plot(m, color=color)
        
        if i < len(dimensions):
            axes[0, i].set_title(f'{dim}-dimensional Feature Space')

        axes[0, i].set_xlabel('Number of Queries')
        if i == 0:
            axes[0, i].set_ylabel(f'{metric_name1} (Mean ± SE)')
        if not use_regret:
            axes[0, i].set_ylim([0, 1])  # Adjust according to your data
        axes[0, i].grid(True, color='lightgrey', linestyle='--')
        axes[0, i].spines['top'].set_visible(False)
        axes[0, i].spines['right'].set_visible(False)
        
        # Plot Metric 2
        for j, (method, color) in enumerate(zip(methods, colors)):
            cumulative_values = metric_data2[method][dim]
            m = np.mean(np.array(cumulative_values), axis=0) 
            print(f'computed QUALITY AUC for {dim}, {method}: {np.trapz(m) / len(m)}')
            std = np.std(np.array(cumulative_values), axis=0) / np.sqrt(30)
            axes[1,i].fill_between(range(len(m)), m-std, m+std, alpha=0.3, color=color)
            axes[1,i].plot(m, color=color)
            
        axes[1, i].set_xlabel('Number of Queries')
        if i == 0:
            axes[1, i].set_ylabel(f'{metric_name2} (Mean ± SE)')
        axes[1, i].set_ylim([-.1, 1])  # Adjust according to your data
        axes[1, i].grid(True, color='lightgrey', linestyle='--')
        axes[1, i].spines['top'].set_visible(False)
        axes[1, i].spines['right'].set_visible(False)
        axes[1, i].spines['bottom'].set_visible(False)
        axes[1, i].axhline(y=0, color='black', linewidth=1)

    # Create a single legend for all subplots
    handles = [plt.Line2D([0], [0], marker='o', color=color, label=method, linestyle='') for method, color in zip(methods, colors)]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Load and plot simulation results for preference learning methods."
    )
    parser.add_argument(
        '--use-regret',
        action='store_true', # Creates a flag; if present, args.use_regret will be True
        help="If set, use 'regret' as the first metric instead of 'alignment'."
    )
    args = parser.parse_args()

    # The first metric is determined by the command line argument (alignment or regret)
    if args.use_regret:
        metric1_name = 'regret'
        print("Using 'Regret' for the first metric.")
    else:
        metric1_name = 'alignment'
        print("Using 'Alignment' for the first metric.")

    metric1_label = metric1_name.capitalize() # 'Alignment' or 'Regret'

    # The second metric is 'per_query_alignment' and label is 'Quality'
    metric2_name = 'per_query_alignment'
    metric2_label = 'Quality' 

    # Load data for the first metric (either alignment or regret)
    metric1_cma_es_ig = {dim: np.load(f'results/CMA-ES-IG_{metric1_name}_4items_dim{dim}.npy') for dim in dimensions}
    metric1_cma_es = {dim: np.load(f'results/CMA-ES_{metric1_name}_4items_dim{dim}.npy') for dim in dimensions}
    metric1_ig = {dim: np.load(f'results/IG_{metric1_name}_4items_dim{dim}.npy') for dim in dimensions}

    # Load data for the second metric (per_query_alignment)
    metric2_cma_es_ig = {dim: np.load(f'results/CMA-ES-IG_{metric2_name}_4items_dim{dim}.npy') for dim in dimensions}
    metric2_cma_es = {dim: np.load(f'results/CMA-ES_{metric2_name}_4items_dim{dim}.npy') for dim in dimensions}
    metric2_ig = {dim: np.load(f'results/IG_{metric2_name}_4items_dim{dim}.npy') for dim in dimensions}

    # Combine metrics into dictionaries 
    metric1_data = {'CMA-ES-IG': metric1_cma_es_ig, 'CMA-ES': metric1_cma_es, 'IG': metric1_ig}
    metric2_data = {'CMA-ES-IG': metric2_cma_es_ig, 'CMA-ES': metric2_cma_es, 'IG': metric2_ig}

    plot_metric_subplots(dimensions, metric1_data, metric2_data, metric1_label, metric2_label, args.use_regret)

    print("\nPlotting complete.")