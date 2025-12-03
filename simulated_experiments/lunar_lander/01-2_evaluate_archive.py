import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ribs.visualize import parallel_axes_plot

df = pd.read_csv("./data/embeddings/lunar_lander_handcrafted.csv")

# Print ranges for each column in the dataframe
print("Ranges for each measure in the dataframe:")
print("=" * 50)

for column in df.columns:
    if 'measure' in column:
        min_val = df[column].min()
        max_val = df[column].max()
        print(f"{column}: [{min_val:.4f}, {max_val:.4f}]")

print("=" * 50)

# Get measure columns
measure_columns = [col for col in df.columns if 'measure' in col]
print(f"Found {len(measure_columns)} measure columns: {measure_columns}")

if len(measure_columns) > 1:
    # Create scatter plot matrix
    n_measures = len(measure_columns)
    # Adjust figure size based on number of measures, but keep it reasonable
    fig_size = min(12, max(6, n_measures * 2))
    fig, axes = plt.subplots(n_measures, n_measures, figsize=(fig_size, fig_size))
    
    # Handle case where there's only one row/column
    if n_measures == 1:
        axes = np.array([[axes]])
    elif n_measures == 2:
        axes = axes.reshape(2, 2)
    
    for i, col1 in enumerate(measure_columns):
        for j, col2 in enumerate(measure_columns):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: show histogram
                ax.hist(df[col1], bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col1}', fontsize=8)
            else:
                # Off-diagonal: show scatter plot
                ax.scatter(df[col2], df[col1], alpha=0.6, s=15)
            
            # Set labels only on edges
            if i == n_measures - 1:  # Bottom row
                ax.set_xlabel(col2, fontsize=8)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
                
            if j == 0:  # Left column
                ax.set_ylabel(col1, fontsize=8)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            
            # Smaller tick labels
            ax.tick_params(labelsize=6)
    
    plt.subplots_adjust(hspace=0.1, wspace=0.1)  # Reduce spacing
    plt.show()

else:
    print("Need at least 2 measure columns to create scatter plots")



