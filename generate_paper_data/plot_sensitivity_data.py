import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('cmaes_param_sensitivity.csv')
# df.query('sigma > 0.2', inplace=True)


# Set font to serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] 
# plt.rcParams['text.usetex'] = True
plt.rcParams['axes.titlesize'] = 16  # Title font size
plt.rcParams['axes.labelsize'] = 12   # Axis labels font size
plt.rcParams['legend.fontsize'] = 16

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot regret on the first axis
sns.lineplot(data=df, x='sigma', y='regret', hue='dim_embedding', style='generator', ax=axes[1])
axes[1].set_title('Regret (\u2193)')
axes[1].set_ylabel('Regret')
axes[1].set_xlabel('Initial Step Size ($\sigma$)')
sns.despine(ax=axes[0])

# Plot alignment on the second axis
sns.lineplot(data=df, x='sigma', y='alignment', hue='dim_embedding', style='generator', ax=axes[0])
axes[0].set_title('Alignment (\u2191)')
axes[0].set_ylabel('Alignment')
axes[0].set_xlabel('Initial Step Size ($\sigma$)')
sns.despine(ax=axes[1])

# Plot per_query_alignment on the third axis
sns.lineplot(data=df, x='sigma', y='per_query_alignment', hue='dim_embedding', style='generator', ax=axes[2])
axes[2].set_title('Quality (\u2191)')
axes[2].set_ylabel('Quality')
axes[2].set_xlabel('Initial Step Size ($\sigma$)')
sns.despine(ax=axes[2])

# Remove individual legends from all axes
for ax in axes:
    ax.get_legend().set_visible(False)

# Get the handles and labels from any of the axes (using the first one here)
handles, labels = axes[0].get_legend_handles_labels()
labels[0] = 'Feature Dimension'
labels[4] = 'Generator'
labels[5] = 'CMA-ES'
labels[6] = 'CMA-ES-IG'

print(labels)
# Create a single legend for the entire figure, placed below the subplots
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=7)

# Adjust the layout by resizing the axes (increasing the bottom margin to make space for the legend)
fig.subplots_adjust(bottom=0.2, left=0.05, right=1)

# Show the plot
# plt.tight_layout()
plt.show()