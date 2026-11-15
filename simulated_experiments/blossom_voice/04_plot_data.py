import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("./data/results/CMA-ES_32dim_4items.csv")
df = pd.read_csv("./data/results/CMA-ES-IG_32dim_4items.csv")
df2 = pd.read_csv("./data/results/Random_32dim_4items.csv")

# sns.barplot(data=df, x='query_num', y='alignment')
sns.barplot(data=df2, x='query_num', y='alignment')
sns.barplot(data=df, x='query_num', y='alignment')
# sns.barplot(data=df, x='query_num', y='query_quality_avg')
# sns.barplot(data=df2, x='query_num', y='query_quality_avg')

plt.show()