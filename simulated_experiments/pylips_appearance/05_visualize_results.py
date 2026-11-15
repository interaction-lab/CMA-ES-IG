import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

def calc_aucs(df, metric):
    aucs = []
    method = df['method'].iloc[0]
    for trial in df['trial'].unique():
        df_trial = df[df['trial'] == trial]
        x = df_trial['query_num']
        y = df_trial[metric]
        auc = np.trapz(y, x) / (x.max() - x.min())
        aucs.append({
            'auc': auc,
            'trial': trial,
            'method': method
        })
    return pd.DataFrame(aucs)



dim = 6
exp_dfs = []
auc_dfs = []
for method in ['CMA-ES', 'CMA-ES-IG', 'Random']:
    df = pd.read_csv(f'./data/results/{method}_{dim}dim_4items.csv')
    exp_dfs.append(df)
    aucs = calc_aucs(df, 'quality_avg')
    auc_dfs.append(aucs)

exp_df = pd.concat(exp_dfs, ignore_index=True)
auc_df = pd.concat(auc_dfs, ignore_index=True)

# statistical analysis of AUCs
print(pg.normality(auc_df, dv='auc', group='method'))
print(pg.homoscedasticity(auc_df, dv='auc', group='method'))
print(pg.anova(auc_df, dv='auc', between='method'))
print(pg.pairwise_ttests(auc_df, dv='auc', between='method', padjust='bonf'))

sns.barplot(data=auc_df, x='method', y='auc', errorbar='se')
plt.show()

# plot alignment over queries
sns.lineplot(data=exp_df, x='query_num', y='regret', hue='method', errorbar='se')
plt.show()

sns.lineplot(data=exp_df, x='query_num', y='quality_median', hue='method', errorbar='se')
plt.show()



    
    
