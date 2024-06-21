#%%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#%%
df = pd.read_excel(r"arrays_to_excel.xlsx", header=1, names=['Column1', 'Column2', 'Column3', 'Column4'])
df.rename(columns={'Column1': 'Model1', 'Column2': 'Model2', 'Column3': 'Model3', 'Column4': 'Model4'}, inplace=True)
# %%
model1=df[["Model1"]]
model2=df[["Model2"]]
model3=df[["Model3"]]
model4=df[["Model4"]]



# %%

stats.probplot(df["Model1"], dist="norm", plot=plt)
plt.title('Q-Q plot for "Baseline"')
plt.show()

stats.probplot(df["Model1"], dist="norm", plot=plt)
plt.title('Q-Q plot or Logistic Regression')
plt.show()

stats.probplot(df["Model1"], dist="norm", plot=plt)
plt.title('Q-Q plot for Neural Network')
plt.show()

stats.probplot(df["Model1"], dist="norm", plot=plt)
plt.title('Q-Q plot for Random Forest')
plt.show()
# %%
# Perform Bartlett's test
stat, p = stats.bartlett(df['Model1'],
                   df['Model2'],
                   df['Model3'],
                   df['Model4'],)
print(f'Bartlett’s Test: Statistics={stat}, p={p}')
# %%
# Perform Shapiro-Wilk test for normality
for model in ['Model1', 'Model2', 'Model3', 'Model4']:
    stat, p = stats.shapiro(df[model].dropna())
    print(f'Shapiro-Wilk Test for {model}: Statistics={stat:.4f}, p-value={p:.4g}')
# %%
# Perform Kruskal-Wallis test
k_statistic, k_pvalue = stats.kruskal(df['Model1'].dropna(), 
                                      df['Model2'].dropna(), 
                                      df['Model3'].dropna(), 
                                      df['Model4'].dropna())

print(f"Kruskal-Wallis Test: Statistic={k_statistic}, p-value={k_pvalue}")
#%%
f_statistic, p_value = stats.f_oneway(model1, model2, model3, model4)

print(f"F-statistic: {f_statistic}, P-value: {p_value}")