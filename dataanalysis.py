#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


#%%
data=pd.read_csv("HR_data.csv").drop("Unnamed: 0",axis=1)
#%%
data.isna().sum()
# %%

plt.hist(data["Frustrated"],bins=20)
# %%
firstfigdata=data[["HR_Mean","HR_Median","HR_std","HR_Min","HR_Max"]]
# HR boxplots
plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
firstfigdata.boxplot()
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()
#%%
plt.figure(figsize=(6, 6))  # Adjust the figure size as needed
data[["HR_AUC"]].boxplot()
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.show()
# %%
# Maybe something on stratification?
# sns.pairplot(data)
# plt.show()
# %%
# DATA BY ROUNDS
round1_df = data[data['Round'] == 'round_1']
round2_df = data[data['Round'] == 'round_2']
round3_df = data[data['Round'] == 'round_3']
round4_df = data[data['Round'] == 'round_4']

round_mean_values = data.groupby('Round')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()

# %%
# DATA BY PUZZLER
puzzler_df=data[data['Puzzler'] == 1]
non_puzzler_df=data[data['Puzzler'] == 0]
puzzler_mean_values = data.groupby('Puzzler')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
puzzler_rounds_mean_values = puzzler_df.groupby('Round')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
nonpuzzler_rounds_mean_values = non_puzzler_df.groupby('Round')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()

# %%
# DATA BY PHASES
phase1_df = data[data['Phase'] == 'phase1']
phase2_df = data[data['Phase'] == 'phase2']
phase3_df = data[data['Phase'] == 'phase3']
phase_mean_values = data.groupby('Phase')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
phase1_by_rounds = phase1_df.groupby("Round")[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
phase2_by_rounds = phase2_df.groupby("Round")[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
phase3_by_rounds = phase3_df.groupby("Round")[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()

# %%
# DATA BY PUZZLER

puzzler_mean_values = data.groupby('Puzzler')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
puzzler_rounds_mean_values = puzzler_df.groupby('Round')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
nonpuzzler_rounds_mean_values = non_puzzler_df.groupby('Round')[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()

#%%
# Initialize a dictionary to hold the dataframes
round_phase_dfs = {}
round_phase_means = {}

# List of unique rounds and phases
rounds = ['round_1', 'round_2', 'round_3', 'round_4']
phases = ['phase1', 'phase2', 'phase3']

# Iterate over each round and phase, filter the data, and store the result
for round_ in rounds:
    for phase in phases:
        print("round:",round_,"phase:",phase)
        # Filter data for the current round and phase
        filtered_df = data[(data['Round'] == round_) & (data['Phase'] == phase)]
        
        # Store the filtered dataframe in the dictionary with a unique key
        round_phase_dfs[f"{round_}_{phase}"] = filtered_df
        round_phase_means[f"{round_}_{phase}_mean"]=filtered_df[["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC","Frustrated"]].mean()
# At this point, round_phase_dfs contains 12 unique dataframes for each round-phase pair


# Define HR variables
hr_variables = ["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC"]

# Create subplots with 2 columns; rows based on the number of HR variables
fig, axs = plt.subplots(len(hr_variables), 2, figsize=(12, 6 * len(hr_variables)))

# Iterate over HR variables to generate QQ-plots
for i, var in enumerate(hr_variables):
    # QQ-plot for puzzler_df
    stats.probplot(puzzler_df[var], dist="norm", plot=axs[i, 0])
    axs[i, 0].set_title(f'QQ-plot for Puzzlers: {var}')
    
    # QQ-plot for non_puzzler_df
    stats.probplot(non_puzzler_df[var], dist="norm", plot=axs[i, 1])
    axs[i, 1].set_title(f'QQ-plot for Non-Puzzlers: {var}')

plt.tight_layout()
plt.show()

#%%
# Define HR variables (assuming they are already defined in your script)
hr_variables = ["HR_Mean", "HR_Median", "HR_std", "HR_Min", "HR_Max", "HR_AUC"]

# Perform Shapiro-Wilk test for each HR variable
for var in hr_variables:
    # Shapiro-Wilk test for puzzler_df
    sw_stat_puzzler, sw_pvalue_puzzler = stats.shapiro(puzzler_df[var].dropna())
    
    # Shapiro-Wilk test for non_puzzler_df
    sw_stat_nonpuzzler, sw_pvalue_nonpuzzler = stats.shapiro(non_puzzler_df[var].dropna())
    
    # Print results
    print(f"{var} - Puzzler: Shapiro-Wilk Test Stat={sw_stat_puzzler:.4f}, p-value={sw_pvalue_puzzler:.4g}")
    print(f"{var} - Non-Puzzler: Shapiro-Wilk Test Stat={sw_stat_nonpuzzler:.4f}, p-value={sw_pvalue_nonpuzzler:.4g}\n")
# %%
