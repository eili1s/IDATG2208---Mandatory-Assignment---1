from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# fetch dataset
superconductivity_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivity_data.data.features
y = superconductivity_data.data.targets

df = pd.concat([X, y], axis=1)
print(df.shape)

df.info()

# Compute 81x81 correlation matrix for all features
corr_matrix = X.corr()

# Get correlation of all features with target variable
target_corr = X.corrwith(y["critical_temp"])

# Find the top 15 features with the highest absolute correlation with target variable
top_15_features = target_corr.abs().sort_values(
    ascending=False
).head(15).index


# Create 15x15 correlation matrix for the top 15 features
top_15_corr_matrix = corr_matrix.loc[top_15_features, top_15_features]

plt.figure(figsize = (12, 8))


sns.heatmap(
    top_15_corr_matrix,
    cmap="coolwarm", # Use diverging colormap to highlight variation
    annot=True,
    fmt=".2f",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=.5,
    annot_kws={"size": 8},
)

plt.title("Heatmap of the 15 Features Most Correlated with critical_temp")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# Find the strongest positive and negative correlations with target variable
strongest_positive = target_corr.idxmax()
strongest_negative = target_corr.idxmin()

print(f"Strongest positive correlation with critical_temp: "
      f"{strongest_positive} ({target_corr[strongest_positive]:.4f})"
)

print(f"Strongest negative correlation with critical_temp: "
      f"{strongest_negative} ({target_corr[strongest_negative]:.4f})"
)
