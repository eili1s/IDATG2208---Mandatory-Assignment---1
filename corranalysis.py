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

print(f"\nStrongest positive correlation with critical_temp: "
      f"{strongest_positive} ({target_corr[strongest_positive]:.4f})"
)

print(f"Strongest negative correlation with critical_temp: "
      f"{strongest_negative} ({target_corr[strongest_negative]:.4f})\n"
)

# Initialize empty list to store highly correlated feature pairs
highly_corr_pairs = []

for i in range (corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        correlation_pair = corr_matrix.iloc[i, j] # Get correlation value for the pair of features

        if abs(correlation_pair) > 0.9:
            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]

            # Store the features with correlation above 0.9 in the list
            highly_corr_pairs.append((feature_1, feature_2, correlation_pair))

for feature_1, feature_2, correlation_pair in highly_corr_pairs[:10]:
    print(
        f"'{feature_1}' and '{feature_2}': " 
        f"{correlation_pair:.4f}"
    )

# Find the feature with the weakest correlation with the target variable
weak_predictor = target_corr.abs().idxmin()
weak_corr = target_corr[weak_predictor]

print(f"\nWeakest correlation with critical_temp: "
        f"{weak_predictor} ({weak_corr:.4f})"
)




