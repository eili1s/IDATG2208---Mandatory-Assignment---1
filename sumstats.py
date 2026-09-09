from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets

df = pd.concat([X, y], axis=1)
print(df.shape)

df.info()

# summary statistics transposed
summary = df.describe().T

# all features + target variable
print(summary)

# only features
features_summary = summary.loc[X.columns]

# highest variation features sorted by std
highest_variation = features_summary.sort_values(
    by="std",
    ascending=False
)

print("Features with highest variation:\n",
      highest_variation.head(10).to_string())

print("\nTarget:")
print(summary.loc[y.columns[0]])



