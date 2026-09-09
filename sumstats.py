from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


y["critical_temp"].hist(bins=50)
plt.title("Distribution of critical_temp")
plt.xlabel("critical_temp (Kelvin)")
plt.ylabel("Frequency")
plt.show()

log_critical_temp = np.log(y["critical_temp"])

log_critical_temp.hist(bins=50)
plt.title("Distribution of log(critical_temp)")
plt.xlabel("ln(critical_temp)")
plt.ylabel("Frequency")
plt.show()

print("Raw skewness:",
      round(y["critical_temp"].skew(), 4))

print("Log skewness:",
      round(np.log(y["critical_temp"]).skew(), 4))
