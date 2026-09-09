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