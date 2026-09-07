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
summary = df.describe().T
pd.set_option('display.max_columns', 100)
print(summary)


