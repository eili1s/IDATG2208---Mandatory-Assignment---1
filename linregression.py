from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets

# select weak feature
X_gmean_fie = X[["gmean_fie"]]

# target variable, 1D array reshaped into 2D array for scikit-learn
critical_temp = y["critical_temp"].to_numpy().reshape(-1, 1)

std_scaler = StandardScaler()
# standardize the weak feature
gmean_fie_std_scaled = std_scaler.fit_transform(X_gmean_fie)

# number of samples
m = len(gmean_fie_std_scaled)

# add bias term to feature matrix
X_b = np.c_[np.ones((m, 1)), gmean_fie_std_scaled]

# gradient descent parameters
eta = 0.1  # learning rate
n_epochs = 1000  # number of iterations
np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

# batch gradient descent
for epoch in range(n_epochs):
    gradients = (2 / m) * X_b.T @ (X_b @ theta - critical_temp)
    theta -= eta * gradients

print(f"\nLinear Regression using batch gradient descent: "
        f"Bias: {theta[0][0]}, Weight: {theta[1][0]}")


lin_reg = LinearRegression()
lin_reg.fit(gmean_fie_std_scaled, critical_temp)

print(f"\nLinear Regression using scikit-learn: "
      f"Bias: , {lin_reg.intercept_[0]}, Weight: {lin_reg.coef_[0, 0]}")











