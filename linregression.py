from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets

# select weak feature
X_gmean_fie = X[["gmean_fie"]]

# target variable reshaped into 2D column vector for matrix operations
critical_temp = y["critical_temp"].to_numpy().reshape(-1, 1)

std_scaler = StandardScaler()
# standardize the weak feature
X_weak_scaled = std_scaler.fit_transform(X_gmean_fie)

# number of observations
m = len(X_weak_scaled)

# add column for bias term to feature matrix
X_b = np.c_[np.ones((m, 1)), X_weak_scaled]

# gradient descent parameters
eta = 0.1  # learning rate
n_epochs = 1000  # number of iterations
np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

# batch gradient descent linear regression
for epoch in range(n_epochs):
    gradients = (2 / m) * X_b.T @ (X_b @ theta - critical_temp)
    theta -= eta * gradients

# gradient descent predictions
weak_predict = X_b @ theta

# select strong predictor
X_strong = X[["wtd_std_ThermalConductivity"]]

lin_reg_strong = LinearRegression()
# fit linear regression model using strong predictor
lin_reg_strong.fit(X_strong, critical_temp)

# predict critical temperature using the fitted model
strong_predict = lin_reg_strong.predict(X_strong)

weak_intercept = theta[0, 0]
weak_coeff = theta[1, 0]

weak_mse = mean_squared_error(critical_temp, weak_predict)
weak_r2 = r2_score(critical_temp, weak_predict)

print("\nWeak Predictor - Batch Gradient Descent:")
print(f"Intercept: {weak_intercept:.4f}")
print(f"Coefficient: {weak_coeff:.4f}")
print(f"MSE: {weak_mse:.4f}")
print(f"R^2: {weak_r2:.4f}")

strong_intercept = lin_reg_strong.intercept_[0]
strong_coeff = lin_reg_strong.coef_[0, 0]

strong_mse = mean_squared_error(critical_temp, strong_predict)
strong_r2 = r2_score(critical_temp, strong_predict)

print("\nStrong predictor - Linear Regression:")
print(f"Intercept: {strong_intercept:.4f}")
print(f"Coefficient: {strong_coeff:.4f}")
print(f"MSE: {strong_mse:.4f}")
print(f"R^2: {strong_r2:.4f}")

# sort standardized predictor values from lowest to highest x-value
weak_sort = np.argsort(X_weak_scaled[:, 0])

plt.figure(figsize=(12, 8))

# plot observed relationship between standardized weak predictor and critical temperature values
plt.scatter(X_weak_scaled[:, 0], critical_temp[:, 0], alpha=0.2, s=10, label="Observations")

# plot regression line using batch gradient descent model
plt.plot(X_weak_scaled[weak_sort, 0], weak_predict[weak_sort, 0], linewidth=2, label="Batch Gradient Descent Regression Line", color="red")

plt.xlabel("gmean_fie standardized")
plt.ylabel("critical_temp (K)")
plt.title("Linear Regression using Weak Predictor")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# sort predictor values from lowest to highest x-values
strong_sort = np.argsort(X_strong["wtd_std_ThermalConductivity"].to_numpy())

# apply sorting order to predictor values and corresponding predicted critical temperatures
X_strong_sorted = X_strong["wtd_std_ThermalConductivity"].to_numpy()[strong_sort]

strong_sort_predict = strong_predict.flatten()[strong_sort]

plt.figure(figsize=(12, 8))

# plot observed relationship strong predictor and critical temperature values
plt.scatter(X_strong["wtd_std_ThermalConductivity"], critical_temp.flatten(), alpha=0.2, s=10, label="Observations")

# plot regression line
plt.plot(X_strong_sorted, strong_sort_predict, linewidth=2, label="Linear Regression Line", color="red")

plt.xlabel("wtd_std_ThermalConductivity")
plt.ylabel("critical_temp (K)")
plt.title("Linear Regression using Strong Predictor")
plt.legend()
plt.grid(alpha=0.3)
plt.show()







