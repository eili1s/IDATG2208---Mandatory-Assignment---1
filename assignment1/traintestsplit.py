from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, add_dummy_feature
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets

# target variable reshaped into 2D column vector for matrix operations
critical_temp = y["critical_temp"].to_numpy().reshape(-1, 1)

# batch gradient descent function
def batch_gradient_descent(X_train, y_train):
    # add column of ones to make theta[0] represent the intercept term
    X_b = add_dummy_feature(X_train)

    # gradient descent hyperparameters
    eta = 0.01 # learning rate
    n_epochs = 1000 # number of iterations
    m = len(X_b) # number of observations

    np.random.seed(42) # fixed, for reproducibility
    theta = np.random.randn(2, 1) # random initialization

    # batch gradient descent on the entire training set
    for epoch in range(n_epochs):
        gradients = (2 / m) * X_b.T @ (X_b @ theta - y_train)
        theta -= eta * gradients

    return theta

# linear regression evaluation function for a single feature
def evaluate_feature(feature_name):
    # select predictor
    X_feature = X[[feature_name]].to_numpy()

    # perform 5-fold cross-validation giving 80/20 train/test split
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    y_test_all = []
    y_pred_all = []

    # repeat training and evaluation for each fold
    for fold, (train_indices, test_indices) in enumerate(k_fold.split(X_feature), start=1):
        # split data into training and testing sets
        X_train, X_test = X_feature[train_indices], X_feature[test_indices]
        y_train, y_test = critical_temp[train_indices], critical_temp[test_indices]

        scaler = StandardScaler()

        # fit scaler on training data, apply transformation to both training and testing data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # linear regression using batch gradient descent on standardized training data
        theta = batch_gradient_descent(X_train_scaled, y_train)

        # add intercept column to scaled training and testing features
        X_train_b = add_dummy_feature(X_train_scaled)
        X_test_b = add_dummy_feature(X_test_scaled)

        # make predictions of critical_temp for both training and testing observations
        y_train_pred = X_train_b @ theta
        y_test_pred = X_test_b @ theta

        # calculate performance metrics for test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_test_pred)

        # calculate performance metrics for training set
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        # store results for current fold
        results.append({
            "fold": fold,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })

        y_test_all.append(y_test)
        y_pred_all.append(y_test_pred)

    y_test_all = np.vstack(y_test_all)
    y_pred_all = np.vstack(y_pred_all)

    return pd.DataFrame(results), y_test_all, y_pred_all

# strong predictor evaluation
strong_results, strong_actual, strong_pred = evaluate_feature("wtd_std_ThermalConductivity")

plt.figure(figsize=(12, 8))

plt.scatter(
    strong_actual.ravel(),
    strong_pred.ravel(),
    alpha=0.3
)

plt.plot(
    [-30, 190],
    [-30, 190],
    color='red',
    linestyle='--',
    label="Perfect prediction"
)

plt.xlim(-30, 190)
plt.ylim(-30, 190)
plt.xlabel("Actual critical_temp (K)")
plt.ylabel("Predicted critical_temp (K)")
plt.title("Simple Linear Regression - Predicted vs Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.show()