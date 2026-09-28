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
    # expand theta into 82 parameters, 81 features + 1 intercept term
    theta = np.random.randn(X_b.shape[1], 1)

    # store training MSE
    cost_history = []

    # batch gradient descent on the entire training set
    for epoch in range(n_epochs):
        gradients = (2 / m) * X_b.T @ (X_b @ theta - y_train)
        theta -= eta * gradients

        # record training cost after parameter update
        y_pred = X_b @ theta
        mse = mean_squared_error(y_train, y_pred)
        cost_history.append(mse)

    return theta, cost_history

# multiple linear regression evaluation function
def evaluate_multip_regr():
    # select all predictor features
    X_all = X.to_numpy()

    # perform 5-fold cross-validation giving 80/20 train/test split
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    # store performance metrics and cost history for each fold
    results = []
    cost_histories = []

    # store actual and predicted test values
    y_test_all = []
    y_pred_all = []

    # repeat training and evaluation for each fold
    for fold, (train_indices, test_indices) in enumerate(k_fold.split(X_all), start=1):
        # split data into training and testing sets
        X_train, X_test = X_all[train_indices], X_all[test_indices]
        y_train, y_test = critical_temp[train_indices], critical_temp[test_indices]

        scaler = StandardScaler()

        # fit scaler on training data, apply transformation to both training and testing data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train multiple linear regression on standardized training data
        theta, cost_history = batch_gradient_descent(X_train_scaled, y_train)

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
        # store cost at each epoch for current fold
        cost_histories.append(cost_history)

        # store actual and predicted test values for current fold
        y_test_all.append(y_test)
        y_pred_all.append(y_test_pred)

    # combine test values from all folds
    y_test_all = np.vstack(y_test_all)
    y_pred_all = np.vstack(y_pred_all)

    return pd.DataFrame(results), cost_histories, y_test_all, y_pred_all

# run 5-fold evaluation
multip_results, multip_costs, multip_actual, multip_pred = evaluate_multip_regr()

plt.figure(figsize=(12, 8))

plt.scatter(
    multip_actual,
    multip_pred,
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
plt.title("Multiple Linear Regression - Predicted vs Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

