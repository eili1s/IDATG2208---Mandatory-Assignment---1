from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, r2

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
X = superconductivty_data.data.features
y = superconductivty_data.data.targets

# target variable
critical_temp = y["critical_temp"].to_numpy()

corr_matrix = X.corr()

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    critical_temp,
    test_size=0.2,
    random_state=42
)

# train random forest regression model
random_forest = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features=1.0
)

random_forest.fit(X_train, y_train)

# predictions on unseen test data
y_pred = random_forest.predict(X_test)

# model performance on test data
mse, rmse, r2 = calculate_metrics(y_test, y_pred)

print("Random Forest Regression:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# relative feature importance from random forest
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": random_forest.feature_importances_
})

# rank features from most to least important
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 most important features:")
print(feature_importance.head(15).to_string(index=False))

top_15 = feature_importance.head(15).sort_values(
    by="importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top_15["feature"],
    top_15["importance"]
)

plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for Predicting critical_temp")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# check correlations between the top 15 important features
top_15_features = feature_importance.head(15)["feature"].tolist()

top_15_corr = corr_matrix.loc[
    top_15_features,
    top_15_features
]

highly_corr_pairs = []

# check each unique pair of top ranked features
for i in range(len(top_15_features)):
    for j in range(i + 1, len(top_15_features)):
        feature_1 = top_15_features[i]
        feature_2 = top_15_features[j]
        correlation = top_15_corr.loc[feature_1, feature_2]

        if abs(correlation) > 0.9:
            highly_corr_pairs.append(
                (feature_1, feature_2, correlation)
            )

# display highly correlated pairs to interpret feature importance results
print("\nHighly correlated pairs among the top 15 features:")

for feature_1, feature_2, correlation in highly_corr_pairs:
    print(
        f"{feature_1} and {feature_2}: "
        f"{correlation:.4f}"
    )

# select top 10 features from Q2.1
top_10_features = feature_importance.head(10)["feature"].tolist()

X_train_top10 = X_train[top_10_features]
X_test_top10 = X_test[top_10_features]

linear_top10 = LinearRegression()
linear_top10.fit(X_train_top10, y_train)

y_train_pred_linear = linear_top10.predict(X_train_top10)
y_test_pred_linear = linear_top10.predict(X_test_top10)

# calculate performance metrics
linear_train_mse, linear_train_rmse, linear_train_r2 = calculate_metrics(
    y_train,
    y_train_pred_linear
)

linear_test_mse, linear_test_rmse, linear_test_r2 = calculate_metrics(
    y_test,
    y_test_pred_linear
)

print("\nLinear Regression - Top 10 Features:")

print("\nTraining:")
print(f"MSE: {linear_train_mse:.4f}")
print(f"RMSE: {linear_train_rmse:.4f}")
print(f"R^2: {linear_train_r2:.4f}")

print("\nTest:")
print(f"MSE: {linear_test_mse:.4f}")
print(f"RMSE: {linear_test_rmse:.4f}")
print(f"R^2: {linear_test_r2:.4f}")


# create quadratic and interaction features
poly_features = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=False
)

# transform top 10 features into quadratic and interaction terms
X_train_poly = poly_features.fit_transform(X_train_top10)
X_test_poly = poly_features.transform(X_test_top10)

# train linear regression model on expanded polynomial feature space
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# make predictions
y_train_pred_poly = poly_reg.predict(X_train_poly)
y_test_pred_poly = poly_reg.predict(X_test_poly)

# calculate performance metrics
train_mse, train_rmse, train_r2 = calculate_metrics(
    y_train,
    y_train_pred_poly
)

test_mse, test_rmse, test_r2 = calculate_metrics(
    y_test,
    y_test_pred_poly
)

print("\nNumber of original features:", X_train_top10.shape[1])
print("Number of polynomial features:", X_train_poly.shape[1])

print("\nTraining:")
print(f"MSE: {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R^2: {train_r2:.4f}")

print("\nTest:")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R^2: {test_r2:.4f}")

# create polynomial regression pipeline
polynomial_regression = make_pipeline(
    PolynomialFeatures(
        degree=2,
        include_bias=False,
        interaction_only=False
    ),
    LinearRegression()
    , memory=None)

# create KFold cross-validation object
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# compute learning curves for polynomial regression model
train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression,
    X_train_top10,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

# convert negative sklearn scores to positive RMSE
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

# plot learning curves
plt.figure(figsize=(12, 8))

plt.plot(
    train_sizes,
    train_errors,
    label="Training RMSE"
)

plt.plot(
    train_sizes,
    valid_errors,
    label="Validation RMSE"
)

plt.xlabel("Training set size")
plt.ylabel("RMSE (K)")
plt.title("Learning Curves - Polynomial Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# scale the training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_cv = RidgeCV(
        alphas=np.logspace(-2, 2, 20),
        cv=kf,
        scoring="neg_root_mean_squared_error"
)


ridge_cv.fit(X_train_scaled, y_train)

# use cross-validation to select alpha
lasso_cv = LassoCV(
    alphas=np.logspace(-3, -1, 15),
    cv=kf,
    max_iter=20000,
    tol=2e-3,
    n_jobs=-1
)

lasso_cv.fit(X_train_scaled, y_train)

# create final Lasso model with best alpha from cross-validation
final_lasso = Lasso(
    alpha=lasso_cv.alpha_,
    max_iter=100000,
    tol=1e-4
)

final_lasso.fit(X_train_scaled, y_train)

lasso_pred = final_lasso.predict(X_test_scaled)
ridge_pred = ridge_cv.predict(X_test_scaled)

linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# predictions on the test set using the best models
linear_pred = linear_reg.predict(X_test_scaled)

linear_mse, linear_rmse, linear_r2 = calculate_metrics(
    y_test,
    linear_pred
)

print("\nLinear Regression:")
print(f"MSE: {linear_mse:.4f}")
print(f"RMSE: {linear_rmse:.4f}")
print(f"R^2: {linear_r2:.4f}")

# performance metrics
ridge_mse, ridge_rmse, ridge_r2 = calculate_metrics(
    y_test,
    ridge_pred
)

lasso_mse, lasso_rmse, lasso_r2 = calculate_metrics(
    y_test,
    lasso_pred
)

print("\nRidge Regression:")
print(f"Best alpha: {ridge_cv.alpha_:.6f}")
print(f"MSE: {ridge_mse:.4f}")
print(f"RMSE: {ridge_rmse:.4f}")
print(f"R^2: {ridge_r2:.4f}")

print("\nLasso Regression:")
print(f"Best alpha: {lasso_cv.alpha_:.6f}")
print(f"MSE: {lasso_mse:.4f}")
print(f"RMSE: {lasso_rmse:.4f}")
print(f"R^2: {lasso_r2:.4f}")

# get coefficients from best models
ridge_coefficients = ridge_cv.coef_
lasso_coefficients = final_lasso.coef_

coefficient_df = pd.DataFrame({
    "feature": X.columns,
    "ridge_coefficient": ridge_coefficients,
    "lasso_coefficient": lasso_coefficients
})

print("\nCoefficients:")
print(coefficient_df.to_string(index=False))

# features eliminated by Lasso
eliminated_features = coefficient_df[
    np.isclose(
        coefficient_df["lasso_coefficient"],
        0.0,
        atol=1e-10
    )
]["feature"].tolist()

print("\nFeatures eliminated by Lasso:")
for feature in eliminated_features:
    print(feature)

print(
    "\nNumber of eliminated features:",
    len(eliminated_features)
)

# find highly correlated pairs among all 81 features
multicollinear_pairs = []

# same logic as in Q1.2.3 with all features
for i in range(corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        correlation = corr_matrix.iloc[i, j]

        if abs(correlation) > 0.9:
            feature_1 = corr_matrix.columns[i]
            feature_2 = corr_matrix.columns[j]

            multicollinear_pairs.append(
                (feature_1, feature_2, correlation)
            )

print("\nMulticollinear pairs involving a Lasso-eliminated feature:")

# display multicollinear pairs that involve a feature eliminated by Lasso
for feature_1, feature_2, correlation in multicollinear_pairs:
    if (
        feature_1 in eliminated_features or
        feature_2 in eliminated_features
    ):
        eliminated = []

        if feature_1 in eliminated_features:
            eliminated.append(feature_1)

        if feature_2 in eliminated_features:
            eliminated.append(feature_2)

        print(
            f"{feature_1} and {feature_2}: "
            f"{correlation:.4f} "
            f"(Lasso eliminated: {', '.join(eliminated)})"
        )

print("Lasso iterations:", final_lasso.n_iter_)
print("Maximum iterations:", final_lasso.max_iter)