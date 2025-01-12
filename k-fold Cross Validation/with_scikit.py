import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Define the ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can tune alpha for regularization strength

# Define the number of folds for k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(ridge_model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert negative MSE scores to positive values for interpretability
mse_scores = -cv_scores

# Output the results
print(f"Cross-validation MSE scores: {mse_scores}")
print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Standard Deviation of MSE: {np.std(mse_scores):.4f}")
