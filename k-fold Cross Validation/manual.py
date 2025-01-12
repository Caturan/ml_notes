import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Define the number of folds
k = 5
n_samples = X.shape[0]
fold_size = n_samples // k

# Shuffle the dataset
indices = np.arange(n_samples)
np.random.seed(42)
np.random.shuffle(indices)

# Initialize lists to store scores
mse_scores = []

# Perform k-fold cross-validation
for fold in range(k):
    # Split the data into training and validation sets
    start, end = fold * fold_size, (fold + 1) * fold_size
    val_indices = indices[start:end]
    train_indices = np.concatenate([indices[:start], indices[end:]])
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # Fit Ridge Regression model on training data
    alpha = 1.0  # Regularization strength
    ridge_model = Ridge(alpha=alpha, fit_intercept=True)
    ridge_model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = ridge_model.predict(X_val)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((y_val - y_pred) ** 2)
    mse_scores.append(mse)

# Output the results
print(f"Cross-validation MSE scores: {mse_scores}")
print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Standard Deviation of MSE: {np.std(mse_scores):.4f}")
