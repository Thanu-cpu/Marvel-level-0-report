# linear_regression_scratch.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# 1. Load California Housing Data
# --------------------------
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features (important for gradient descent!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --------------------------
# 2. Helper Functions
# --------------------------
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, weights, bias, lr, epochs):
    n_samples = X.shape[0]
    for i in range(epochs):
        y_pred = predict(X, weights, bias)

        # Gradients
        dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
        db = -(2/n_samples) * np.sum(y - y_pred)

        # Update
        weights -= lr * dw
        bias -= lr * db

        # Print progress
        if i % 100 == 0:
            print(f"Epoch {i}, MSE: {mse(y, y_pred):.4f}")

    return weights, bias

def evaluate(y_true, y_pred, model_name):
    print(f"\n---- {model_name} ----")
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R²:", r2_score(y_true, y_pred))

# --------------------------
# 3. Train Custom Model
# --------------------------
n_samples, n_features = X.shape
weights = np.zeros(n_features)
bias = 0.0

# Gradient descent
weights, bias = gradient_descent(X, y, weights, bias, lr=0.01, epochs=1000)
y_pred_custom = predict(X, weights, bias)

# --------------------------
# 4. Train Sklearn Model
# --------------------------
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_sklearn = lr_model.predict(X)

# --------------------------
# 5. Evaluate Both Models
# --------------------------
evaluate(y, y_pred_custom, "Custom Gradient Descent Model")
evaluate(y, y_pred_sklearn, "Sklearn LinearRegression Model")

# --------------------------
# 6. Visualization
# --------------------------
# For visualization, plot one feature (Median Income vs Target)
plt.scatter(X[:,0], y, alpha=0.3, label="Data points")
plt.scatter(X[:,0], y_pred_custom, color="red", alpha=0.2, label="Custom Model Predictions")
plt.xlabel("Median Income (standardized)")
plt.ylabel("House Value")
plt.legend()
plt.title("California Housing: Custom Linear Regression Fit")
plt.show()
