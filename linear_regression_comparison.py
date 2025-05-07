import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tracemalloc
from scipy.stats import zscore

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Target")

# ========== PLOTTING RESULTS ==========

def plot_residuals(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{title} - Residual Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--')
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{title} - Residuals vs. Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()


# ========== BASELINE LINEAR REGRESSION ==========

def baseline_linear_regression(X, y):
    # Start performance tracking
    tracemalloc.start()
    start_time = time.time()
    
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Fit simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # End performance tracking
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nBaseline Linear Regression:")
    print(f"RMSE:        {rmse:.4f}")
    print(f"R²:          {r2:.4f}")
    print(f"Runtime:     {(end_time - start_time):.4f} seconds")
    print(f"Peak Memory: {peak / 1024 / 1024:.4f} MB")

    return model, X_test, y_test


# ========== ENHANCED LINEAR REGRESSION ==========

def enhanced_linear_regression(X, y):
    # Start performance tracking
    tracemalloc.start()
    start_time = time.time()
    
    # 1. Outlier removal
    X_z = X.apply(zscore)
    mask = (np.abs(X_z) < 3).all(axis=1)
    X, y = X[mask], y[mask]

    # 2. Skew correction
    skewed = X.skew().sort_values(ascending=False)
    skewed_feats = skewed[skewed > 0.75].index
    pt = PowerTransformer(method='yeo-johnson')
    X[skewed_feats] = pt.fit_transform(X[skewed_feats])

    # 3. Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)

    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. ElasticNet regression
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)

    # 7. Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # End performance tracking
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\nEnhanced Linear Regression:")
    print(f"RMSE:        {rmse:.4f}")
    print(f"R²:          {r2:.4f}")
    print(f"Runtime:     {(end_time - start_time):.4f} seconds")
    print(f"Peak Memory: {peak / 1024 / 1024:.4f} MB")

    return model, X_test, y_test

# Run both models
baseline_model = baseline_linear_regression(X.copy(), y.copy())
enhanced_model = enhanced_linear_regression(X.copy(), y.copy())

# ========== Run and Plot ==========

# Baseline
baseline_model, X_test_base, y_test_base = baseline_linear_regression(X.copy(), y.copy())
plot_residuals(baseline_model, X_test_base, y_test_base, "Baseline Linear Regression")

# Enhanced
enhanced_model, X_test_enh, y_test_enh = enhanced_linear_regression(X.copy(), y.copy())
plot_residuals(enhanced_model, X_test_enh, y_test_enh, "Enhanced Linear Regression")
