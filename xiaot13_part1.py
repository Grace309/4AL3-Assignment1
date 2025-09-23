import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Loads the dataset, filters to 2018, drops rows with missing values,
    and keeps records with happiness score > 4.5 (as in the class example).
    Returns:
        X (n, 2): design matrix with column of ones and standardized GDP
        Y (n, 1): standardized happiness
        x_stats: (mean, std) for the GDP feature (raw space)
        y_stats: (mean, std) for the happiness target (raw space)
        x_raw: array of raw GDP used (n,)
        y_raw: array of raw happiness used (n,)
        x_std_1d, y_std_1d: standardized 1-D arrays for plotting (n,)
    """

    # 每次运行都提示用户输入路径
    csv_path = input("请输入 gdp-vs-happiness.csv 的完整路径: ").strip()
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")
    print("使用的数据路径:", csv_path)

    df = pd.read_csv(csv_path)

    # Keep only 2018 data and required columns
    by_year = df[df['Year'] == 2018].drop(columns=["World regions according to OWID","Code"])
    # Drop NA in the two relevant columns
    by_year = by_year[
        by_year['Cantril ladder score'].notna() &
        by_year['GDP per capita, PPP (constant 2021 international $)'].notna()
    ]
    # Keep countries with happiness > 4.5 (matches the reference code/workflow)
    by_year = by_year[by_year['Cantril ladder score'] > 4.5]

    # Extract raw arrays
    y_raw = by_year['Cantril ladder score'].to_numpy()  # target: happiness
    x_raw = by_year['GDP per capita, PPP (constant 2021 international $)'].to_numpy()  # feature: GDP

    # Standardize both X and Y (compute stats from the current subset)
    x_mean, x_std = float(np.mean(x_raw)), float(np.std(x_raw))
    y_mean, y_std = float(np.mean(y_raw)), float(np.std(y_raw))

    x_std_1d = (x_raw - x_mean) / (x_std if x_std != 0 else 1.0)
    y_std_1d = (y_raw - y_mean) / (y_std if y_std != 0 else 1.0)

    # Build design matrix with intercept
    X = np.column_stack([np.ones_like(x_std_1d), x_std_1d])
    Y = y_std_1d.reshape(-1, 1)

    return X, Y, (x_mean, x_std), (y_mean, y_std), x_raw, y_raw, x_std_1d, y_std_1d

def mse(y_true, y_pred):
    """Mean Squared Error."""
    diff = y_true - y_pred
    return float(np.mean(diff ** 2))

def ols_beta(X, Y):
    """
    Closed-form OLS using the normal equation:
        beta = (X^T X)^(-1) X^T Y
    Uses pinv for numerical stability.
    """
    XtX = X.T @ X
    XtY = X.T @ Y
    beta = np.linalg.pinv(XtX) @ XtY  # (2,1)
    return beta

def predict(X, beta):
    """Vectorized prediction: y_hat = X @ beta"""
    return X @ beta

def gradient_descent(X, Y, lr=0.01, epochs=1000):
    """
    Batch Gradient Descent for linear regression with MSE loss.
    Initializes beta at zeros.
    Updates:
        beta := beta - lr * (1/n) * X^T (X beta - Y)
    Returns the learned beta (2x1).
    """
    n, d = X.shape
    beta = np.zeros((d, 1))  # (2,1)

    for _ in range(epochs):
        y_hat = X @ beta  # (n,1)
        grad = (X.T @ (y_hat - Y)) / n
        beta = beta - lr * grad

    return beta

def denormalize_beta(beta_std, x_stats, y_stats):
    """
    Convert beta' (standardized space) back to raw-space coefficients.
    Standardized model: y_std = b0' + b1' * x_std
    Raw-space equivalent: y = a + b * x
      where b = (y_std * b1') / x_std = (y_std / x_std) * b1'
            a = y_mean + y_std*b0' - b*x_mean
    Returns (intercept_raw, slope_raw)
    """
    b0p = float(beta_std[0, 0])
    b1p = float(beta_std[1, 0])
    x_mean, x_std = x_stats
    y_mean, y_std = y_stats

    if x_std == 0:
        slope_raw = 0.0
    else:
        slope_raw = (y_std / x_std) * b1p
    intercept_raw = y_mean + y_std * b0p - slope_raw * x_mean
    return intercept_raw, slope_raw

def part1_experiment():
    # Load data
    X, Y, x_stats, y_stats, x_raw, y_raw, x_std_1d, y_std_1d = load_and_prepare_data()

    # OLS (closed-form) baseline in standardized space
    beta_ols = ols_beta(X, Y)
    yhat_ols = predict(X, beta_ols)
    mse_ols = mse(Y, yhat_ols)

    # Gradient Descent experiments (choose 4–8 combos to satisfy the spec)
    combos = [
        (0.001, 5000),
        (0.01, 2000),
        (0.05, 1500),
        (0.1, 1000),
        (0.2, 800),
        (0.01, 500),
        (0.05, 500),
        (0.1, 500),
    ]

    results = []
    for lr, epochs in combos:
        beta_gd = gradient_descent(X, Y, lr=lr, epochs=epochs)
        yhat_gd = predict(X, beta_gd)
        m = mse(Y, yhat_gd)
        results.append({
            "lr": lr,
            "epochs": epochs,
            "beta": beta_gd,
            "mse": m
        })

    # Sort by MSE ascending (lower is better) to pick "best" GD run
    results_sorted = sorted(results, key=lambda r: r["mse"])
    best = results_sorted[0]

    # -----------------------------
    # Plot 1: Scatter + multiple GD lines (standardized space)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x_std_1d, Y, s=16)
    for r in results_sorted[:8]:  # up to 8 lines
        yhat = predict(X, r["beta"])
        # Don't specify colors (per instructions); use labels for clarity
        plt.plot(x_std_1d, yhat, label=f"lr={r['lr']}, epochs={r['epochs']}")

    plt.title("Standardized Happiness vs Standardized GDP — Gradient Descent Fits")
    plt.xlabel("Standardized GDP per capita")
    plt.ylabel("Standardized Happiness (Cantril ladder)")
    plt.legend()
    plot1_path = "part1_plot1_gd_lines.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # Plot 2: Scatter + OLS vs Best GD (standardized space)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x_std_1d, Y, s=16)
    plt.plot(x_std_1d, yhat_ols, label=f"OLS (MSE={mse_ols:.4f})")
    yhat_best = predict(X, best["beta"])
    plt.plot(x_std_1d, yhat_best, label=f"Best GD lr={best['lr']}, epochs={best['epochs']} (MSE={best['mse']:.4f})")
    plt.title("Standardized Happiness vs Standardized GDP — OLS vs Best Gradient Descent")
    plt.xlabel("Standardized GDP per capita")
    plt.ylabel("Standardized Happiness (Cantril ladder)")
    plt.legend()
    plot2_path = "part1_plot2_ols_vs_bestgd.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # Print coefficients (β′) and raw-space equivalents
    # -----------------------------
    print("\n=== OLS (standardized space) ===")
    print(f"beta_prime (intercept, slope): {beta_ols.ravel()}")
    ols_intercept_raw, ols_slope_raw = denormalize_beta(beta_ols, x_stats, y_stats)
    print(f"Raw-space coefficients: intercept={ols_intercept_raw:.6f}, slope={ols_slope_raw:.6f}")
    print(f"MSE (standardized): {mse_ols:.6f}")

    print("\n=== Gradient Descent runs (standardized space) ===")
    for r in results_sorted:
        b = r["beta"].ravel()
        inter_raw, slope_raw = denormalize_beta(r["beta"], x_stats, y_stats)
        print(f"lr={r['lr']:<6} epochs={r['epochs']:<5}  beta_prime=[{b[0]:+.6f}, {b[1]:+.6f}]  "
              f"MSE={r['mse']:.6f}  | raw: intercept={inter_raw:.6f}, slope={slope_raw:.6f}")

    print("\n=== Best GD (standardized space) ===")
    b = best["beta"].ravel()
    best_inter_raw, best_slope_raw = denormalize_beta(best["beta"], x_stats, y_stats)
    print(f"beta_prime=[{b[0]:+.6f}, {b[1]:+.6f}]  MSE={best['mse']:.6f}")
    print(f"Raw-space best: intercept={best_inter_raw:.6f}, slope={best_slope_raw:.6f}")
    print(f"Saved plots:\n  - {plot1_path}\n  - {plot2_path}")

    return plot1_path, plot2_path, beta_ols, best


if __name__ == "__main__":
    part1_experiment()
