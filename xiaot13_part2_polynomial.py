import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


# ---------- loading and prepare data ----------
def load_data():
    # ask user for dataset path every time (no hard-code)
    csv_path = input("Please input the full path of your dataset (e.g. training_data.csv): ").strip()
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"could not input file path: {csv_path}")
    print("used file path:", csv_path)

    df = pd.read_csv(csv_path)

    # 1) Age
    if "Rings" not in df.columns:
        raise ValueError("Column 'Rings' not found. Cannot compute Age.")
    df["Age"] = df["Rings"] + 1.5

    # 2) helper for tolerant name matching
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "")

    # 3) required numeric features (7 cols)
    target_norm_features = [
        "length",
        "diameter",
        "height",
        "wholeweight",
        "shuckedweight",
        "visceraweight",
        "shellweight",
    ]

    # 4) drop index-like columns if present
    drop_like = []
    for c in df.columns:
        cn = norm(c)
        if cn.startswith("unnamed") or cn in {"index", "id"}:
            drop_like.append(c)
    if drop_like:
        df = df.drop(columns=drop_like)

    # 5) map normalized name -> real name
    norm_map = {norm(c): c for c in df.columns}
    missing, feature_cols = [], []
    for key in target_norm_features:
        if key in norm_map:
            feature_cols.append(norm_map[key])
        else:
            missing.append(key)
    if missing:
        raise ValueError(
            "Missing required feature columns (case/space/underscore tolerant):\n"
            + ", ".join(missing)
            + f"\nCurrent columns: {list(df.columns)}"
        )

    # 6) drop NA on features + target
    df = df.dropna(subset=feature_cols + ["Age"]).copy()

    # 7) X/Y
    X = df[feature_cols].to_numpy()               # (n, 7) raw scale
    Y = df["Age"].to_numpy().reshape(-1, 1)       # (n, 1)

    print("PATH of CSV:", csv_path)
    print("detected feature columns:", feature_cols)
    return df, X, Y, feature_cols, csv_path


# ---------- OLS ----------
def ols_beta(Xb, Y):
    """
    Xb: design matrix WITH intercept column already included
    Y : (n,1)
    """
    return np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ Y)


def predict_with_design(Xb, beta):
    return Xb @ beta


# ---------- metrics ----------
def metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae


# ---------- standardization (fit on train, apply on test) ----------
def fit_standardize(X):
    mu = X.mean(axis=0, keepdims=True)     # (1, d)
    sigma = X.std(axis=0, keepdims=True)   # (1, d)
    sigma = np.where(sigma == 0, 1.0, sigma)
    Z = (X - mu) / sigma
    return Z, mu, sigma

def apply_standardize(X, mu, sigma):
    return (X - mu) / sigma


# ---------- polynomial feature builder (WITH interactions) ----------
def build_poly_with_interactions(Z, degree):
    """
    Z: (n, d) standardized features
    degree >= 1
    Returns:
        Xb: (n, 1 + num_monomials) with intercept as the first column
        terms: list of tuples describing each monomial
               e.g. () is intercept, (0,) is z0, (1,1) is z1^2, (0,2) is z0*z2, etc.
    Logic:
        For k in [0..degree], take all combinations_with_replacement of feature indices of length k,
        multiply corresponding columns (k=0 gives bias).
    """
    n, d = Z.shape
    cols = [np.ones((n, 1))]   # bias
    terms = [()]               # record term structure

    for k in range(1, degree + 1):
        for comb in combinations_with_replacement(range(d), k):
            col = np.ones((n, 1))
            for j in comb:
                col *= Z[:, [j]]
            cols.append(col)
            terms.append(comb)

    Xb = np.hstack(cols)  # (n, 1 + sum_{k=1..degree} C(d+k-1, k))
    return Xb, terms


# ---------- Univariate visualization (scatter + polynomial OLS on single feature) ----------
def build_poly_univariate(x_1d, degree):
    X = np.ones((x_1d.shape[0], 1))
    for p in range(1, degree + 1):
        X = np.column_stack([X, x_1d ** p])
    return X

def visualize_feature_scatter_with_fit(df, feature, target_col="Age",
                                       out_dir="part2_figures", degree: int = 1):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    x = df[feature].to_numpy().reshape(-1, 1)
    y = df[target_col].to_numpy().reshape(-1, 1)

    # univariate polynomial OLS (raw scale on x)
    X_poly = build_poly_univariate(x, degree)
    beta = ols_beta(X_poly, y)
    y_hat = X_poly @ beta
    mse_uv = float(np.mean((y - y_hat) ** 2))

    # smooth curve
    xg = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    Xg = build_poly_univariate(xg, degree)
    yg = (Xg @ beta).ravel()

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.plot(xg, yg, linestyle="--", linewidth=1.5, label=f"deg={degree}")
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.title(f"{feature} vs {target_col} (univariate poly OLS, deg={degree})")
    plt.legend()
    plt.tight_layout()

    out_name = os.path.join(out_dir, f"{feature}_vs_{target_col}_poly_deg{degree}.png")
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Univariate deg={degree}] {feature}: MSE={mse_uv:.6f}")
    return out_name


def visualize_features_grid(df, features, target_col="Age",
                            out_dir="part2_figures", filename=None, degree: int = 1):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if filename is None:
        filename = f"feature_grid_poly_deg{degree}.png"

    n_feats = len(features)
    nrows, ncols = 2, 4  # 7 features fit in 2x4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharey=True)
    axes = axes.ravel()

    y = df[target_col].to_numpy().reshape(-1, 1)

    for i, feat in enumerate(features):
        ax = axes[i]
        x = df[feat].to_numpy().reshape(-1, 1)
        Xp = build_poly_univariate(x, degree)
        beta = ols_beta(Xp, y)

        xg = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        Xg = build_poly_univariate(xg, degree)
        yg = (Xg @ beta).ravel()

        ax.scatter(x, y, s=10, alpha=0.6)
        ax.plot(xg, yg, linestyle='--', linewidth=1.5, label=f'deg={degree}')
        ax.set_title(feat)
        ax.set_xlabel(feat)
        if i % ncols == 0:
            ax.set_ylabel(target_col)
        ax.legend(loc='best', fontsize=8)

    # hide extra subplot if any
    for j in range(n_feats, nrows * ncols):
        axes[j].axis('off')

    fig.suptitle(f"{target_col} vs Features (univariate poly OLS, deg={degree})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated grid figure: {out_path}")
    return out_path


# ---------- Training/Evaluation Process ----------
def run():
    df, X, Y, features, csv_path = load_data()

    # ask degree once
    try:
        degree = int(input("Degree for polynomial regression (1=linear, 2=quadratic, ...): ").strip())
        if degree < 1:
            raise ValueError
    except Exception:
        degree = 1
        print("Invalid input. Fallback to degree=1 (linear).")

    if degree >= 3:
        print(f"NOTE: degree={degree} will generate many features and may overfit.")

    # --- split train/test ---
    n = X.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # --- standardize ---
    Z_train, mu, sigma = fit_standardize(X_train)
    Z_test = apply_standardize(X_test, mu, sigma)

    # --- polynomial features (with interactions) ---
    Xb_train, terms = build_poly_with_interactions(Z_train, degree)
    Xb_test, _ = build_poly_with_interactions(Z_test, degree)

    print(f"\n[Model] Polynomial OLS with degree={degree}")
    print(f"Design matrix shapes: train {Xb_train.shape}, test {Xb_test.shape}")

    # --- OLS ---
    beta = ols_beta(Xb_train, Y_train)

    # --- evaluate ---
    y_pred = predict_with_design(Xb_test, beta)
    mse, rmse, mae = metrics(Y_test, y_pred)
    print("\n=== Test set error ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    # --- visualization (all use same degree) ---
    out_dir = "part2_figures"
    per_plot_paths = []
    for feat in features:
        p = visualize_feature_scatter_with_fit(df, feat, target_col="Age",
                                               out_dir=out_dir, degree=degree)
        per_plot_paths.append(p)

    grid_path = visualize_features_grid(df, features, target_col="Age",
                                        out_dir=out_dir,
                                        filename=f"feature_grid_poly_deg{degree}.png",
                                        degree=degree)

    print("\nSaved figures:")
    for p in per_plot_paths:
        print(" -", p)
    print(" -", grid_path)



if __name__ == "__main__":
    run()
