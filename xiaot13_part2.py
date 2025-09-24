import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- read CSV PATH（avoid hard-code） ----------
def resolve_csv_path():
    """ask user to input full file path of dataset"""
    user_p = input("please input complete PATH of dataset: ").strip()
    if not user_p:
        raise FileNotFoundError("no CSV PATH input detected。")
    if not os.path.exists(user_p):
        raise FileNotFoundError(f"input CSV PATH not exist：{user_p}")
    return user_p

# ---------- loading and prepare data ----------
def load_data():
    csv_path = resolve_csv_path()
    df = pd.read_csv(csv_path)

    if "Rings" not in df.columns:
        raise ValueError("There is no 'Ring' Column in CSV, Cannot calculate Age. Please check the dataset file")

    # 1) Calculate Age
    df["Age"] = df["Rings"] + 1.5

    # 2) standarize the name of the column（Internal mapping used only for feature matching, without changing the original df column names）
    #    Make a "standardized version" of the column names: all in lowercase, without Spaces and underscores
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "")

    # 3) Define the "specification names" of the seven target features
    target_norm_features = [
        "length",
        "diameter",
        "height",
        "wholeweight",
        "shuckedweight",
        "visceraweight",
        "shellweight",
    ]

    # 4) Automatically discard the "index /ID class" column (common: Unnamed: 0, index, id, etc.)
    drop_like = []
    for c in df.columns:
        cn = norm(c)
        if cn.startswith("unnamed") or cn in {"index", "id"}:
            drop_like.append(c)
    if drop_like:
        df = df.drop(columns=drop_like)

    # 5) Locate these 7 feature columns from the data (compatible with Spaces/underscores/case)
    #    Create a map: Standard name -> original column name
    norm_map = {norm(c): c for c in df.columns}

    missing = []
    feature_cols = []
    for key in target_norm_features:
        if key in norm_map:
            feature_cols.append(norm_map[key])
        else:
            missing.append(key)

    if missing:
        raise ValueError(
            "The following required feature columns (compatible with Spaces/underscores/case) could not be found：\n"
            + ", ".join(missing)
            + f"\ncurrent columns：{list(df.columns)}"
        )

    # 6) X/Y
    X = df[feature_cols].to_numpy()
    Y = df["Age"].to_numpy().reshape(-1, 1)

    # 7) Check the CSV PATH and featured columns which we finally used
    print("PATH of CSV：", csv_path)
    print("detected featured columns：", feature_cols)

    return df, X, Y, feature_cols, csv_path

# ---------- OLS ----------
def ols_beta(X, Y):
    Xb = np.column_stack([np.ones(X.shape[0]), X])  # intercept
    beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ Y)
    return beta  # (1 + d, 1)

def predict(X, beta):
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    return Xb @ beta

# ---------- index ----------
def metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae

# ---------- Visualization of univariate relationships (scattered points + univariate OLS straight lines) ----------
def visualize_feature_scatter_with_fit(df, feature, target_col="Age", out_dir="part2_figures"):
    # make sure the folder exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    x = df[feature].to_numpy().reshape(-1, 1)
    y = df[target_col].to_numpy().reshape(-1, 1)
    Xb = np.column_stack([np.ones(x.shape[0]), x])
    beta_uv = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)
    y_hat = Xb @ beta_uv

    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], df[target_col], s=10, alpha=0.6)
    plt.plot(df[feature], y_hat, label="OLS fit", color="red")
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.title(f"{feature} vs {target_col}")
    plt.legend()
    plt.tight_layout()

    out_name = os.path.join(out_dir, f"{feature}_vs_{target_col}.png")
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close()

def visualize_features_grid(df, features, target_col="Age", out_dir="part2_figures", filename="feature_grid.png"):
    """
    Compare all features in one graph: Each subgraph = the scatter of the feature + the univariate OLS fitting line
    It will be saved to part2_figures/feature_grid.png
    
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # generates the overall diagram of all the graphs. 
    # A 2-row by 4-column grid (with exactly 7 features fitting in, leaving 1 empty space)
    n_feats = len(features)
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharey=True)
    axes = axes.ravel()

    y = df[target_col].to_numpy().reshape(-1, 1)

    for i, feat in enumerate(features):
        ax = axes[i]
        x = df[feat].to_numpy().reshape(-1, 1)

        # Univariate OLS Linear: y = b0 + b1 * x
        Xb = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)

        # To make the straight lines smoother, sort by x first
        idx = np.argsort(x[:, 0])
        x_sorted = x[idx]
        Xb_sorted = np.column_stack([np.ones(x_sorted.shape[0]), x_sorted])
        y_line = (Xb_sorted @ beta).ravel()

        # Plot scatter points + Fit line
        ax.scatter(x, y, s=10, alpha=0.6)
        ax.plot(x_sorted, y_line, linestyle='--', linewidth=1.5, label='OLS fit')
        ax.set_title(feat)
        ax.set_xlabel(feat)
        if i % ncols == 0:
            ax.set_ylabel(target_col)
        ax.legend(loc='best', fontsize=8)

    # Hide the redundant subplot (the 8th one)
    if n_feats < nrows * ncols:
        for j in range(n_feats, nrows * ncols):
            axes[j].axis('off')

    fig.suptitle(f"{target_col} vs Features (scatter + univariate OLS)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"The comparison summary chart has been generated.：{out_path}")


# ---------- Training/Evaluation Process ----------
def run():
    df, X, Y, features, csv_path = load_data()

    # Shuffle and split (without relying on sklearn)
    n = X.shape[0]
    rng = np.random.default_rng(42)  # Storing seeds ensures reproducibility
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Training OLS
    beta = ols_beta(X_train, Y_train)

    # Printing coefficient
    print("=== source of code ===")
    print(csv_path)
    print("\n=== regression coefficient β (OLS，multivariable) ===")
    print(f"Intercept: {beta[0, 0]:.6f}")
    for i, feat in enumerate(features):
        print(f"{feat}: {beta[i+1, 0]:.6f}")

    # evaluation
    y_pred = predict(X_test, beta)
    mse, rmse, mae = metrics(Y_test, y_pred)
    print("\n=== Test set error ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    # Visualization (one graph for each feature)
    for feat in features:
        visualize_feature_scatter_with_fit(df, feat, target_col="Age")
    #print("\nSeven scatter plots have been generated (with filenames of the form '<Feature>_vs_Age.png').")

    visualize_features_grid(df, features, target_col="Age", out_dir="part2_figures", filename="feature_grid.png")

if __name__ == "__main__":
    run()
