# Part 2: Abalone Age Prediction (OLS, no hard-coding, no extra libs)
# Allowed libs only: os, pandas, numpy, matplotlib, seaborn, random
# Usage (优先级从高到低):
#   1) 设置环境变量 CSV_PATH 指向 csv
#      Windows PowerShell:
#         $env:CSV_PATH="C:\path\to\training_data.csv"; python xiaot13_part2.py
#      macOS/Linux:
#         CSV_PATH="/path/to/training_data.csv" python xiaot13_part2.py
#   2) 将 training_data.csv 放在脚本同目录，直接运行：python xiaot13_part2.py
#   3) 若找不到，会提示你输入路径

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- 读取 CSV 路径（避免 hard-code） ----------
def resolve_csv_path(default_name="training_data.csv"):
    # 1) 环境变量优先
    p = os.environ.get("CSV_PATH", "").strip()
    if p and os.path.exists(p):
        return p

    # 2) 当前目录下的默认文件名
    here = os.getcwd()
    candidate = os.path.join(here, default_name)
    if os.path.exists(candidate):
        return candidate

    # 3) 交互输入
    print("未在当前目录找到 training_data.csv，且未设置 CSV_PATH 环境变量。")
    user_p = input("请输入 CSV 完整路径（例如 C:\\Users\\you\\training_data.csv）：").strip()
    if not user_p:
        raise FileNotFoundError("未提供 CSV 路径。")
    if not os.path.exists(user_p):
        raise FileNotFoundError(f"提供的路径不存在：{user_p}")
    return user_p

# ---------- 加载与准备数据 ----------
def load_data():
    csv_path = resolve_csv_path()
    df = pd.read_csv(csv_path)

    if "Rings" not in df.columns:
        raise ValueError("CSV 中未找到 'Rings' 列，无法计算 Age。请检查数据文件。")

    # 1) 先计算 Age
    df["Age"] = df["Rings"] + 1.5

    # 2) 规范化列名（仅用于特征匹配的内部映射，不改变原 df 列名）
    #    把列名做一个“标准化版本”：全小写，去空格和下划线
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "")

    # 3) 定义 7 个目标特征的“规范名”
    target_norm_features = [
        "length",
        "diameter",
        "height",
        "wholeweight",
        "shuckedweight",
        "visceraweight",
        "shellweight",
    ]

    # 4) 自动丢弃“索引/ID 类”列（常见：Unnamed: 0, index, id 等）
    drop_like = []
    for c in df.columns:
        cn = norm(c)
        if cn.startswith("unnamed") or cn in {"index", "id"}:
            drop_like.append(c)
    if drop_like:
        df = df.drop(columns=drop_like)

    # 5) 从数据里定位这 7 个特征列（兼容空格/下划线/大小写）
    #    建一个 map：规范名 -> 原始列名
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
            "找不到以下必需特征列（已兼容空格/下划线/大小写）：\n"
            + ", ".join(missing)
            + f"\n现有列：{list(df.columns)}"
        )

    # 6) X/Y
    X = df[feature_cols].to_numpy()
    Y = df["Age"].to_numpy().reshape(-1, 1)

    # 7) 打印一下最终使用的 CSV 路径和特征列，便于确认
    print("使用的 CSV 路径：", csv_path)
    print("检测到的特征列：", feature_cols)

    return df, X, Y, feature_cols, csv_path

# ---------- OLS ----------
def ols_beta(X, Y):
    Xb = np.column_stack([np.ones(X.shape[0]), X])  # 截距
    beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ Y)
    return beta  # (1 + d, 1)

def predict(X, beta):
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    return Xb @ beta

# ---------- 指标 ----------
def metrics(y_true, y_pred):
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae

# ---------- 可视化单变量关系（散点 + 单变量 OLS 直线） ----------
# ---------- 可视化单变量关系（散点 + 单变量 OLS 直线） ----------
def visualize_feature_scatter_with_fit(df, feature, target_col="Age", out_dir="part2_figures"):
    # 确保文件夹存在
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
    在一张图里对比所有特征：每个子图 = 该特征的散点 + 单变量 OLS 拟合线
    会保存到 part2_figures/feature_grid.png
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2 行 4 列的网格（7 个特征刚好放得下，留 1 个空位）
    n_feats = len(features)
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8), sharey=True)
    axes = axes.ravel()

    y = df[target_col].to_numpy().reshape(-1, 1)

    for i, feat in enumerate(features):
        ax = axes[i]
        x = df[feat].to_numpy().reshape(-1, 1)

        # 单变量 OLS 直线：y = b0 + b1 * x
        Xb = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)

        # 为了画直线更顺滑，先按 x 排序
        idx = np.argsort(x[:, 0])
        x_sorted = x[idx]
        Xb_sorted = np.column_stack([np.ones(x_sorted.shape[0]), x_sorted])
        y_line = (Xb_sorted @ beta).ravel()

        # 画散点 + 拟合线
        ax.scatter(x, y, s=10, alpha=0.6)
        ax.plot(x_sorted, y_line, linestyle='--', linewidth=1.5, label='OLS fit')
        ax.set_title(feat)
        ax.set_xlabel(feat)
        if i % ncols == 0:
            ax.set_ylabel(target_col)
        ax.legend(loc='best', fontsize=8)

    # 隐藏多余子图（第 8 个）
    if n_feats < nrows * ncols:
        for j in range(n_feats, nrows * ncols):
            axes[j].axis('off')

    fig.suptitle(f"{target_col} vs Features (scatter + univariate OLS)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"已生成对比总图：{out_path}")


# ---------- 训练/评估流程 ----------
def run():
    df, X, Y, features, csv_path = load_data()

    # 打乱并切分（不依赖 sklearn）
    n = X.shape[0]
    rng = np.random.default_rng(42)  # 固定种子便于复现
    idx = rng.permutation(n)
    split = int(0.7 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # 训练 OLS
    beta = ols_beta(X_train, Y_train)

    # 打印系数
    print("=== 数据来源 ===")
    print(csv_path)
    print("\n=== 回归系数 β (OLS，多变量) ===")
    print(f"Intercept: {beta[0, 0]:.6f}")
    for i, feat in enumerate(features):
        print(f"{feat}: {beta[i+1, 0]:.6f}")

    # 评估
    y_pred = predict(X_test, beta)
    mse, rmse, mae = metrics(Y_test, y_pred)
    print("\n=== 测试集误差 ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    # 可视化（每个特征一张图）
    for feat in features:
        visualize_feature_scatter_with_fit(df, feat, target_col="Age")
    #print("\n已生成 7 张散点图（文件名形如 '<Feature>_vs_Age.png'）。")

    visualize_features_grid(df, features, target_col="Age", out_dir="part2_figures", filename="feature_grid.png")

if __name__ == "__main__":
    run()
