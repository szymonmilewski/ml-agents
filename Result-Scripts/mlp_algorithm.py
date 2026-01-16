import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

#Replace the path with whichever folder the target csv file is in
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

def load_run_csv(run_id: str, csv_name: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / run_id / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV was not found")
    return pd.read_csv(csv_path)

def select_target(df: pd.DataFrame) -> str:
    if "reward_rolling_mean" in df.columns:
        return "reward_rolling_mean"
    if "mean_reward" in df.columns:
        return "mean_reward"
    if"cumulative_reward" in df.columns:
        return "cumulative_reward"
    raise ValueError("No valid reward column was found")

def build_xy(df: pd.DataFrame, target: str):
    if "step" not in df.columns or target not in df.columns:
        raise ValueError(f"the CSV must contain 'step' and '{target}'")
    data = df[["step", target]].dropna()
    step_values = data["step"].astype(float).to_numpy()
    y = data[target].astype(float).to_numpy()
    X = step_values.reshape(-1, 1)
    return step_values, X, y

def time_split(X, y, train_frac = 0.8):
    split_idx = int(len(X) * train_frac)
    if split_idx < 10 or (len(X) - split_idx) < 5:
        raise ValueError("Not enough samples for splitting")
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

def get_step_ticks(max_step: float) -> np.ndarray:
    if max_step <=  1_000_000:
        step = 100_000
    elif max_step <= 5_000_000:
        step = 500_000
    elif max_step <= 10_000_000:
        step = 1_000_000
    else:
        step = 2_500_000
    ticks = np.arange(0.0, max_step + step, step, dtype = float)
    return np.atleast_1d(ticks)

def build_mlp_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes = (64, 64),
            activation = "relu",
            solver = "adam",
            alpha = 0.0001,
            learning_rate_init = 0.001,
            max_iter = 3000,
            early_stopping = True,
            validation_fraction = 0.15,
            n_iter_no_change = 30,
            random_state = seed))
])    

def evaluate(model: Pipeline, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    }
    if np.var(y_test) < 1e-12:
        metrics["test_r2"] = float("nan")
    return metrics

def save_fit_plot(run_id: str, X_all, y_all, y_all_pred):
    out_path = RESULTS_DIR / run_id / "mlp_reward_fit.png"

    fig, ax = plt.subplots(figsize= (10, 6))

    ax.scatter(X_all, y_all, s = 10, alpha = 0.25, label = "Real Reward", color = "black")
    ax.plot(X_all, y_all_pred, linewidth = 1, color = "red", label = "MLP Fit")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Reward")
    ax.set_title(f"{run_id}: MLP fit of reward vs Steps")
    ax.grid(True, alpha = 0.3)
    ax.legend()

    max_step = float(np.max(X_all))
    ticks = get_step_ticks(max_step)

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    return out_path


def save_residuals_plot(run_id: str, X_all, residuals):
    out_path = RESULTS_DIR / run_id / "mlp_residuals.png"
    
    fig, ax = plt.subplots(figsize= (10, 6))

    ax.scatter(X_all, residuals, s = 10, alpha = 0.25,label = "Residuals (Errors)", color = "black")
    ax.axhline(0, color = "red", linewidth = 1, label = "Zero Error Prediction")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{run_id}: MLP Residuals")
    ax.grid(True, alpha = 0.3)
    ax.legend()

    max_step = float(np.max(X_all))
    ticks = get_step_ticks(max_step)

    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
    plt.savefig(out_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    return out_path


def analyze_mlp(run_id: str, csv_name: str = "prototype.csv") -> None:
    df = load_run_csv(run_id, csv_name)
    target = select_target(df)
    steps, X_all, y_all = build_xy(df, target)

    X_train, X_test, y_train, y_test = time_split(X_all, y_all, train_frac = 0.8)
    model = build_mlp_model(seed = 42)

    metrics = evaluate(model, X_train, X_test, y_train, y_test)

    mlp = model.named_steps["mlp"]
    print(f"MLP Post Analysis")
    print(f"Run ID: {run_id}")
    print(f"CSV: {csv_name}")
    print(f"Target: {target}")
    print("\nModel details:")
    print(f"Hidden Layers: {mlp.hidden_layer_sizes}")
    print(f"Activation: {mlp.activation}")
    print(f"Solver: {mlp.solver}")
    print(f"L2 alpha: {mlp.alpha}")
    print(f"Early stop: {mlp.early_stopping}")
    print(f"Training iterations: {mlp.n_iter_}")
    print("\nPerformance:")
    print(f"Train R^2: {metrics['train_r2']:.4f} Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Test R^2: {metrics['test_r2']:.4f} Test RMSE: {metrics['test_rmse']:.4f}")

    y_all_pred = model.predict(X_all)
    fit_path = save_fit_plot(run_id, steps, y_all, y_all_pred)
    res_path = save_residuals_plot(run_id, steps, y_all - y_all_pred)

    print(f"\nSaved Plot: {fit_path}")
    print(f"Saved Plot: {res_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mlp_algorithm.py <run_id> [csv_name]")
        raise SystemExit(1)
    run_id_arg = sys.argv[1]
    csv_name_arg = sys.argv[2] if len(sys.argv) > 2 else "prototype.csv"
    analyze_mlp(run_id_arg, csv_name_arg)