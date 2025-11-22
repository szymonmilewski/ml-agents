import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


def analyze_single_run(run_id: str, csv_name: str = "prototype.csv"):
    """Perform linear regression analysis on a single training run."""

    csv_path = os.path.join(r"C:\Users\Anna\Documents\UM\Project 2-1\ml-agents\ml-agents\results", run_id, csv_name)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\nAnalyzing run: {run_id}")
    print(f"Total steps: {len(df)}\n")

    # Learning progress: reward vs steps
    print("Learning progression (reward vs steps)")
    X = df[['step']].values
    y = df['cumulative_reward'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"Slope (reward increase per step): {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R² Score: {r2_score(y, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

    # Episode length vs steps (efficiency)
    print("\nEfficiency (episode length vs steps)")
    y_length = df['episode_length'].dropna().values
    X_length = df.loc[df['episode_length'].notna(), 'step'].values.reshape(-1, 1)

    model_length = LinearRegression()
    model_length.fit(X_length, y_length)
    y_length_pred = model_length.predict(X_length)

    print(f"Slope (length change per step): {model_length.coef_[0]:.6f}")
    print(f"R² Score: {r2_score(y_length, y_length_pred):.4f}")

    # Correlation analysis
    print("\nCorrelation analysis")

    metrics = ['cumulative_reward', 'episode_length', 'entropy', 'value_estimate',
               'policy_loss', 'value_loss', 'learning_rate']

    available_metrics = [m for m in metrics if m in df.columns]
    corr_matrix = df[available_metrics].corr()

    print("\nCorrelation with Cumulative Reward:")
    reward_corr = corr_matrix['cumulative_reward'].sort_values(ascending=False)
    print(reward_corr)

    # Creating the plots (save to files)
    output_dir = os.path.join(r"C:\Users\Anna\Documents\UM\Project 2-1\ml-agents-develop\ml-agents-develop\results",
                              run_id)

    # Plot 1: Reward Progression with Linear Regression
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Y: Cumulative Reward
    axes[0, 0].scatter(X, y, alpha=0.5, s=20, label='Actual')
    axes[0, 0].plot(X, y_pred, color='red', linewidth=2, label=f'Linear Fit (R²={r2_score(y, y_pred):.3f})')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].set_title(f'{run_id}: Cumulative Reward vs Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # X: Episode Length
    axes[0, 1].scatter(X_length, y_length, alpha=0.5, s=20, label='Actual')
    axes[0, 1].plot(X_length, y_length_pred, color='red', linewidth=2,
                    label=f'Linear Fit (R²={r2_score(y_length, y_length_pred):.3f})')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title(f'{run_id}: Episode Length vs Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Rolling Mean Reward
    if 'reward_rolling_mean' in df.columns:
        axes[1, 0].plot(df['step'], df['cumulative_reward'], alpha=0.3, label='Raw Reward')
        axes[1, 0].plot(df['step'], df['reward_rolling_mean'], linewidth=2, label='Rolling Mean (10 steps)')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title(f'{run_id}: Reward Smoothing')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Correlation Heatmap

    corr_to_plot = corr_matrix.loc[available_metrics[:5], available_metrics[:5]]
    im = axes[1, 1].imshow(corr_to_plot, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr_to_plot.columns)))
    axes[1, 1].set_yticks(range(len(corr_to_plot.index)))
    axes[1, 1].set_xticklabels(corr_to_plot.columns, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(corr_to_plot.index)
    axes[1, 1].set_title('Correlation Heatmap')

    # Add correlation values to heatmap
    for i in range(len(corr_to_plot.index)):
        for j in range(len(corr_to_plot.columns)):
            text = axes[1, 1].text(j, i, f'{corr_to_plot.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'regression_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    plt.close()

    # print results
    results = {
        'run_id': run_id,
        'total_steps': len(df),
        'reward_slope': model.coef_[0],
        'reward_r2': r2_score(y, y_pred),
        'length_slope': model_length.coef_[0],
        'length_r2': r2_score(y_length, y_length_pred),
        'final_reward': df['cumulative_reward'].iloc[-1],
        'mean_reward': df['cumulative_reward'].mean()
    }

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        # Multiple runs comparison
        run_ids = sys.argv[1:]
        print(f"Comparing {len(run_ids)} runs: {run_ids}")
        # compare_multiple_runs(run_ids)
    elif len(sys.argv) == 2:
        # Single run analysis
        run_id = sys.argv[1]
        analyze_single_run(run_id)
    else:
        print("Usage:")
        print("  Single run:     python linear_regression.py <run_id>")
        # print("  Multiple runs:  python linear_regression.py <run1> <run2> <run3> ...")
