import pandas as pd
import statsmodels.api as sm
import os
import numpy as np

# Load your aggregated data
csv_path = "results/3rdPyramidsRun/prototype.csv"
df = pd.read_csv(csv_path)

# Select features and target
""" features = [
    'batch_size', 'buffer_size', 'learning_rate', 'network_size',
    'ram_size_gb', 'cpu_type', 'gpu_type'
]
df = pd.get_dummies(df, columns=['cpu_type', 'gpu_type'], drop_first=True)
X = df[features + [col for col in df.columns if col.startswith('cpu_type_') or col.startswith('gpu_type_')]]
y = df['ram_usage_percent'] """



# Example: Predict cumulative_reward from learning_rate and entropy
features = [
    'step', 'time_elapsed', 'episode_length', 'extrinsic_reward', 'value_estimate',
    'entropy', 'policy_loss', 'value_loss', 'learning_rate', 'epsilon', 'beta',
    'reward_rolling_mean', 'reward_rolling_std', 'episode_length_rolling_mean'
]
# Target variable
y = df['cumulative_reward']
X = df[features]
X = sm.add_constant(X)

# Add constant for intercept
X = sm.add_constant(X)

# Remove rows with NaN or infinite values in features or target
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)
valid = X.notnull().all(axis=1) & y.notnull()
X = X[valid]
y = y[valid]

# Fit regression model
model = sm.OLS(y, X).fit()

# Print summary (shows coefficients and p-values)
print(model.summary())

# After model = sm.OLS(y, X).fit()
summary_text = model.summary().as_text()

# Save to a .txt file in the same folder as your prototype.csv
import os
csv_path = "results/3rdPyramidsRun/prototype.csv"
output_path = os.path.join(os.path.dirname(csv_path), "linreg_full_summary.txt")
with open(output_path, "w") as f:
    f.write(summary_text)

print(f"Full regression summary saved to: {output_path}")

# Extract main results table
results_df = model.summary2().tables[1]  # This is the coefficients table

# Save to CSV in same folder as prototype.csv
output_path = os.path.join(os.path.dirname(csv_path), "linreg_full_summary.csv")
results_df.to_csv(output_path)

print(f"Main regression results table saved to: {output_path}")
