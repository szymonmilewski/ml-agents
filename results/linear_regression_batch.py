from linear_regression import analyze_ram_usage
from pathlib import Path
import pandas as pd
from datetime import datetime


# Analyze linear regression across runs

# Add your files here manually
csv_files = [
    "ml-agents/config/multi/BasicPPO.1.AK/BasicPPO.1.AK.csv",
    "ml-agents/config/multi/BasicPPO.2.AK/BasicPPO.2.AK.csv",
    "ml-agents/config/multi/BasicPPO.4.AK/BasicPPO.4.AK.csv",
    "ml-agents/config/multi/BasicPPOEpsilon.1.AK/BasicPPOEpsilon.1.AK.csv",
    "ml-agents/config/multi/BasicPPOEpsilon.2.AK/BasicPPOEpsilon.2.AK.csv",
    "ml-agents/config/multi/BasicPPOEpsilon.3.AK/BasicPPOEpsilon.3.AK.csv",
    "ml-agents/config/multi/BasicPPOEpsilon.4.AK/BasicPPOEpsilon.4.AK.csv",


]


print("Combined analysis across multiple runs")

# Load and combine all CSV files
all_dataframes = []
for csv_path in csv_files:
    csv_path = Path(csv_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # new column for tracking run number 
        df['source_run'] = csv_path.parent.name
        all_dataframes.append(df)
        print(f" Loaded {len(df)} runs from: {csv_path}")
    else:
        print(f"File not found: {csv_path}")

if not all_dataframes:
    print("\nError: No valid CSV files found")
    exit(1)

# Combine all dataframes
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Combined dataset: {len(combined_df)} total training runs from {len(all_dataframes)} experiments")

# Show breakdown by source
print("Runs per experiment:")
for run_name in combined_df['source_run'].unique():
    count = len(combined_df[combined_df['source_run'] == run_name])
    print(f"  {run_name}: {count} runs")

# Create unique output directory with timestamp (to avoid overwriting prev results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = Path("results/ram_analysis_combined")
output_dir = base_output_dir / f"analysis_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nResults will be saved to: {output_dir}")

# Save combined CSV
combined_csv_path = output_dir / "combined_runs.csv"
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved to: {combined_csv_path}")

# Run unified analysis on combined data
print("Unified LinReg Analysis")

analyze_ram_usage(str(combined_csv_path), output_dir=str(output_dir))

# Correlation matrix for numeric columns
print("Correlation analysis")

# Select only numeric columns (hyperparams + RAM metrics)
numeric_cols = combined_df.select_dtypes(include=['number']).columns
corr_matrix = combined_df[numeric_cols].corr()

# Show correlation with RAM usage
ram_correlations = corr_matrix['ram_mb_used'].sort_values(ascending=False)
print("Correlation with RAM usage (ram_mb_used):")
print("-"*50)
for feature, corr_value in ram_correlations.items():
    if feature != 'ram_mb_used':  # Skip self-correlation
        strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
        direction = "positive" if corr_value > 0 else "negative"
        print(f"  {feature:<25} {corr_value:>8.3f} ({strength} {direction})")

# Save correlation with RAM as CSV
corr_df = pd.DataFrame({
    'Feature': ram_correlations.index,
    'Correlation': ram_correlations.values,
    'Abs_Correlation': ram_correlations.abs().values
})
corr_df = corr_df[corr_df['Feature'] != 'ram_mb_used'] 
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
corr_output_path = output_dir / "ram_correlations.csv"
corr_df.to_csv(corr_output_path, index=False)
print(f"RAM correlations saved to: {corr_output_path}")

# Save full correlation matrix
corr_matrix_path = output_dir / "correlation_matrix.csv"
corr_matrix.to_csv(corr_matrix_path)
print(f"Full correlation matrix saved to: {corr_matrix_path}")

# Save run metadata
metadata_path = output_dir / "run_metadata.txt"
with open(metadata_path, 'w', encoding='utf-8') as f:
    f.write(f"Analysis Run: {timestamp}\n")
    f.write(f"{'='*80}\n\n")
    f.write(f"Source files:\n")
    for csv_file in csv_files:
        f.write(f"  - {csv_file}\n")
    f.write(f"\nTotal runs: {len(combined_df)}\n")
    f.write(f"Experiments combined: {len(all_dataframes)}\n\n")
    f.write(f"Runs per experiment:\n")
    for run_name in combined_df['source_run'].unique():
        count = len(combined_df[combined_df['source_run'] == run_name])
        f.write(f"  {run_name}: {count} runs\n")

print(f"Run metadata saved to: {metadata_path}")

print("Conclusions across runs:")
print(f"Based on {len(combined_df)} combined training runs from {len(all_dataframes)} experiments,")
print("we can identify which hyperparameters consistently affect RAM usage by checking")
print("  - statistically significant hyperparameters (p < 0.05)")
print("  - the amount the hyperparameters effect RAM (coefficients)")
print("  - overall prediction accuracy (R²)")
print("  - visual plots in: results/ram_analysis_combined/ folder")

print(f"\nAll results saved in: {output_dir}")