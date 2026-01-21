from linear_regression import analyze_ram_usage
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys


# Analyze linear regression across runs

def find_csv_files(folder_path):
    #Find folder of training run csvs (terminal input)
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return []
    
    # Find all CSV files recursively
    csv_files = list(folder.rglob("*.csv"))
    return [str(csv_file) for csv_file in csv_files]


def main():
    if len(sys.argv) < 2:
        print("Usage: python linear_regression_batch.py <folder_path>")
        print("\nExample:")
        print("  python results/linear_regression_batch.py ml-agents/config/multi/batch_train_files")
        print("\nThis will find all CSV files in the folder and run combined linear regression analysis.")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # print(f"Scanning folder: {folder_path}")
    csv_files = find_csv_files(folder_path)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        sys.exit(1)
    
    # print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        # print(f"  - {csv_file}")
        pass
    
    # print("\nCombined analysis across multiple runs")

    # Load and combine all CSV files
    all_dataframes = []
    for csv_path_str in csv_files:
        csv_path = Path(csv_path_str)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # new column for tracking run number 
            df['source_run'] = csv_path.parent.name
            all_dataframes.append(df)
            # print(f" Loaded {len(df)} runs from: {csv_path.name}")
        else:
            print(f"File not found: {csv_path}")

    if not all_dataframes:
        print("\nError: No valid CSV files found")
        sys.exit(1)

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    # print(f"Combined dataset: {len(combined_df)} total training runs from {len(all_dataframes)} experiments")

    # Show breakdown by source
    # print("Runs per experiment:")
    for run_name in combined_df['source_run'].unique():
        count = len(combined_df[combined_df['source_run'] == run_name])
        # print(f"  {run_name}: {count} runs")
        pass

    # Create unique output directory with timestamp (to avoid overwriting prev results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("results/ram_analysis_combined")
    output_dir = base_output_dir / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_file = output_dir / f"batch_analysis_log_{timestamp}.txt"
    
    def log_and_print(message):
        #both print to terminal and log to file
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    log_and_print(f"Batch Linear Regression Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"Scanning folder: {folder_path}")
    
    if not csv_files:
        log_and_print(f"No CSV files found in {folder_path}")
        sys.exit(1)
    
    log_and_print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        log_and_print(f"  - {csv_file}")
    
    log_and_print("\nCombined analysis across multiple runs")

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    log_and_print(f"Combined dataset: {len(combined_df)} total training runs from {len(all_dataframes)} experiments")

    # Show breakdown by source
    log_and_print("Runs per experiment:")
    for run_name in combined_df['source_run'].unique():
        count = len(combined_df[combined_df['source_run'] == run_name])
        log_and_print(f"  {run_name}: {count} runs")

    log_and_print(f"\nResults will be saved to: {output_dir}")

    # Save combined CSV
    combined_csv_path = output_dir / "combined_runs.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    log_and_print(f"Combined CSV saved to: {combined_csv_path}")

    # Run unified analysis on combined data
    log_and_print("\n" + "="*80)
    log_and_print("Unified LinReg Analysis - RAM Usage Percentage (normalized across machines)")
    log_and_print("="*80)
    
    analyze_ram_usage(str(combined_csv_path), output_dir=str(output_dir), use_percentage=True)
    
    log_and_print("\n" + "="*80)
    log_and_print("For comparison: Analysis using absolute RAM (MB)")
    log_and_print("="*80)
    
    # Create subfolder for absolute analysis
    absolute_output_dir = output_dir / "absolute_ram_analysis"
    absolute_output_dir.mkdir(exist_ok=True)
    analyze_ram_usage(str(combined_csv_path), output_dir=str(absolute_output_dir), use_percentage=False)
    
    # Correlation matrix for numeric columns
    log_and_print("Correlation analysis")

    # Select only numeric columns (hyperparams + RAM metrics)
    numeric_cols = combined_df.select_dtypes(include=['number']).columns
    corr_matrix = combined_df[numeric_cols].corr()

    # Show correlation with RAM usage
    ram_correlations = corr_matrix['ram_mb_used'].sort_values(ascending=False)
    log_and_print("Correlation with RAM usage (ram_mb_used):")
    log_and_print("-"*50)
    for feature, corr_value in ram_correlations.items():
        if feature != 'ram_mb_used':  # Skip self-correlation
            strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
            direction = "positive" if corr_value > 0 else "negative"
            log_and_print(f"  {feature:<25} {corr_value:>8.3f} ({strength} {direction})")

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
    log_and_print(f"RAM correlations saved to: {corr_output_path}")

    # Save full correlation matrix
    corr_matrix_path = output_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_matrix_path)
    log_and_print(f"Full correlation matrix saved to: {corr_matrix_path}")

    # Save run metadata
    metadata_path = output_dir / "run_metadata.txt"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"Analysis Run: {timestamp}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Source folder: {folder_path}\n")
        f.write(f"CSV files found:\n")
        for csv_file in csv_files:
            f.write(f"  - {csv_file}\n")
        f.write(f"\nTotal runs: {len(combined_df)}\n")
        f.write(f"Experiments combined: {len(all_dataframes)}\n\n")
        f.write(f"Runs per experiment:\n")
        for run_name in combined_df['source_run'].unique():
            count = len(combined_df[combined_df['source_run'] == run_name])
            f.write(f"  {run_name}: {count} runs\n")

    log_and_print(f"Run metadata saved to: {metadata_path}")

    log_and_print("Conclusions across runs:")
    log_and_print(f"Based on {len(combined_df)} combined training runs from {len(all_dataframes)} experiments,")
    log_and_print("we can identify which hyperparameters consistently affect RAM usage by checking")
    log_and_print("  - statistically significant hyperparameters (p < 0.05)")
    log_and_print("  - the amount the hyperparameters effect RAM (coefficients)")
    log_and_print("  - overall prediction accuracy (R²)")
    log_and_print("  - visual plots in: results/ram_analysis_combined/ folder")

    log_and_print(f"\nAll results saved in: {output_dir}")
    log_and_print(f"Batch analysis log saved to: {log_file}")


if __name__ == "__main__":
    main()