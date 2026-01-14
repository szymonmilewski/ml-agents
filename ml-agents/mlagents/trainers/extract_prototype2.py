import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics(run_id: str, output_csv: str = "prototype.csv") -> None:
    logdir = os.path.join("/Users/Sebastian/PycharmProjects/ml-agents/results", run_id)
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    print(f"Found event file: {event_files[0]}")
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    available_keys = ea.Tags()["scalars"]
    print(f"Available scalar keys: {available_keys}")

    metric_mapping = {
        "Environment/Cumulative Reward": "cumulative_reward",
    }

    # Initialize data dictionary with step and wall_time
    data = {"step": [], "wall_time": []}

    for csv_name in metric_mapping.values():
        data[csv_name] = []

    primary_key = "Environment/Cumulative Reward"
    if primary_key in available_keys:
        events = ea.Scalars(primary_key)
        for event in events:
            data["step"].append(event.step)
            data["wall_time"].append(event.wall_time)
            data[metric_mapping[primary_key]].append(event.value)
    else:
        raise ValueError(f"Primary metric '{primary_key}' not found in TensorBoard logs")

    num_steps = len(data["step"])
    print(f"Total training steps found: {num_steps}")

    # Extract all other metrics
    for tb_key, csv_name in metric_mapping.items():
        if tb_key == primary_key:
            continue

        if tb_key in available_keys:
            events = ea.Scalars(tb_key)
            values = [event.value for event in events]

            if len(values) < num_steps:
                print(f"Warning: {tb_key} has {len(values)} entries, padding to {num_steps}")
                values.extend([None] * (num_steps - len(values)))
            elif len(values) > num_steps:
                print(f"Warning: {tb_key} has {len(values)} entries, truncating to {num_steps}")
                values = values[:num_steps]

            data[csv_name] = values
        else:
            print(f"Warning: {tb_key} not found in logs, filling with None")
            data[csv_name] = [None] * num_steps

    df = pd.DataFrame(data)

    df["time_elapsed"] = df["wall_time"] - df["wall_time"].iloc[0]

    df = df.drop("wall_time", axis=1)

    df["reward_rolling_mean"] = df["cumulative_reward"].rolling(window=10, min_periods=1).mean()

    column_order = [
        "step",
        "cumulative_reward",
    ]

    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    out_path = os.path.join(logdir, output_csv)
    df.to_csv(out_path, index=False)

    print(f"\n✓ CSV saved: {out_path}")
    print(f"✓ Total rows: {len(df)}")
    print(f"✓ Total columns: {len(df.columns)}")
    print(f"\nColumn summary:")
    print(df.describe())
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    import sys


    run_id = sys.argv[1] if len(sys.argv) > 1 else "smoke-3dball-2" #run id from folder


    output_csv = sys.argv[2] if len(sys.argv) > 2 else "prototype.csv"

    print(f"Extracting metrics from run: {run_id}")
    print(f"Output file: {output_csv}\n")

    extract_metrics(run_id, output_csv)
