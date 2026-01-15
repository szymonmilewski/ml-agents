import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

FILE_PATH = Path(__file__).resolve() 
REPO = FILE_PATH.parents[3]

def extract_metrics(run_id: str, output_csv: str = "prototype.csv", summary_freq: int = 2000):
    logdir = REPO / "results" / run_id
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    if "Environment/Cumulative Reward" not in ea.Tags()["scalars"]:
        raise ValueError("Missing Environment/Cumulative Reward")

    events = ea.Scalars("Environment/Cumulative Reward")

    df = pd.DataFrame(
        [(e.step, e.value) for e in events],
        columns=["step", "episode_reward"],
    )

    df["window"] = df["step"] // summary_freq

    summary = (
        df.groupby("window")
        .agg(
            step=("step", "max"),
            mean_reward=("episode_reward", "mean"),
            std_reward=("episode_reward", "std"),
        )
        .reset_index(drop=True)
    )

    summary["mean_reward"] = summary["mean_reward"].round(3)
    summary["std_reward"] = summary["std_reward"].round(3)

    out_path = os.path.join(logdir, output_csv)
    summary.to_csv(out_path, index=False)

    print(f"✓ CSV saved: {out_path}")
    return out_path


if __name__ == "_main_":
    import sys


    run_id = sys.argv[1] if len(sys.argv) > 1 else "smoke-3dball-2" #run id from folder


    output_csv = sys.argv[2] if len(sys.argv) > 2 else "prototype.csv"

    print(f"Extracting metrics from run: {run_id}")
    print(f"Output file: {output_csv}\n")

    extract_metrics(run_id, output_csv)
