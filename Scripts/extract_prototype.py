import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics(run_id: str, output_csv: str = "prototype.csv") -> None:
    logdir = os.path.join("/Users/Sebastian/Documents/ml-agents/results", run_id)
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    print(f"Found event file: {event_files[0]}")
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    print("Available scalar keys:", ea.Tags()["scalars"])

    # Extract metrics
    data = {"step": [], "mean_reward": [], "episode_length": [], "wall_time": []}

    for event in ea.Scalars("Environment/Cumulative Reward"):
        data["step"].append(event.step)
        data["mean_reward"].append(event.value)
        data["wall_time"].append(event.wall_time)

    for event in ea.Scalars("Environment/Episode Length"):
        data["episode_length"].append(event.value)

    if len(data["episode_length"]) != len(data["step"]):
        print(f"Length mismatch: rewards={len(data['step'])}, episodes={len(data['episode_length'])}")
        min_len = min(len(data["step"]), len(data["episode_length"]))
        for key in data:
            data[key] = data[key][:min_len]

    df = pd.DataFrame(data)

    df["time_elapsed"] = df["wall_time"] - df["wall_time"].iloc[0]
    df = df.drop("wall_time", axis=1)

    df["reward_rolling_std"] = df["mean_reward"].rolling(5, min_periods=1).std()

    out_path = os.path.join(logdir, output_csv)
    df.to_csv(out_path, index=False)
    print(f"✓ CSV saved: {out_path}")

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "smoke-3dball-2"
    extract_metrics(run_id)
