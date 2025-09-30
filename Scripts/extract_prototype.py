import os
import glob
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_metrics(run_id: str, output_csv: str = "prototype.csv") -> None:
    """
    Extract training metrics from a Unity ML-Agents run and save them to a CSV file.

    Args:
        run_id (str): The name of the run folder inside `results/`.
                      Example: "smoke-3dball-2" will look in `results/smoke-3dball-2/`.
        output_csv (str): Name of the output CSV file to save.
                          It will be saved inside the run folder.

    Returns:
        None. A CSV file is written to disk.
    """

    # Path to the results folder for this run
    logdir = os.path.join("results", run_id)

    # Find TensorBoard event files (.tfevents.*). --> this file contains all the training logs
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {logdir}")

    print(f"Found event file: {event_files[0]}")

    # Load the event file with TensorBoard's EventAccumulator
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    print("Available scalar keys:", ea.Tags()["scalars"]) #print the available for debugging

    # Storage for the metrics we care about
    steps, mean_rewards, episode_lengths = [], [], []

    # Extract cumulative rewards across steps
    for event in ea.Scalars("Environment/Cumulative Reward"):
        steps.append(event.step)        # training step number
        mean_rewards.append(event.value)  # mean reward at that step

    # Extract episode lengths across steps
    for event in ea.Scalars("Environment/Episode Length"):
        episode_lengths.append(event.value)

    # Build a DataFrame with the metrics
    df = pd.DataFrame({
        "step": steps,
        "mean_reward": mean_rewards,
        "episode_length": episode_lengths[:len(steps)]  # align length
    })

    # Add rolling standard deviation of rewards (stability measure)
    df["std_reward"] = df["mean_reward"].rolling(5).std().fillna(0)

    # Add a simple "time elapsed" index (not real time, just step index)
    df["time_elapsed"] = range(len(df))

    # Save the DataFrame to CSV in the run folder
    out_path = os.path.join(logdir, output_csv)
    df.to_csv(out_path, index=False)
    print(f"✅ Prototype CSV saved at {out_path}")


if __name__ == "__main__":
    import sys
    # Allow run_id to be passed as a command-line argument
    run_id = sys.argv[1] if len(sys.argv) > 1 else "smoke-3dball-2"
    extract_metrics(run_id)
