# read_results.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import pandas as pd
from functools import reduce

RUN_ID = "walker-1"
BEHAVIOR = "Walker"

logdir = Path("../results") / RUN_ID / BEHAVIOR
events = list(logdir.glob("events.out.tfevents.*"))
if not events:
    raise FileNotFoundError(f"No event files under: {logdir}")

ea = EventAccumulator(str(events[0])); ea.Reload()
scalar_tags = ea.Tags().get("scalars", [])
if not scalar_tags:
    raise RuntimeError("No scalar tags found in the event file.")

preferred = [
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "Losses/Policy Loss",
    "Losses/Value Loss",
    "Policy/Entropy",
    "Policy/Learning Rate",
    "Policy/Epsilon",
]
chosen = [t for t in preferred if t in scalar_tags] or \
         [t for t in scalar_tags if ("reward" in t.lower() or "episode" in t.lower())]

frames = []
for tag in chosen:
    vals = ea.Scalars(tag)
    frames.append(pd.DataFrame({"Step": [e.step for e in vals], tag: [e.value for e in vals]}))

out = reduce(lambda a,b: pd.merge(a,b,on="Step",how="outer"), frames).sort_values("Step")
out.to_csv("training_results_walker.csv", index=False)
print(out.head(12))
print("\n✅ Saved -> training_results_walker.csv")
print("Included tags:"); [print(" -", t) for t in chosen]

#Plot Creation
import matplotlib.pyplot as plt

#Smoother Averages
out["Reward_sMA"] = out["Environment/Cumulative Reward"].rolling(10, min_periods=1).mean()
out["EpLen_sMA"]  = out["Environment/Episode Length"].rolling(10, min_periods=1).mean()

#Plot Reward
plt.figure()
out.plot(x="Step", y=["Environment/Cumulative Reward", "Reward_sMA"])
plt.title("Walker — Cumulative Reward")

#Plot Episode Length
plt.figure()
out.plot(x="Step", y=["Environment/Episode Length", "EpLen_sMA"])
plt.title("Walker — Episode Length")

plt.show()

