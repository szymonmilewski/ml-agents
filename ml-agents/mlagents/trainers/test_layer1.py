from pathlib import Path
from automate import csv_to_dict_list, find_step_at_benchmark

csv_path = Path("/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/mlagents/trainers/prototype.csv")
rows = csv_to_dict_list(str(csv_path))

step = find_step_at_benchmark(rows, benchmark_reward=0.85)
print("Step at benchmark:", step)

assert step == 500000, f"Expected 500000, got {step}"
print("✅ Layer 1 passed")
