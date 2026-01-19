import pandas as pd

files = [
    "/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/config/multi/BasicPP0Epsilon.3.SS/BasicPP0Epsilon.3.SS.csv",
    "/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/config/multi/BasicPP0Epsilon.4.SS/BasicPP0Epsilon.4.SS.csv",
    "/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/config/multi/BasicPP0Epsilon.5.SS/BasicPP0Epsilon.5.SS.csv",
    "/Users/Sebastian/PycharmProjects/ml-agents/ml-agents/config/multi/BasicPP0Epsilon.6.SS/BasicPP0Epsilon.6.SS.csv",
]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

df = df.drop_duplicates(
    subset=["batch_size", "epsilon", "num_epoch"]
)

df.to_csv("BasicPP0Epsilon.7.SS.csv", index=False)
