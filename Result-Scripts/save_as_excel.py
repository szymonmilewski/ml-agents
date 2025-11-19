import pandas as pd
from pathlib import Path

#Load the CSV results
src = Path(r"C:\Users\Franz\OneDrive\Documents\University\University Material\ML_AI\ml-agents\results\walker-1\training_results_walker.csv")
df = pd.read_csv(src)

#Create the excel file
numeric_cols = [c for c in df.columns if c != "step"]
df = df.sort_values("step").reset_index(drop=True)
df[numeric_cols] = df[numeric_cols].round(6)

#Output the excel file
out = src.with_suffix(".xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    sheet = "Results"
    df.to_excel(xl, index=False, sheet_name=sheet)
    ws = xl.sheets[sheet]

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    from openpyxl.utils import get_column_letter
    for i, col in enumerate(df.columns, start=1):
        maxlen = max(len(str(col)), *(len(str(v)) for v in df[col].head(200)))
        if col == "step":
            ws.column_dimensions[get_column_letter(i)].width = maxlen + 5
        else:
            ws.column_dimensions[get_column_letter(i)].width = min(maxlen + 2, 40)

    for j, col in enumerate(df.columns, start=1):
        if col != "step":
            for cell in ws.iter_cols(min_col=j, max_col=j, min_row=2, max_row=ws.max_row):
                for c in cell: c.number_format = "0.000000"

print(f"Saved-> {out}")
