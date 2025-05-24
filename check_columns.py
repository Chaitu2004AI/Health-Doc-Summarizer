import pandas as pd

df = pd.read_csv("dataset.csv")
print("Actual column names:")
for col in df.columns:
    print(f">>> '{col}'")
