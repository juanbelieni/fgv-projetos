import os
from utils.data import load_data_as_df


open("./scripts/results.txt", "w").close()

df = load_data_as_df()[:500]

options = [
    "track_name_semantic", "lyrics_semantic",
    "track_name_bm25", "lyrics_bm25"
    ]

# Testing exact track name
for idx, name in enumerate(df["track_name"]):
    print(idx, name)
    os.system(f"python3 ./scripts/run-query.py -r {options[0]} -q {name}")
    

print(df["track_id"][0])
print(df.head())
