from tqdm import tqdm
from utils.data import load_data_as_df
from utils.vespa import get_relevant_songs
from utils.embeddings import load_model_embeddings
import pandas as pd
import re

number_of_songs = 500

# Results header:
with open("./data/results.csv", "w") as f:
    f.write("search_id;query;rank;result_id;track_name;relevance;option;search_mode\n")
    f.close()

# Load datasets:
df = load_data_as_df()[:500]
df_queries = pd.read_csv("data/lyrics_queries.csv", sep=";")

model = "all-MiniLM-L6-v2"

options = [
    "track_name_semantic", "lyrics_semantic",
    "track_name_bm25", "lyrics_bm25"
    ]

for option in options:
    print(f'[{option}]')
    if "semantic" in option:
        embeddings = load_model_embeddings(model)
    else:
        embeddings = None
    
    for _, track_id in tqdm(enumerate(df["track_id"][:number_of_songs])):
        title = df[df["track_id"] == track_id]["track_name"].to_list()[0]

        # Prepare queries:
        query_title = " ".join([re.sub(r'\W+', '', token) for token in title.lower().split(" ")])
        query_lyrics = df_queries[df_queries["track_id"] == track_id]["query"].to_list()[0]

        # Search the exact track title:
        songs = get_relevant_songs(query_title, option, hits=5, embeddings=embeddings)
        # Register results:
        with open("./data/results.csv", "a") as f:
            for i, song in enumerate(songs):
                # print(song)
                f.write(f'{track_id};\"{query_title}\";{i+1};{song["fields"]["track_id"]};')
                f.write(f'\"{song["fields"]["track_name"]}\";{song["relevance"]};')
                f.write(f'{option};title\n')

        # Search a random part of the lyrics:
        songs = get_relevant_songs(query_lyrics, option, hits=5, embeddings=embeddings)
        # Register results:
        with open("./data/results.csv", "a") as f:
            for i, song in enumerate(songs):
                f.write(f'{track_id};\"{query_lyrics}\";{i+1};{song["fields"]["track_id"]};')
                f.write(f'\"{song["fields"]["track_name"]}\";{song["relevance"]};')
                f.write(f'{option};lyrics\n')
    print("\n---")
