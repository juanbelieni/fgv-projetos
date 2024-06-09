from utils.data import load_data_as_df
import random
import re

df = load_data_as_df()[:500]

print(df["lyrics"].shape)

# Header
with open("data/lyrics_queries.csv", "w") as f:
    f.write("track_id;track_name;query\n")
    f.close()

file = open("data/lyrics_queries.csv", "a")

for track_id in df["track_id"]:
    lyrics = df[df["track_id"] == track_id]["lyrics"].to_list()[0]
    tokenized_lyrics = [re.sub(r'\W+', '', token.lower()) for token in lyrics.split(" ")]
    size = random.randint(
        min(5, len(tokenized_lyrics)),
        min(15, len(tokenized_lyrics))
        )
    begin = random.randint(0, len(tokenized_lyrics)-size)
    query = " ".join(tokenized_lyrics[begin:begin+size])
    track_name = df[df["track_id"] == track_id]["track_name"].to_list()[0]

    print(f'Song name: {track_name}\nQuery: {query.lower()}\n')
    file.write(f'{track_id};\"{track_name}\";\"{query}\"\n')

file.close()