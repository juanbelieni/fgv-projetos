import pandas as pd

df = pd.read_csv("./data/results.csv", sep=";")

df_title = df[df["search_mode"] == "title"]
df_lyrics = df[df["search_mode"] == "lyrics"]

options = [
    "track_name_semantic", "lyrics_semantic",
    "track_name_bm25", "lyrics_bm25"
    ]

results_dict = {
    "title": {
        "track_name_semantic": [],
        "lyrics_semantic": [],
        "track_name_bm25": [], 
        "lyrics_bm25": []
    },
    "lyrics": {
        "track_name_semantic": [],
        "lyrics_semantic": [],
        "track_name_bm25": [], 
        "lyrics_bm25": []
    }
}

# Title-search mode
for search_id in df["search_id"].unique():
    for option in options:
        cur_df_title = df_title[df_title["search_id"] == search_id][df_title["option"] == option]
        if search_id in cur_df_title["result_id"].to_list():
            results_dict["title"][option].append(
                cur_df_title[cur_df_title["result_id"] == search_id]["rank"].to_list()[0]
            )
        else:
            results_dict["title"][option].append(0)

# Lyrics-search mode
for search_id in df["search_id"].unique():
    for option in options:
        cur_df_lyrics = df_lyrics[df_lyrics["search_id"] == search_id][df_lyrics["option"] == option]
        if search_id in cur_df_lyrics["result_id"].to_list():
            results_dict["lyrics"][option].append(
                cur_df_lyrics[cur_df_lyrics["result_id"] == search_id]["rank"].to_list()[0]
            )
        else:
            results_dict["lyrics"][option].append(0)

print("\n---\n  RESULTS:\n")
for mode in results_dict.keys():
    for option in results_dict[mode].keys():
        MRR = 0
        for i in results_dict[mode][option]:
            if i != 0:
                MRR += 1/i
        MRR = MRR/len(results_dict[mode][option])
        print(f'Mode: {mode, option}\nMRR = {MRR}\n')
