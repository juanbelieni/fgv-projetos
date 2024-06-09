import pandas as pd

# Not working!!!

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

for search_id in df["search_id"].unique()[:2]:
    for option in options:
        # Tittle-search mode
        cur_df_title = df_title[df_title["search_id"] == search_id][df_title["option"] == option]
        if search_id in cur_df_title["result_id"].to_list():
            results_dict["title"][option].append(
                cur_df_title[cur_df_title["result_id"] == search_id]["rank"].to_list()[0]
            )
        else:
            results_dict["title"][option].append(0)
        
        # Lyrics-search mode
        cur_df_lyrics = df_lyrics[df_lyrics["search_id"] == search_id][df_title["option"] == option]
        if search_id in cur_df_lyrics["result_id"].to_list():
            results_dict["lyrics"][option].append(
                cur_df_lyrics[cur_df_lyrics["result_id"] == search_id]["rank"].to_list()[0]
            )
        else:
            results_dict["lyrics"][option].append(0)


print(results_dict)