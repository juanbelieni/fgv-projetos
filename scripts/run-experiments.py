from tqdm import tqdm
from utils.data import load_data_as_df, load_queries_as_df, data_path
from utils.vespa import get_relevant_songs
from utils.embeddings import load_model_embeddings
import pandas as pd
from itertools import product

num_songs = 5
model = "all-MiniLM-L12-v2"

rank_profiles = [
    "track_name_semantic", "lyrics_semantic",
    "track_name_bm25", "lyrics_bm25"
]

query_types = [
    "title",
    "llm"
]

ks = [2, 5, 10]


if __name__ == "__main__":
    data_df = load_data_as_df()
    queries_df = load_queries_as_df()

    df = pd.merge(data_df, queries_df, on="track_id")
    df = df.sample(n=num_songs)

    embeddings = load_model_embeddings(model)
    results = []

    for rank_profile, query_type in product(rank_profiles, query_types):
        print(f"Running experiment {rank_profile=}, {query_type=}")

        experiment_results = {
            k: dict(precision=0, mrr=0)
            for k in ks
        }

        for _, row in tqdm(df.iterrows(), total=num_songs):
            track_id = row["track_id"]

            match query_type:
                case "title": query = row["track_name"]
                case "llm":   query = row["query"]

            songs = get_relevant_songs(
                query=query,
                rank_profile=rank_profile,
                hits=10,
                embeddings=embeddings)

            track_ids = [s["fields"]["track_id"] for s in songs]

            for k in ks:
                if track_id in track_ids[:k]:
                    rank = track_ids[:k].index(track_id) + 1
                    experiment_results[k]["precision"] += 1
                    experiment_results[k]["mrr"] += 1 / rank

        for k in ks:
            experiment_results[k]["precision"] /= num_songs
            experiment_results[k]["mrr"] /= num_songs

            results.append(dict(
                rank_profile=rank_profile,
                query_type=query_type,
                k=k,
                precision=experiment_results[k]["precision"],
                mrr=experiment_results[k]["mrr"]
            ))

            print(
                f"Result for {k=}: "
                f"precision={experiment_results[k]['precision']:.4f} "
                f"mrr={experiment_results[k]['mrr']:.4f}")

    results_df = pd.DataFrame(results)
    print(results_df.to_csv(index=False))
