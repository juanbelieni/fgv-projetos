import pathlib as pl
import argparse as ap
from prompt_toolkit import prompt
from utils.embeddings import load_model_embeddings, model_list
from utils.vespa import get_relevant_songs

DATA_PATH = pl.Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-l", "--loop", action=ap.BooleanOptionalAction)
    parser.add_argument("-r", "--rank-profile", required=True, choices=[
        "track_name_semantic", "lyrics_semantic",
        "track_name_bm25", "lyrics_bm25"])
    parser.add_argument(
        "-m",
        "--model",
        choices=model_list,
        default="all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    if "semantic" in args.rank_profile:
        embeddings = load_model_embeddings(args.model)
    else:
        embeddings = None

    while True:
        query = prompt("Enter a query (q to quit): ")

        if query == "q":
            break

        get_relevant_songs(query, args.rank_profile, hits=5, embeddings=embeddings)

        if not args.loop:
            break
