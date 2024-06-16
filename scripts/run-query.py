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
        default="all-MiniLM-L12-v2",
    )
    parser.add_argument("-q", "--query", default="none")
    
    args = parser.parse_args()

    if "semantic" in args.rank_profile:
        embeddings = load_model_embeddings(args.model)
    else:
        embeddings = None

    while True:
        if args.query != "none":
            test_mode = True
            print("[Using test mode...]")
            query = args.query
        else: 
            test_mode = False
            query  = prompt("Enter a query (q to quit): ")

        if query == "q":
            break

        songs = get_relevant_songs(query, args.rank_profile, hits=5, embeddings=embeddings)
            
        for song in songs: 
            print(song["fields"]["track_id"], song["fields"]["track_name"], song["relevance"])

        if not args.loop:
            break
