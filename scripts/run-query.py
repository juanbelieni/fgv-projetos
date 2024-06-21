import pathlib as pl
import argparse as ap
from prompt_toolkit import prompt
from utils.embeddings import load_model_embeddings
from utils.vespa import get_relevant_songs

DATA_PATH = pl.Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-l", "--loop", action=ap.BooleanOptionalAction)
    parser.add_argument("-r", "--rank-profile", required=True, choices=[
        "track_name_semantic", "lyrics_semantic",
        "track_name_bm25", "lyrics_bm25", "hybrid"])
    # parser.add_argument("-q", "--query", default="none")
    
    args = parser.parse_args()

    # Define args.model directly
    args.model = "all-MiniLM-L12-v2"

    if "semantic" or "hybrid" in args.rank_profile:
        embeddings = load_model_embeddings(args.model)
    else:
        embeddings = None

    while True:
        query  = prompt("Enter a query (q to quit): ")

        if query == "q":
            break

        songs = get_relevant_songs(query, args.rank_profile, hits=5, embeddings=embeddings)
            
        if args.rank_profile == "hybrid":
            for song in songs:
                print(song['track_id'], song['track_name'], "\n   BM25 score:",song['bm25_score'])
                print("   Semantic score:", song['semantic_score'],"\n   Combined score:",song['combined_score'],"\n\n")
        else:
            for song in songs: 
                print(song["fields"]["track_id"], song["fields"]["track_name"], song["relevance"])

        if not args.loop:
            break