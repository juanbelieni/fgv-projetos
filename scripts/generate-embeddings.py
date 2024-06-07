import argparse as ap

from utils.embeddings import load_model_embeddings, model_list
from utils.data import load_data_as_df
from utils.vespa import vespa_app
from vespa.io import VespaResponse

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=model_list,
        default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    print("[Loading embeddings]")

    embeddings = load_model_embeddings(args.model)

    print("[Loading data from CSV]")

    df = load_data_as_df()[:1000]

    print("[Generating embeddings]")

    df["track_name_embedding"] = embeddings.embed_documents(df["track_name"].tolist())
    df["lyrics_embedding"] = embeddings.embed_documents(df["lyrics"].tolist())

    iter_data = [
        dict(
            id=row["track_id"],
            fields=dict(
                track_id=row["track_id"],
                track_name=row["track_name"],
                lyrics=row["lyrics"],
                track_name_embedding=row["track_name_embedding"],
                lyrics_embedding=row["lyrics_embedding"]))
        for row in df.to_dict("records")]

    print("[Feeding data to Vespa]")

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(f"Error when feeding document {id}: {response.get_json()}")

    vespa_app.feed_iterable(
        iter=iter_data,
        callback=callback,
    )
