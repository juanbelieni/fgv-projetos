import argparse as ap
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.data import data_path, load_data_as_df
from tqdm import tqdm


def get_arguments():
    parser = ap.ArgumentParser()

    parser.add_argument(
        "-m", "--model",
        choices=["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
        default="all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "-f", "--frac", type=float, default=1,
    )

    parser.add_argument(
        "-k", type=int, default=5,
    )

    args = parser.parse_args()

    model_name = args.model
    frac = args.frac
    k = args.k

    return model_name, frac, k


def get_model_embeddings(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model_name}",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False},
    )

    return embeddings


if __name__ == "__main__":
    model_name, f, k = get_arguments()

    print("[Loading embeddings]")

    embeddings = get_model_embeddings(model_name)

    print("[Loading vector store index]")

    index_path = data_path / f"{model_name}-index"
    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True)

    print("[Loading data]")

    df = load_data_as_df()

    print("[Testing model]")

    sample = df.sample(frac=f)
    results = pd.DataFrame(
        columns=["track_id", "track_artist", "track_name", "result"])

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        track_id = row["track_id"]
        track_artist = row["track_artist"]
        track_name = row["track_name"]

        query = f"{track_artist} - {track_name}"
        query_docs = db.similarity_search(query, k=k)

        if row["track_id"] in [doc.metadata["track_id"] for doc in query_docs]:
            result = "success"
        else:
            result = "failure"

        results = pd.concat([results, pd.DataFrame([{
            "track_id": track_id,
            "track_artist": track_artist,
            "track_name": track_name,
            "result": result,
        }])], axis=0, ignore_index=True)

    print(results["result"].value_counts(normalize=True))
