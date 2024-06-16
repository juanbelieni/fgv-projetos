import argparse as ap
import pandas as pd
from langchain_community.vectorstores import FAISS
from utils.data import data_path, load_data_as_df
from tqdm import tqdm
from utils.embeddings import load_model_embeddings

def get_arguments():
    parser = ap.ArgumentParser()

    parser.add_argument("-f", "--frac", type=float, default=1)

    parser.add_argument("-k", type=int, default=5)

    args = parser.parse_args()
    # Define standard model
    model = "all-MiniLM-L12-v2"
    frac = args.frac
    k = args.k

    return model, frac, k


if __name__ == "__main__":
    model, frac, k = get_arguments()

    print("[Loading embeddings]")

    embeddings = load_model_embeddings(model)

    print("[Loading vector store index]")

    index_path = data_path / f"{model}-index"
    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True)

    print("[Loading data]")

    df = load_data_as_df()

    print("[Testing model]")

    sample = df.sample(frac=frac)
    results = pd.DataFrame(columns=["result", "reciprocal_rank"])

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        track_id = row["track_id"]
        track_artist = row["track_artist"]
        track_name = row["track_name"]

        query = f"{track_artist} - {track_name}"
        query_docs = db.similarity_search(query, k=k)

        result = "failure"
        reciprocal_rank = 0

        for i, doc in enumerate(query_docs):
            if row["track_id"] == doc.metadata["track_id"]:
                result = "success"
                reciprocal_rank = 1/(i+1)
                break

        results = pd.concat([results, pd.DataFrame([{
            "result": result,
            "reciprocal_rank": reciprocal_rank,
        }])], axis=0, ignore_index=True)

    precision = results['result'].value_counts(normalize=True)['success']
    mrr = results["reciprocal_rank"].mean()

    print(f"---\nRESULTS (model = {model}, frac={frac*100}%, k = {k}):\n")
    print(">> Precision:", precision)
    print(">> MRR:", mrr)