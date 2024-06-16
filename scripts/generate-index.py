import argparse as ap
from langchain_community.vectorstores import FAISS
from utils.data import data_path, load_data_as_documents
from utils.embeddings import load_model_embeddings, model_list

if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        choices=model_list,
        default="all-MiniLM-L12-v2",
    )

    args = parser.parse_args()
    model = args.model

    print("[Loading embeddings]")

    embeddings = load_model_embeddings(model)

    print("[Loading data]")

    documents = load_data_as_documents()

    print("[Generating vector store index]")

    db = FAISS.from_documents(documents, embeddings)

    print("[Saving vector store index]")

    db.save_local(data_path / f"{args.model}-index")
