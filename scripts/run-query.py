import pathlib as pl
import argparse as ap
from langchain_community.vectorstores import FAISS
from prompt_toolkit import prompt
from utils.embeddings import load_model_embeddings, model_list

DATA_PATH = pl.Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        choices=model_list,
        default="all-MiniLM-L6-v2",
    )

    parser.add_argument("-l", "--loop", action=ap.BooleanOptionalAction)

    args = parser.parse_args()
    model = args.model
    loop = args.loop

    index_path = DATA_PATH / f"{model}-index"

    if not index_path.exists():
        print("Index not found. Please run generate-index.py first.")
        exit(1)

    print("[Loading embeddings]")

    embeddings = load_model_embeddings(model)

    print("[Loading vector store index]")

    db = FAISS.load_local(index_path,
                          embeddings,
                          allow_dangerous_deserialization=True)

    while True:
        query = prompt("Enter a query (q to quit): ")

        if query == "q":
            break

        doc, score = db.similarity_search_with_score(query)[0]
        print("Score   ", score)
        print("Meta    ", doc.metadata)
        print("Content ", doc.page_content)

        if not loop:
            break
