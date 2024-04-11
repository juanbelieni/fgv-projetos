import pathlib as pl
import argparse as ap
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = pl.Path(__file__).parent.parent / "data"

if __name__ == "__main__":

    #
    # Instantiate parser
    #

    parser = ap.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        choices=["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
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

    #
    # Load embeddings from a model
    #

    print("[Loading embeddings]")

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model}",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )

    print("Loading vector store index]")

    db = FAISS.load_local(index_path,
                          embeddings,
                          allow_dangerous_deserialization=True)

    print("[Vector store index loaded]")

    query = input("Enter a query: ")

    doc, score = db.similarity_search_with_score(query)[0]
    print("Score ", score)
    print("Meta ", doc.metadata)
    print("Content ", doc.page_content)

    if loop:
        while True:
            query = input("Enter a query (q to quit): ")

            if query == "q":
                break

            doc, score = db.similarity_search_with_score(query)[0]
            print("Score ", score)
            print("Meta ", doc.metadata)
            print("Content ", doc.page_content)
    else:
        print("Exiting")
