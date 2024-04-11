from fastapi import FastAPI
import pathlib as pl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = pl.Path(__file__).parent.parent / "data"

app = FastAPI()


@app.get("/query/{model}/{query}")
def read_query(model: str, query: str):
    index_path = DATA_PATH / f"{model}-index"

    if not index_path.exists():
        return {"message": "Index not found. Please run generate-index.py first."}

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model}",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )

    db = FAISS.load_local(index_path,
                          embeddings,
                          allow_dangerous_deserialization=True)

    doc, score = db.similarity_search_with_score(query)[0]
    
    return {"score": float(score), "metadata": doc.metadata}
