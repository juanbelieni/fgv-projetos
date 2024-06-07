import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

model_list = [
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
]


def load_model_embeddings(model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model}",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings
