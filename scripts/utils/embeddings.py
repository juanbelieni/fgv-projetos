import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_model_embeddings(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model_name}",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False},
    )

    return embeddings
