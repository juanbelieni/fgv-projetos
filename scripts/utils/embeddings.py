import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_model_embeddings(model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading embeddings from {model} using {device}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model}",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings
