from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from utils.data import data_path
from utils.embeddings import load_model_embeddings
from utils.vespa import get_relevant_songs

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = 'all-MiniLM-L12-v2'
rank_profiles = ["track_name_semantic",
                 "lyrics_semantic",
                 "track_name_bm25",
                 "lyrics_bm25"]

print("== Pre-loading the models ==")

model_embeddings = load_model_embeddings(model)

print("== Models pre-loaded ==")


@app.get("/query/{rank_profile}/{k}")
def read_query(rank_profile: str, k: int, query: str):

    if rank_profile not in rank_profiles:
        return {"error": "Rank Profile not found"}

    if "semantic" in rank_profile:
        embeddings = model_embeddings
    else:
        embeddings = None

    songs = get_relevant_songs(query,
                               rank_profile,
                               hits=5,
                               embeddings=embeddings)
    
    return_list = []
    for i, song in enumerate(songs):
        return_list.append({
            "score": float(song["relevance"]),
            "track_id": song["fields"]["track_id"]
        })
    
    return return_list
