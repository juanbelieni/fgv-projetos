from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from utils.data import data_path
from utils.embeddings import load_model_embeddings, model_list

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("== Pre-loading the models ==")

db_dict = {}
for model in model_list:
    embeddings = load_model_embeddings(model)
    db = FAISS.load_local(data_path / f"{model}-index",
                          embeddings,
                          allow_dangerous_deserialization=True)
    db_dict[model] = db
    del embeddings, db

print("== Models pre-loaded ==")


@app.get("/query/{model}/{k}")
def read_query(model: str, k: int, query: str):

    if model not in db_dict.keys():
        return {"error": "Model not found"}

    db = db_dict[model]

    results = db.similarity_search_with_score(query, k=k)
    
    return_list = []
    for i, result in enumerate(results):
        return_list.append({
            "score": float(result[1]),
            "metadata": result[0].metadata,
            "content": result[0].page_content
        })
    
    return return_list
