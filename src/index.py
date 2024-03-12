import pandas as pd
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

tokenizer = AutoTokenizer.from_pretrained("gpt2")
data = pd.read_csv("data/lyrics-mini.csv").dropna()

data = data.drop(columns=["Unnamed: 0"])

data["Tokens"] = [
    tokenizer(lyric, truncation=True).input_ids
    for lyric in data["Lyric"].to_list()
]

bm25 = BM25Okapi(data["Tokens"].to_list())

# Trecho de Chega de Saudade
query = "Não há beleza é só tristeza e a melancolia"
tokenized_query = tokenizer(query, truncation=True).input_ids

scores = bm25.get_scores(tokenized_query)
match = scores.argsort()[-1]
print(data.iloc[match])
