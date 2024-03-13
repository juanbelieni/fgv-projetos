import pathlib
import pandas as pd
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

path = pathlib.Path(__file__).parent.parent
data_path = path / "data/lyrics-mini.csv"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
data = pd.read_csv(data_path).dropna()

data = data.drop(columns=["Unnamed: 0"])

data["Tokens"] = [
    tokenizer(lyric, truncation=True).input_ids
    for lyric in data["Lyric"].to_list()
]

bm25 = BM25Okapi(data["Tokens"].to_list())

def main():
    while True:
        query = input("Digite a letra da música: ")
        tokenized_query = tokenizer(query, truncation=True).input_ids

        scores = bm25.get_scores(tokenized_query)
        match = scores.argsort()[-1]
        print(data.iloc[match])
        
        if input("Deseja continuar? (s/n): ") == "n":
            break

if __name__ == "__main__":
    main()

# # Trecho de Chega de Saudade
# query = "Não há beleza é só tristeza e a melancolia"
# tokenized_query = tokenizer(query, truncation=True).input_ids

# scores = bm25.get_scores(tokenized_query)
# match = scores.argsort()[-1]
# print(data.iloc[match])
