from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader


if __name__ == "__main__":
    #
    # Load embeddings from a model
    #

    print("[Loading embeddings]")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )

    #
    # Load the data as documents
    #

    print("[Loading data]")

    loader = CSVLoader(
        file_path="./data/spotify-songs.csv",
        metadata_columns=[
            "track_id",
            "track_popularity",
            "track_album_id",
            "track_album_release_date",
            "playlist_id",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "language",
        ])

    documents = loader.load()

    #
    # Generate vector store index and save it
    #

    print("[Generating vector store index]")

    db = FAISS.from_documents(documents, embeddings)

    print("[Saving vector store index]")

    db.save_local("data/index")
