
import pathlib as pl
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader

data_path = pl.Path(__file__).parent.parent.parent / "data"


def load_data_as_documents():
    loader = CSVLoader(
        file_path=data_path / "spotify-songs.csv",
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
        ],
    )

    return loader.load()


def load_data_as_df() -> pd.DataFrame:
    df = pd.read_csv(data_path / "spotify-songs.csv")
    df = df[df["lyrics"].notnull()]
    return df

