import os
import pandas as pd
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Literal

from vespa.package import (
    ApplicationPackage,
    Field,
    RankProfile,
    FieldSet
)
from vespa.application import Vespa

CLOUD = True

vespa_app_package = ApplicationPackage(name="crazyfrogger")

vespa_app_package.schema.add_fields(
    Field(
        name="track_id",
        type="string",
        indexing=["attribute", "summary"]
    ),
    Field(
        name="track_name",
        type="string",
        indexing=["index", "summary"],
        index="enable-bm25",
    ),
    Field(
        name="lyrics",
        type="string",
        indexing=["index", "summary"],
        index="enable-bm25"
    ),
    Field(
        name="track_name_embedding",
        type="tensor<float>(x[384])",
        indexing=["attribute", "summary"],
        attribute=["distance-metric: angular"],
    ),
    Field(
        name="lyrics_embedding",
        type="tensor<float>(x[384])",
        indexing=["attribute", "summary"],
        attribute=["distance-metric: angular"],
    ),
)

vespa_app_package.schema.add_field_set(
    FieldSet(name="default", fields=["track_name", "lyrics"])
)

vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="track_name_semantic",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
        first_phase="closeness(field, track_name_embedding)",
    )
)

vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="lyrics_semantic",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
        first_phase="closeness(field, lyrics_embedding)",
    )
)

vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="track_name_bm25",
        first_phase="bm25(track_name)",
    ),
)


vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="lyrics_bm25",
        first_phase="bm25(lyrics)",
    ),
)

if not CLOUD:
    vespa_app = Vespa(
        url="http://localhost",
        port="8080",
        application_package=vespa_app_package)
else:
    endpoint = "https://afa26c4b.c65fc8bc.z.vespa-app.cloud/"
    
    os.environ["TENANT_NAME"] = "crazyfrogger"  # Replace with your tenant name
    application = "crazyfrogger"
    
    cert_path = (
        Path.home()
        / ".vespa"
        / f"{os.environ['TENANT_NAME']}.{application}.default/data-plane-public-cert.pem"
    )
    key_path = (
        Path.home()
        / ".vespa"
        / f"{os.environ['TENANT_NAME']}.{application}.default/data-plane-private-key.pem"
    )
    
    print(f"Connecting to Vespa Cloud at {endpoint}")
    vespa_app = Vespa(url=endpoint,
                      application_package=vespa_app_package,
                      cert=cert_path,
                      key=key_path)


def get_relevant_songs(
        query: str,
        rank_profile: Literal[
            "track_name_semantic", "lyrics_semantic",
            "track_name_bm25", "lyrics_bm25"],
        hits: int,
        embeddings: HuggingFaceEmbeddings = None) -> pd.DataFrame:

    if "bm25" in rank_profile:
        response = vespa_app.query(
            yql="select * from sources * where userQuery()",
            hits=hits,
            query=query,
            ranking=rank_profile,
        )

        return response.hits

    else:
        assert embeddings is not None

        query_embedding = embeddings.embed_query(query)

        response = vespa_app.query(
            body={
                "yql": f"select * from sources * where ({{targetHits:{hits}}}nearestNeighbor(track_name_embedding, query_embedding))",
                "ranking.profile": rank_profile,
                "input.query(query_embedding)": query_embedding,
            },
        )

        return response.hits

