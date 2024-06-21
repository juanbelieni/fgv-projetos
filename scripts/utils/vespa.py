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

    print(f"Connecting to Vespa Cloud at {endpoint}...")
    vespa_app = Vespa(url=endpoint,
                      application_package=vespa_app_package,
                      cert=cert_path,
                      key=key_path)


def normalize_scores(results):
    scores = [hit['relevance'] for hit in results['root']['children']]
    min_score = min(scores)
    max_score = max(scores)
    if min_score != max_score:
        for hit in results['root']['children']:
            hit['relevance'] = (hit['relevance'] - min_score) / (max_score - min_score)
    return results


def combine_scores(bm25_results, semantic_results):
    bm25_results = normalize_scores(bm25_results)
    semantic_results = normalize_scores(semantic_results)

    combined_results = {}
    for bm25_hit in bm25_results['root']['children']:
        doc_id = bm25_hit['fields']['track_id']
        doc_name = bm25_hit['fields']['track_name']
        bm25_score = bm25_hit['relevance']
        if doc_id not in combined_results:
            combined_results[doc_id] = {'track_id': doc_id, 'track_name': doc_name, 'bm25_score': bm25_score, 'semantic_score': 0}
        else:
            combined_results[doc_id]['bm25_score'] = bm25_score

    for semantic_hit in semantic_results['root']['children']:
        doc_id = semantic_hit['fields']['track_id']
        doc_name = semantic_hit['fields']['track_name']
        semantic_score = semantic_hit['relevance']
        if doc_id not in combined_results:
            combined_results[doc_id] = {'track_id': doc_id, 'track_name': doc_name, 'bm25_score': 0, 'semantic_score': semantic_score}
        else:
            combined_results[doc_id]['semantic_score'] = semantic_score

    for doc_id in combined_results:
        combined_results[doc_id]['combined_score'] = (combined_results[doc_id]['bm25_score'] + combined_results[doc_id]['semantic_score']) / 2

    combined_results_list = list(combined_results.values())
    combined_results_list.sort(key=lambda x: x['combined_score'], reverse=True)

    return combined_results_list



def get_relevant_songs(
        query: str,
        rank_profile: Literal[
            "track_name_semantic", "lyrics_semantic",
            "track_name_bm25", "lyrics_bm25", "hybrid"],
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

    elif "semantic" in rank_profile:
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
    
    else:  # Hybrid
        assert embeddings is not None

        # Track name semantic
        query_embedding = embeddings.embed_query(query)

        response_semantic = vespa_app.query(
            body={
                "yql": f"select * from sources * where ({{targetHits:{hits}}}nearestNeighbor(track_name_embedding, query_embedding))",
                "ranking.profile": "track_name_semantic",
                "input.query(query_embedding)": query_embedding,
            },
        )

        # Lyrics BM25
        response_bm25 = vespa_app.query(
            yql="select * from sources * where userQuery()",
            query=query,
            hits=hits,
            ranking="lyrics_bm25",
        )

        bm25_results = response_bm25.json
        semantic_results = response_semantic.json
        hybrid_results = combine_scores(bm25_results, semantic_results)

        return hybrid_results[:hits]

