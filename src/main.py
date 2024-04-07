from vespa.io import VespaResponse, VespaQueryResponse
import pandas as pd
from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage, Field, Schema, Document, RankProfile, HNSW, RankProfile, Component, Parameter, FieldSet, GlobalPhaseRanking, Function

print("[Starting script]")

package = ApplicationPackage(
    name="hybridsearch",
    schema=[Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="artist_link", type="string",
                          indexing=["summary"]),
                    Field(name="song_link", type="string",
                          indexing=["summary"]),
                    Field(name="song_name", type="string", indexing=[
                          "index", "summary"], index="enable-bm25"),
                    Field(name="lyrics", type="string", indexing=[
                          "index", "summary"], index="enable-bm25", bolding=True),
                    Field(name="embedding", type="tensor<float>(x[384])",
                          indexing=["input song_name . \" \" . input lyrics",
                                    "embed", "index", "attribute"],
                          ann=HNSW(distance_metric="angular"),
                          is_document_field=False)
                ]
            ),
            fieldsets=[
                FieldSet(name="default", fields=["song_name", "lyrics"])
            ],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[Function(
                        name="bm25sum", expression="bm25(song_name) + bm25(lyrics)"
                    )],
                    first_phase="bm25sum"
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)"
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                        rerank_count=1000
                    )
                )
            ]
            )
            ],
    components=[Component(
        id="e5",
        type="hugging-face-embedder",
        parameters=[
            Parameter("transformer-model", {
                "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"}),
            Parameter(
                "tokenizer-model", {"url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"})
        ])]
)

print("[`package` defined]")

vespa_docker = VespaDocker()

print("[`vespa_docker` defined]")

app = vespa_docker.deploy(application_package=package)

print("[App deployed]")

dataset = pd.read_csv("lyrics.csv")

print("[Dataset imported]")

vespa_feed = [{
    "id": data["song_link"],
    "fields": {
        "artist_link": data["artist_link"],
        "song_link": data["song_link"],
        "song_name": data["song_name"],
        "lyrics": data["lyrics"],
    }}
    for _, data in dataset.iterrows()]


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(f"Error when feeding document {id}: {response.get_json()}")


app.feed_iterable(vespa_feed, schema="doc",
                  namespace="tutorial", callback=callback)

print("[`app.feed_iterable` called]")
