from vespa.package import (
    ApplicationPackage,
    Field,
    RankProfile,
    Function
)

vespa_app_package = ApplicationPackage(name="crazyfrogger")

vespa_app_package.schema.add_fields(
    Field(
        name="track_id",
        type="string",
        indexing=["summary"]
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
        first_phase="closeness(field, track_name_embedding)",
    )
)

vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="track_name_bm25",
        inputs=[("query(query)", "tensor<float>(x[384])")],
        functions=[Function(name="bm25sum", expression="bm25(track_name)")],
        first_phase="bm25sum",
    ),
)


vespa_app_package.schema.add_rank_profile(
    RankProfile(
        name="lyrics_bm25",
        inputs=[("query(query)", "tensor<float>(x[384])")],
        functions=[Function(name="bm25sum", expression="bm25(lyrics)")],
        first_phase="bm25sum",
    ),
)
