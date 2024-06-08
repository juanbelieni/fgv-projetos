from vespa.package import ApplicationPackage, Field, RankProfile
from vespa.deployment import VespaDocker
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.vespa import VespaStore
import numpy as np

# Create application package
app_package = ApplicationPackage(name="crazyfrogger")

# Add fields to schema
app_package.schema.add_fields(
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
        name="embedding",
        type="tensor<float>(x[384])",
        indexing=["attribute", "summary"],
        attribute=["distance-metric: angular"],
    ),
)

# Add rank profile
app_package.schema.add_rank_profile(
    RankProfile(
        name="default",
        first_phase="closeness(field, embedding)",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
    )
)

# Save configuration files
app_package.to_files("config/vespa")

# Deploy Vespa application using Docker
vespa_docker = VespaDocker()
vespa_docker.deploy_from_disk(application_package="config/vespa")

# Load documents
documents = CSVLoader("path_to_your_csv_file.csv").load()

# Define embedding functions
embedding_functions = [
    SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
]

# Configure Vespa store
vespa_config = dict(
    page_content_field="lyrics",
    embedding_field="embedding",
    input_field="query_embedding",
)

# Create Vespa store for each embedding function
vespa_stores = []
for embedding_function in embedding_functions:
    db = VespaStore.from_documents(
        documents[:1000],
        embedding_function,
        app=vespa_docker,
        **vespa_config
    )
    vespa_stores.append(db)

# Perform similarity search using each Vespa store
query = "Queen"
for db in vespa_stores:
    results = db.similarity_search(query, k=10)
    for result in results:
        print(result.page_content)
        print(result.metadata)
