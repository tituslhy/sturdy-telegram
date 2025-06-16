
from airflow.sdk import chain, dag, task
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.empty import EmptyOperator
from llama_index.core.schema import Document

import pandas as pd

import os
from datetime import datetime
from typing import List, Literal

# ─── CONFIG ────────────────────────────────────────────────────────────────

__curdir__ = os.getcwd()
if "notebooks" in __curdir__:
    DOCS_FOLDER = "../../docs"
    DATA_PATH = "../../data/last_read.csv"
elif "dag" in __curdir__:
    DOCS_FOLDER = "../docs"
    DATA_PATH = "../data/last_read.csv"
else:
    DOCS_FOLDER = "docs"
    DATA_PATH = "data/last_read.csv"

COLLECTION_NAME = "Airflow_Experiment"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # Adjust based on the model used
QDRANT_URL= "http://localhost:6333"

# ─── TASKS ─────────────────────────────────────────────────────────────────
@task
def load_last_read(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["last_read_date"])
    return pd.DataFrame(columns=["file_path", "last_read_date"])

@task
def find_new_files(folder: str, last_read: pd.DataFrame) -> List[str]:
    all_pdfs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf")
    ]
    new_files = []
    for path in all_pdfs:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        prev    = last_read.loc[last_read.file_path == path, "last_read_date"]
        if prev.empty or mtime > prev.iloc[0]:
            new_files.append(path)
    return sorted(new_files)

@task
def read_new_documents(paths: List[str]) -> List[Document] | List[None]:
    from llama_index.core import SimpleDirectoryReader
    if not paths:
        return []
    return SimpleDirectoryReader(input_files=paths).load_data()

@task
def ingest_new_documents(documents: List[Document]) -> None:
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, AsyncQdrantClient

    client = QdrantClient(url=QDRANT_URL)
    aclient = AsyncQdrantClient(url=QDRANT_URL)
    embed_model= OllamaEmbedding(model_name="nomic-embed-text")
    vs = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=COLLECTION_NAME,
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(documents, storage_context=storage, embed_model=embed_model)


@task
def update_last_read(paths: List[str], csv_path: str) -> None:
    df = pd.read_csv(csv_path, parse_dates=["last_read_date"]) \
         if os.path.exists(csv_path) \
         else pd.DataFrame(columns=["file_path", "last_read_date"])
    for path in paths:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        if path in df.file_path.values:
            df.loc[df.file_path == path, "last_read_date"] = mtime
        else:
            df = pd.concat([df, pd.DataFrame([{"file_path": path, "last_read_date": mtime}])])
    df.to_csv(csv_path, index=False)

@task
def check_qdrant_collection_exists(
    name: str
) -> Literal["create_collection", "skip_create"]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL)
    cols   = client.get_collections().collections
    if any(c.name == name for c in cols):
        return "skip_create"
    return "create_collection"

@task
def create_qdrant_collection_if_not_exists(name: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams

    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance="Cosine"),
    )

# ─── DAG DEFINITION ────────────────────────────────────────────────────────
@dag(
    dag_id="qdrant_document_ingestion",
    start_date=datetime(2025,1,1),
    schedule=None,
    catchup=False,
    default_args={"owner":"airflow","retries":2},
    tags=["qdrant","vector_db","doc_ingest"],
)
def qdrant_document_ingestion():
    # 1) Wait until the folder exists
    wait_dir = FileSensor(
        task_id="wait_for_docs_folder",
        filepath=DOCS_FOLDER,
        poke_interval=15,
        timeout=600,
        mode="poke",
    )
    skip = EmptyOperator(task_id="skip_create_collection")

    # 2) Branch: create collection if needed
    branch = check_qdrant_collection_exists(COLLECTION_NAME)
    create = create_qdrant_collection_if_not_exists(COLLECTION_NAME)

    # 3) Load state & find new files
    last_read_df = load_last_read(DATA_PATH)
    new_paths = find_new_files(DOCS_FOLDER, last_read_df)

    # 4) Read & ingest
    docs = read_new_documents(new_paths)
    ingest = ingest_new_documents(docs)

    # 5) Update state
    update = update_last_read(new_paths, DATA_PATH)

    # 6) Chain dependencies
    # Wait -> Branch -> [Create, Skip] -> Load state -> Find -> Read -> Ingest -> Update
    chain(
        wait_dir,
        branch,
        [create, skip],
        last_read_df,
        new_paths,
        docs,
        ingest,
        update
    )

qdrant_document_ingestion()
