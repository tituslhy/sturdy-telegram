
from airflow.decorators import dag, task
from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models.baseoperator import chain
from datetime import datetime
from typing import List, Literal, Dict, Any
import os
import pandas as pd

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
EMBEDDING_DIMENSION = 768
QDRANT_URL = "http://localhost:6333"
# QDRANT_URL = "http://qdrant:6333"

# ─── TASKS ─────────────────────────────────────────────────────────────────
@task
def load_last_read(csv_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["last_read_date"])
        return df.to_dict(orient="records")
    return []

@task(
    trigger_rule="none_failed_min_one_success"
)
def find_new_files(folder: str, last_read_records: List[Dict[str, Any]]) -> List[str]:
    from datetime import datetime as _dt
    last_read = pd.DataFrame(last_read_records) if last_read_records else pd.DataFrame(columns=["file_path", "last_read_date"])
    all_pdfs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    new_files = []
    for path in all_pdfs:
        mtime = _dt.fromtimestamp(os.path.getmtime(path))
        prev = last_read.loc[last_read.file_path == path, "last_read_date"]
        if prev.empty or mtime > prev.iloc[0]:
            new_files.append(path)
    return sorted(new_files)

@task
def serialize_documents(paths: List[str]) -> List[Dict[str, Any]]:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.schema import Document
    if not paths:
        return []
    docs = SimpleDirectoryReader(input_files=paths).load_data()
    serialized = []
    for doc in docs:
        serialized.append(doc.model_dump())
    return serialized

@task
def ingest_serialized_documents(serialized_docs: List[Dict[str, Any]]) -> None:
    from llama_index.core.schema import Document
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, AsyncQdrantClient

    if not serialized_docs:
        return
    # reconstruct Document objects
    documents = [Document(**d) for d in serialized_docs]
    # setup Qdrant vector store
    client = QdrantClient(url=QDRANT_URL)
    aclient = AsyncQdrantClient(url=QDRANT_URL)
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        # base_url="http://ollama:11434",  # Default is localhost:11434
    )
    vs = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=COLLECTION_NAME,
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vs)
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_ctx,
        embed_model = embed_model,
    )

@task
def update_last_read(paths: List[str], csv_path: str) -> None:
    df = pd.read_csv(csv_path, parse_dates=["last_read_date"]) if os.path.exists(csv_path) else pd.DataFrame(columns=["file_path", "last_read_date"])
    from datetime import datetime as _dt
    for path in paths:
        mtime = _dt.fromtimestamp(os.path.getmtime(path))
        if path in df.file_path.values:
            df.loc[df.file_path == path, "last_read_date"] = mtime
        else:
            df = pd.concat([df, pd.DataFrame([{"file_path": path, "last_read_date": mtime}])])
    df.to_csv(csv_path, index=False)

@task.branch
def check_or_create_collection(name: str) -> Literal["create_collection", "skip_create_collection"]:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL)
    cols = client.get_collections().collections
    return "skip_create_collection" if any(c.name == name for c in cols) else "create_collection"

@task
def create_collection(name: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams
    client = QdrantClient(url=QDRANT_URL)
    client.create_collection(collection_name=name, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance="Cosine"))

# ─── DAG DEFINITION ────────────────────────────────────────────────────────
@dag(
    dag_id="qdrant_document_ingestion",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
    default_args={"owner": "airflow", "retries": 2},
    tags=["qdrant", "vector_db", "doc_ingest"],
)
def qdrant_document_ingestion():
    wait_dir = FileSensor(
        task_id="wait_for_docs_folder",
        fs_conn_id="fs_default",
        filepath=DOCS_FOLDER,
        poke_interval=15,
        timeout=600,
        mode="poke",
    )
    skip = EmptyOperator(task_id="skip_create_collection")
    branch = check_or_create_collection(COLLECTION_NAME)
    create = create_collection(COLLECTION_NAME)

    last_read = load_last_read(DATA_PATH)
    new_paths = find_new_files(DOCS_FOLDER, last_read)
    serialized = serialize_documents(new_paths)
    ingest = ingest_serialized_documents(serialized)
    update = update_last_read(new_paths, DATA_PATH)

    chain(
        wait_dir,
        branch,
        [create, skip],
        new_paths,
        serialized,
        ingest,
        update,
    )

qdrant_document_ingestion()

if __name__ == "__main__":
    print("File successfully loaded.")
