"""Vector database adapters."""

from .base import IndexStats, QueryResult, VectorDBAdapter
from .chroma_adapter import ChromaAdapter
from .elasticsearch_adapter import ElasticsearchAdapter
from .faiss_adapter import FaissAdapter
from .milvus_adapter import MilvusAdapter
from .pgvector_adapter import PgvectorAdapter
from .pinecone_adapter import PineconeAdapter
from .qdrant_adapter import QdrantAdapter
from .weaviate_adapter import WeaviateAdapter

__all__ = [
    "VectorDBAdapter",
    "QueryResult",
    "IndexStats",
    "ChromaAdapter",
    "QdrantAdapter",
    "PgvectorAdapter",
    "MilvusAdapter",
    "PineconeAdapter",
    "WeaviateAdapter",
    "FaissAdapter",
    "ElasticsearchAdapter",
]


def get_adapter(db_name: str, config: dict) -> VectorDBAdapter:
    """Factory function to get database adapter by name."""
    adapters = {
        "chroma": ChromaAdapter,
        "qdrant": QdrantAdapter,
        "pgvector": PgvectorAdapter,
        "milvus": MilvusAdapter,
        "pinecone": PineconeAdapter,
        "weaviate": WeaviateAdapter,
        "faiss": FaissAdapter,
        "elasticsearch": ElasticsearchAdapter,
    }

    if db_name.lower() not in adapters:
        raise ValueError(f"Unknown database: {db_name}. Available: {list(adapters.keys())}")

    return adapters[db_name.lower()](config)
