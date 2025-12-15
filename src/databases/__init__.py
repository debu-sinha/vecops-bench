"""Vector database adapters."""

from .base import VectorDBAdapter, QueryResult, IndexStats
from .chroma_adapter import ChromaAdapter
from .qdrant_adapter import QdrantAdapter
from .pgvector_adapter import PgvectorAdapter
from .milvus_adapter import MilvusAdapter
from .pinecone_adapter import PineconeAdapter
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
    }

    if db_name.lower() not in adapters:
        raise ValueError(f"Unknown database: {db_name}. Available: {list(adapters.keys())}")

    return adapters[db_name.lower()](config)
