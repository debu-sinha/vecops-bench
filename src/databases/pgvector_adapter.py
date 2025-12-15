"""pgvector (PostgreSQL) adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

from .base import VectorDBAdapter, QueryResult, IndexStats


class PgvectorAdapter(VectorDBAdapter):
    """Adapter for pgvector (PostgreSQL with vector extension)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pgvector adapter.

        Config options:
            host: str - PostgreSQL host
            port: int - PostgreSQL port (default: 5432)
            database: str - Database name
            user: str - Username
            password: str - Password
        """
        super().__init__("pgvector", config)
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to PostgreSQL with pgvector."""
        self.conn = psycopg2.connect(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            database=self.config.get("database", "vectordb"),
            user=self.config.get("user", "postgres"),
            password=self.config.get("password", "")
        )
        self.cursor = self.conn.cursor()

        # Ensure pgvector extension is installed
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()

        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cursor = None
        self._is_connected = False

    def create_index(
        self,
        collection_name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new table with vector column and index."""
        if not self._is_connected:
            raise RuntimeError("Not connected to PostgreSQL")

        # Map metric to pgvector operator
        metric_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "euclidean": "vector_l2_ops",
            "ip": "vector_ip_ops",
            "inner_product": "vector_ip_ops"
        }
        ops_class = metric_map.get(metric.lower(), "vector_cosine_ops")

        # Store metric for search queries
        self._current_metric = metric.lower()

        # Drop table if exists
        self.cursor.execute(f"DROP TABLE IF EXISTS {collection_name}")

        # Create table
        self.cursor.execute(f"""
            CREATE TABLE {collection_name} (
                id SERIAL PRIMARY KEY,
                doc_id TEXT UNIQUE,
                embedding vector({dimensions}),
                metadata JSONB
            )
        """)

        # Create index (HNSW or IVFFlat)
        index_type = kwargs.get("index_type", "hnsw")

        if index_type == "hnsw":
            m = kwargs.get("m", 16)
            ef_construction = kwargs.get("ef_construction", 64)
            self.cursor.execute(f"""
                CREATE INDEX {collection_name}_embedding_idx
                ON {collection_name}
                USING hnsw (embedding {ops_class})
                WITH (m = {m}, ef_construction = {ef_construction})
            """)
        elif index_type == "ivfflat":
            lists = kwargs.get("lists", 100)
            self.cursor.execute(f"""
                CREATE INDEX {collection_name}_embedding_idx
                ON {collection_name}
                USING ivfflat (embedding {ops_class})
                WITH (lists = {lists})
            """)

        # Create index on doc_id for faster lookups
        self.cursor.execute(f"""
            CREATE INDEX {collection_name}_doc_id_idx
            ON {collection_name} (doc_id)
        """)

        self.conn.commit()

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Insert vectors into the table."""
        if not self._is_connected:
            raise RuntimeError("Not connected to PostgreSQL")

        import json

        batch_size = 1000
        start_time = time.perf_counter()

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_ids)

            # Prepare data for bulk insert
            data = [
                (doc_id, str(vec), json.dumps(meta))
                for doc_id, vec, meta in zip(batch_ids, batch_vectors, batch_metadata)
            ]

            execute_values(
                self.cursor,
                f"""
                INSERT INTO {collection_name} (doc_id, embedding, metadata)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE
                SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                """,
                data,
                template="(%s, %s::vector, %s::jsonb)"
            )

        self.conn.commit()
        return time.perf_counter() - start_time

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Search for similar vectors."""
        if not self._is_connected:
            raise RuntimeError("Not connected to PostgreSQL")

        # Select distance operator based on metric
        metric = getattr(self, '_current_metric', 'cosine')
        if metric in ('cosine', 'cos'):
            distance_op = "<=>"  # Cosine distance
            score_expr = "1 - (embedding <=> %s::vector)"  # Convert to similarity
        elif metric in ('l2', 'euclidean'):
            distance_op = "<->"  # L2 distance
            score_expr = "-1 * (embedding <-> %s::vector)"  # Negative distance as score
        elif metric in ('ip', 'inner_product', 'dot'):
            distance_op = "<#>"  # Inner product (negative)
            score_expr = "-1 * (embedding <#> %s::vector)"  # Inner product as score
        else:
            distance_op = "<=>"
            score_expr = "1 - (embedding <=> %s::vector)"

        # Build query
        query_vector_str = str(query_vector)

        where_clause = ""
        if filter:
            conditions = self._build_filter_conditions(filter)
            if conditions:
                where_clause = f"WHERE {conditions}"

        query = f"""
            SELECT doc_id, {score_expr} as score, metadata
            FROM {collection_name}
            {where_clause}
            ORDER BY embedding {distance_op} %s::vector
            LIMIT %s
        """

        start_time = time.perf_counter()

        self.cursor.execute(query, (query_vector_str, query_vector_str, top_k))
        results = self.cursor.fetchall()

        latency_ms = (time.perf_counter() - start_time) * 1000

        ids = [row[0] for row in results]
        scores = [float(row[1]) for row in results]
        metadatas = [row[2] if row[2] else {} for row in results]

        return QueryResult(
            ids=ids,
            scores=scores,
            latency_ms=latency_ms,
            metadata=metadatas
        )

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> QueryResult:
        """
        Hybrid search combining dense and sparse (full-text) retrieval.

        Uses PostgreSQL full-text search combined with vector search.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to PostgreSQL")

        query_vector_str = str(query_vector)

        # Check if full-text search column exists
        self.cursor.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{collection_name}' AND column_name = 'text_content'
        """)

        if not self.cursor.fetchone():
            # Fall back to dense search if no text column
            return self.search(collection_name, query_vector, top_k)

        # Hybrid query using RRF (Reciprocal Rank Fusion)
        query = f"""
            WITH vector_search AS (
                SELECT doc_id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank
                FROM {collection_name}
                LIMIT %s
            ),
            text_search AS (
                SELECT doc_id, ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', text_content), plainto_tsquery('english', %s)) DESC) as rank
                FROM {collection_name}
                WHERE to_tsvector('english', text_content) @@ plainto_tsquery('english', %s)
                LIMIT %s
            )
            SELECT
                COALESCE(v.doc_id, t.doc_id) as doc_id,
                (COALESCE(1.0 / (60 + v.rank), 0) * %s + COALESCE(1.0 / (60 + t.rank), 0) * %s) as score
            FROM vector_search v
            FULL OUTER JOIN text_search t ON v.doc_id = t.doc_id
            ORDER BY score DESC
            LIMIT %s
        """

        start_time = time.perf_counter()

        self.cursor.execute(query, (
            query_vector_str, top_k * 2,
            query_text, query_text, top_k * 2,
            alpha, 1 - alpha,
            top_k
        ))
        results = self.cursor.fetchall()

        latency_ms = (time.perf_counter() - start_time) * 1000

        ids = [row[0] for row in results]
        scores = [float(row[1]) for row in results]

        return QueryResult(
            ids=ids,
            scores=scores,
            latency_ms=latency_ms,
            metadata=None
        )

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the table/index."""
        if not self._is_connected:
            raise RuntimeError("Not connected to PostgreSQL")

        # Get row count
        self.cursor.execute(f"SELECT COUNT(*) FROM {collection_name}")
        num_vectors = self.cursor.fetchone()[0]

        # Get vector dimensions
        self.cursor.execute(f"""
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = '{collection_name}'::regclass AND attname = 'embedding'
        """)
        result = self.cursor.fetchone()
        dimensions = result[0] if result else 0

        # Get table size
        self.cursor.execute(f"SELECT pg_total_relation_size('{collection_name}')")
        index_size = self.cursor.fetchone()[0]

        return IndexStats(
            num_vectors=num_vectors,
            dimensions=dimensions,
            index_size_bytes=index_size,
            build_time_seconds=0,
            memory_usage_bytes=0
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete a table."""
        if self.cursor:
            try:
                self.cursor.execute(f"DROP TABLE IF EXISTS {collection_name}")
                self.conn.commit()
            except Exception:
                self.conn.rollback()

    def _build_filter_conditions(self, filter: Dict[str, Any]) -> str:
        """Build SQL WHERE conditions from filter dict."""
        conditions = []

        for key, value in filter.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(f"metadata->>'{key}' = '{val}'")
                    elif op == "$gt":
                        conditions.append(f"(metadata->>'{key}')::float > {val}")
                    elif op == "$gte":
                        conditions.append(f"(metadata->>'{key}')::float >= {val}")
                    elif op == "$lt":
                        conditions.append(f"(metadata->>'{key}')::float < {val}")
                    elif op == "$lte":
                        conditions.append(f"(metadata->>'{key}')::float <= {val}")
                    elif op == "$ne":
                        conditions.append(f"metadata->>'{key}' != '{val}'")
            else:
                conditions.append(f"metadata->>'{key}' = '{value}'")

        return " AND ".join(conditions)

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
