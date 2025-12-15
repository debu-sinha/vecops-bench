"""
Ingestion Speed Measurement

Measures vectors per second during bulk load.
Optimized for 100M scale with streaming/batching.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np


@dataclass
class IngestionResult:
    """Results from ingestion benchmark."""
    database: str
    total_vectors: int
    total_seconds: float
    vectors_per_second: float
    batches: int
    batch_size: int
    time_to_100m_estimate_hours: float
    memory_delta_mb: float
    index_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "total_vectors": self.total_vectors,
            "total_seconds": self.total_seconds,
            "vectors_per_second": self.vectors_per_second,
            "batches": self.batches,
            "batch_size": self.batch_size,
            "time_to_100m_estimate_hours": self.time_to_100m_estimate_hours,
            "memory_delta_mb": self.memory_delta_mb,
            "index_size_bytes": self.index_size_bytes
        }


def vector_batch_generator(
    total_vectors: int,
    dimensions: int,
    batch_size: int = 10000,
    seed: int = 42
) -> Generator[Tuple[List[str], List[List[float]]], None, None]:
    """
    Generator that yields batches of random vectors.

    CRITICAL: Does NOT load all vectors into RAM at once.
    Memory efficient for 100M+ scale.

    Yields:
        Tuple of (ids, vectors) for each batch
    """
    rng = np.random.default_rng(seed)
    vector_id = 0

    while vector_id < total_vectors:
        current_batch_size = min(batch_size, total_vectors - vector_id)

        # Generate random normalized vectors
        vectors = rng.random((current_batch_size, dimensions), dtype=np.float32)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        # Generate IDs
        ids = [f"doc_{vector_id + i}" for i in range(current_batch_size)]

        yield ids, vectors.tolist()

        vector_id += current_batch_size


def measure_ingestion_speed(
    adapter,
    collection_name: str,
    total_vectors: int,
    dimensions: int,
    batch_size: int = 10000,
    create_index_first: bool = True
) -> IngestionResult:
    """
    Measure ingestion speed with streaming batches.

    Optimized for large scale:
    - Uses generator to avoid loading all data into RAM
    - Batches insertions for efficiency
    - Measures memory delta

    Args:
        adapter: Database adapter
        collection_name: Target collection
        total_vectors: Number of vectors to insert
        dimensions: Vector dimensionality
        batch_size: Vectors per batch (tune for DB)
        create_index_first: Whether to create index before insert

    Returns:
        IngestionResult with timing and throughput
    """
    import psutil
    process = psutil.Process()

    # Record initial memory
    mem_before = process.memory_info().rss

    # Create index if needed
    if create_index_first:
        adapter.create_index(collection_name, dimensions, metric="cosine")

    # Ingest with batching
    total_time = 0.0
    batches_completed = 0

    generator = vector_batch_generator(total_vectors, dimensions, batch_size)

    for ids, vectors in generator:
        batch_start = time.perf_counter()
        adapter.insert_vectors(collection_name, ids, vectors)
        batch_time = time.perf_counter() - batch_start

        total_time += batch_time
        batches_completed += 1

        # Progress logging for long runs
        if batches_completed % 100 == 0:
            progress = (batches_completed * batch_size) / total_vectors * 100
            current_rate = (batches_completed * batch_size) / total_time
            print(f"    Progress: {progress:.1f}% | Rate: {current_rate:.0f} vec/s")

    # Final stats
    mem_after = process.memory_info().rss
    stats = adapter.get_index_stats(collection_name)

    vectors_per_second = total_vectors / total_time
    time_to_100m = (100_000_000 / vectors_per_second) / 3600  # hours

    return IngestionResult(
        database=adapter.name,
        total_vectors=total_vectors,
        total_seconds=total_time,
        vectors_per_second=vectors_per_second,
        batches=batches_completed,
        batch_size=batch_size,
        time_to_100m_estimate_hours=time_to_100m,
        memory_delta_mb=(mem_after - mem_before) / (1024 * 1024),
        index_size_bytes=stats.index_size_bytes
    )


def stream_dataset_from_disk(
    file_path: str,
    batch_size: int = 10000
) -> Generator[Tuple[List[str], List[List[float]]], None, None]:
    """
    Stream pre-embedded vectors from disk (Parquet/HDF5).

    For real datasets like LAION-10M that are too large for RAM.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        ids = df["id"].tolist()
        vectors = df["embedding"].tolist()
        yield ids, vectors


def stream_from_huggingface(
    dataset_name: str,
    embedding_column: str = "embedding",
    id_column: str = "id",
    batch_size: int = 10000,
    max_vectors: Optional[int] = None
) -> Generator[Tuple[List[str], List[List[float]]], None, None]:
    """
    Stream embeddings from HuggingFace dataset.

    Uses streaming mode to avoid downloading full dataset.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, streaming=True, split="train")

    batch_ids = []
    batch_vectors = []
    count = 0

    for sample in dataset:
        if max_vectors and count >= max_vectors:
            break

        batch_ids.append(sample.get(id_column, f"doc_{count}"))
        batch_vectors.append(sample[embedding_column])
        count += 1

        if len(batch_ids) >= batch_size:
            yield batch_ids, batch_vectors
            batch_ids = []
            batch_vectors = []

    # Yield remaining
    if batch_ids:
        yield batch_ids, batch_vectors


def optimal_batch_size_for_db(db_name: str) -> int:
    """
    Return recommended batch size for each database.

    Based on empirical testing and documentation.
    """
    optimal_sizes = {
        "milvus": 10000,     # Milvus handles large batches well
        "qdrant": 5000,      # Qdrant prefers medium batches
        "pgvector": 1000,    # PostgreSQL has transaction overhead
        "chroma": 5000,      # Chroma is memory-constrained
        "weaviate": 10000,   # Weaviate optimized for batches
        "elasticsearch": 5000,
        "faiss": 50000,      # In-memory, can handle large batches
    }
    return optimal_sizes.get(db_name.lower(), 5000)
