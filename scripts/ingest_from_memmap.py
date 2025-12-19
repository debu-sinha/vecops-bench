#!/usr/bin/env python3
"""
VecOps-Bench: Ingest from Local Memmap

Ingests pre-prepared corpus data into a vector database.
Uses LOCAL memmap files - no HuggingFace streaming.

This fixes the issue where baseline benchmark tried to stream from HuggingFace.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.databases import get_adapter

# Configuration
BATCH_SIZE = 10_000
DIMENSIONS = 768

# Database connection configs
DB_CONFIGS = {
    "pgvector": {
        "host": "localhost",
        "port": 5432,
        "database": "vectordb_bench",
        "user": "postgres",
        "password": "postgres",
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333,
    },
    "milvus": {
        "host": "localhost",
        "port": 19530,
    },
    "weaviate": {
        "host": "localhost",
        "port": 8080,
    },
    "chroma": {
        "host": "localhost",
        "port": 8000,
    },
    "faiss": {},
}

# HNSW parameters (standardized)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128


def ingest_from_memmap(
    adapter,
    collection_name: str,
    corpus_path: str,
    ids_path: str,
    batch_size: int = BATCH_SIZE
):
    """Ingest vectors from local memmap file."""

    # Load corpus shape
    shape = tuple(np.load(corpus_path + '.shape.npy'))
    corpus = np.memmap(corpus_path, dtype='float32', mode='r', shape=shape)

    # Load IDs
    with open(ids_path, 'r') as f:
        corpus_ids = json.load(f)

    print(f"\n{'='*60}")
    print(f"INGESTION FROM LOCAL MEMMAP: {adapter.name}")
    print(f"  Corpus: {corpus.shape}")
    print(f"  IDs: {len(corpus_ids):,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"{'='*60}\n")

    # Create index (delete existing if any)
    print("Creating index...")
    try:
        adapter.delete_index(collection_name)
    except:
        pass

    # For Milvus, we need to explicitly define metadata fields in the schema
    create_kwargs = {
        "dimensions": DIMENSIONS,
        "metric": "cosine",
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
    }
    if adapter.name == "milvus":
        create_kwargs["metadata_fields"] = {"category": "string"}

    adapter.create_index(collection_name, **create_kwargs)

    # Ingest in batches
    start_time = time.time()
    total_ingested = 0

    for i in range(0, len(corpus_ids), batch_size):
        batch_end = min(i + batch_size, len(corpus_ids))
        batch_vectors = np.array(corpus[i:batch_end])  # Load batch into RAM
        batch_ids = corpus_ids[i:batch_end]

        # Add category metadata for filtered search
        batch_metadata = [{"category": chr(65 + (j % 10))} for j in range(i, batch_end)]

        adapter.insert_vectors(collection_name, batch_ids, batch_vectors.tolist(), batch_metadata)
        total_ingested += len(batch_ids)

        elapsed = time.time() - start_time
        rate = total_ingested / elapsed if elapsed > 0 else 0
        eta = (len(corpus_ids) - total_ingested) / rate if rate > 0 else 0

        print(f"  Ingested: {total_ingested:,}/{len(corpus_ids):,} ({rate:.0f} vec/s, ETA: {eta/60:.1f}min)")

    total_time = time.time() - start_time
    final_rate = len(corpus_ids) / total_time

    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE: {adapter.name}")
    print(f"  Total vectors: {len(corpus_ids):,}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Rate: {final_rate:.0f} vectors/second")
    print(f"{'='*60}\n")

    return {
        "database": adapter.name,
        "collection": collection_name,
        "total_vectors": len(corpus_ids),
        "total_time_s": total_time,
        "ingestion_rate": final_rate,
        "timestamp": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest from local memmap")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to ingest into")
    parser.add_argument("--collection", type=str, default="bench_v2",
                        help="Collection name prefix")
    parser.add_argument("--data-dir", type=str, default="data/recall_test",
                        help="Directory with memmap files")
    parser.add_argument("--output", type=str, default="results_v2/ingestion",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for ingestion")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    corpus_path = os.path.join(args.data_dir, "corpus.memmap")
    ids_path = os.path.join(args.data_dir, "corpus_ids.json")

    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        return

    if not os.path.exists(ids_path):
        print(f"ERROR: IDs not found at {ids_path}")
        return

    # Connect with proper config
    config = DB_CONFIGS.get(args.database, {})
    adapter = get_adapter(args.database, config)
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"

    # Ingest
    result = ingest_from_memmap(
        adapter, collection_name,
        corpus_path, ids_path,
        args.batch_size
    )

    # Save result
    output_file = os.path.join(
        args.output,
        f"{args.database}_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_file}")

    adapter.disconnect()


if __name__ == "__main__":
    main()
