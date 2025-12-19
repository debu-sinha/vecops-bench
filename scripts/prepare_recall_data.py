#!/usr/bin/env python3
"""
VecOps-Bench: Prepare Data for Proper Recall Testing

This script:
1. Streams vectors from Cohere Wikipedia dataset
2. Saves corpus vectors to memory-mapped file (for GT computation)
3. Holds out last N vectors as queries (NOT inserted into any index)
4. Saves query vectors separately
5. Saves ID mapping

Run this BEFORE ingestion to ensure queries are truly held out.

Usage:
    python scripts/prepare_recall_data.py --scale 10000000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DIMENSIONS = 768
DEFAULT_HELD_OUT = 10_000


def stream_cohere_wikipedia(num_vectors: int, batch_size: int = 50000):
    """Stream vectors from Cohere Wikipedia dataset."""
    from datasets import load_dataset

    print(f"Loading Cohere Wikipedia dataset (streaming {num_vectors:,} vectors)...")

    dataset = load_dataset(
        "Cohere/wikipedia-22-12-en-embeddings",
        split="train",
        streaming=True
    )

    vectors_yielded = 0
    batch_vectors = []
    batch_ids = []

    for i, item in enumerate(dataset):
        if vectors_yielded >= num_vectors:
            break

        emb = item["emb"]
        # Normalize for cosine similarity
        emb = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        batch_vectors.append(emb)
        batch_ids.append(f"wiki_{i}")
        vectors_yielded += 1

        if len(batch_vectors) >= batch_size:
            yield batch_ids, np.array(batch_vectors, dtype=np.float32)
            batch_vectors = []
            batch_ids = []

            if vectors_yielded % 500000 == 0:
                print(f"  Streamed {vectors_yielded:,} / {num_vectors:,} vectors")

    # Yield remaining
    if batch_vectors:
        yield batch_ids, np.array(batch_vectors, dtype=np.float32)


def prepare_data(
    output_dir: str,
    num_vectors: int,
    held_out: int = DEFAULT_HELD_OUT,
    use_real_embeddings: bool = True
):
    """
    Prepare corpus and query data for recall testing.

    Args:
        output_dir: Directory to save memmap files
        num_vectors: Total vectors to process (corpus + queries)
        held_out: Number of vectors to hold out as queries
        use_real_embeddings: Use Cohere Wikipedia (True) or random (False)
    """
    os.makedirs(output_dir, exist_ok=True)

    corpus_size = num_vectors - held_out
    print(f"\n{'='*60}")
    print(f"PREPARING RECALL TEST DATA")
    print(f"  Total vectors: {num_vectors:,}")
    print(f"  Corpus size: {corpus_size:,}")
    print(f"  Held-out queries: {held_out:,}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Create memory-mapped files
    corpus_path = os.path.join(output_dir, "corpus.memmap")
    query_path = os.path.join(output_dir, "queries.memmap")
    ids_path = os.path.join(output_dir, "corpus_ids.json")

    # Initialize memmap files
    corpus_fp = np.memmap(
        corpus_path, dtype='float32', mode='w+',
        shape=(corpus_size, DIMENSIONS)
    )
    query_fp = np.memmap(
        query_path, dtype='float32', mode='w+',
        shape=(held_out, DIMENSIONS)
    )

    corpus_ids = []
    query_ids = []
    vectors_written = 0

    if use_real_embeddings:
        generator = stream_cohere_wikipedia(num_vectors)
    else:
        # Random vectors for testing
        print("Using random vectors (for testing only)")
        rng = np.random.default_rng(42)

        def random_generator():
            batch_size = 50000
            for start in range(0, num_vectors, batch_size):
                end = min(start + batch_size, num_vectors)
                size = end - start
                vectors = rng.random((size, DIMENSIONS), dtype=np.float32)
                vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                ids = [f"vec_{i}" for i in range(start, end)]
                yield ids, vectors

        generator = random_generator()

    # Stream and save
    for batch_ids, batch_vectors in generator:
        for i, (vid, vec) in enumerate(zip(batch_ids, batch_vectors)):
            if vectors_written < corpus_size:
                # Write to corpus
                corpus_fp[vectors_written] = vec
                corpus_ids.append(vid)
            else:
                # Write to queries (held out)
                query_idx = vectors_written - corpus_size
                query_fp[query_idx] = vec
                query_ids.append(vid)

            vectors_written += 1

            if vectors_written >= num_vectors:
                break

        if vectors_written >= num_vectors:
            break

    # Flush and save metadata
    corpus_fp.flush()
    query_fp.flush()
    del corpus_fp
    del query_fp

    # Save shapes
    np.save(corpus_path + '.shape.npy', np.array([corpus_size, DIMENSIONS]))
    np.save(query_path + '.shape.npy', np.array([held_out, DIMENSIONS]))

    # Save IDs
    with open(ids_path, 'w') as f:
        json.dump(corpus_ids, f)

    query_ids_path = os.path.join(output_dir, "query_ids.json")
    with open(query_ids_path, 'w') as f:
        json.dump(query_ids, f)

    print(f"\n{'='*60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"  Corpus: {corpus_path} ({corpus_size:,} vectors)")
    print(f"  Queries: {query_path} ({held_out:,} vectors)")
    print(f"  Corpus IDs: {ids_path}")
    print(f"  Query IDs: {query_ids_path}")
    print(f"{'='*60}\n")

    # Verify
    print("Verifying files...")
    corpus_check = np.memmap(corpus_path, dtype='float32', mode='r',
                              shape=(corpus_size, DIMENSIONS))
    query_check = np.memmap(query_path, dtype='float32', mode='r',
                             shape=(held_out, DIMENSIONS))
    print(f"  Corpus shape: {corpus_check.shape}")
    print(f"  Query shape: {query_check.shape}")
    print(f"  Corpus[0] norm: {np.linalg.norm(corpus_check[0]):.4f}")
    print(f"  Query[0] norm: {np.linalg.norm(query_check[0]):.4f}")

    return corpus_path, query_path, ids_path


def main():
    parser = argparse.ArgumentParser(description="Prepare data for recall testing")
    parser.add_argument("--scale", type=int, default=10_000_000,
                        help="Total number of vectors (corpus + queries)")
    parser.add_argument("--held-out", type=int, default=DEFAULT_HELD_OUT,
                        help="Number of vectors to hold out as queries")
    parser.add_argument("--output-dir", type=str, default="data/recall_test",
                        help="Output directory for memmap files")
    parser.add_argument("--random", action="store_true",
                        help="Use random vectors instead of Cohere Wikipedia")
    args = parser.parse_args()

    prepare_data(
        args.output_dir,
        args.scale,
        args.held_out,
        use_real_embeddings=not args.random
    )

    print("\nNext steps:")
    print("1. Run ingestion with ONLY corpus vectors (exclude held-out queries)")
    print("2. Run: python scripts/recall_fix.py --database <db> --data-dir data/recall_test")


if __name__ == "__main__":
    main()
