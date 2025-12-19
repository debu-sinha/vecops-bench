#!/usr/bin/env python3
"""
VecOps-Bench: SIFT1M Dataset Preparation

Downloads and prepares the classic SIFT1M benchmark dataset.
This is the gold standard for ANN benchmark papers.

Dataset: http://corpus-texmex.irisa.fr/
- 1,000,000 base vectors (128 dimensions)
- 10,000 query vectors (held out)
- 100 ground truth neighbors per query (pre-computed)

This gives us:
1. A second dataset for generalization
2. Pre-computed ground truth (no need to brute force)
3. Direct comparison with ann-benchmarks results
"""

import os
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

# SIFT1M URLs
SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_FILENAME = "sift.tar.gz"

# Output paths
DATA_DIR = "data/sift1m"


def download_sift1m(output_dir: str) -> str:
    """Download SIFT1M dataset if not already present."""
    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, SIFT1M_FILENAME)

    if os.path.exists(tar_path):
        print(f"  {SIFT1M_FILENAME} already exists, skipping download")
        return tar_path

    print(f"  Downloading SIFT1M from {SIFT1M_URL}...")
    print(f"  This may take a few minutes (~160MB)...")

    urllib.request.urlretrieve(SIFT1M_URL, tar_path)
    print(f"  Downloaded to {tar_path}")

    return tar_path


def extract_sift1m(tar_path: str, output_dir: str) -> str:
    """Extract SIFT1M tar.gz file."""
    extract_dir = os.path.join(output_dir, "sift")

    if os.path.exists(extract_dir):
        print(f"  Already extracted to {extract_dir}")
        return extract_dir

    print(f"  Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)

    print(f"  Extracted to {extract_dir}")
    return extract_dir


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs file format (float vectors)."""
    with open(filename, "rb") as f:
        # Read all data
        data = f.read()

    # Parse vectors
    vectors = []
    offset = 0
    while offset < len(data):
        # First 4 bytes: dimension (int32)
        dim = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4

        # Next dim*4 bytes: vector (float32)
        vec = struct.unpack(f'{dim}f', data[offset:offset+dim*4])
        vectors.append(vec)
        offset += dim * 4

    return np.array(vectors, dtype=np.float32)


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs file format (integer vectors - for ground truth)."""
    with open(filename, "rb") as f:
        data = f.read()

    vectors = []
    offset = 0
    while offset < len(data):
        dim = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        vec = struct.unpack(f'{dim}i', data[offset:offset+dim*4])
        vectors.append(vec)
        offset += dim * 4

    return np.array(vectors, dtype=np.int32)


def prepare_sift1m(output_dir: str = DATA_DIR):
    """
    Download, extract, and prepare SIFT1M dataset.

    Creates:
    - corpus.memmap (1M x 128 float32)
    - queries.memmap (10K x 128 float32)
    - ground_truth.npy (10K x 100 int32)
    - corpus_ids.json
    """
    print("\n" + "="*60)
    print("PREPARING SIFT1M DATASET")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    # Download
    print("[1/4] Downloading...")
    tar_path = download_sift1m(output_dir)

    # Extract
    print("\n[2/4] Extracting...")
    extract_dir = extract_sift1m(tar_path, output_dir)

    # Read raw files
    print("\n[3/4] Reading vectors...")
    base_file = os.path.join(extract_dir, "sift_base.fvecs")
    query_file = os.path.join(extract_dir, "sift_query.fvecs")
    gt_file = os.path.join(extract_dir, "sift_groundtruth.ivecs")

    print(f"  Reading base vectors from {base_file}...")
    base_vectors = read_fvecs(base_file)
    print(f"    Shape: {base_vectors.shape}")

    print(f"  Reading query vectors from {query_file}...")
    query_vectors = read_fvecs(query_file)
    print(f"    Shape: {query_vectors.shape}")

    print(f"  Reading ground truth from {gt_file}...")
    ground_truth = read_ivecs(gt_file)
    print(f"    Shape: {ground_truth.shape}")

    # Normalize vectors for cosine similarity
    print("\n[4/4] Normalizing and saving...")

    # Normalize base vectors
    norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_vectors = base_vectors / norms

    # Normalize query vectors
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / norms

    # Save as memmap
    corpus_path = os.path.join(output_dir, "corpus.memmap")
    query_path = os.path.join(output_dir, "queries.memmap")
    gt_path = os.path.join(output_dir, "ground_truth.npy")
    ids_path = os.path.join(output_dir, "corpus_ids.json")

    # Corpus
    corpus_fp = np.memmap(corpus_path, dtype='float32', mode='w+', shape=base_vectors.shape)
    corpus_fp[:] = base_vectors[:]
    corpus_fp.flush()
    del corpus_fp
    np.save(corpus_path + '.shape.npy', np.array(base_vectors.shape))

    # Queries
    query_fp = np.memmap(query_path, dtype='float32', mode='w+', shape=query_vectors.shape)
    query_fp[:] = query_vectors[:]
    query_fp.flush()
    del query_fp
    np.save(query_path + '.shape.npy', np.array(query_vectors.shape))

    # Ground truth (already computed - just save)
    np.save(gt_path, ground_truth)

    # IDs
    import json
    corpus_ids = [f"sift_{i}" for i in range(len(base_vectors))]
    with open(ids_path, 'w') as f:
        json.dump(corpus_ids, f)

    print(f"\n  Saved corpus: {corpus_path} ({base_vectors.shape})")
    print(f"  Saved queries: {query_path} ({query_vectors.shape})")
    print(f"  Saved ground truth: {gt_path} ({ground_truth.shape})")
    print(f"  Saved IDs: {ids_path}")

    print("\n" + "="*60)
    print("SIFT1M PREPARATION COMPLETE")
    print("="*60)
    print(f"\nDataset summary:")
    print(f"  Corpus: 1,000,000 vectors x 128 dimensions")
    print(f"  Queries: 10,000 vectors (held out)")
    print(f"  Ground truth: 100 neighbors per query (pre-computed)")
    print(f"  Normalized: Yes (for cosine similarity)")

    return {
        "corpus_path": corpus_path,
        "query_path": query_path,
        "gt_path": gt_path,
        "ids_path": ids_path,
        "corpus_size": len(base_vectors),
        "query_size": len(query_vectors),
        "dimensions": base_vectors.shape[1],
    }


if __name__ == "__main__":
    prepare_sift1m()
