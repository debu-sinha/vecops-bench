#!/usr/bin/env python3
r"""
VectorDB-Bench: Local Test Script

Run this FIRST to verify the benchmark code works before spinning up EC2.
Tests with Chroma only (no Docker required).

Usage:
    cd C:\Users\dsinh\research\vectordb-bench
    python scripts/test_local.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Importing modules")
    print("=" * 60)

    errors = []

    modules = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("tqdm", "tqdm"),
        ("yaml", "pyyaml"),
        ("psutil", "psutil"),
    ]

    for module_name, pip_name in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [X] {module_name} - FAILED")
            errors.append(f"pip install {pip_name}")

    # Test our custom modules
    custom_modules = [
        "src.datasets",
        "src.databases",
        "src.metrics",
        "src.workloads",
        "src.drift",
        "src.cost",
        "src.operational",
    ]

    print("\n  Custom modules:")
    for module_name in custom_modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except ImportError as e:
            print(f"  [X] {module_name} - {e}")
            errors.append(f"Check {module_name}")

    if errors:
        print(f"\n  FAILED - Run: {'; '.join(errors)}")
        return False

    print("\n  [OK] All imports successful!")
    return True


def test_chroma_basic():
    """Test basic Chroma operations."""
    print("\n" + "=" * 60)
    print("TEST 2: Chroma basic operations")
    print("=" * 60)

    import chromadb
    import numpy as np

    try:
        # Create ephemeral client (in-memory)
        client = chromadb.Client()
        print("  [OK] Created Chroma client")

        # Create collection
        collection = client.create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("  [OK] Created collection")

        # Insert vectors
        n_vectors = 100
        dim = 128
        ids = [f"doc_{i}" for i in range(n_vectors)]
        embeddings = np.random.randn(n_vectors, dim).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings
        )
        print(f"  [OK] Inserted {n_vectors} vectors")

        # Query
        query_vector = np.random.randn(dim).tolist()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )
        print(f"  [OK] Query returned {len(results['ids'][0])} results")

        # Cleanup
        client.delete_collection("test_collection")
        print("  [OK] Cleanup successful")

        print("\n  [OK] Chroma test passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Chroma test failed: {e}")
        return False


def test_embedding_model():
    """Test embedding model loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Embedding model (may take 1-2 min first time)")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        print("  Loading model (downloading if first time)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model for testing
        print("  [OK] Model loaded")

        # Test embedding
        texts = ["Hello world", "Test document"]
        embeddings = model.encode(texts)
        print(f"  [OK] Generated embeddings: shape {embeddings.shape}")

        print("\n  [OK] Embedding model test passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Embedding test failed: {e}")
        return False


def test_metrics():
    """Test metric calculations."""
    print("\n" + "=" * 60)
    print("TEST 4: Metrics module")
    print("=" * 60)

    try:
        from src.metrics import recall_at_k, precision_at_k, ndcg_at_k, latency_percentiles

        # Test recall
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c", "f"]

        r = recall_at_k(retrieved, relevant, 5)
        print(f"  [OK] recall@5 = {r:.4f} (expected: 0.6667)")

        p = precision_at_k(retrieved, relevant, 5)
        print(f"  [OK] precision@5 = {p:.4f} (expected: 0.4000)")

        # Test latency percentiles
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        percs = latency_percentiles(latencies)
        print(f"  [OK] latency p50 = {percs['p50']:.2f} (expected: ~5.5)")

        print("\n  [OK] Metrics test passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_module():
    """Test cost tracking module."""
    print("\n" + "=" * 60)
    print("TEST 5: Cost tracking module")
    print("=" * 60)

    try:
        from src.cost import CostTracker, COST_MODELS

        print(f"  [OK] Available cost models: {list(COST_MODELS.keys())}")

        # Test tracker
        tracker = CostTracker(cost_model_name="self_hosted_medium", database_name="test")
        tracker.start()

        import time
        time.sleep(0.5)
        tracker.record_queries(1000)

        tracker.stop()
        breakdown = tracker.get_cost_breakdown()

        print(f"  [OK] Cost breakdown: ${breakdown.cost_per_million_queries_usd:.4f}/M queries")

        print("\n  [OK] Cost module test passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Cost module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_operational_module():
    """Test operational complexity module."""
    print("\n" + "=" * 60)
    print("TEST 6: Operational complexity module")
    print("=" * 60)

    try:
        from src.operational import (
            DATABASE_OPERATIONAL_PROFILES,
            compute_complexity_score,
            generate_complexity_report
        )

        print(f"  [OK] Profiles available for: {list(DATABASE_OPERATIONAL_PROFILES.keys())}")

        # Test scoring
        metrics = DATABASE_OPERATIONAL_PROFILES["qdrant"]
        score = compute_complexity_score(metrics, "qdrant")

        print(f"  [OK] Qdrant complexity score: {score.overall_score:.1f}/100")
        print(f"    - Deployment: {score.deployment_score:.1f}")
        print(f"    - Recommended team: {score.recommended_team_size}")

        print("\n  [OK] Operational module test passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Operational module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_benchmark():
    """Run a minimal end-to-end benchmark."""
    print("\n" + "=" * 60)
    print("TEST 7: Mini end-to-end benchmark (Chroma)")
    print("=" * 60)

    try:
        import numpy as np
        import chromadb
        from src.metrics import recall_at_k, latency_percentiles
        from src.cost import CostTracker
        import time

        # Create test data
        n_docs = 500
        n_queries = 50
        dim = 128

        print(f"  Creating test data: {n_docs} docs, {n_queries} queries, {dim} dims")

        np.random.seed(42)
        doc_embeddings = np.random.randn(n_docs, dim).astype(np.float32)
        query_embeddings = np.random.randn(n_queries, dim).astype(np.float32)

        # Normalize for cosine similarity
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Ground truth: brute force nearest neighbors
        print("  Computing ground truth...")
        similarities = query_embeddings @ doc_embeddings.T
        ground_truth = [np.argsort(-sim)[:10].tolist() for sim in similarities]

        # Setup Chroma
        print("  Setting up Chroma...")
        client = chromadb.Client()
        collection = client.create_collection(
            name="mini_bench",
            metadata={"hnsw:space": "cosine"}
        )

        # Insert
        print("  Inserting vectors...")
        ids = [f"doc_{i}" for i in range(n_docs)]
        collection.add(ids=ids, embeddings=doc_embeddings.tolist())

        # Benchmark queries
        print("  Running queries...")
        tracker = CostTracker(cost_model_name="local_development", database_name="chroma")
        tracker.start()

        latencies = []
        recalls = []

        for i, query_vec in enumerate(query_embeddings):
            start = time.perf_counter()
            results = collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=10
            )
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Compute recall
            retrieved_ids = [int(id.split("_")[1]) for id in results['ids'][0]]
            recall = len(set(retrieved_ids) & set(ground_truth[i])) / 10
            recalls.append(recall)

            tracker.record_queries(1)

        tracker.stop()

        # Results
        avg_recall = np.mean(recalls)
        lat_stats = latency_percentiles(latencies)
        qps = 1000 / lat_stats['mean']

        print(f"\n  Results:")
        print(f"    Recall@10:    {avg_recall:.4f}")
        print(f"    Latency p50:  {lat_stats['p50']:.2f} ms")
        print(f"    Latency p99:  {lat_stats['p99']:.2f} ms")
        print(f"    QPS:          {qps:.1f}")

        # Cleanup
        client.delete_collection("mini_bench")

        print("\n  [OK] Mini benchmark passed!")
        return True

    except Exception as e:
        print(f"\n  [X] Mini benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n")
    print("+============================================================+")
    print("|     VectorDB-Bench: Local Environment Test                 |")
    print("|     Run this before deploying to EC2!                      |")
    print("+============================================================+")
    print()

    tests = [
        ("Imports", test_imports),
        ("Chroma Basic", test_chroma_basic),
        ("Embedding Model", test_embedding_model),
        ("Metrics", test_metrics),
        ("Cost Module", test_cost_module),
        ("Operational Module", test_operational_module),
        ("Mini Benchmark", test_mini_benchmark),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [X] {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[OK] PASS" if p else "[X] FAIL"
        print(f"  {status}  {name}")

    print()
    print(f"  {passed}/{total} tests passed")
    print()

    if passed == total:
        print("+============================================================+")
        print("|  [OK] ALL TESTS PASSED - Ready for EC2 deployment!        |")
        print("+============================================================+")
        return 0
    else:
        print("+============================================================+")
        print("|  [X] SOME TESTS FAILED - Fix issues before deploying      |")
        print("+============================================================+")
        return 1


if __name__ == "__main__":
    sys.exit(main())
