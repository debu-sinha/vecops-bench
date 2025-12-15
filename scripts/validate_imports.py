#!/usr/bin/env python3
"""Validate all module imports after code changes."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modified modules import correctly."""
    results = []

    # Test 1: Stats module
    print("  [1/6] src.stats...", end=" ")
    try:
        from src.stats import (
            compute_confidence_interval,
            check_normality,
            mann_whitney_u,
            compare_databases,
            StatisticalResult,
            ComparisonResult
        )
        print("OK")
        results.append(("stats", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("stats", False, str(e)))

    # Test 2: Analysis module
    print("  [2/6] src.analysis...", end=" ")
    try:
        from src.analysis import (
            capture_pgvector_query_plan,
            capture_generic_timing,
            QueryPlanResult
        )
        print("OK")
        results.append(("analysis", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("analysis", False, str(e)))

    # Test 3: Resilience modules
    print("  [3/6] src.resilience...", end=" ")
    try:
        from src.resilience.cold_start import ColdStartResult, measure_cold_start
        from src.resilience.crash_recovery import verify_data_integrity, CrashRecoveryResult
        from src.resilience.ingestion_speed import vector_batch_generator, measure_ingestion_speed
        print("OK")
        results.append(("resilience", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("resilience", False, str(e)))

    # Test 4: Faiss adapter
    print("  [4/6] src.databases.faiss_adapter...", end=" ")
    try:
        from src.databases.faiss_adapter import FaissAdapter
        print("OK")
        results.append(("faiss_adapter", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("faiss_adapter", False, str(e)))

    # Test 5: Elasticsearch adapter
    print("  [5/6] src.databases.elasticsearch_adapter...", end=" ")
    try:
        from src.databases.elasticsearch_adapter import ElasticsearchAdapter
        print("OK")
        results.append(("elasticsearch_adapter", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("elasticsearch_adapter", False, str(e)))

    # Test 6: Main benchmark runner
    print("  [6/6] scripts.run_benchmark_v2...", end=" ")
    try:
        # Just check syntax by compiling
        import py_compile
        py_compile.compile(
            str(Path(__file__).parent / "run_benchmark_v2.py"),
            doraise=True
        )
        print("OK")
        results.append(("run_benchmark_v2", True, None))
    except Exception as e:
        print(f"FAIL: {e}")
        results.append(("run_benchmark_v2", False, str(e)))

    return results


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")

    # Test stats
    print("  [1/3] Statistical functions...", end=" ")
    try:
        from src.stats import compute_confidence_interval, check_normality
        import numpy as np

        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std, ci_low, ci_high = compute_confidence_interval(data)
        assert ci_low < mean < ci_high, "CI should contain mean"

        is_normal, p_val = check_normality(data)
        assert 0 <= p_val <= 1, "p-value should be in [0,1]"

        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # Test vector generator
    print("  [2/3] Vector batch generator...", end=" ")
    try:
        from src.resilience.ingestion_speed import vector_batch_generator

        gen = vector_batch_generator(100, 768, batch_size=50)
        batch1_ids, batch1_vecs = next(gen)
        batch2_ids, batch2_vecs = next(gen)

        assert len(batch1_ids) == 50, "First batch should have 50 vectors"
        assert len(batch2_ids) == 50, "Second batch should have 50 vectors"
        assert batch1_ids[0] != batch2_ids[0], "IDs should be unique across batches"

        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")

    # Test Faiss adapter (without actual faiss operations)
    print("  [3/3] Faiss adapter init...", end=" ")
    try:
        from src.databases.faiss_adapter import FaissAdapter

        adapter = FaissAdapter({"M": 32, "ef_construction": 200})
        assert adapter.M == 32
        assert adapter.ef_construction == 200

        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("VectorDB-Bench v2.0 - Import Validation")
    print("=" * 60)
    print("\nTesting module imports...")

    results = test_imports()

    test_basic_functionality()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    if passed == total:
        print(f"SUCCESS: All {total} import tests passed!")
        sys.exit(0)
    else:
        print(f"FAILED: {passed}/{total} import tests passed")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
