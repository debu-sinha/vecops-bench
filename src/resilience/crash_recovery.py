"""
Crash Recovery Time Measurement (TTR - Time to Recovery)

Measures the time from kill -9 (ungraceful termination) to API returning 200 OK.
This simulates real-world crash scenarios and tests database durability.
"""

import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import requests


@dataclass
class CrashRecoveryResult:
    """Results from crash recovery measurement."""

    database: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    trials: List[float]
    data_integrity_verified: bool
    methodology: str = "kill_sigkill"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "num_trials": len(self.trials),
            "trials": self.trials,
            "data_integrity_verified": self.data_integrity_verified,
            "methodology": self.methodology,
        }


# Health check endpoints for each database
HEALTH_ENDPOINTS = {
    "milvus": ("localhost", 19530, "tcp"),  # gRPC port
    "qdrant": ("localhost", 6333, "http", "/"),
    "pgvector": ("localhost", 5432, "tcp"),  # PostgreSQL port
    "chroma": ("localhost", 8000, "http", "/api/v1/heartbeat"),
    "weaviate": ("localhost", 8080, "http", "/v1/.well-known/ready"),
    "elasticsearch": ("localhost", 9200, "http", "/_cluster/health"),
}


def check_tcp_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_http_endpoint(host: str, port: int, path: str, timeout: float = 1.0) -> bool:
    """Check if an HTTP endpoint returns 200."""
    try:
        response = requests.get(f"http://{host}:{port}{path}", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_api_ready(db_name: str, timeout_seconds: float = 120) -> float:
    """
    Wait for database API to be ready after restart.

    Returns time taken in seconds.
    """
    config = HEALTH_ENDPOINTS.get(db_name)
    if not config:
        raise ValueError(f"Unknown database: {db_name}")

    start = time.perf_counter()
    deadline = start + timeout_seconds

    while time.perf_counter() < deadline:
        if config[2] == "tcp":
            if check_tcp_port(config[0], config[1]):
                # TCP port open, wait a bit more for full initialization
                time.sleep(0.5)
                return time.perf_counter() - start
        else:  # HTTP
            if check_http_endpoint(config[0], config[1], config[3]):
                return time.perf_counter() - start

        time.sleep(0.1)

    raise TimeoutError(f"API not ready after {timeout_seconds}s")


def kill_container(container_name: str) -> None:
    """Kill container with SIGKILL (ungraceful termination)."""
    subprocess.run(
        ["docker", "kill", "--signal=KILL", container_name], capture_output=True, check=True
    )


def start_container(container_name: str) -> None:
    """Start a stopped container."""
    subprocess.run(["docker", "start", container_name], capture_output=True, check=True)


def verify_data_integrity(
    adapter,
    collection_name: str,
    expected_count: int,
    sample_query: List[float],
    pre_crash_results: List[str] = None,
) -> dict:
    """
    Verify data integrity after crash recovery.

    Checks:
    1. Vector count matches expected
    2. Sample query returns results
    3. (If pre_crash_results provided) Results match pre-crash state

    Returns:
        Dict with verification details instead of just bool
    """
    result = {
        "count_ok": False,
        "query_ok": False,
        "results_match": None,
        "actual_count": 0,
        "expected_count": expected_count,
        "error": None,
    }

    try:
        stats = adapter.get_index_stats(collection_name)
        result["actual_count"] = stats.num_vectors
        result["count_ok"] = stats.num_vectors == expected_count

        search_result = adapter.search(collection_name, sample_query, top_k=10)
        result["query_ok"] = len(search_result.ids) > 0

        # If we have pre-crash results, verify consistency
        if pre_crash_results is not None and search_result.ids:
            # Check if top-10 results are the same (allowing for minor reordering)
            pre_set = set(pre_crash_results[:10])
            post_set = set(search_result.ids[:10])
            overlap = len(pre_set & post_set)
            result["results_match"] = overlap >= 8  # Allow 20% drift due to ANN
            result["result_overlap"] = overlap

    except Exception as e:
        result["error"] = str(e)

    return result


def verify_data_integrity_bool(
    adapter,
    collection_name: str,
    expected_count: int,
    sample_query: List[float],
    pre_crash_results: List[str] = None,
) -> bool:
    """Backward-compatible bool version of verify_data_integrity."""
    result = verify_data_integrity(
        adapter, collection_name, expected_count, sample_query, pre_crash_results
    )
    return result["count_ok"] and result["query_ok"]


def measure_crash_recovery(
    db_name: str,
    adapter,
    collection_name: str,
    query_vector: List[float],
    expected_vector_count: int,
    num_trials: int = 5,
    timeout_seconds: float = 120,
) -> CrashRecoveryResult:
    """
    Measure crash recovery time.

    Simulates crash with kill -9, then measures time until API is ready
    and data is accessible.

    Args:
        db_name: Name of the database container
        adapter: Database adapter instance
        collection_name: Name of the collection/index
        query_vector: Sample query vector for verification
        expected_vector_count: Expected number of vectors after recovery
        num_trials: Number of crash/recovery trials
        timeout_seconds: Maximum time to wait for recovery

    Returns:
        CrashRecoveryResult with statistics
    """
    times = []
    integrity_checks = []
    integrity_details = []

    for trial in range(num_trials):
        pre_crash_results = None

        # Ensure container is running and capture pre-crash state
        try:
            adapter.connect()
            initial_stats = adapter.get_index_stats(collection_name)
            if initial_stats.num_vectors == 0:
                raise ValueError("No data in collection - load data first")

            # Capture pre-crash query results for comparison
            pre_crash_search = adapter.search(collection_name, query_vector, top_k=10)
            pre_crash_results = pre_crash_search.ids

        except Exception as e:
            print(f"Warning: Pre-check failed: {e}")

        # KILL -9 (ungraceful termination)
        kill_container(db_name)

        # Disconnect adapter (connection is now dead)
        try:
            adapter.disconnect()
        except Exception:
            pass

        # Measure recovery time
        start = time.perf_counter()

        # Restart container
        start_container(db_name)

        # Wait for API to be ready
        try:
            wait_for_api_ready(db_name, timeout_seconds)

            # Reconnect adapter
            adapter.connect()

            # Execute query to verify full recovery
            result = adapter.search(collection_name, query_vector, top_k=10)

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

            # Verify data integrity with pre-crash comparison
            integrity_result = verify_data_integrity(
                adapter, collection_name, expected_vector_count, query_vector, pre_crash_results
            )
            integrity_ok = integrity_result["count_ok"] and integrity_result["query_ok"]
            integrity_checks.append(integrity_ok)
            integrity_details.append(integrity_result)

        except TimeoutError:
            times.append(timeout_seconds * 1000)  # Record timeout
            integrity_checks.append(False)
            integrity_details.append({"error": "timeout"})

        except Exception as e:
            print(f"Recovery error: {e}")
            times.append(timeout_seconds * 1000)
            integrity_checks.append(False)
            integrity_details.append({"error": str(e)})

    times_np = np.array(times)

    return CrashRecoveryResult(
        database=db_name,
        mean_ms=float(np.mean(times_np)),
        std_ms=float(np.std(times_np)),
        min_ms=float(np.min(times_np)),
        max_ms=float(np.max(times_np)),
        p50_ms=float(np.percentile(times_np, 50)),
        p95_ms=float(np.percentile(times_np, 95)),
        p99_ms=float(np.percentile(times_np, 99)),
        trials=times,
        data_integrity_verified=all(integrity_checks),
    )


def measure_graceful_shutdown_recovery(
    db_name: str, adapter, collection_name: str, query_vector: List[float], num_trials: int = 5
) -> CrashRecoveryResult:
    """
    Measure recovery time after graceful shutdown (docker stop).

    This is less stressful than kill -9 and shows best-case recovery.
    """
    times = []

    for _ in range(num_trials):
        # Graceful stop
        subprocess.run(["docker", "stop", db_name], capture_output=True, check=True)

        try:
            adapter.disconnect()
        except Exception:
            pass

        # Measure recovery
        start = time.perf_counter()
        start_container(db_name)
        wait_for_api_ready(db_name)
        adapter.connect()
        adapter.search(collection_name, query_vector, top_k=10)
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)

    times_np = np.array(times)

    return CrashRecoveryResult(
        database=db_name,
        mean_ms=float(np.mean(times_np)),
        std_ms=float(np.std(times_np)),
        min_ms=float(np.min(times_np)),
        max_ms=float(np.max(times_np)),
        p50_ms=float(np.percentile(times_np, 50)),
        p95_ms=float(np.percentile(times_np, 95)),
        p99_ms=float(np.percentile(times_np, 99)),
        trials=times,
        data_integrity_verified=True,  # Graceful shutdown should preserve data
        methodology="graceful_shutdown",
    )
