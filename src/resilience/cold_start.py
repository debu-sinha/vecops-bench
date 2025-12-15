"""
Cold Start Latency Measurement (TTFT - Time to First Token)

Measures the time from container start to first successful query.
Critical for serverless and auto-scaling deployments.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ColdStartResult:
    """
    Results from cold start measurement.

    We separate measurements into:
    - container_start_ms: Time for Docker container to start (orchestration overhead)
    - api_ready_ms: Time from container running to API healthy (database initialization)
    - first_query_ms: Time for first query after API ready (index loading/warmup)
    - total_ms: Sum of all phases

    This separation allows reviewers to understand WHERE time is spent.
    """
    database: str
    mean_ms: float  # Total time (legacy compatibility)
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    trials: List[float]
    # Breakdown by phase (new in v2)
    container_start_ms: float = 0.0
    api_ready_ms: float = 0.0
    first_query_ms: float = 0.0
    methodology: str = "container_restart"

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
            "methodology": self.methodology,
            # Breakdown (averages across trials)
            "breakdown": {
                "container_start_ms": self.container_start_ms,
                "api_ready_ms": self.api_ready_ms,
                "first_query_ms": self.first_query_ms,
            }
        }


def get_container_id(db_name: str) -> Optional[str]:
    """Get Docker container ID for a database."""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={db_name}"],
        capture_output=True,
        text=True
    )
    container_id = result.stdout.strip()
    return container_id if container_id else None


def wait_for_healthy(db_name: str, timeout_seconds: float = 60) -> float:
    """
    Wait for container to be healthy and return time taken.

    Returns time in seconds.
    """
    start = time.perf_counter()
    deadline = start + timeout_seconds

    while time.perf_counter() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", db_name],
            capture_output=True,
            text=True
        )
        status = result.stdout.strip()

        if status == "healthy":
            return time.perf_counter() - start

        # Also check if running without health check
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", db_name],
            capture_output=True,
            text=True
        )
        if result.stdout.strip() == "running" and status == "":
            # No health check defined, assume healthy after short delay
            time.sleep(0.5)
            return time.perf_counter() - start

        time.sleep(0.1)

    raise TimeoutError(f"Container {db_name} not healthy after {timeout_seconds}s")


def execute_test_query(adapter, collection_name: str, query_vector: List[float]) -> float:
    """Execute a test query and return latency in ms."""
    result = adapter.search(collection_name, query_vector, top_k=10)
    return result.latency_ms


def measure_cold_start(
    db_name: str,
    adapter,
    collection_name: str,
    query_vector: List[float],
    num_trials: int = 10,
    include_data_load: bool = False
) -> ColdStartResult:
    """
    Measure cold start latency with phase breakdown.

    This measures the time from container restart to first successful query,
    broken down into three phases:
    1. Container start: Docker starting the container
    2. API ready: Database initializing until health check passes
    3. First query: Time to execute first query (includes index loading)

    Args:
        db_name: Name of the database container
        adapter: Database adapter instance
        collection_name: Name of the collection/index
        query_vector: Sample query vector
        num_trials: Number of restart trials
        include_data_load: If True, include index loading time

    Returns:
        ColdStartResult with statistics and phase breakdown
    """
    total_times = []
    container_start_times = []
    api_ready_times = []
    first_query_times = []

    for trial in range(num_trials):
        # Stop container
        subprocess.run(
            ["docker", "stop", db_name],
            capture_output=True,
            check=True
        )

        # Disconnect adapter
        try:
            adapter.disconnect()
        except Exception:
            pass

        # Phase 1: Container start
        phase1_start = time.perf_counter()

        subprocess.run(
            ["docker", "start", db_name],
            capture_output=True,
            check=True
        )

        # Container is now "running" but may not be healthy
        phase1_end = time.perf_counter()
        container_start_ms = (phase1_end - phase1_start) * 1000

        # Phase 2: Wait for API ready (health check)
        phase2_start = time.perf_counter()
        try:
            wait_for_healthy(db_name)
            phase2_end = time.perf_counter()
            api_ready_ms = (phase2_end - phase2_start) * 1000
        except TimeoutError:
            api_ready_ms = 60000  # Timeout

        # Phase 3: First query
        phase3_start = time.perf_counter()
        try:
            adapter.connect()
            result = adapter.search(collection_name, query_vector, top_k=10)
            phase3_end = time.perf_counter()
            first_query_ms = (phase3_end - phase3_start) * 1000
        except Exception as e:
            first_query_ms = 60000  # Timeout marker

        # Record all times
        total_ms = container_start_ms + api_ready_ms + first_query_ms
        total_times.append(total_ms)
        container_start_times.append(container_start_ms)
        api_ready_times.append(api_ready_ms)
        first_query_times.append(first_query_ms)

    times_np = np.array(total_times)

    return ColdStartResult(
        database=db_name,
        mean_ms=float(np.mean(times_np)),
        std_ms=float(np.std(times_np)),
        min_ms=float(np.min(times_np)),
        max_ms=float(np.max(times_np)),
        p50_ms=float(np.percentile(times_np, 50)),
        p95_ms=float(np.percentile(times_np, 95)),
        p99_ms=float(np.percentile(times_np, 99)),
        trials=total_times,
        # Phase breakdowns (averages)
        container_start_ms=float(np.mean(container_start_times)),
        api_ready_ms=float(np.mean(api_ready_times)),
        first_query_ms=float(np.mean(first_query_times)),
    )


def measure_cold_start_without_docker(
    adapter,
    collection_name: str,
    query_vector: List[float],
    num_trials: int = 10
) -> ColdStartResult:
    """
    Measure cold start at the adapter level (without Docker restart).

    Useful for embedded databases or when Docker control isn't available.
    Measures disconnect -> reconnect -> first query time.
    """
    times = []

    for _ in range(num_trials):
        # Disconnect
        adapter.disconnect()

        # Measure reconnect + query
        start = time.perf_counter()
        adapter.connect()
        result = adapter.search(collection_name, query_vector, top_k=10)
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)

    times_np = np.array(times)

    return ColdStartResult(
        database=adapter.name,
        mean_ms=float(np.mean(times_np)),
        std_ms=float(np.std(times_np)),
        min_ms=float(np.min(times_np)),
        max_ms=float(np.max(times_np)),
        p50_ms=float(np.percentile(times_np, 50)),
        p95_ms=float(np.percentile(times_np, 95)),
        p99_ms=float(np.percentile(times_np, 99)),
        trials=times,
        methodology="adapter_reconnect"
    )
