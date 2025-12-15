"""
Memory Pressure Testing

Tests database behavior under artificial RAM constraints.
This reveals which databases gracefully degrade vs crash under memory pressure.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class MemoryPressureResult:
    """Results from memory pressure test."""
    database: str
    memory_limit: str
    status: str  # "success", "degraded", "crashed", "oom"
    qps: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p99_ms: Optional[float]
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "memory_limit": self.memory_limit,
            "status": self.status,
            "qps": self.qps,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "error": self.error
        }


def restart_container_with_memory_limit(
    container_name: str,
    memory_limit: str,
    compose_file: str = "docker-compose.yaml"
) -> bool:
    """
    Restart container with new memory limit.

    Uses docker update for running containers or recreates with limit.
    """
    # Stop existing container
    subprocess.run(
        ["docker", "stop", container_name],
        capture_output=True
    )

    # Update memory limit and restart
    result = subprocess.run(
        ["docker", "update", f"--memory={memory_limit}",
         f"--memory-swap={memory_limit}", container_name],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # Container might need recreation - use docker-compose
        # Set env var and recreate
        env = {"MEMORY_LIMIT": memory_limit}
        subprocess.run(
            ["docker-compose", "-f", compose_file, "up", "-d", container_name],
            capture_output=True,
            env={**subprocess.os.environ, **env}
        )

    # Start container
    subprocess.run(
        ["docker", "start", container_name],
        capture_output=True
    )

    # Wait for container to be ready
    time.sleep(5)

    # Check if container is still running (didn't OOM immediately)
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True
    )

    return result.stdout.strip() == "true"


def run_memory_constrained_benchmark(
    db_name: str,
    adapter,
    collection_name: str,
    query_vectors: List[List[float]],
    memory_limits: List[str] = ["64g", "32g", "16g", "8g", "4g"],
    qps_duration: float = 30.0
) -> Dict[str, MemoryPressureResult]:
    """
    Test database behavior under varying memory constraints.

    Args:
        db_name: Container name
        adapter: Database adapter
        collection_name: Collection to query
        query_vectors: Test query vectors
        memory_limits: List of memory limits to test (descending)
        qps_duration: Duration for QPS measurement

    Returns:
        Dict mapping memory limit to result
    """
    results = {}

    for mem_limit in memory_limits:
        print(f"  Testing {db_name} with --memory={mem_limit}")

        try:
            # Restart with new memory limit
            is_running = restart_container_with_memory_limit(db_name, mem_limit)

            if not is_running:
                results[mem_limit] = MemoryPressureResult(
                    database=db_name,
                    memory_limit=mem_limit,
                    status="crashed",
                    qps=None,
                    latency_p50_ms=None,
                    latency_p99_ms=None,
                    error="Container failed to start (likely OOM)"
                )
                continue

            # Reconnect adapter
            adapter.disconnect()
            adapter.connect()

            # Run QPS benchmark
            qps, latencies = adapter.benchmark_qps(
                collection_name,
                query_vectors,
                top_k=10,
                duration_seconds=qps_duration
            )

            latencies_np = np.array(latencies)

            # Check for degradation (p99 > 10x p50)
            p50 = np.percentile(latencies_np, 50)
            p99 = np.percentile(latencies_np, 99)

            status = "success"
            if p99 > p50 * 10:
                status = "degraded"

            results[mem_limit] = MemoryPressureResult(
                database=db_name,
                memory_limit=mem_limit,
                status=status,
                qps=qps,
                latency_p50_ms=float(p50),
                latency_p99_ms=float(p99),
                error=None
            )

        except MemoryError as e:
            results[mem_limit] = MemoryPressureResult(
                database=db_name,
                memory_limit=mem_limit,
                status="oom",
                qps=None,
                latency_p50_ms=None,
                latency_p99_ms=None,
                error=str(e)
            )

        except Exception as e:
            results[mem_limit] = MemoryPressureResult(
                database=db_name,
                memory_limit=mem_limit,
                status="error",
                qps=None,
                latency_p50_ms=None,
                latency_p99_ms=None,
                error=str(e)
            )

    return results


def find_minimum_viable_memory(
    db_name: str,
    adapter,
    collection_name: str,
    query_vectors: List[List[float]],
    min_qps: float = 10.0,
    max_p99_ms: float = 1000.0
) -> str:
    """
    Binary search to find minimum memory for acceptable performance.

    Returns the smallest memory limit that achieves min_qps and max_p99.
    """
    # Memory limits in MB
    memory_values = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for mem_mb in memory_values:
        mem_limit = f"{mem_mb}m"
        try:
            is_running = restart_container_with_memory_limit(db_name, mem_limit)
            if not is_running:
                continue

            adapter.disconnect()
            adapter.connect()

            qps, latencies = adapter.benchmark_qps(
                collection_name,
                query_vectors[:100],  # Quick test
                top_k=10,
                duration_seconds=10
            )

            p99 = np.percentile(latencies, 99)

            if qps >= min_qps and p99 <= max_p99_ms:
                return mem_limit

        except Exception:
            continue

    return "128g"  # No limit found, needs full RAM
