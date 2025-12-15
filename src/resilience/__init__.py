"""
Resilience Testing Module

Replaces the hardcoded "Operational Complexity" with time-based measurements.

The 3 Pillars of Operations:
1. Cold Start Latency (TTFT) - Time to first token/query
2. Crash Recovery Time (TTR) - Time to recover from kill -9
3. Ingestion Speed - Vectors per second during bulk load
"""

from .cold_start import ColdStartResult, measure_cold_start
from .crash_recovery import CrashRecoveryResult, measure_crash_recovery
from .ingestion_speed import IngestionResult, measure_ingestion_speed
from .memory_pressure import MemoryPressureResult, run_memory_constrained_benchmark

__all__ = [
    "measure_cold_start",
    "ColdStartResult",
    "measure_crash_recovery",
    "CrashRecoveryResult",
    "run_memory_constrained_benchmark",
    "MemoryPressureResult",
    "measure_ingestion_speed",
    "IngestionResult",
]
