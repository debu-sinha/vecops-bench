"""
Cost Tracking for Vector Database Benchmarking.

This module tracks and models the costs associated with running
vector databases, enabling cost-normalized performance metrics.

Key metrics:
- $/million queries
- $/recall point (cost to achieve specific recall level)
- Cost-quality Pareto frontiers
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import psutil


@dataclass
class ResourceUsage:
    """Captures resource usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class CostModel:
    """
    Cost model for a vector database deployment.

    Supports both cloud (hourly pricing) and self-hosted (resource-based) costing.
    """
    name: str

    # Cloud pricing (if applicable)
    hourly_rate_usd: float = 0.0          # $/hour for managed service
    per_million_queries_usd: float = 0.0  # $/million queries (serverless)
    storage_per_gb_month_usd: float = 0.0 # $/GB/month for storage

    # Self-hosted pricing (compute costs)
    compute_hourly_rate_usd: float = 0.0  # $/hour for compute
    memory_per_gb_hour_usd: float = 0.0   # $/GB/hour for memory

    # Infrastructure constants
    instance_type: str = "unknown"
    vcpus: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0

    # Operational costs
    setup_time_hours: float = 0.0        # Time to deploy
    maintenance_hours_per_month: float = 0.0  # Ongoing maintenance

    def compute_hourly_cost(self, memory_usage_gb: float = 0) -> float:
        """Compute hourly cost based on model."""
        if self.hourly_rate_usd > 0:
            # Managed service pricing
            return self.hourly_rate_usd
        else:
            # Self-hosted pricing
            return (
                self.compute_hourly_rate_usd +
                memory_usage_gb * self.memory_per_gb_hour_usd
            )

    def compute_query_cost(self, num_queries: int, duration_hours: float) -> float:
        """Compute cost for a number of queries over a duration."""
        if self.per_million_queries_usd > 0:
            # Serverless pricing
            return (num_queries / 1_000_000) * self.per_million_queries_usd
        else:
            # Time-based pricing
            return self.compute_hourly_cost() * duration_hours


# Pre-defined cost models for common deployments
COST_MODELS = {
    # Managed cloud services (approximate pricing as of 2024)
    "pinecone_serverless": CostModel(
        name="Pinecone Serverless",
        per_million_queries_usd=2.00,  # ~$2/million queries
        storage_per_gb_month_usd=0.33,
        instance_type="serverless",
    ),
    "pinecone_standard": CostModel(
        name="Pinecone Standard",
        hourly_rate_usd=0.096,  # p1.x1 pod
        storage_per_gb_month_usd=0.33,
        instance_type="p1.x1",
        vcpus=2,
        memory_gb=8,
    ),

    # Self-hosted on cloud compute (AWS pricing)
    "self_hosted_small": CostModel(
        name="Self-hosted (Small)",
        compute_hourly_rate_usd=0.0832,  # t3.large
        memory_per_gb_hour_usd=0.01,
        instance_type="t3.large",
        vcpus=2,
        memory_gb=8,
        setup_time_hours=2.0,
        maintenance_hours_per_month=4.0,
    ),
    "self_hosted_medium": CostModel(
        name="Self-hosted (Medium)",
        compute_hourly_rate_usd=0.166,  # t3.xlarge
        memory_per_gb_hour_usd=0.01,
        instance_type="t3.xlarge",
        vcpus=4,
        memory_gb=16,
        setup_time_hours=2.0,
        maintenance_hours_per_month=4.0,
    ),
    "self_hosted_large": CostModel(
        name="Self-hosted (Large)",
        compute_hourly_rate_usd=0.333,  # t3.2xlarge
        memory_per_gb_hour_usd=0.01,
        instance_type="t3.2xlarge",
        vcpus=8,
        memory_gb=32,
        setup_time_hours=4.0,
        maintenance_hours_per_month=8.0,
    ),

    # Free tier / local development
    "local_development": CostModel(
        name="Local Development",
        compute_hourly_rate_usd=0.0,
        instance_type="local",
    ),
}


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a benchmark run."""
    database: str
    cost_model: str
    duration_seconds: float
    num_queries: int

    # Computed costs
    compute_cost_usd: float
    storage_cost_usd: float
    query_cost_usd: float
    total_cost_usd: float

    # Cost efficiency metrics
    cost_per_query_usd: float
    cost_per_million_queries_usd: float
    queries_per_dollar: float

    # Resource utilization
    avg_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float

    # Performance-normalized costs
    cost_per_recall_point: Optional[float] = None  # $/0.01 recall improvement
    cost_per_qps: Optional[float] = None  # $/QPS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database": self.database,
            "cost_model": self.cost_model,
            "duration_seconds": self.duration_seconds,
            "num_queries": self.num_queries,
            "compute_cost_usd": self.compute_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "query_cost_usd": self.query_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_query_usd": self.cost_per_query_usd,
            "cost_per_million_queries_usd": self.cost_per_million_queries_usd,
            "queries_per_dollar": self.queries_per_dollar,
            "avg_cpu_percent": self.avg_cpu_percent,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "cost_per_recall_point": self.cost_per_recall_point,
            "cost_per_qps": self.cost_per_qps,
        }


class CostTracker:
    """
    Tracks resource usage and costs during benchmark execution.

    Usage:
        tracker = CostTracker("pinecone_serverless")
        tracker.start()
        # ... run benchmark ...
        tracker.record_queries(1000)
        tracker.stop()
        breakdown = tracker.get_cost_breakdown()
    """

    def __init__(
        self,
        cost_model_name: str = "local_development",
        custom_cost_model: Optional[CostModel] = None,
        database_name: str = "unknown"
    ):
        """
        Initialize cost tracker.

        Args:
            cost_model_name: Key from COST_MODELS dict
            custom_cost_model: Override with custom cost model
            database_name: Name of database being tracked
        """
        if custom_cost_model:
            self.cost_model = custom_cost_model
        elif cost_model_name in COST_MODELS:
            self.cost_model = COST_MODELS[cost_model_name]
        else:
            self.cost_model = COST_MODELS["local_development"]

        self.database_name = database_name
        self.cost_model_name = cost_model_name

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._resource_samples: List[ResourceUsage] = []
        self._query_count: int = 0
        self._storage_gb: float = 0.0

        # Baseline resource usage
        self._baseline_disk_read = 0
        self._baseline_disk_write = 0
        self._baseline_net_sent = 0
        self._baseline_net_recv = 0

    def start(self) -> None:
        """Start tracking."""
        self._start_time = time.perf_counter()
        self._resource_samples = []
        self._query_count = 0

        # Record baseline I/O counters
        try:
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            self._baseline_disk_read = disk_io.read_bytes if disk_io else 0
            self._baseline_disk_write = disk_io.write_bytes if disk_io else 0
            self._baseline_net_sent = net_io.bytes_sent if net_io else 0
            self._baseline_net_recv = net_io.bytes_recv if net_io else 0
        except Exception:
            pass

        # Record initial sample
        self._sample_resources()

    def stop(self) -> None:
        """Stop tracking."""
        self._end_time = time.perf_counter()
        self._sample_resources()

    def record_queries(self, count: int) -> None:
        """Record number of queries executed."""
        self._query_count += count

    def set_storage_gb(self, gb: float) -> None:
        """Set storage usage in GB."""
        self._storage_gb = gb

    def sample(self) -> None:
        """Take a resource usage sample (call periodically during benchmark)."""
        self._sample_resources()

    def _sample_resources(self) -> None:
        """Internal: sample current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            sample = ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_read_mb=(disk_io.read_bytes - self._baseline_disk_read) / (1024 * 1024) if disk_io else 0,
                disk_write_mb=(disk_io.write_bytes - self._baseline_disk_write) / (1024 * 1024) if disk_io else 0,
                network_sent_mb=(net_io.bytes_sent - self._baseline_net_sent) / (1024 * 1024) if net_io else 0,
                network_recv_mb=(net_io.bytes_recv - self._baseline_net_recv) / (1024 * 1024) if net_io else 0,
            )
            self._resource_samples.append(sample)
        except Exception:
            pass

    def get_cost_breakdown(
        self,
        recall: Optional[float] = None,
        qps: Optional[float] = None
    ) -> CostBreakdown:
        """
        Compute cost breakdown for the tracked period.

        Args:
            recall: Optional recall score for cost-per-recall computation
            qps: Optional QPS for cost-per-QPS computation

        Returns:
            CostBreakdown with all cost metrics
        """
        if self._start_time is None:
            raise ValueError("Tracking not started")

        end_time = self._end_time or time.perf_counter()
        duration_seconds = end_time - self._start_time
        duration_hours = duration_seconds / 3600

        # Compute costs
        compute_cost = self.cost_model.compute_hourly_cost() * duration_hours
        storage_cost = self._storage_gb * self.cost_model.storage_per_gb_month_usd / (30 * 24) * duration_hours
        query_cost = self.cost_model.compute_query_cost(self._query_count, duration_hours)

        # For serverless, query cost is primary; for self-hosted, compute cost is primary
        if self.cost_model.per_million_queries_usd > 0:
            total_cost = query_cost + storage_cost
        else:
            total_cost = compute_cost + storage_cost

        # Cost efficiency
        cost_per_query = total_cost / self._query_count if self._query_count > 0 else 0
        cost_per_million = cost_per_query * 1_000_000
        queries_per_dollar = 1 / cost_per_query if cost_per_query > 0 else float('inf')

        # Resource utilization
        avg_cpu = sum(s.cpu_percent for s in self._resource_samples) / len(self._resource_samples) if self._resource_samples else 0
        avg_memory = sum(s.memory_mb for s in self._resource_samples) / len(self._resource_samples) if self._resource_samples else 0
        peak_memory = max((s.memory_mb for s in self._resource_samples), default=0)

        # Performance-normalized costs
        cost_per_recall = None
        if recall is not None and recall > 0:
            cost_per_recall = total_cost / (recall * 100)  # Cost per recall percentage point

        cost_per_qps = None
        if qps is not None and qps > 0:
            cost_per_qps = total_cost / qps

        return CostBreakdown(
            database=self.database_name,
            cost_model=self.cost_model_name,
            duration_seconds=duration_seconds,
            num_queries=self._query_count,
            compute_cost_usd=compute_cost,
            storage_cost_usd=storage_cost,
            query_cost_usd=query_cost,
            total_cost_usd=total_cost,
            cost_per_query_usd=cost_per_query,
            cost_per_million_queries_usd=cost_per_million,
            queries_per_dollar=queries_per_dollar,
            avg_cpu_percent=avg_cpu,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            cost_per_recall_point=cost_per_recall,
            cost_per_qps=cost_per_qps,
        )


def estimate_monthly_cost(
    cost_model: CostModel,
    queries_per_day: int,
    storage_gb: float,
    hours_per_day: float = 24.0
) -> Dict[str, float]:
    """
    Estimate monthly cost for a deployment.

    Args:
        cost_model: Cost model to use
        queries_per_day: Expected queries per day
        storage_gb: Storage requirement in GB
        hours_per_day: Hours of operation per day

    Returns:
        Dict with cost breakdown
    """
    days_per_month = 30

    # Compute costs
    compute_monthly = cost_model.compute_hourly_cost() * hours_per_day * days_per_month
    storage_monthly = storage_gb * cost_model.storage_per_gb_month_usd
    query_monthly = (queries_per_day * days_per_month / 1_000_000) * cost_model.per_million_queries_usd

    # Operational costs (engineer time)
    engineer_hourly_rate = 75.0  # Assumed $/hour for engineer
    operational_monthly = cost_model.maintenance_hours_per_month * engineer_hourly_rate

    total = compute_monthly + storage_monthly + query_monthly + operational_monthly

    return {
        "compute_monthly_usd": compute_monthly,
        "storage_monthly_usd": storage_monthly,
        "query_monthly_usd": query_monthly,
        "operational_monthly_usd": operational_monthly,
        "total_monthly_usd": total,
        "cost_per_query_usd": total / (queries_per_day * days_per_month) if queries_per_day > 0 else 0,
    }
