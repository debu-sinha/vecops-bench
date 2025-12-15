"""Workload generators for realistic query patterns."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
from numpy.random import default_rng


@dataclass
class QueryInstance:
    """A single query instance in a workload."""
    query_id: str
    vector: List[float]
    text: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    top_k: int = 10
    expected_ids: List[str] = field(default_factory=list)
    timestamp: float = 0.0  # For timing-based workloads


class WorkloadGenerator(ABC):
    """Abstract base class for workload generators."""

    def __init__(self, seed: int = 42):
        self.rng = default_rng(seed)
        self.seed = seed

    @abstractmethod
    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate query instances."""
        pass

    def get_name(self) -> str:
        """Get workload name."""
        return self.__class__.__name__


class UniformWorkload(WorkloadGenerator):
    """Uniform random query selection."""

    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        top_k: int = 10,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate uniformly distributed queries."""
        indices = self.rng.choice(len(query_ids), size=num_queries, replace=True)

        for i, idx in enumerate(indices):
            yield QueryInstance(
                query_id=query_ids[idx],
                vector=query_vectors[idx].tolist(),
                top_k=top_k,
                timestamp=time.time()
            )


class BurstyWorkload(WorkloadGenerator):
    """Bursty traffic pattern using Poisson arrivals."""

    def __init__(self, seed: int = 42, rate: float = 10.0, burst_factor: float = 5.0):
        """
        Initialize bursty workload.

        Args:
            rate: Base queries per second
            burst_factor: Multiplier during burst periods
        """
        super().__init__(seed)
        self.rate = rate
        self.burst_factor = burst_factor

    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        top_k: int = 10,
        burst_probability: float = 0.2,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate bursty query pattern."""
        current_time = 0.0
        in_burst = False
        queries_generated = 0

        while queries_generated < num_queries:
            # Determine if we're in a burst
            if self.rng.random() < 0.05:  # 5% chance to toggle burst
                in_burst = not in_burst

            # Compute inter-arrival time
            current_rate = self.rate * self.burst_factor if in_burst else self.rate
            inter_arrival = self.rng.exponential(1.0 / current_rate)
            current_time += inter_arrival

            # Select random query
            idx = self.rng.integers(len(query_ids))

            yield QueryInstance(
                query_id=query_ids[idx],
                vector=query_vectors[idx].tolist(),
                top_k=top_k,
                timestamp=current_time
            )

            queries_generated += 1


class FilteredWorkload(WorkloadGenerator):
    """Workload with metadata filtering."""

    def __init__(
        self,
        seed: int = 42,
        filter_fields: Optional[List[str]] = None,
        filter_values: Optional[Dict[str, List[Any]]] = None
    ):
        """
        Initialize filtered workload.

        Args:
            filter_fields: List of metadata fields to filter on
            filter_values: Possible values for each filter field
        """
        super().__init__(seed)
        self.filter_fields = filter_fields or ["category", "year"]
        self.filter_values = filter_values or {
            "category": ["A", "B", "C", "D"],
            "year": [2020, 2021, 2022, 2023, 2024]
        }

    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        top_k: int = 10,
        filter_probability: float = 0.5,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate queries with varying filter complexity."""
        for i in range(num_queries):
            idx = self.rng.integers(len(query_ids))

            # Decide whether to add filter
            query_filter = None
            if self.rng.random() < filter_probability:
                # Random number of filter conditions (1-3)
                num_conditions = self.rng.integers(1, min(4, len(self.filter_fields) + 1))
                fields_to_use = self.rng.choice(
                    self.filter_fields,
                    size=num_conditions,
                    replace=False
                )

                query_filter = {}
                for field in fields_to_use:
                    if field in self.filter_values:
                        value = self.rng.choice(self.filter_values[field])
                        query_filter[field] = {"$eq": value}

            yield QueryInstance(
                query_id=query_ids[idx],
                vector=query_vectors[idx].tolist(),
                filter=query_filter,
                top_k=top_k,
                timestamp=time.time()
            )


class SkewedWorkload(WorkloadGenerator):
    """Zipfian (power-law) query distribution - some queries are much more popular."""

    def __init__(self, seed: int = 42, alpha: float = 1.2):
        """
        Initialize skewed workload.

        Args:
            alpha: Zipf distribution parameter (higher = more skewed)
        """
        super().__init__(seed)
        self.alpha = alpha

    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        top_k: int = 10,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate queries following Zipfian distribution."""
        n = len(query_ids)

        # Pre-compute Zipf probabilities
        ranks = np.arange(1, n + 1)
        weights = 1.0 / np.power(ranks, self.alpha)
        probabilities = weights / weights.sum()

        # Generate indices according to Zipf distribution
        indices = self.rng.choice(n, size=num_queries, p=probabilities)

        for idx in indices:
            yield QueryInstance(
                query_id=query_ids[idx],
                vector=query_vectors[idx].tolist(),
                top_k=top_k,
                timestamp=time.time()
            )


class TemporalDriftWorkload(WorkloadGenerator):
    """Workload that simulates query distribution drift over time."""

    def __init__(self, seed: int = 42, drift_rate: float = 0.1):
        """
        Initialize temporal drift workload.

        Args:
            drift_rate: How quickly the query distribution shifts
        """
        super().__init__(seed)
        self.drift_rate = drift_rate

    def generate(
        self,
        query_vectors: np.ndarray,
        query_ids: List[str],
        num_queries: int,
        top_k: int = 10,
        **kwargs
    ) -> Generator[QueryInstance, None, None]:
        """Generate queries with drifting distribution."""
        n = len(query_ids)

        # Initialize weights (start uniform)
        weights = np.ones(n) / n

        for i in range(num_queries):
            # Sample according to current distribution
            idx = self.rng.choice(n, p=weights)

            yield QueryInstance(
                query_id=query_ids[idx],
                vector=query_vectors[idx].tolist(),
                top_k=top_k,
                timestamp=time.time()
            )

            # Drift the distribution
            # Increase weight for recently queried items, decrease for others
            weights *= (1 - self.drift_rate)
            weights[idx] += self.drift_rate

            # Re-normalize
            weights /= weights.sum()


def generate_workload(
    workload_type: str,
    query_vectors: np.ndarray,
    query_ids: List[str],
    num_queries: int,
    seed: int = 42,
    **kwargs
) -> Generator[QueryInstance, None, None]:
    """
    Factory function to generate workloads.

    Args:
        workload_type: "uniform", "bursty", "filtered", "skewed", or "drift"
        query_vectors: Array of query vectors
        query_ids: List of query IDs
        num_queries: Number of queries to generate
        seed: Random seed
        **kwargs: Additional arguments for specific workload types

    Returns:
        Generator yielding QueryInstance objects
    """
    workloads = {
        "uniform": UniformWorkload,
        "bursty": BurstyWorkload,
        "filtered": FilteredWorkload,
        "skewed": SkewedWorkload,
        "drift": TemporalDriftWorkload,
    }

    if workload_type.lower() not in workloads:
        raise ValueError(f"Unknown workload type: {workload_type}. "
                        f"Available: {list(workloads.keys())}")

    generator_class = workloads[workload_type.lower()]

    # Extract class-specific kwargs
    if workload_type.lower() == "bursty":
        generator = generator_class(
            seed=seed,
            rate=kwargs.get("rate", 10.0),
            burst_factor=kwargs.get("burst_factor", 5.0)
        )
    elif workload_type.lower() == "filtered":
        generator = generator_class(
            seed=seed,
            filter_fields=kwargs.get("filter_fields"),
            filter_values=kwargs.get("filter_values")
        )
    elif workload_type.lower() == "skewed":
        generator = generator_class(
            seed=seed,
            alpha=kwargs.get("alpha", 1.2)
        )
    elif workload_type.lower() == "drift":
        generator = generator_class(
            seed=seed,
            drift_rate=kwargs.get("drift_rate", 0.1)
        )
    else:
        generator = generator_class(seed=seed)

    return generator.generate(
        query_vectors=query_vectors,
        query_ids=query_ids,
        num_queries=num_queries,
        **kwargs
    )
