"""
Query Plan Capture and Analysis

Captures execution plans for root cause analysis.
Explains WHY certain databases are faster/slower for specific operations.
"""

import cProfile
import io
import pstats
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json


def validate_identifier(name: str) -> str:
    """
    Validate and sanitize SQL identifier to prevent injection.

    Only allows alphanumeric characters and underscores.
    Raises ValueError if invalid.
    """
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


@dataclass
class QueryPlanResult:
    """Query execution plan result."""
    database: str
    query_type: str  # "vector_search", "filtered_search", "hybrid"
    planning_time_ms: float
    execution_time_ms: float
    rows_scanned: int
    index_used: Optional[str]
    strategy: str  # e.g., "index_scan", "seq_scan", "hnsw_search"
    raw_plan: Dict[str, Any]
    insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "query_type": self.query_type,
            "planning_time_ms": self.planning_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "rows_scanned": self.rows_scanned,
            "index_used": self.index_used,
            "strategy": self.strategy,
            "raw_plan": self.raw_plan,
            "insights": self.insights
        }


# =============================================================================
# pgvector Query Plan Capture (SQL EXPLAIN ANALYZE)
# =============================================================================

def capture_pgvector_query_plan(
    cursor,
    query_vector: List[float],
    collection_name: str,
    top_k: int = 10,
    filter_value: Optional[str] = None
) -> QueryPlanResult:
    """
    Capture PostgreSQL execution plan for pgvector query.

    This reveals WHY pgvector gets FASTER with filters (-31%).
    """
    # Validate collection name to prevent SQL injection
    safe_collection_name = validate_identifier(collection_name)

    # Build query with validated identifier
    # Note: Table names cannot be parameterized in PostgreSQL,
    # so we validate the identifier instead
    if filter_value:
        query = f"""
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT id, embedding <=> %s::vector AS distance
        FROM {safe_collection_name}
        WHERE category = %s
        ORDER BY distance
        LIMIT %s
        """
        params = (query_vector, filter_value, top_k)
    else:
        query = f"""
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT id, embedding <=> %s::vector AS distance
        FROM {safe_collection_name}
        ORDER BY distance
        LIMIT %s
        """
        params = (query_vector, top_k)

    cursor.execute(query, params)
    plan_json = cursor.fetchone()[0]

    # Parse plan
    plan = plan_json[0] if isinstance(plan_json, list) else plan_json

    planning_time = plan.get("Planning Time", 0)
    execution_time = plan.get("Execution Time", 0)

    # Extract details from plan tree
    plan_tree = plan.get("Plan", {})
    rows_scanned = plan_tree.get("Actual Rows", 0)
    index_used = _extract_index_from_plan(plan_tree)
    strategy = _determine_strategy(plan_tree)

    # Generate insights
    insights = _generate_pgvector_insights(plan_tree, filter_value is not None)

    return QueryPlanResult(
        database="pgvector",
        query_type="filtered_search" if filter_value else "vector_search",
        planning_time_ms=planning_time,
        execution_time_ms=execution_time,
        rows_scanned=rows_scanned,
        index_used=index_used,
        strategy=strategy,
        raw_plan=plan,
        insights=insights
    )


def _extract_index_from_plan(plan: Dict) -> Optional[str]:
    """Recursively find index name in plan tree."""
    if "Index Name" in plan:
        return plan["Index Name"]

    for child in plan.get("Plans", []):
        result = _extract_index_from_plan(child)
        if result:
            return result

    return None


def _determine_strategy(plan: Dict) -> str:
    """Determine the search strategy from plan."""
    node_type = plan.get("Node Type", "")

    if "Index" in node_type:
        if "Scan" in node_type:
            return "index_scan"
        return "index_lookup"
    elif "Seq Scan" in node_type:
        return "seq_scan"
    elif "Sort" in node_type:
        return "sort_then_limit"

    return node_type.lower().replace(" ", "_")


def _generate_pgvector_insights(plan: Dict, has_filter: bool) -> List[str]:
    """Generate human-readable insights from plan."""
    insights = []

    node_type = plan.get("Node Type", "")
    rows = plan.get("Actual Rows", 0)

    if has_filter:
        if "Index" in node_type:
            insights.append(
                "PostgreSQL used index-first strategy: filter by category "
                "index, then compute vector distances on reduced set."
            )
            insights.append(
                f"Only {rows} rows needed distance computation instead of full table."
            )
            insights.append(
                "This explains the -31% overhead: fewer distance calculations = faster."
            )
        else:
            insights.append(
                "PostgreSQL fell back to sequential scan with filter. "
                "Consider adding a B-tree index on the filter column."
            )
    else:
        if "Index" in node_type:
            insights.append("Using HNSW or IVFFlat index for approximate search.")
        else:
            insights.append("Full table scan - index may not be built yet.")

    return insights


# =============================================================================
# Chroma Profiling (Python cProfile)
# =============================================================================

def profile_chroma_search(
    collection,
    query_vector: List[float],
    top_k: int = 10,
    filter_dict: Optional[Dict] = None
) -> QueryPlanResult:
    """
    Profile Chroma's search execution using cProfile.

    This reveals WHY Chroma is 30x slower with filters (+2978%).
    """
    profiler = cProfile.Profile()

    # Profile the search
    profiler.enable()
    start = time.perf_counter()

    if filter_dict:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_dict
        )
    else:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

    execution_time = (time.perf_counter() - start) * 1000
    profiler.disable()

    # Analyze profile
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    profile_text = stats_stream.getvalue()

    # Extract top functions
    top_functions = _extract_top_functions(stats)

    # Generate insights
    insights = _generate_chroma_insights(top_functions, filter_dict is not None)

    return QueryPlanResult(
        database="chroma",
        query_type="filtered_search" if filter_dict else "vector_search",
        planning_time_ms=0,  # No explicit planning phase
        execution_time_ms=execution_time,
        rows_scanned=len(results.get("ids", [[]])[0]),
        index_used="HNSW",
        strategy="hnsw_then_filter" if filter_dict else "hnsw_search",
        raw_plan={"profile": profile_text, "top_functions": top_functions},
        insights=insights
    )


def _extract_top_functions(stats: pstats.Stats) -> List[Dict[str, Any]]:
    """Extract top time-consuming functions from profile."""
    top = []
    for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:10]:
        top.append({
            "function": f"{func[0]}:{func[2]}",
            "cumulative_time_ms": ct * 1000,
            "calls": nc
        })
    return sorted(top, key=lambda x: -x["cumulative_time_ms"])


def _generate_chroma_insights(
    top_functions: List[Dict],
    has_filter: bool
) -> List[str]:
    """Generate insights from Chroma profile."""
    insights = []

    if has_filter:
        # Look for filtering overhead
        filter_time = sum(
            f["cumulative_time_ms"]
            for f in top_functions
            if "filter" in f["function"].lower() or "where" in f["function"].lower()
        )
        total_time = sum(f["cumulative_time_ms"] for f in top_functions)

        if filter_time > total_time * 0.5:
            insights.append(
                f"Filtering consumes {filter_time/total_time*100:.1f}% of execution time."
            )
            insights.append(
                "Chroma performs HNSW search first, then applies Python filter "
                "to all candidates. This negates HNSW's O(log n) advantage."
            )
            insights.append(
                "For filtered search, Chroma effectively becomes O(n) - explaining "
                "the +2978% overhead."
            )
        else:
            insights.append("Filter overhead is not the primary bottleneck.")

    else:
        insights.append("Pure HNSW search without filtering overhead.")

    return insights


# =============================================================================
# Generic Adapter for Other DBs
# =============================================================================

def capture_generic_timing(
    adapter,
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10,
    filter_dict: Optional[Dict] = None,
    num_samples: int = 100
) -> QueryPlanResult:
    """
    Capture timing-based "pseudo-plan" for databases without EXPLAIN.

    Measures overhead by comparing filtered vs unfiltered.
    """
    # Measure unfiltered
    unfiltered_times = []
    for _ in range(num_samples):
        result = adapter.search(collection_name, query_vector, top_k)
        unfiltered_times.append(result.latency_ms)

    # Measure filtered if applicable
    filtered_times = []
    if filter_dict:
        for _ in range(num_samples):
            result = adapter.search(collection_name, query_vector, top_k, filter_dict)
            filtered_times.append(result.latency_ms)

    import numpy as np
    avg_unfiltered = np.mean(unfiltered_times)
    avg_filtered = np.mean(filtered_times) if filtered_times else None

    insights = []
    if avg_filtered:
        overhead = (avg_filtered / avg_unfiltered - 1) * 100
        if overhead < 0:
            insights.append(
                f"Filtered search is {-overhead:.1f}% FASTER. "
                "Likely using pre-filtering or selectivity-aware optimization."
            )
        elif overhead > 100:
            insights.append(
                f"Filtered search has {overhead:.1f}% overhead. "
                "Likely post-filtering after full ANN search."
            )
        else:
            insights.append(f"Moderate filter overhead: {overhead:.1f}%")

    return QueryPlanResult(
        database=adapter.name,
        query_type="filtered_search" if filter_dict else "vector_search",
        planning_time_ms=0,
        execution_time_ms=avg_filtered if filter_dict else avg_unfiltered,
        rows_scanned=0,  # Unknown without introspection
        index_used="HNSW",  # Assumed
        strategy="unknown",
        raw_plan={
            "unfiltered_latency_ms": avg_unfiltered,
            "filtered_latency_ms": avg_filtered,
            "overhead_percent": (avg_filtered / avg_unfiltered - 1) * 100 if avg_filtered else None
        },
        insights=insights
    )
