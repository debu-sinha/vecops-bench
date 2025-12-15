"""Operational complexity scoring for vector databases.

IMPORTANT: For scientific validity, use RuntimeComplexityProber to get
MEASURED values. The hardcoded DATABASE_OPERATIONAL_PROFILES should only
be used as fallback when containers aren't running.
"""

from .complexity import (
    OperationalComplexityScore,
    OperationalMetrics,
    compute_complexity_score,
    DATABASE_OPERATIONAL_PROFILES,  # DEPRECATED: Use RuntimeComplexityProber
    generate_complexity_report,
    score_all_databases,
    DeploymentMethod,
)

from .runtime_prober import (
    RuntimeComplexityProber,
    RuntimeMetrics,
    compute_runtime_complexity_score,
)

__all__ = [
    # Runtime measurement (PREFERRED for scientific validity)
    "RuntimeComplexityProber",
    "RuntimeMetrics",
    "compute_runtime_complexity_score",
    # Legacy hardcoded (fallback only)
    "OperationalComplexityScore",
    "OperationalMetrics",
    "compute_complexity_score",
    "DATABASE_OPERATIONAL_PROFILES",
    "generate_complexity_report",
    "score_all_databases",
    "DeploymentMethod",
]
