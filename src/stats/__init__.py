"""Statistical Analysis Module."""

from .statistical_analysis import (
    compute_confidence_interval,
    compute_stats_for_trials,
    check_normality,
    two_sample_ttest,
    mann_whitney_u,
    cohens_d,
    compare_databases,
    anova_multiple_databases,
    aggregate_trial_results,
    format_result_with_ci,
    generate_latex_table,
    StatisticalResult,
    ComparisonResult,
)

__all__ = [
    "compute_confidence_interval",
    "compute_stats_for_trials",
    "check_normality",
    "two_sample_ttest",
    "mann_whitney_u",
    "cohens_d",
    "compare_databases",
    "anova_multiple_databases",
    "aggregate_trial_results",
    "format_result_with_ci",
    "generate_latex_table",
    "StatisticalResult",
    "ComparisonResult",
]
