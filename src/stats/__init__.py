"""Statistical Analysis Module."""

from .statistical_analysis import (
    ComparisonResult,
    StatisticalResult,
    aggregate_trial_results,
    anova_multiple_databases,
    check_normality,
    cohens_d,
    compare_databases,
    compute_confidence_interval,
    compute_stats_for_trials,
    format_result_with_ci,
    generate_latex_table,
    mann_whitney_u,
    two_sample_ttest,
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
