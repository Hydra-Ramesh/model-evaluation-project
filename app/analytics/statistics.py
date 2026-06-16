import numpy as np
from scipy import stats
from typing import Dict, Any, List

class StatisticalAnalyzer:
    """
    Computes statistical significance and confidence intervals for benchmark results.
    Used to rigorously determine if Model A is truly better than Model B.
    """
    
    @staticmethod
    def t_test_compare(scores_model_a: List[float], scores_model_b: List[float]) -> Dict[str, float]:
        """Independent Welch's t-test for comparing two models' continuous scores."""
        if not scores_model_a or not scores_model_b:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant_at_05": False}
            
        t_stat, p_value = stats.ttest_ind(scores_model_a, scores_model_b, equal_var=False)
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_05": bool(p_value < 0.05)
        }

    @staticmethod
    def mann_whitney_u(scores_model_a: List[float], scores_model_b: List[float]) -> Dict[str, float]:
        """Non-parametric Mann-Whitney U test, robust for non-normal distributions."""
        if not scores_model_a or not scores_model_b:
            return {"u_statistic": 0.0, "p_value": 1.0, "significant_at_05": False}
            
        u_stat, p_value = stats.mannwhitneyu(scores_model_a, scores_model_b, alternative='two-sided')
        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant_at_05": bool(p_value < 0.05)
        }

    @staticmethod
    def anova_compare(*args) -> Dict[str, float]:
        """One-way ANOVA to test differences across multiple models concurrently."""
        if len(args) < 2:
            return {"f_statistic": 0.0, "p_value": 1.0, "significant_at_05": False}
            
        f_stat, p_value = stats.f_oneway(*args)
        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant_at_05": bool(p_value < 0.05)
        }

    @staticmethod
    def bootstrap_confidence_interval(data: List[float], num_resamples: int = 1000, alpha: float = 0.05) -> Dict[str, float]:
        """Calculates bootstrap confidence intervals for the mean accuracy/score."""
        if not data:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            
        data_arr = np.array(data)
        resamples = np.random.choice(data_arr, size=(num_resamples, len(data_arr)), replace=True)
        means = np.mean(resamples, axis=1)
        lower_bound = np.percentile(means, 100 * (alpha / 2))
        upper_bound = np.percentile(means, 100 * (1 - alpha / 2))
        
        return {
            "mean": float(np.mean(data_arr)),
            "ci_lower": float(lower_bound),
            "ci_upper": float(upper_bound)
        }
