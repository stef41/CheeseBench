"""
CheeseBench — Statistical Testing Module

Provides effect sizes, hypothesis tests, power analysis, and
learning-curve fitting for rigorous NeurIPS-level analysis.

All functions operate on EnvResult objects from analysis.py.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from analysis import EnvResult, ANIMAL_BASELINES, COGNITIVE_DIMENSIONS, ENV_COGNITIVE_MAP


# ============================================================================
# Effect Sizes
# ============================================================================

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for comparing two proportions.
    Preferred over Cohen's d for binomial (success/fail) data."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
    """
    Cliff's delta: non-parametric effect size for ordinal data.
    Returns (delta, interpretation).
    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (more - less) / (n_x * n_y)
    ad = abs(delta)
    if ad < 0.147:
        interp = "negligible"
    elif ad < 0.33:
        interp = "small"
    elif ad < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return delta, interp


# ============================================================================
# Hypothesis Tests
# ============================================================================

def proportion_z_test(
    successes_a: int, n_a: int,
    successes_b: int, n_b: int,
) -> Tuple[float, float]:
    """
    Two-proportion z-test. Returns (z_stat, p_value).
    H0: p_a = p_b.
    """
    p_a = successes_a / n_a if n_a > 0 else 0
    p_b = successes_b / n_b if n_b > 0 else 0
    p_pool = (successes_a + successes_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0

    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se > 0 else 0.0

    # Two-tailed p-value from standard normal
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return z, p_value


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy needed)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).
    Returns (U_statistic, approx_p_value).
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, 1.0

    # Count how many times x > y
    u = sum(1 for xi in x for yi in y if xi > yi) + \
        0.5 * sum(1 for xi in x for yi in y if xi == yi)

    # Normal approximation for p-value
    mu = n_x * n_y / 2
    sigma = math.sqrt(n_x * n_y * (n_x + n_y + 1) / 12)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return u, p_value


def permutation_test(
    x: List[bool], y: List[bool], n_permutations: int = 10000, seed: int = 42,
) -> float:
    """
    Exact permutation test for difference in proportions.
    Returns p-value. More robust than z-test for small samples.
    """
    rng = np.random.RandomState(seed)
    observed_diff = abs(np.mean(x) - np.mean(y))
    combined = np.array(list(x) + list(y), dtype=float)
    n_x = len(x)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(combined[:n_x].mean() - combined[n_x:].mean())
        if perm_diff >= observed_diff:
            count += 1
    return count / n_permutations


# ============================================================================
# Power Analysis
# ============================================================================

def required_sample_size(
    effect_size_h: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Required N per group to detect effect size h (Cohen's h for proportions)
    with given alpha and power.
    Formula: n = ((z_alpha + z_beta) / h)^2
    """
    if abs(effect_size_h) < 0.01:
        return 9999
    from math import ceil
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)
    n = ((z_alpha + z_beta) / effect_size_h) ** 2
    return int(ceil(n))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Beasley-Springer-Moro approximation)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0

    # Rational approximation for central region
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ============================================================================
# Learning Curve Fitting
# ============================================================================

def fit_learning_curve(
    success_by_trial: List[bool],
    block_size: int = 4,
) -> Dict:
    """
    Fit a logistic curve to trial-by-trial success data.
    Returns parameters and goodness-of-fit.

    Model: p(t) = L / (1 + exp(-k*(t - t0)))
    L = asymptote, k = learning rate, t0 = inflection point.
    """
    if len(success_by_trial) < block_size * 2:
        return {"fit": False, "reason": "too few trials"}

    # Block averages
    blocks = []
    for i in range(0, len(success_by_trial), block_size):
        chunk = success_by_trial[i:i + block_size]
        if chunk:
            blocks.append(sum(chunk) / len(chunk))

    if len(blocks) < 3:
        return {"fit": False, "reason": "too few blocks"}

    x = np.arange(len(blocks))
    y = np.array(blocks)

    # Grid search for best logistic fit (no scipy.optimize needed)
    best_mse = float("inf")
    best_params = (1.0, 0.5, len(blocks) / 2)

    for L in np.arange(0.3, 1.05, 0.1):
        for k in np.arange(0.1, 3.0, 0.2):
            for t0 in np.arange(0, len(blocks), 0.5):
                pred = L / (1 + np.exp(-k * (x - t0)))
                mse = np.mean((pred - y) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (L, k, t0)

    L, k, t0 = best_params
    pred = L / (1 + np.exp(-k * (x - t0)))
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "fit": True,
        "asymptote": round(float(L), 3),
        "learning_rate": round(float(k), 3),
        "inflection_trial": round(float(t0 * block_size), 1),
        "r_squared": round(float(r_squared), 4),
        "block_means": [round(float(b), 4) for b in blocks],
        "predicted": [round(float(p), 4) for p in pred],
    }


# ============================================================================
# Comparison Reports
# ============================================================================

@dataclass
class ComparisonResult:
    """Statistical comparison between two conditions."""
    env_name: str
    condition_a: str  # e.g. model name or "LLM"
    condition_b: str  # e.g. "Random" or "animal"
    sr_a: float
    sr_b: float
    n_a: int
    n_b: int
    effect_size_h: float
    effect_interpretation: str
    z_stat: float
    p_value: float
    significant: bool  # p < 0.05
    n_needed_for_power: int

    def to_dict(self) -> Dict:
        return {
            "env": self.env_name,
            "a": self.condition_a,
            "b": self.condition_b,
            "sr_a": round(self.sr_a, 4),
            "sr_b": round(self.sr_b, 4),
            "n_a": self.n_a,
            "n_b": self.n_b,
            "h": round(self.effect_size_h, 3),
            "interpretation": self.effect_interpretation,
            "z": round(self.z_stat, 3),
            "p": round(self.p_value, 4),
            "significant": self.significant,
            "n_needed": self.n_needed_for_power,
        }


def compare_agents(
    results_a: List[EnvResult],
    results_b: List[EnvResult],
    label_a: str = "LLM",
    label_b: str = "Random",
) -> List[ComparisonResult]:
    """
    Compare two agents across all environments.
    Returns per-environment statistical comparisons.
    """
    # Index by env (best view mode)
    def best_by_env(results):
        by_env = defaultdict(list)
        for r in results:
            by_env[r.env_name].append(r)
        return {env: max(rs, key=lambda r: r.success_rate) for env, rs in by_env.items()}

    a_by_env = best_by_env(results_a)
    b_by_env = best_by_env(results_b)

    comparisons = []
    for env_name in sorted(set(a_by_env) | set(b_by_env)):
        ra = a_by_env.get(env_name)
        rb = b_by_env.get(env_name)
        if not ra or not rb:
            continue

        succ_a = sum(1 for t in ra.trials if t.success)
        succ_b = sum(1 for t in rb.trials if t.success)
        n_a = len(ra.trials)
        n_b = len(rb.trials)

        sr_a = succ_a / n_a if n_a else 0
        sr_b = succ_b / n_b if n_b else 0

        h = cohens_h(sr_a, sr_b)
        z, p = proportion_z_test(succ_a, n_a, succ_b, n_b)
        n_needed = required_sample_size(h) if abs(h) > 0.01 else 9999

        ad = abs(h)
        if ad < 0.2:
            interp = "negligible"
        elif ad < 0.5:
            interp = "small"
        elif ad < 0.8:
            interp = "medium"
        else:
            interp = "large"

        comparisons.append(ComparisonResult(
            env_name=env_name,
            condition_a=label_a,
            condition_b=label_b,
            sr_a=sr_a, sr_b=sr_b,
            n_a=n_a, n_b=n_b,
            effect_size_h=h,
            effect_interpretation=interp,
            z_stat=z, p_value=p,
            significant=p < 0.05,
            n_needed_for_power=n_needed,
        ))

    return comparisons


def compare_to_animal(
    results: List[EnvResult],
    label: str = "LLM",
) -> List[ComparisonResult]:
    """
    Compare agent results to published animal baselines.
    Uses the final learning-curve value as the animal success rate.
    """
    by_env = defaultdict(list)
    for r in results:
        by_env[r.env_name].append(r)

    comparisons = []
    for env_name, ers in by_env.items():
        best = max(ers, key=lambda r: r.success_rate)
        baseline = ANIMAL_BASELINES.get(env_name, {})
        lc = baseline.get("learning_curve", [])
        if not lc:
            continue
        animal_sr = lc[-1]

        succ = sum(1 for t in best.trials if t.success)
        n = len(best.trials)
        sr = succ / n if n else 0

        h = cohens_h(sr, animal_sr)
        # Approximate z-test (treat animal as known proportion)
        if n > 0:
            se = math.sqrt(animal_sr * (1 - animal_sr) / n) if 0 < animal_sr < 1 else 0.01
            z = (sr - animal_sr) / se if se > 0 else 0
            p = 2 * (1 - _norm_cdf(abs(z)))
        else:
            z, p = 0, 1

        ad = abs(h)
        interp = "negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large"
        n_needed = required_sample_size(h) if abs(h) > 0.01 else 9999

        comparisons.append(ComparisonResult(
            env_name=env_name,
            condition_a=label,
            condition_b=f"Animal ({baseline.get('species', '?')})",
            sr_a=sr, sr_b=animal_sr,
            n_a=n, n_b=0,
            effect_size_h=h,
            effect_interpretation=interp,
            z_stat=z, p_value=p,
            significant=p < 0.05,
            n_needed_for_power=n_needed,
        ))

    return comparisons


def generate_stats_report(results: List[EnvResult]) -> Dict:
    """Full statistical report for all results."""
    from analysis import group_by_agent

    by_agent = group_by_agent(results)
    report = {"comparisons": {}, "learning_fits": {}, "power_analysis": {}}

    # Agent vs Random
    if "LLM" in by_agent and "Random" in by_agent:
        comps = compare_agents(by_agent["LLM"], by_agent["Random"], "LLM", "Random")
        report["comparisons"]["LLM_vs_Random"] = [c.to_dict() for c in comps]

    # Agent vs Animal
    if "LLM" in by_agent:
        comps = compare_to_animal(by_agent["LLM"], "LLM")
        report["comparisons"]["LLM_vs_Animal"] = [c.to_dict() for c in comps]

    # Learning curve fits
    for r in results:
        if r.agent_type != "LLM":
            continue
        successes = [t.success for t in r.trials]
        fit = fit_learning_curve(successes)
        report["learning_fits"][f"{r.env_name}_{r.view_mode}"] = fit

    # Power analysis summary
    if "LLM" in by_agent and "Random" in by_agent:
        comps = compare_agents(by_agent["LLM"], by_agent["Random"])
        for c in comps:
            report["power_analysis"][c.env_name] = {
                "current_n": c.n_a,
                "effect_size_h": c.effect_size_h,
                "n_needed_80pct_power": c.n_needed_for_power,
                "adequately_powered": c.n_a >= c.n_needed_for_power,
            }

    return report


def print_stats_report(results: List[EnvResult]):
    """Pretty-print statistical comparisons."""
    from analysis import group_by_agent

    by_agent = group_by_agent(results)

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    if "LLM" in by_agent and "Random" in by_agent:
        comps = compare_agents(by_agent["LLM"], by_agent["Random"])
        print(f"\n{'─' * 60}")
        print("LLM vs. Random Baseline")
        print(f"{'─' * 60}")
        print(f"{'Environment':<20} {'LLM':>6} {'Rand':>6} {'h':>6} {'Effect':>10} {'p':>8} {'Sig':>4}")
        print("-" * 66)
        for c in comps:
            sig = "***" if c.p_value < 0.001 else "**" if c.p_value < 0.01 else "*" if c.p_value < 0.05 else ""
            print(f"{c.env_name:<20} {c.sr_a:>5.0%} {c.sr_b:>5.0%} {c.effect_size_h:>+5.2f} "
                  f"{c.effect_interpretation:>10} {c.p_value:>7.4f} {sig:>4}")

    if "LLM" in by_agent:
        comps = compare_to_animal(by_agent["LLM"])
        print(f"\n{'─' * 60}")
        print("LLM vs. Animal Baselines (published)")
        print(f"{'─' * 60}")
        print(f"{'Environment':<20} {'LLM':>6} {'Animal':>7} {'h':>6} {'Effect':>10} {'p':>8}")
        print("-" * 62)
        for c in comps:
            sig = "***" if c.p_value < 0.001 else "**" if c.p_value < 0.01 else "*" if c.p_value < 0.05 else ""
            print(f"{c.env_name:<20} {c.sr_a:>5.0%} {c.sr_b:>6.0%} {c.effect_size_h:>+5.2f} "
                  f"{c.effect_interpretation:>10} {c.p_value:>7.4f} {sig}")

        # Power
        print(f"\n{'─' * 60}")
        print("Power Analysis (LLM vs Random, 80% power, α=0.05)")
        print(f"{'─' * 60}")
        comps2 = compare_agents(by_agent["LLM"], by_agent.get("Random", []))
        for c in comps2:
            adequate = "✓" if c.n_a >= c.n_needed_for_power else "✗"
            print(f"  {c.env_name:<20} N={c.n_a:>3}, need N≥{c.n_needed_for_power:>4} {adequate}")
