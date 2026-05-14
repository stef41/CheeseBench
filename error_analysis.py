"""
CheeseBench — Error Analysis Pipeline

Automatically categorizes WHY models fail at each task, extracting
actionable failure modes for the paper's error analysis section.

Failure modes:
  - loop_stuck:     Agent repeats the same action >70% of the time
  - boundary_hug:   Agent spends >60% of actions hitting walls
  - no_exploration:  Agent stays in a small area (low unique positions)
  - parse_failure:  Agent's output couldn't be parsed into valid actions
  - timeout:        Hit max steps without success
  - wrong_strategy: Got close to goal but didn't complete task
  - slow_learner:   Succeeded but took >3× more steps than animal baseline

All categorizations are computed from the trial data (actions, steps,
reward, success) without needing the raw LLM trace logs.
"""

import math
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from analysis import EnvResult, TrialMetrics, ANIMAL_BASELINES


# ============================================================================
# Failure Mode Classification
# ============================================================================

@dataclass
class FailureAnalysis:
    """Analysis of a single failed trial."""
    env_name: str
    view_mode: str
    trial_idx: int
    steps: int
    reward: float
    failure_mode: str
    details: str


def classify_failure(
    trial: TrialMetrics,
    env_name: str,
    max_steps: int = 200,
) -> Tuple[str, str]:
    """
    Classify the failure mode of a single trial.
    Returns (mode, details).
    """
    actions = trial.actions
    steps = trial.steps
    n = len(actions)

    if n == 0:
        return "parse_failure", "No actions produced"

    counts = Counter(a.upper() for a in actions)
    most_common_action, most_common_count = counts.most_common(1)[0]
    repetition = most_common_count / n

    # 1. Loop stuck: single action dominates
    if n > 10 and repetition > 0.70:
        return "loop_stuck", f"Action '{most_common_action}' repeated {repetition:.0%} of time"

    # 2. Parse failures (proxy: lots of FORWARD-only, suggests fallback)
    parse_failures = getattr(trial, 'parse_failures', 0)
    if parse_failures > 0 or (n > 10 and counts.get("FORWARD", 0) == n):
        return "parse_failure", f"Parse failures detected or all-FORWARD fallback"

    # 3. Boundary hugging: lots of rotations with little forward movement
    fwd = counts.get("FORWARD", 0)
    rot = counts.get("ROTATE_LEFT", 0) + counts.get("ROTATE_RIGHT", 0)
    if n > 15 and fwd < n * 0.15 and rot > n * 0.60:
        return "boundary_hug", f"Forward={fwd}/{n}, Rotate={rot}/{n} — stuck at walls"

    # 4. Timeout: hit max steps
    if steps >= max_steps * 0.95 and not trial.success:
        # Was the agent making diverse actions? If so, exploration failure
        if trial.action_entropy > 1.0:
            return "timeout", f"Exhausted {steps} steps with diverse exploration"
        else:
            return "timeout", f"Exhausted {steps} steps with low-entropy strategy"

    # 5. No exploration: too much STAY
    stay_count = counts.get("STAY", 0)
    if stay_count > n * 0.40:
        return "no_exploration", f"STAY used {stay_count}/{n} times ({stay_count/n:.0%})"

    # 6. Oscillation: rapid direction changes
    if trial.direction_changes > n * 0.4 and n > 10:
        return "wrong_strategy", f"Excessive direction changes ({trial.direction_changes}/{n})"

    # 7. General inefficiency
    baseline = ANIMAL_BASELINES.get(env_name, {})
    animal_lc = baseline.get("learning_curve", [])
    if animal_lc and trial.success:
        # This shouldn't be called for failures, but defensive
        return "slow_learner", f"Needed {steps} steps"

    return "wrong_strategy", f"Failed after {steps} steps, reward={trial.reward:.2f}"


# ============================================================================
# Aggregate Failure Analysis
# ============================================================================

@dataclass
class EnvFailureReport:
    """Aggregated failure report for one environment."""
    env_name: str
    n_total: int = 0
    n_failures: int = 0
    mode_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    examples: List[FailureAnalysis] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.n_failures / self.n_total if self.n_total > 0 else 0

    @property
    def mode_distribution(self) -> Dict[str, float]:
        if self.n_failures == 0:
            return {}
        return {mode: count / self.n_failures
                for mode, count in sorted(self.mode_counts.items(), key=lambda x: -x[1])}

    def to_dict(self) -> Dict:
        return {
            "env_name": self.env_name,
            "n_total": self.n_total,
            "n_failures": self.n_failures,
            "failure_rate": round(self.failure_rate, 4),
            "mode_distribution": {k: round(v, 4) for k, v in self.mode_distribution.items()},
            "top_examples": [
                {"trial": e.trial_idx, "mode": e.failure_mode, "details": e.details}
                for e in self.examples[:5]
            ],
        }


def analyze_failures(
    results: List[EnvResult],
    max_steps: int = 200,
    agent_filter: str = "LLM",
) -> Dict[str, EnvFailureReport]:
    """
    Analyze failures across all environments for a given agent type.
    Returns env_name → EnvFailureReport.
    """
    reports: Dict[str, EnvFailureReport] = {}

    for r in results:
        if r.agent_type != agent_filter:
            continue
        key = r.env_name
        if key not in reports:
            reports[key] = EnvFailureReport(env_name=key)
        report = reports[key]

        for idx, trial in enumerate(r.trials):
            report.n_total += 1
            if trial.success:
                continue
            report.n_failures += 1

            mode, details = classify_failure(trial, r.env_name, max_steps)
            report.mode_counts[mode] += 1

            # Keep up to 5 examples per env
            if len(report.examples) < 5:
                report.examples.append(FailureAnalysis(
                    env_name=r.env_name,
                    view_mode=r.view_mode,
                    trial_idx=idx,
                    steps=trial.steps,
                    reward=trial.reward,
                    failure_mode=mode,
                    details=details,
                ))

    return reports


# ============================================================================
# Success Analysis (for qualitative examples)
# ============================================================================

@dataclass
class SuccessProfile:
    """Profile of successful trials for an environment."""
    env_name: str
    n_successes: int = 0
    avg_steps: float = 0
    avg_entropy: float = 0
    avg_forward_ratio: float = 0
    fastest_trial_idx: int = 0
    fastest_steps: int = 999

    def to_dict(self) -> Dict:
        return {
            "env_name": self.env_name,
            "n_successes": self.n_successes,
            "avg_steps": round(self.avg_steps, 1),
            "avg_entropy": round(self.avg_entropy, 3),
            "avg_forward_ratio": round(self.avg_forward_ratio, 3),
            "fastest_trial_idx": self.fastest_trial_idx,
            "fastest_steps": self.fastest_steps,
        }


def analyze_successes(
    results: List[EnvResult],
    agent_filter: str = "LLM",
) -> Dict[str, SuccessProfile]:
    """Profile successful trials for cherry-picking qualitative examples."""
    profiles: Dict[str, SuccessProfile] = {}

    for r in results:
        if r.agent_type != agent_filter:
            continue
        key = r.env_name
        if key not in profiles:
            profiles[key] = SuccessProfile(env_name=key)
        p = profiles[key]

        succ_trials = [(i, t) for i, t in enumerate(r.trials) if t.success]
        if not succ_trials:
            continue

        p.n_successes += len(succ_trials)
        p.avg_steps = sum(t.steps for _, t in succ_trials) / len(succ_trials)
        p.avg_entropy = sum(t.action_entropy for _, t in succ_trials) / len(succ_trials)
        p.avg_forward_ratio = sum(t.forward_ratio for _, t in succ_trials) / len(succ_trials)

        fastest_idx, fastest_trial = min(succ_trials, key=lambda x: x[1].steps)
        if fastest_trial.steps < p.fastest_steps:
            p.fastest_steps = fastest_trial.steps
            p.fastest_trial_idx = fastest_idx

    return profiles


# ============================================================================
# Full Error Report
# ============================================================================

def generate_error_report(results: List[EnvResult]) -> Dict:
    """Comprehensive error analysis report."""
    failures = analyze_failures(results)
    successes = analyze_successes(results)

    # Aggregate failure modes across all envs
    global_modes: Dict[str, int] = defaultdict(int)
    total_failures = 0
    for report in failures.values():
        for mode, count in report.mode_counts.items():
            global_modes[mode] += count
            total_failures += count

    return {
        "global_failure_distribution": {
            mode: round(count / total_failures, 4) if total_failures > 0 else 0
            for mode, count in sorted(global_modes.items(), key=lambda x: -x[1])
        },
        "total_failures": total_failures,
        "per_environment": {
            env: report.to_dict() for env, report in sorted(failures.items())
        },
        "success_profiles": {
            env: prof.to_dict() for env, prof in sorted(successes.items())
        },
    }


def print_error_report(results: List[EnvResult]):
    """Pretty-print error analysis."""
    failures = analyze_failures(results)
    successes = analyze_successes(results)

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    # Global summary
    global_modes: Dict[str, int] = defaultdict(int)
    total_failures = 0
    total_trials = 0
    for report in failures.values():
        total_trials += report.n_total
        total_failures += report.n_failures
        for mode, count in report.mode_counts.items():
            global_modes[mode] += count

    print(f"\nOverall: {total_failures}/{total_trials} trials failed ({total_failures/total_trials:.0%})" if total_trials > 0 else "")
    print(f"\nGlobal failure mode distribution:")
    for mode, count in sorted(global_modes.items(), key=lambda x: -x[1]):
        pct = count / total_failures if total_failures > 0 else 0
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"  {mode:<18} {bar} {pct:>5.0%} ({count})")

    # Per-environment
    print(f"\n{'─' * 60}")
    print(f"{'Environment':<20} {'Fail%':>6} {'Top Failure Mode':<20} {'%':>5}")
    print("-" * 55)
    for env_name in sorted(failures):
        report = failures[env_name]
        dist = report.mode_distribution
        top_mode = next(iter(dist), "—")
        top_pct = dist.get(top_mode, 0)
        print(f"{env_name:<20} {report.failure_rate:>5.0%} {top_mode:<20} {top_pct:>4.0%}")

    # Example failures
    print(f"\n{'─' * 60}")
    print("Example Failures:")
    for env_name in sorted(failures):
        report = failures[env_name]
        if report.examples:
            ex = report.examples[0]
            print(f"\n  {env_name} — {ex.failure_mode}")
            print(f"    Trial {ex.trial_idx}: {ex.steps} steps, reward={ex.reward:.2f}")
            print(f"    {ex.details}")
