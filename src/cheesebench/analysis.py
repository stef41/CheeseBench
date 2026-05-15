"""
CheeseBench Analysis Pipeline

Computes cognitive profiles, learning curves, strategy metrics,
and generates publication-quality figures from benchmark results.
"""

import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Cognitive Taxonomy — maps environments to cognitive dimensions
# ============================================================================

COGNITIVE_DIMENSIONS = [
    "Allocentric Spatial Learning",
    "Egocentric Navigation",
    "Working Memory",
    "Instrumental Conditioning",
    "Avoidance Learning",
    "Associative Learning",
]

# Each env maps to one or more dimensions with a weight (0-1)
ENV_COGNITIVE_MAP: Dict[str, Dict[str, float]] = {
    "MorrisWaterMaze": {
        "Allocentric Spatial Learning": 1.0,
    },
    "BarnesMaze": {
        "Allocentric Spatial Learning": 1.0,
    },
    "TMaze": {
        "Egocentric Navigation": 0.7,
        "Working Memory": 0.3,
    },
    "StarMaze": {
        "Allocentric Spatial Learning": 0.5,
        "Egocentric Navigation": 0.5,
    },
    "RadialArmMaze": {
        "Working Memory": 0.7,
        "Allocentric Spatial Learning": 0.3,
    },
    "OperantChamber": {
        "Instrumental Conditioning": 1.0,
    },
    "ShuttleBox": {
        "Avoidance Learning": 1.0,
    },
    "PlacePreference": {
        "Associative Learning": 1.0,
    },
    "DNMSTask": {
        "Working Memory": 1.0,
    },
}

# Neural circuit dependencies (for discussion section)
NEURAL_CIRCUITS: Dict[str, str] = {
    "Allocentric Spatial Learning": "Hippocampus (place cells, grid cells)",
    "Egocentric Navigation": "Dorsomedial striatum, parietal cortex",
    "Working Memory": "Prefrontal cortex, hippocampus",
    "Instrumental Conditioning": "Dorsolateral striatum, nucleus accumbens",
    "Avoidance Learning": "Amygdala, periaqueductal gray",
    "Associative Learning": "Ventral tegmental area, nucleus accumbens",
}


# ============================================================================
# Published Animal Baselines — extracted from source papers
# ============================================================================

ANIMAL_BASELINES: Dict[str, Dict] = {
    "MorrisWaterMaze": {
        "species": "C57BL/6 mice",
        "source": "PMC3259155 (de Fiebre et al., 2006)",
        "trials_to_criterion": 15,
        "success_rate_session_1": 0.30,  # ~30% find platform in first session
        "success_rate_session_5": 0.85,  # ~85% by session 5
        "learning_curve": [0.30, 0.50, 0.65, 0.78, 0.85],  # per-session
        "avg_latency_session_1_s": 55.0,  # seconds
        "avg_latency_session_5_s": 15.0,
        "notes": "Platform acquisition over 5 sessions, 5 trials/session",
    },
    "TMaze": {
        "species": "NMRI mice",
        "source": "PMC3399492 (Shoji et al., 2012)",
        "trials_to_criterion": 20,
        "success_rate_session_1": 0.50,  # chance level
        "success_rate_session_4": 0.80,
        "learning_curve": [0.50, 0.60, 0.72, 0.80],
        "notes": "Forced alternation, 10 trials/session over 4 days",
    },
    "BarnesMaze": {
        "species": "B6C3F1/J mice",
        "source": "PMC1783636 (Harrison et al., 2006)",
        "trials_to_criterion": 20,
        "success_rate_session_1": 0.20,
        "success_rate_session_5": 0.80,
        "learning_curve": [0.20, 0.40, 0.55, 0.70, 0.80],
        "avg_latency_session_1_s": 120.0,
        "avg_latency_session_5_s": 25.0,
        "notes": "12 holes, 4 trials/session over 5 sessions",
    },
    "RadialArmMaze": {
        "species": "C57BL/6 mice",
        "source": "PMC4030456 (Penley et al., 2013)",
        "trials_to_criterion": 36,
        "success_rate_session_1": 0.15,
        "success_rate_session_6": 0.70,
        "learning_curve": [0.15, 0.25, 0.40, 0.50, 0.60, 0.70],
        "working_memory_errors_session_1": 3.5,
        "working_memory_errors_session_6": 0.8,
        "notes": "8 arms, 4 baited. Errors = revisits to depleted arms",
    },
    "OperantChamber": {
        "species": "C57BL/6 mice",
        "source": "PMC6619163 (Jurado-Parras et al., 2013)",
        "trials_to_criterion": 50,
        "success_rate_session_1": 0.40,
        "success_rate_session_5": 0.90,
        "learning_curve": [0.40, 0.60, 0.75, 0.85, 0.90],
        "avg_lever_presses_session_1": 12.0,
        "avg_lever_presses_session_5": 45.0,
        "notes": "FR-1 schedule, 30 min sessions",
    },
    "ShuttleBox": {
        "species": "Sprague Dawley rats",
        "source": "PMC4633642 (Chacon et al., 2016)",
        "trials_to_criterion": 90,
        "success_rate_session_1": 0.10,
        "success_rate_session_5": 0.70,
        "learning_curve": [0.10, 0.25, 0.45, 0.60, 0.70],
        "notes": "Active avoidance, 30 trials/session over 5 sessions",
    },
    "PlacePreference": {
        "species": "C57BL/6 mice",
        "source": "PMC6101638 (Blanco-Gandía et al., 2018)",
        "trials_to_criterion": 6,
        "success_rate_session_1": 0.50,  # chance
        "success_rate_session_6": 0.75,
        "learning_curve": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "time_in_paired_chamber_pct_pre": 50.0,
        "time_in_paired_chamber_pct_post": 65.0,
        "notes": "3-phase CPP protocol, 6 conditioning sessions",
    },
    "StarMaze": {
        "species": "C57BL/6 mice",
        "source": "PMC3695082 (Rondi-Reig et al., 2006)",
        "trials_to_criterion": 40,
        "success_rate_session_1": 0.25,
        "success_rate_session_10": 0.80,
        "learning_curve": [0.25, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "notes": "5-arm maze, 4 trials/session over 10 sessions",
    },
    "DNMSTask": {
        "species": "Long-Evans rats",
        "source": "PMC3982138 (Oomen et al., 2013)",
        "trials_to_criterion": 2856,
        "success_rate_session_1": 0.50,  # chance (2AFC)
        "success_rate_session_30": 0.80,
        "learning_curve": [0.50, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.78, 0.79, 0.79, 0.79, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
        "notes": "TUNL task, 84 trials/session, ~30 sessions to criterion",
    },
}


# ============================================================================
# Metric Computation
# ============================================================================

@dataclass
class TrialMetrics:
    """Rich metrics for a single trial."""
    steps: int = 0
    reward: float = 0.0
    success: bool = False
    actions: List[str] = field(default_factory=list)

    # Computed metrics
    action_entropy: float = 0.0
    forward_ratio: float = 0.0
    rotation_ratio: float = 0.0
    stay_ratio: float = 0.0
    action_repetition_rate: float = 0.0  # same action back-to-back
    direction_changes: int = 0  # how often rotation direction changes


@dataclass
class EnvResult:
    """Aggregated results for one (env, view_mode, agent) combination."""
    env_name: str
    view_mode: str
    agent_type: str
    trials: List[TrialMetrics] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    @property
    def success_rate_ci(self) -> Tuple[float, float]:
        """95% Wilson score confidence interval for binomial proportion."""
        n = len(self.trials)
        if n == 0:
            return (0.0, 0.0)
        p = self.success_rate
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return (max(0.0, center - spread), min(1.0, center + spread))

    @property
    def avg_steps_success(self) -> Optional[float]:
        succ = [t.steps for t in self.trials if t.success]
        return sum(succ) / len(succ) if succ else None

    @property
    def avg_steps_failure(self) -> Optional[float]:
        fail = [t.steps for t in self.trials if not t.success]
        return sum(fail) / len(fail) if fail else None

    @property
    def avg_reward(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.reward for t in self.trials) / len(self.trials)

    @property
    def timeout_rate(self) -> float:
        """Fraction of trials that hit max steps without success."""
        if not self.trials:
            return 0.0
        max_s = max(t.steps for t in self.trials) if self.trials else 200
        timeouts = sum(1 for t in self.trials if not t.success and t.steps >= max_s * 0.95)
        return timeouts / len(self.trials)

    def learning_curve(self, window: int = 4) -> List[float]:
        """Smoothed success rate over trials (rolling window)."""
        if len(self.trials) < window:
            return [self.success_rate]
        curve = []
        for i in range(len(self.trials) - window + 1):
            block = self.trials[i:i + window]
            curve.append(sum(1 for t in block if t.success) / len(block))
        return curve

    def learning_curve_blocks(self, block_size: int = 5) -> List[float]:
        """Success rate per block of trials (non-overlapping)."""
        curve = []
        for i in range(0, len(self.trials), block_size):
            block = self.trials[i:i + block_size]
            if block:
                curve.append(sum(1 for t in block if t.success) / len(block))
        return curve

    def to_dict(self) -> Dict:
        sr_ci = self.success_rate_ci
        return {
            "env_name": self.env_name,
            "view_mode": self.view_mode,
            "agent_type": self.agent_type,
            "n_trials": self.n_trials,
            "success_rate": round(self.success_rate, 4),
            "success_rate_ci_95": [round(sr_ci[0], 4), round(sr_ci[1], 4)],
            "avg_steps_success": round(self.avg_steps_success, 1) if self.avg_steps_success else None,
            "avg_steps_failure": round(self.avg_steps_failure, 1) if self.avg_steps_failure else None,
            "avg_reward": round(self.avg_reward, 4),
            "timeout_rate": round(self.timeout_rate, 4),
            "learning_curve_blocks": [round(v, 4) for v in self.learning_curve_blocks()],
            "trials": [
                {
                    "steps": t.steps,
                    "reward": round(t.reward, 4),
                    "success": t.success,
                    "n_actions": len(t.actions),
                    "action_entropy": round(t.action_entropy, 4),
                    "forward_ratio": round(t.forward_ratio, 4),
                    "rotation_ratio": round(t.rotation_ratio, 4),
                }
                for t in self.trials
            ],
        }


def compute_trial_metrics(actions: List[str]) -> TrialMetrics:
    """Compute rich strategy metrics from an action sequence."""
    m = TrialMetrics()
    m.actions = actions
    m.steps = len(actions)
    if not actions:
        return m

    counts = defaultdict(int)
    for a in actions:
        counts[a.upper()] += 1

    total = len(actions)
    m.forward_ratio = counts.get("FORWARD", 0) / total
    m.rotation_ratio = (counts.get("ROTATE_LEFT", 0) + counts.get("ROTATE_RIGHT", 0)) / total
    m.stay_ratio = counts.get("STAY", 0) / total

    # Shannon entropy of action distribution
    probs = [c / total for c in counts.values() if c > 0]
    m.action_entropy = -sum(p * math.log2(p) for p in probs) if len(probs) > 1 else 0.0

    # Repetition rate
    repeats = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    m.action_repetition_rate = repeats / (len(actions) - 1) if len(actions) > 1 else 0.0

    # Direction changes (left→right or right→left)
    dir_changes = 0
    last_rot = None
    for a in actions:
        au = a.upper()
        if au in ("ROTATE_LEFT", "ROTATE_RIGHT"):
            if last_rot and last_rot != au:
                dir_changes += 1
            last_rot = au
    m.direction_changes = dir_changes

    return m


# ============================================================================
# Cognitive Profile
# ============================================================================

def compute_cognitive_profile(
    results: List[EnvResult],
) -> Dict[str, float]:
    """
    Compute a cognitive profile (radar chart values) from benchmark results.

    Returns a dict mapping each cognitive dimension to a score in [0, 1],
    computed as the weighted average of success rates from contributing envs.
    """
    dim_scores: Dict[str, List[Tuple[float, float]]] = {d: [] for d in COGNITIVE_DIMENSIONS}

    for r in results:
        mapping = ENV_COGNITIVE_MAP.get(r.env_name, {})
        for dim, weight in mapping.items():
            dim_scores[dim].append((r.success_rate, weight))

    profile = {}
    for dim in COGNITIVE_DIMENSIONS:
        entries = dim_scores[dim]
        if not entries:
            profile[dim] = 0.0
        else:
            total_w = sum(w for _, w in entries)
            profile[dim] = sum(sr * w for sr, w in entries) / total_w if total_w > 0 else 0.0

    return profile


def compute_animal_profile() -> Dict[str, float]:
    """
    Compute the 'animal baseline' cognitive profile from published data.
    Uses the final-session success rate for each environment.
    """
    results = []
    for env_name, baseline in ANIMAL_BASELINES.items():
        lc = baseline.get("learning_curve", [])
        final_sr = lc[-1] if lc else 0.5
        r = EnvResult(env_name=env_name, view_mode="N/A", agent_type="animal")
        # Fake trials to get the right success rate
        n = 20
        n_succ = round(final_sr * n)
        for i in range(n):
            t = TrialMetrics(success=(i < n_succ), steps=50)
            r.trials.append(t)
        results.append(r)
    return compute_cognitive_profile(results)


# ============================================================================
# Result Loading & Parsing
# ============================================================================

def load_results(path: str) -> List[EnvResult]:
    """Load benchmark results JSON and parse into EnvResult objects."""
    with open(path) as f:
        data = json.load(f)

    env_results = []
    for entry in data.get("results", []):
        r = EnvResult(
            env_name=entry["env_name"],
            view_mode=entry["view_mode"],
            agent_type=entry["agent_type"],
        )
        for t in entry.get("trials", []):
            actions = t.get("actions", [])
            m = compute_trial_metrics(actions)
            m.reward = t.get("reward", 0.0)
            m.success = t.get("success", False)
            m.steps = t.get("steps", len(actions))
            r.trials.append(m)
        env_results.append(r)
    return env_results


def group_by_agent(results: List[EnvResult]) -> Dict[str, List[EnvResult]]:
    """Group results by agent type."""
    groups: Dict[str, List[EnvResult]] = defaultdict(list)
    for r in results:
        groups[r.agent_type].append(r)
    return dict(groups)


def group_by_env(results: List[EnvResult]) -> Dict[str, List[EnvResult]]:
    """Group results by environment name."""
    groups: Dict[str, List[EnvResult]] = defaultdict(list)
    for r in results:
        groups[r.env_name].append(r)
    return dict(groups)


# ============================================================================
# Summary Report
# ============================================================================

def generate_summary(results: List[EnvResult]) -> Dict:
    """Generate a comprehensive summary report."""
    by_agent = group_by_agent(results)

    summary = {
        "agents": {},
        "cognitive_profiles": {},
        "animal_baseline_profile": compute_animal_profile(),
    }

    for agent_name, agent_results in by_agent.items():
        # Per-environment results (best view mode)
        by_env = group_by_env(agent_results)
        env_summaries = {}
        best_per_env = []

        for env_name, env_results in by_env.items():
            # Pick best view mode by success rate
            best = max(env_results, key=lambda r: r.success_rate)
            best_per_env.append(best)
            env_summaries[env_name] = {
                "best_view_mode": best.view_mode,
                "best_success_rate": best.success_rate,
                "by_view_mode": {
                    r.view_mode: {
                        "success_rate": r.success_rate,
                        "ci_95": list(r.success_rate_ci),
                        "avg_steps_success": r.avg_steps_success,
                        "learning_curve": r.learning_curve_blocks(),
                    }
                    for r in env_results
                },
            }

        # Cognitive profile for this agent (using best per-env)
        profile = compute_cognitive_profile(best_per_env)

        summary["agents"][agent_name] = {
            "overall_success_rate": sum(r.success_rate for r in best_per_env) / len(best_per_env) if best_per_env else 0,
            "environments": env_summaries,
        }
        summary["cognitive_profiles"][agent_name] = profile

    return summary


def print_report(results: List[EnvResult]):
    """Print a formatted text report to stdout."""
    by_agent = group_by_agent(results)

    print("\n" + "=" * 80)
    print("CHEESEBENCH RESULTS REPORT")
    print("=" * 80)

    for agent_name, agent_results in sorted(by_agent.items()):
        print(f"\n{'─' * 60}")
        print(f"Agent: {agent_name}")
        print(f"{'─' * 60}")

        by_env = group_by_env(agent_results)
        for env_name in sorted(by_env.keys()):
            env_results = by_env[env_name]
            print(f"\n  {env_name}:")
            for r in sorted(env_results, key=lambda x: x.view_mode):
                ci = r.success_rate_ci
                steps_str = f"steps={r.avg_steps_success:.0f}" if r.avg_steps_success else "N/A"
                print(
                    f"    {r.view_mode:<14} "
                    f"SR={r.success_rate:.0%} [{ci[0]:.0%}-{ci[1]:.0%}]  "
                    f"{steps_str}  "
                    f"timeout={r.timeout_rate:.0%}"
                )
                lc = r.learning_curve_blocks()
                if len(lc) > 1:
                    lc_str = " → ".join(f"{v:.0%}" for v in lc)
                    print(f"{'':>20}learning: {lc_str}")

        # Cognitive profile
        best_per_env = []
        for env_results in by_env.values():
            best_per_env.append(max(env_results, key=lambda r: r.success_rate))
        profile = compute_cognitive_profile(best_per_env)
        print(f"\n  Cognitive Profile:")
        for dim in COGNITIVE_DIMENSIONS:
            bar = "█" * int(profile[dim] * 20) + "░" * (20 - int(profile[dim] * 20))
            print(f"    {dim:<32} {bar} {profile[dim]:.0%}")

    # Animal baseline
    animal = compute_animal_profile()
    print(f"\n{'─' * 60}")
    print("Animal Baselines (final session, from published literature):")
    print(f"{'─' * 60}")
    for dim in COGNITIVE_DIMENSIONS:
        bar = "█" * int(animal[dim] * 20) + "░" * (20 - int(animal[dim] * 20))
        print(f"  {dim:<32} {bar} {animal[dim]:.0%}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis.py <results.json>")
        sys.exit(1)

    results = load_results(sys.argv[1])
    print_report(results)

    summary = generate_summary(results)
    out_path = sys.argv[1].replace(".json", "_analysis.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")
