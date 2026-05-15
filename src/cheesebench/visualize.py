#!/usr/bin/env python3
"""
CheeseBench Visualization Pipeline

Generates publication-quality figures from benchmark results:
1. Cognitive radar chart (model vs animal baselines)
2. Per-environment learning curves (model overlaid on rodent curves)
3. View mode comparison heatmap
4. Strategy analysis (action distributions, entropy)

Usage:
    python visualize.py results/benchmark_results.json
    python visualize.py results/benchmark_results.json --output figures/
"""

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List

from .analysis import (
    COGNITIVE_DIMENSIONS,
    ENV_COGNITIVE_MAP,
    ANIMAL_BASELINES,
    NEURAL_CIRCUITS,
    load_results,
    group_by_agent,
    group_by_env,
    compute_cognitive_profile,
    compute_animal_profile,
    compute_trial_metrics,
    EnvResult,
)

# ============================================================================
# Style constants
# ============================================================================

COLORS = {
    "LLM": "#2196F3",       # blue
    "Random": "#9E9E9E",    # grey
    "animal": "#4CAF50",    # green
}
FONT_SIZE = 10
TITLE_SIZE = 12
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# ============================================================================
# 1. Cognitive Radar Chart
# ============================================================================

def plot_cognitive_radar(
    results: List[EnvResult],
    output_path: Path,
):
    """
    Radar chart comparing agent cognitive profiles to animal baselines.
    """
    by_agent = {}
    for r in results:
        by_agent.setdefault(r.agent_type, []).append(r)

    # Compute profiles
    profiles = {}
    for agent_name, agent_results in by_agent.items():
        # Best view mode per env
        by_env = {}
        for r in agent_results:
            by_env.setdefault(r.env_name, []).append(r)
        best = [max(ers, key=lambda x: x.success_rate) for ers in by_env.values()]
        profiles[agent_name] = compute_cognitive_profile(best)

    profiles["Animal"] = compute_animal_profile()

    # Radar plot
    dims = COGNITIVE_DIMENSIONS
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for agent_name, profile in profiles.items():
        values = [profile[d] for d in dims]
        values += values[:1]
        color = COLORS.get(agent_name, "#FF9800")
        ls = "--" if agent_name == "Animal" else "-"
        lw = 2.5 if agent_name == "Animal" else 2.0
        ax.plot(angles, values, ls, linewidth=lw, label=agent_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    # Short labels
    short_dims = [d.replace("Allocentric Spatial Learning", "Allocentric\nSpatial")
                   .replace("Egocentric Navigation", "Egocentric\nNavigation")
                   .replace("Instrumental Conditioning", "Instrumental\nConditioning")
                   .replace("Avoidance Learning", "Avoidance\nLearning")
                   .replace("Associative Learning", "Associative\nLearning")
                   .replace("Working Memory", "Working\nMemory")
                  for d in dims]
    ax.set_xticklabels(short_dims, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Cognitive Profile: Model vs. Rodent Baselines", pad=20, fontsize=13, fontweight="bold")

    fig.savefig(output_path / "cognitive_radar.pdf")
    fig.savefig(output_path / "cognitive_radar.png")
    plt.close(fig)
    print(f"  Saved cognitive_radar.{{pdf,png}}")


# ============================================================================
# 2. Learning Curves (per environment)
# ============================================================================

def plot_learning_curves(
    results: List[EnvResult],
    output_path: Path,
):
    """
    Per-environment learning curves: model block success rate vs animal baseline.
    """
    by_agent = {}
    for r in results:
        by_agent.setdefault(r.agent_type, []).append(r)

    # Get all env names
    all_envs = sorted(set(r.env_name for r in results))
    n_envs = len(all_envs)
    cols = 3
    rows = (n_envs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, env_name in enumerate(all_envs):
        ax = axes[idx]

        # Animal baseline
        baseline = ANIMAL_BASELINES.get(env_name, {})
        animal_lc = baseline.get("learning_curve", [])
        if animal_lc:
            x_animal = np.arange(1, len(animal_lc) + 1)
            ax.plot(x_animal, animal_lc, "o--", color=COLORS["animal"], label="Rodent", markersize=5, linewidth=1.5)
            ax.fill_between(x_animal, 0, animal_lc, alpha=0.08, color=COLORS["animal"])

        # VLM learning curve (best view mode)
        for agent_name in ["LLM", "Random"]:
            agent_results = by_agent.get(agent_name, [])
            env_results = [r for r in agent_results if r.env_name == env_name]
            if not env_results:
                continue
            best = max(env_results, key=lambda r: r.success_rate)
            lc = best.learning_curve_blocks(block_size=4)
            if lc:
                x = np.arange(1, len(lc) + 1)
                color = COLORS.get(agent_name, "#FF9800")
                ax.plot(x, lc, "s-", color=color, label=agent_name, markersize=4, linewidth=1.5)

        ax.set_title(env_name, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Success Rate")
        ax.set_xlabel("Session")
        ax.axhline(y=0.5, ls=":", color="#ccc", lw=0.8)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    # Hide unused axes
    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Learning Curves: Model vs. Rodent Baselines", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "learning_curves.pdf")
    fig.savefig(output_path / "learning_curves.png")
    plt.close(fig)
    print(f"  Saved learning_curves.{{pdf,png}}")


# ============================================================================
# 3. View Mode Comparison Heatmap
# ============================================================================

def plot_view_mode_heatmap(
    results: List[EnvResult],
    output_path: Path,
):
    """
    Heatmap of success rates: environments × view modes (LLM agent only).
    """
    llm_results = [r for r in results if r.agent_type == "LLM"]
    if not llm_results:
        print("  Skipping heatmap — no LLM results")
        return

    envs = sorted(set(r.env_name for r in llm_results))
    modes = sorted(set(r.view_mode for r in llm_results))

    data = np.full((len(envs), len(modes)), np.nan)
    for r in llm_results:
        i = envs.index(r.env_name)
        j = modes.index(r.view_mode)
        data[i, j] = r.success_rate

    fig, ax = plt.subplots(figsize=(max(5, len(modes) * 1.8), max(4, len(envs) * 0.6)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=45, ha="right")
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels(envs)

    # Annotate cells
    for i in range(len(envs)):
        for j in range(len(modes)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        color="white" if val < 0.4 or val > 0.8 else "black", fontsize=9)

    ax.set_title("Success Rate by Environment × View Mode (LLM Agent)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Success Rate")
    fig.tight_layout()
    fig.savefig(output_path / "view_mode_heatmap.pdf")
    fig.savefig(output_path / "view_mode_heatmap.png")
    plt.close(fig)
    print(f"  Saved view_mode_heatmap.{{pdf,png}}")


# ============================================================================
# 4. Strategy Analysis (action distributions)
# ============================================================================

def plot_strategy_analysis(
    results: List[EnvResult],
    output_path: Path,
):
    """
    Bar chart of action distributions and entropy per environment.
    """
    llm_results = [r for r in results if r.agent_type == "LLM"]
    if not llm_results:
        print("  Skipping strategy — no LLM results")
        return

    # Pick best view mode per env
    by_env = {}
    for r in llm_results:
        by_env.setdefault(r.env_name, []).append(r)

    envs = sorted(by_env.keys())
    fwd_ratios = []
    rot_ratios = []
    stay_ratios = []
    entropies = []

    for env_name in envs:
        best = max(by_env[env_name], key=lambda r: r.success_rate)
        all_fwd = [t.forward_ratio for t in best.trials]
        all_rot = [t.rotation_ratio for t in best.trials]
        all_stay = [t.stay_ratio for t in best.trials]
        all_ent = [t.action_entropy for t in best.trials]
        fwd_ratios.append(np.mean(all_fwd) if all_fwd else 0)
        rot_ratios.append(np.mean(all_rot) if all_rot else 0)
        stay_ratios.append(np.mean(all_stay) if all_stay else 0)
        entropies.append(np.mean(all_ent) if all_ent else 0)

    x = np.arange(len(envs))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar for action distribution
    ax1.bar(x, fwd_ratios, w, label="Forward", color="#2196F3")
    ax1.bar(x, rot_ratios, w, bottom=fwd_ratios, label="Rotate", color="#FF9800")
    bottoms = [f + r for f, r in zip(fwd_ratios, rot_ratios)]
    ax1.bar(x, stay_ratios, w, bottom=bottoms, label="Stay", color="#9E9E9E")
    ax1.set_xticks(x)
    ax1.set_xticklabels(envs, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Action Proportion")
    ax1.set_title("Action Distribution by Environment", fontweight="bold")
    ax1.legend(fontsize=8)

    # Entropy bar
    bars = ax2.bar(x, entropies, color="#4CAF50", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(envs, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Shannon Entropy (bits)")
    ax2.set_title("Action Entropy by Environment", fontweight="bold")
    ax2.axhline(y=2.0, ls="--", color="#999", lw=0.8, label="Uniform (4 actions)")
    ax2.legend(fontsize=8)

    fig.suptitle("Model Strategy Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "strategy_analysis.pdf")
    fig.savefig(output_path / "strategy_analysis.png")
    plt.close(fig)
    print(f"  Saved strategy_analysis.{{pdf,png}}")


# ============================================================================
# 5. Summary Table (LaTeX)
# ============================================================================

def generate_latex_table(
    results: List[EnvResult],
    output_path: Path,
):
    """Generate LaTeX table for paper."""
    by_agent = {}
    for r in results:
        by_agent.setdefault(r.agent_type, []).append(r)

    envs = sorted(set(r.env_name for r in results))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CheeseBench Results: Success rates (\%) with 95\% Wilson CIs.}",
        r"\label{tab:results}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Environment & Random & LLM & Animal & Cog. Dimension \\",
        r"\midrule",
    ]

    animal_profile = compute_animal_profile()

    for env_name in envs:
        # Random
        rand_results = [r for r in by_agent.get("Random", []) if r.env_name == env_name]
        rand_best = max(rand_results, key=lambda r: r.success_rate) if rand_results else None
        rand_str = f"{rand_best.success_rate*100:.0f}" if rand_best else "—"

        # LLM
        llm_results = [r for r in by_agent.get("LLM", []) if r.env_name == env_name]
        llm_best = max(llm_results, key=lambda r: r.success_rate) if llm_results else None
        if llm_best:
            ci = llm_best.success_rate_ci
            llm_str = f"{llm_best.success_rate*100:.0f} [{ci[0]*100:.0f}–{ci[1]*100:.0f}]"
        else:
            llm_str = "—"

        # Animal
        baseline = ANIMAL_BASELINES.get(env_name, {})
        lc = baseline.get("learning_curve", [])
        animal_str = f"{lc[-1]*100:.0f}" if lc else "—"

        # Dimension
        dim = list(ENV_COGNITIVE_MAP.get(env_name, {}).keys())
        dim_str = dim[0] if dim else "—"

        lines.append(f"{env_name} & {rand_str} & {llm_str} & {animal_str} & {dim_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    out_file = output_path / "results_table.tex"
    with open(out_file, "w") as f:
        f.write(tex)
    print(f"  Saved results_table.tex")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CheeseBench Visualization")
    parser.add_argument("results_file", help="Path to benchmark_results.json")
    parser.add_argument("--output", default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()

    results = load_results(args.results_file)
    if not results:
        print("No results found.")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else Path(args.results_file).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures from {args.results_file} → {out_dir}/")
    plot_cognitive_radar(results, out_dir)
    plot_learning_curves(results, out_dir)
    plot_view_mode_heatmap(results, out_dir)
    plot_strategy_analysis(results, out_dir)
    generate_latex_table(results, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
