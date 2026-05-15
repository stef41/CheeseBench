#!/usr/bin/env python3
"""
CheeseBench — Full Experiment Runner for NeurIPS Paper

Runs ALL experiments needed for a competitive paper submission:

  Experiment 1: Multi-model benchmark (6+ models across 9 envs × 3 views)
  Experiment 2: Scaling analysis (3B → 72B, same family)
  Experiment 3: Ablation — prompt variants
  Experiment 4: Ablation — history length
  Experiment 5: Ablation — actions per call
  Experiment 6: Ablation — text-only vs vision (architecture ablation)
  Experiment 7: Image mode — TOPDOWN_2D view across all open-weight VLMs

Each experiment launches a vLLM server, runs the benchmark, saves results,
then shuts down the server before moving to the next model.

Usage:
    # Run everything (full paper)
    python run_experiments.py --all

    # Run specific experiment
    python run_experiments.py --exp multi_model
    python run_experiments.py --exp scaling
    python run_experiments.py --exp ablation_prompt
    python run_experiments.py --exp ablation_history
    python run_experiments.py --exp ablation_actions
    python run_experiments.py --exp ablation_vision

    # Quick smoke test (2 trials, 1 model)
    python run_experiments.py --smoke-test

    # Analyze all existing results
    python run_experiments.py --analyze
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import BenchmarkConfig
from model_server import (
    MODEL_REGISTRY, ModelSpec, get_model,
    launch_vllm_server, stop_server, check_vllm_installed, install_vllm,
)

RESULTS_DIR = Path("results")
VLLM_PORT = 8000


# ============================================================================
# Helpers
# ============================================================================

def run_benchmark_with_config(
    cfg: BenchmarkConfig,
    tag: str,
) -> Optional[Path]:
    """
    Run the benchmark with the given config.
    Returns path to results file, or None on failure.
    """
    out_dir = Path(cfg.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": cfg.model,
            "api_url": cfg.api_url,
            "api_format": cfg.api_format,
            "num_trials": cfg.num_trials,
            "max_steps_per_trial": cfg.max_steps_per_trial,
            "max_actions_per_call": cfg.max_actions_per_call,
            "max_history_messages": cfg.max_history_messages,
            "seed": cfg.seed,
            "prompt_variant": cfg.prompt_variant,
            "view_modes": cfg.view_modes,
        }, f, indent=2)

    # Build command — run benchmark.py as subprocess for isolation
    # Only pass --num-trials / --max-steps if the experiment explicitly overrides them
    # (i.e. differs from BenchmarkConfig defaults), so per-env ENV_PROTOCOL is used otherwise.
    _defaults = BenchmarkConfig()
    cmd = [
        sys.executable, "benchmark.py",
        "--model", cfg.model,
        "--api-url", cfg.api_url,
        "--api-format", cfg.api_format,
        "--seed", str(cfg.seed),
        "--output-dir", str(out_dir),
        "--prompt-variant", cfg.prompt_variant,
        "--max-actions", str(cfg.max_actions_per_call),
        "--max-history", str(cfg.max_history_messages),
        "--view-modes", *cfg.view_modes,
    ]
    if cfg.num_trials != _defaults.num_trials:
        cmd.extend(["--num-trials", str(cfg.num_trials)])
    if cfg.max_steps_per_trial != _defaults.max_steps_per_trial:
        cmd.extend(["--max-steps", str(cfg.max_steps_per_trial)])

    print(f"  Running: {tag}")
    print(f"    Model: {cfg.model}")
    if cfg.num_trials != _defaults.num_trials:
        print(f"    Trials: {cfg.num_trials} (CLI override, all envs)")
    else:
        print(f"    Trials: per-environment protocol")
    if cfg.max_steps_per_trial != _defaults.max_steps_per_trial:
        print(f"    MaxSteps: {cfg.max_steps_per_trial} (CLI override, all envs)")
    else:
        print(f"    MaxSteps: per-environment protocol")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    results_file = out_dir / "benchmark_results.json"
    if results_file.exists():
        print(f"    ✓ Done in {elapsed:.0f}s → {results_file}")
        return results_file
    else:
        print(f"    ✗ FAILED ({elapsed:.0f}s)")
        err_log = out_dir / "error.log"
        with open(err_log, "w") as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")
        print(f"    Error log: {err_log}")
        return None


def serve_and_run(
    model: ModelSpec,
    cfg: BenchmarkConfig,
    tag: str,
) -> Optional[Path]:
    """Launch vLLM server for model, run benchmark, stop server."""
    proc = None
    try:
        proc = launch_vllm_server(model, port=VLLM_PORT)
        cfg.api_url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
        cfg.api_format = "openai"
        cfg.model = model.hf_id
        return run_benchmark_with_config(cfg, tag)
    except Exception as e:
        print(f"    ✗ Server error: {e}")
        return None
    finally:
        if proc:
            stop_server(proc)
            time.sleep(5)  # Let GPU memory free


# ============================================================================
# Experiment 1: Multi-Model Benchmark
# ============================================================================

def exp_multi_model(num_trials: int = 20):
    """
    Run the full benchmark across all models in the registry.
    This is the main results table in the paper.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Multi-Model Benchmark")
    print("=" * 60)

    # Core models to benchmark
    model_names = [
        "qwen2.5vl-3b",
        "qwen2.5vl-7b",
        "internvl2.5-8b",
        "phi-4-mm-14b",
        "qwen2.5vl-32b",
        "qwen2.5vl-72b",
    ]

    results_files = []
    for name in model_names:
        model = get_model(name)
        if model is None:
            print(f"  Skipping {name}: not in registry")
            continue

        cfg = BenchmarkConfig()
        cfg.num_trials = num_trials
        tag = f"multi_model/{name}"

        # Check if already done
        existing = RESULTS_DIR / tag / "benchmark_results.json"
        if existing.exists():
            print(f"  {name}: already done, skipping (delete {existing} to re-run)")
            results_files.append(existing)
            continue

        path = serve_and_run(model, cfg, tag)
        if path:
            results_files.append(path)

    print(f"\nCompleted {len(results_files)}/{len(model_names)} models")
    return results_files


# ============================================================================
# Experiment 2: Scaling Analysis
# ============================================================================

def exp_scaling(num_trials: int = 20):
    """
    Same architecture family (Qwen2.5-VL) at different scales.
    Tests whether bigger = better at behavioral tasks.

    Reuses results from multi_model (same models, same config, same seed)
    by symlinking instead of re-running.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Scaling Analysis (Qwen2.5-VL 3B→72B)")
    print("=" * 60)

    model_names = ["qwen2.5vl-3b", "qwen2.5vl-7b", "qwen2.5vl-32b", "qwen2.5vl-72b"]

    results_files = []
    for name in model_names:
        scaling_dir = RESULTS_DIR / f"scaling/{name}"
        multi_model_dir = RESULTS_DIR / f"multi_model/{name}"
        multi_model_results = multi_model_dir / "benchmark_results.json"

        if scaling_dir.exists():
            existing = scaling_dir / "benchmark_results.json"
            if existing.exists():
                print(f"  {name}: already done, skipping")
                results_files.append(existing)
                continue

        if multi_model_results.exists():
            # Reuse multi_model results (same config, same seed)
            scaling_dir.parent.mkdir(parents=True, exist_ok=True)
            if scaling_dir.exists():
                import shutil
                shutil.rmtree(scaling_dir)
            scaling_dir.symlink_to(multi_model_dir.resolve())
            print(f"  {name}: symlinked from multi_model/{name}")
            results_files.append(scaling_dir / "benchmark_results.json")
        else:
            # multi_model hasn't run yet for this model — run it
            model = get_model(name)
            if model is None:
                continue
            cfg = BenchmarkConfig()
            cfg.num_trials = num_trials
            tag = f"scaling/{name}"
            path = serve_and_run(model, cfg, tag)
            if path:
                results_files.append(path)

    return results_files


# ============================================================================
# Experiment 3: Ablation — Prompt Variants
# ============================================================================

PROMPT_VARIANTS = {
    "default": None,  # Uses build_system_prompt() as-is
    "minimal": """You are an agent. Maximize cumulative reward.
You see an ASCII rendering. Your position has an arrow.

ACTIONS: FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY

Respond:
LEARNINGS: <brief notes>
ACTIONS: <1-{k} comma-separated actions>""",

    "cot": """You are an embodied agent in a behavioral experiment. Maximize cumulative reward.

You receive ASCII renderings of your environment. Walls (#, █) block movement.
Your position/orientation is shown by arrow symbols.

Before acting, reason step by step:
1. Where am I? What do I see around me?
2. What reward did I just receive? What does it tell me?
3. What is my hypothesis about the task goal?
4. What should I try next to test this hypothesis?

ACTIONS: FORWARD (move ahead), ROTATE_LEFT (turn left), ROTATE_RIGHT (turn right), STAY (wait)

RESPONSE FORMAT:
LEARNINGS: <Your step-by-step reasoning and updated strategy. Max 500 chars.>
ACTIONS: <1 to {k} comma-separated actions>""",

    "few_shot": """You are an embodied agent in a behavioral experiment. Maximize cumulative reward.

You see an ASCII rendering. Walls (#, █) block movement. Arrow = your position/facing.

ACTIONS: FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY

Example interaction:
Observation:
  # # # # #
  # . . . #
  # . → . #
  # . . G #
  # # # # #
Reward: +0.00

LEARNINGS: I am at center facing east. I see 'G' to my south-east. I should go forward then rotate right toward it.
ACTIONS: FORWARD, ROTATE_RIGHT, FORWARD

Now it's your turn. Respond with LEARNINGS and up to {k} ACTIONS.""",
}


def exp_ablation_prompt(num_trials: int = 15, model_name: str = "qwen2.5vl-7b"):
    """Test different prompt formats."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ablation — Prompt Variants")
    print("=" * 60)

    model = get_model(model_name)
    if model is None:
        print(f"  Model {model_name} not found")
        return []

    results_files = []
    for variant_name, prompt_text in PROMPT_VARIANTS.items():
        cfg = BenchmarkConfig()
        cfg.num_trials = num_trials
        cfg.prompt_variant = variant_name
        tag = f"ablation_prompt/{variant_name}"

        existing = RESULTS_DIR / tag / "benchmark_results.json"
        if existing.exists():
            print(f"  {variant_name}: already done, skipping")
            results_files.append(existing)
            continue

        # For "default", just use normal config. For others, we need to
        # temporarily override the prompt. This is done via config.prompt_variant
        # which benchmark.py will read when building the system prompt.
        path = serve_and_run(model, cfg, tag)
        if path:
            results_files.append(path)

    return results_files


# ============================================================================
# Experiment 4: Ablation — History Length
# ============================================================================

def exp_ablation_history(num_trials: int = 15, model_name: str = "qwen2.5vl-7b"):
    """Test impact of conversation history length on learning."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Ablation — History Length")
    print("=" * 60)

    model = get_model(model_name)
    if model is None:
        print(f"  Model {model_name} not found")
        return []

    history_lengths = [1, 3, 5, 10]
    results_files = []

    for h in history_lengths:
        cfg = BenchmarkConfig()
        cfg.num_trials = num_trials
        cfg.max_history_messages = h
        tag = f"ablation_history/history_{h}"

        existing = RESULTS_DIR / tag / "benchmark_results.json"
        if existing.exists():
            print(f"  history={h}: already done, skipping")
            results_files.append(existing)
            continue

        path = serve_and_run(model, cfg, tag)
        if path:
            results_files.append(path)

    return results_files


# ============================================================================
# Experiment 5: Ablation — Actions Per Call
# ============================================================================

def exp_ablation_actions(num_trials: int = 15, model_name: str = "qwen2.5vl-7b"):
    """Test impact of multi-action budget on performance."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Ablation — Actions Per Call")
    print("=" * 60)

    model = get_model(model_name)
    if model is None:
        print(f"  Model {model_name} not found")
        return []

    action_budgets = [1, 4, 8, 16]
    results_files = []

    for k in action_budgets:
        cfg = BenchmarkConfig()
        cfg.num_trials = num_trials
        cfg.max_actions_per_call = k
        tag = f"ablation_actions/actions_{k}"

        existing = RESULTS_DIR / tag / "benchmark_results.json"
        if existing.exists():
            print(f"  k={k}: already done, skipping")
            results_files.append(existing)
            continue

        path = serve_and_run(model, cfg, tag)
        if path:
            results_files.append(path)

    return results_files


# ============================================================================
# Experiment 6: Ablation — Text-Only vs Vision
# ============================================================================

def exp_ablation_vision(num_trials: int = 15):
    """
    Compare VLM vs text-only model at same scale.
    Key question: Does vision architecture help on ASCII tasks?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Ablation — Text-Only vs Vision")
    print("=" * 60)

    pairs = [
        ("qwen2.5vl-7b", "qwen2.5-7b-text"),
        ("qwen2.5vl-32b", "qwen2.5-32b-text"),
    ]

    results_files = []
    for vlm_name, text_name in pairs:
        for name in [vlm_name, text_name]:
            model = get_model(name)
            if model is None:
                print(f"  {name}: not in registry, skipping")
                continue

            cfg = BenchmarkConfig()
            cfg.num_trials = num_trials
            tag = f"ablation_vision/{name}"

            existing = RESULTS_DIR / tag / "benchmark_results.json"
            if existing.exists():
                print(f"  {name}: already done, skipping")
                results_files.append(existing)
                continue

            path = serve_and_run(model, cfg, tag)
            if path:
                results_files.append(path)

    return results_files


# ============================================================================
# Experiment 7: Image Mode (TOPDOWN_2D)
# ============================================================================

def exp_image_mode(num_trials: int = 10):
    """
    Run TOPDOWN_2D image view across all VLMs.
    Sends actual rendered images (224×224 PNG) to VLMs via base64.
    Text-only models are skipped automatically by benchmark.py.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Image Mode (TOPDOWN_2D)")
    print("=" * 60)

    model_names = [
        "qwen2.5vl-3b",
        "qwen2.5vl-7b",
        "internvl2.5-8b",
        "phi-4-mm-14b",
        "qwen2.5vl-32b",
        "qwen2.5vl-72b",
    ]

    results_files = []
    for name in model_names:
        model = get_model(name)
        if model is None:
            print(f"  Skipping {name}: not in registry")
            continue

        cfg = BenchmarkConfig()
        cfg.num_trials = num_trials
        cfg.view_modes = ["TOPDOWN_2D"]
        tag = f"image_mode/{name}"

        existing = RESULTS_DIR / tag / "benchmark_results.json"
        if existing.exists():
            print(f"  {name}: already done, skipping (delete {existing} to re-run)")
            results_files.append(existing)
            continue

        path = serve_and_run(model, cfg, tag)
        if path:
            results_files.append(path)

    print(f"\nCompleted {len(results_files)}/{len(model_names)} models")
    return results_files


# ============================================================================
# Post-hoc Analysis
# ============================================================================

def run_analysis():
    """Run full analysis on all existing results."""
    print("\n" + "=" * 60)
    print("POST-HOC ANALYSIS")
    print("=" * 60)

    from analysis import load_results, print_report, generate_summary
    from stat_tests import print_stats_report, generate_stats_report
    from error_analysis import print_error_report, generate_error_report

    # Collect all result files
    result_files = sorted(RESULTS_DIR.rglob("benchmark_results.json"))
    if not result_files:
        print("No results found. Run experiments first.")
        return

    print(f"Found {len(result_files)} result files")

    # Load and merge all results
    all_results = []
    for f in result_files:
        try:
            results = load_results(str(f))
            # Tag with experiment name from path
            exp_name = str(f.parent.relative_to(RESULTS_DIR))
            for r in results:
                r.agent_type = f"{r.agent_type}_{exp_name}" if "/" in exp_name else r.agent_type
            all_results.extend(results)
            print(f"  Loaded {len(results)} results from {f.parent.name}")
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    if not all_results:
        print("No valid results loaded.")
        return

    # Run reports
    print_report(all_results)
    print_stats_report(all_results)
    print_error_report(all_results)

    # Save aggregated analysis
    analysis_dir = RESULTS_DIR / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    summary = generate_summary(all_results)
    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    stats = generate_stats_report(all_results)
    with open(analysis_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    errors = generate_error_report(all_results)
    with open(analysis_dir / "error_analysis.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\nAnalysis saved to {analysis_dir}/")

    # Generate figures
    try:
        from visualize import (
            plot_cognitive_radar, plot_learning_curves,
            plot_view_mode_heatmap, plot_strategy_analysis,
            generate_latex_table,
        )
        fig_dir = analysis_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_cognitive_radar(all_results, fig_dir)
        plot_learning_curves(all_results, fig_dir)
        plot_view_mode_heatmap(all_results, fig_dir)
        plot_strategy_analysis(all_results, fig_dir)
        generate_latex_table(all_results, fig_dir)
        print(f"Figures saved to {fig_dir}/")
    except Exception as e:
        print(f"Figure generation failed: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CheeseBench — Full Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  multi_model      6+ models across all environments (main table)
  scaling          Qwen2.5-VL 3B→72B scaling law
  ablation_prompt  4 prompt format variants
  ablation_history 4 history length settings
  ablation_actions 4 action budget settings
  ablation_vision  VLM vs text-only at same scale
  image_mode       TOPDOWN_2D image view across all open-weight VLMs

Examples:
  python run_experiments.py --all                    # Full paper
  python run_experiments.py --exp multi_model        # Just main table
  python run_experiments.py --smoke-test             # Quick 2-trial test
  python run_experiments.py --analyze                # Analyze existing results
        """,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run ALL experiments (full paper)")
    parser.add_argument("--exp", type=str,
                        choices=["multi_model", "scaling", "ablation_prompt",
                                 "ablation_history", "ablation_actions",
                                 "ablation_vision", "image_mode"],
                        help="Run a specific experiment")
    parser.add_argument("--num-trials", type=int, default=20,
                        help="Trials per env (default: 20)")
    parser.add_argument("--ablation-trials", type=int, default=15,
                        help="Trials for ablation studies (default: 15)")
    parser.add_argument("--ablation-model", type=str, default="qwen2.5vl-7b",
                        help="Model for ablation studies (default: qwen2.5vl-7b)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test: 2 trials, 1 small model")
    parser.add_argument("--analyze", action="store_true",
                        help="Run analysis on existing results")
    parser.add_argument("--install-vllm", action="store_true",
                        help="Install vLLM before running")
    args = parser.parse_args()

    # Ensure vLLM is available
    if args.install_vllm or (not args.analyze and not check_vllm_installed()):
        install_vllm()

    print("=" * 60)
    print("CheeseBench — NeurIPS Experiment Suite")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.analyze:
        run_analysis()
        return

    if args.smoke_test:
        print("\n--- SMOKE TEST (2 trials, qwen2.5vl-3b) ---")
        model = get_model("qwen2.5vl-3b")
        cfg = BenchmarkConfig()
        cfg.num_trials = 2
        cfg.max_steps_per_trial = 50
        serve_and_run(model, cfg, "smoke_test")
        return

    if args.all:
        exp_multi_model(args.num_trials)
        exp_scaling(args.num_trials)
        exp_ablation_prompt(args.ablation_trials, args.ablation_model)
        exp_ablation_history(args.ablation_trials, args.ablation_model)
        exp_ablation_actions(args.ablation_trials, args.ablation_model)
        exp_ablation_vision(args.ablation_trials)
        exp_image_mode(args.num_trials)
        run_analysis()
        return

    if args.exp == "multi_model":
        exp_multi_model(args.num_trials)
    elif args.exp == "scaling":
        exp_scaling(args.num_trials)
    elif args.exp == "ablation_prompt":
        exp_ablation_prompt(args.ablation_trials, args.ablation_model)
    elif args.exp == "ablation_history":
        exp_ablation_history(args.ablation_trials, args.ablation_model)
    elif args.exp == "ablation_actions":
        exp_ablation_actions(args.ablation_trials, args.ablation_model)
    elif args.exp == "ablation_vision":
        exp_ablation_vision(args.ablation_trials)
    elif args.exp == "image_mode":
        exp_image_mode(args.num_trials)
    else:
        parser.print_help()
        return

    # Always run analysis after experiments
    run_analysis()

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
