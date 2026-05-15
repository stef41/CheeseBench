"""CheeseBench Leaderboard — Gradio Space."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

LB_PATH = Path(__file__).with_name("leaderboard.json")
DATA = json.loads(LB_PATH.read_text())

ENVS = DATA["envs"]
BASELINES = DATA["rodent_baselines"]
ENTRIES = DATA["entries"]


def overall_table() -> pd.DataFrame:
    rows = []
    for e in ENTRIES:
        rows.append({
            "Model": e["model"],
            "Source": e.get("source", "—"),
            "Modality": e["modality"],
            "Overall": round((e["overall"] or 0) * 100, 1),
            **{env: round((e["per_environment"][env]["success_rate"] or 0) * 100, 1) for env in ENVS},
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Overall", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df.reset_index()


def _baseline_pct(env: str) -> float | str:
    b = BASELINES.get(env)
    if isinstance(b, dict):
        curve = b.get("learning_curve") or []
        if curve:
            return round(curve[-1] * 100, 1)
    if isinstance(b, (int, float)):
        return round(b * 100, 1)
    return "—"


def baseline_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "Source": "Rodent baseline (peer-reviewed asymptote, %)",
        **{env: _baseline_pct(env) for env in ENVS},
    }])


with gr.Blocks(title="CheeseBench Leaderboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧀 CheeseBench Leaderboard
        Large Language Models (and Vision-Language Models in image mode)
        evaluated on 9 classical rodent behavioral neuroscience paradigms.
        Higher is better. Each cell is the per-environment success rate (%)
        at the model's best view mode.

        - **Paper:** *CheeseBench: Evaluating Large Language Models on Rodent-Level Cognition*
        - **Code:** https://github.com/stef41/CheeseBench
        - **PyPI:** `pip install cheesebench`
        """
    )

    with gr.Tab("Leaderboard"):
        gr.Dataframe(
            value=overall_table(),
            interactive=False,
            wrap=True,
        )
        gr.Markdown("### Reference: rodent baselines from the original protocols")
        gr.Dataframe(value=baseline_row(), interactive=False, wrap=True)

    with gr.Tab("Submit a model"):
        gr.Markdown(
            """
            ## Run CheeseBench on your model

            ```bash
            pip install cheesebench
            cheesebench \\
                --model your-model-name \\
                --api-url https://your-openai-compatible-endpoint/v1/chat/completions \\
                --api-format openai \\
                --num-trials 10
            ```

            Then open a pull request against
            [stef41/CheeseBench](https://github.com/stef41/CheeseBench)
            adding your `benchmark_results.json` under
            `results/multi_model/<your-model>/` (text/ASCII modality) or
            `results/image_mode/<your-model>/` (vision modality). The leaderboard is
            rebuilt automatically on merge.
            """
        )

    with gr.Tab("Citation"):
        gr.Code(
            value=(
                "@inproceedings{cheesebench2025,\n"
                "  title={CheeseBench: Evaluating Large Language Models on Rodent-Level Cognition},\n"
                "  author={CheeseBench Contributors},\n"
                "  booktitle={NeurIPS Datasets and Benchmarks Track},\n"
                "  year={2025}\n"
                "}\n"
            ),
            language="python",
            label="BibTeX",
        )

    gr.Markdown("---")
    gr.Markdown("## How CheeseBench works")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "**9 environments × 1 unified protocol.** Each environment "
                "implements a classical rodent paradigm (Morris Water Maze, "
                "Barnes Maze, T-Maze, Radial Arm Maze, Operant Chamber, "
                "Shuttle Box, Conditioned Place Preference, Star Maze, DNMS). "
                "The model receives no task-specific hint — it must discover "
                "the goal from observation and reward alone."
            )
            gr.Image(
                value="fig_environments.png",
                label="The 9 environments",
                show_label=True,
                interactive=False,
            )

        with gr.Column():
            gr.Markdown(
                "**3 view modes per environment.** ASCII top-down (full map), "
                "ASCII first-person (egocentric partial view), and pseudo-3D "
                "ASCII (depth cues, narrow FOV). The headline score is each "
                "model's *best* per-environment view-mode success rate."
            )
            gr.Image(
                value="fig_viewmodes.png",
                label="View modes (top-down / FPV / pseudo-3D)",
                show_label=True,
                interactive=False,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "**Animated example.** All 9 environments running in their "
                "top-down ASCII mode — the agent's position is shown by an "
                "arrow; walls (`#`) block movement."
            )
            gr.Image(
                value="all_envs_top_down.gif",
                label="All environments — top-down ASCII",
                show_label=True,
                interactive=False,
            )

        with gr.Column():
            gr.Markdown(
                "**Cognitive profile.** Per-cognitive-dimension success rates "
                "for the best open-weight model, overlaid on rodent reference "
                "baselines from the original published protocols."
            )
            gr.Image(
                value="fig_cognitive_radar.png",
                label="Model vs rodent baselines",
                show_label=True,
                interactive=False,
            )

    gr.Markdown(
        "Each environment is grounded in a peer-reviewed rodent study with "
        "quantitative animal baselines (see [task_definitions.json]"
        "(https://github.com/stef41/CheeseBench/blob/main/task_definitions.json)). "
        "Full methodology in the [paper]"
        "(https://github.com/stef41/CheeseBench/blob/main/paper/cheesebench.pdf)."
    )


if __name__ == "__main__":
    demo.launch()
