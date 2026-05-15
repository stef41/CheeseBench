"""
CheeseBench Configuration

Centralized configuration for benchmark runs.
Override via environment variables or CLI arguments.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """All benchmark hyperparameters in one place."""

    # --- Model / API ---
    # api_format: "openai" (vLLM, OpenAI, etc.) or "ollama"
    api_format: str = os.environ.get("CHEESEBENCH_API_FORMAT", "openai")
    api_url: str = os.environ.get(
        "CHEESEBENCH_API_URL", "http://localhost:8000/v1/chat/completions"
    )
    model: str = os.environ.get("CHEESEBENCH_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    api_timeout: int = int(os.environ.get("CHEESEBENCH_TIMEOUT", "120"))
    api_max_retries: int = 2

    # --- Trial Protocol ---
    num_trials: int = 20
    max_steps_per_trial: int = 200
    max_actions_per_call: int = 8

    # --- History Management ---
    max_history_messages: int = 5  # pairs, not individual msgs
    save_thinking_in_history: bool = False
    save_actions_in_history: bool = True  # MUST be True for proper logging

    # --- Reproducibility ---
    seed: int = 42

    # --- View Modes ---
    view_modes: List[str] = field(
        default_factory=lambda: ["ASCII_2D", "ASCII_2D_FPV", "ASCII_3D", "TOPDOWN_2D"]
    )

    # --- Output ---
    output_dir: str = "results"
    save_traces: bool = True
    verbose: bool = True

    # --- Prompt variant (for ablation studies) ---
    prompt_variant: str = "default"  # "default", "minimal", "cot", "few_shot"
