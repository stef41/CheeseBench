#!/usr/bin/env python3
"""
CheeseBench — Local Model Server Manager

Spins up vLLM (or HuggingFace Transformers) serving local VLMs with an
OpenAI-compatible chat/completions API for the benchmark to talk to.

Usage:
    # Serve a single model
    python model_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --port 8000

    # List recommended models for the benchmark
    python model_server.py --list

    # Serve and immediately run benchmark
    python model_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --run-benchmark
"""

import argparse
import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# ============================================================================
# Model Registry — all models to benchmark
# ============================================================================

@dataclass
class ModelSpec:
    """Specification for a local model."""
    name: str               # Short name for results
    hf_id: str              # HuggingFace model ID
    size_b: float           # Approximate size in billions of params
    family: str             # Architecture family
    tp: int                 # Tensor-parallel GPUs needed
    max_model_len: int      # Context window to use
    vision: bool = True     # Whether it's a VLM (vs text-only)
    trust_remote_code: bool = True
    notes: str = ""


# Models selected for diversity: size (3B→72B), architecture (Qwen, Llama,
# InternVL, Phi), and capability (vision vs text-only ablation).
# All fit on 8×H100 80GB; smaller models on fewer GPUs.
MODEL_REGISTRY: List[ModelSpec] = [
    # --- Small (≤ 8B) ---
    ModelSpec(
        name="qwen2.5vl-3b",
        hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        size_b=3.0, family="Qwen2.5-VL", tp=1,
        max_model_len=32768,
        notes="Smallest VLM baseline",
    ),
    ModelSpec(
        name="qwen2.5vl-7b",
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        size_b=7.0, family="Qwen2.5-VL", tp=1,
        max_model_len=32768,
        notes="Mid-small VLM",
    ),
    # --- Medium (8B–30B) ---
    ModelSpec(
        name="internvl2.5-8b",
        hf_id="OpenGVLab/InternVL2_5-8B",
        size_b=8.0, family="InternVL2.5", tp=1,
        max_model_len=32768,
        notes="Different VLM architecture for diversity",
    ),
    ModelSpec(
        name="phi-4-mm-14b",
        hf_id="microsoft/Phi-4-multimodal-instruct",
        size_b=14.0, family="Phi-4", tp=1,
        max_model_len=32768,
        notes="Microsoft VLM, different training approach",
    ),
    # --- Large (30B+) ---
    ModelSpec(
        name="qwen2.5vl-32b",
        hf_id="Qwen/Qwen2.5-VL-32B-Instruct",
        size_b=32.0, family="Qwen2.5-VL", tp=2,
        max_model_len=32768,
        notes="Large VLM, same family as 3B/7B for scaling analysis",
    ),
    ModelSpec(
        name="qwen2.5vl-72b",
        hf_id="Qwen/Qwen2.5-VL-72B-Instruct",
        size_b=72.0, family="Qwen2.5-VL", tp=4,
        max_model_len=32768,
        notes="Largest local VLM — ceiling performance",
    ),
    # --- Text-only ablation (same family, no vision) ---
    ModelSpec(
        name="qwen2.5-7b-text",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        size_b=7.0, family="Qwen2.5", tp=1,
        max_model_len=32768,
        vision=False,
        notes="Text-only ablation — same size as qwen2.5vl-7b",
    ),
    ModelSpec(
        name="qwen2.5-32b-text",
        hf_id="Qwen/Qwen2.5-32B-Instruct",
        size_b=32.0, family="Qwen2.5", tp=2,
        max_model_len=32768,
        vision=False,
        notes="Text-only ablation — same size as qwen2.5vl-32b",
    ),
]


def get_model(name: str) -> Optional[ModelSpec]:
    """Look up a model by short name."""
    for m in MODEL_REGISTRY:
        if m.name == name:
            return m
    return None


def list_models():
    """Print the model registry."""
    print(f"{'Name':<22} {'HF ID':<45} {'Size':>5} {'TP':>3} {'Vision':>6}")
    print("-" * 90)
    for m in MODEL_REGISTRY:
        v = "Yes" if m.vision else "No"
        print(f"{m.name:<22} {m.hf_id:<45} {m.size_b:>4.0f}B {m.tp:>3} {v:>6}")
    print(f"\nTotal: {len(MODEL_REGISTRY)} models")


# ============================================================================
# Server Management
# ============================================================================

def check_vllm_installed() -> bool:
    """Check if vLLM is available."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def install_vllm():
    """Install vLLM."""
    print("Installing vLLM...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "vllm>=0.6.0", "--quiet"
    ])
    print("vLLM installed successfully.")


def wait_for_server(port: int, timeout: int = 300, expected_model: str = None) -> bool:
    """Wait for the vLLM server to be ready, optionally verifying the served model."""
    import requests
    start = time.time()
    url = f"http://localhost:{port}/v1/models"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    served = models[0]['id']
                    if expected_model and served != expected_model:
                        print(f"  WARNING: Server serves {served}, expected {expected_model}. Killing stale server...")
                        _kill_stale_vllm(port)
                        time.sleep(5)
                        continue
                    print(f"  Server ready — model: {served}")
                    return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  ERROR: Server did not start within {timeout}s")
    return False


def _kill_stale_vllm(port: int = 8000):
    """Kill any lingering vLLM processes and wait for GPU memory to free."""
    # Kill anything listening on the target port
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        capture_output=True,
    )
    # Also kill any vLLM processes
    subprocess.run(
        ["pkill", "-9", "-f", "vllm.entrypoints"],
        capture_output=True,
    )
    # Wait for GPU memory to actually free (up to 30s)
    for i in range(30):
        time.sleep(1)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            used = [int(x.strip()) for x in out.strip().split("\n")]
            if all(u < 1000 for u in used):  # < 1GB on all GPUs
                return
        except Exception:
            return
    print("  WARNING: GPU memory did not fully free after 30s")


def launch_vllm_server(
    model: ModelSpec,
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server for the given model."""
    # Kill any stale vLLM processes and wait for GPU memory to free
    _kill_stale_vllm()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model.hf_id,
        "--port", str(port),
        "--tensor-parallel-size", str(model.tp),
        "--max-model-len", str(model.max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", "auto",
    ]
    if model.trust_remote_code:
        cmd.append("--trust-remote-code")

    # vLLM serves at /v1/chat/completions (OpenAI-compatible)
    log_path = Path("results") / "server_logs" / f"{model.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Launching vLLM server for {model.name} ({model.hf_id})...")
    print(f"  TP={model.tp}, port={port}, ctx={model.max_model_len}")
    print(f"  Log: {log_path}")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    if not wait_for_server(port, expected_model=model.hf_id):
        proc.terminate()
        log_file.close()
        # Print last 20 lines of log for debugging
        with open(log_path) as f:
            lines = f.readlines()
            print("  Last 20 lines of server log:")
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")
        raise RuntimeError(f"vLLM server for {model.name} failed to start")

    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop a vLLM server."""
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CheeseBench — Local Model Server Manager"
    )
    parser.add_argument("--list", action="store_true",
                        help="List all models in the registry")
    parser.add_argument("--model", type=str,
                        help="Model short name (from --list) or HF ID")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    parser.add_argument("--install-vllm", action="store_true",
                        help="Install vLLM if not present")
    parser.add_argument("--run-benchmark", action="store_true",
                        help="After launching server, run benchmark")
    parser.add_argument("--num-trials", type=int, default=20,
                        help="Trials per env (if --run-benchmark)")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.install_vllm or not check_vllm_installed():
        install_vllm()
        if not args.model:
            return

    if not args.model:
        parser.print_help()
        return

    # Resolve model
    spec = get_model(args.model)
    if spec is None:
        # Treat as raw HF ID
        spec = ModelSpec(
            name=args.model.split("/")[-1].lower(),
            hf_id=args.model,
            size_b=0, family="custom", tp=1,
            max_model_len=4096,
        )

    proc = launch_vllm_server(spec, port=args.port)

    if args.run_benchmark:
        # Run benchmark against this server
        api_url = f"http://localhost:{args.port}/v1/chat/completions"
        cmd = [
            sys.executable, "benchmark.py",
            "--model", spec.hf_id,
            "--api-url", api_url,
            "--num-trials", str(args.num_trials),
            "--output-dir", f"results/{spec.name}",
        ]
        print(f"\nRunning benchmark: {' '.join(cmd)}")
        subprocess.run(cmd)
        stop_server(proc)
    else:
        print(f"\nServer running at http://localhost:{args.port}/v1/chat/completions")
        print("Press Ctrl+C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            stop_server(proc)
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
