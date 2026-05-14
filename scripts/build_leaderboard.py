"""Build a leaderboard JSON by scanning every benchmark_results.json under
results/ — guaranteeing no data gaps even if results/analysis/summary.json
is stale.

Output: leaderboard/leaderboard.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
TASKS = ROOT / "task_definitions.json"
OUT = ROOT / "leaderboard" / "leaderboard.json"

ENV_ORDER = [
    "MorrisWaterMaze", "TMaze", "BarnesMaze", "RadialArmMaze",
    "OperantChamber", "ShuttleBox", "PlacePreference", "StarMaze", "DNMSTask",
]

# Friendly model display names
DISPLAY = {
    "internvl2.5-8b": "InternVL2.5-8B",
    "phi-4-mm-14b": "Phi-4-Multimodal-14B",
    "qwen2.5vl-3b": "Qwen2.5-VL-3B",
    "qwen2.5vl-7b": "Qwen2.5-VL-7B",
    "qwen2.5vl-32b": "Qwen2.5-VL-32B",
    "qwen2.5vl-72b": "Qwen2.5-VL-72B",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4.7": "Claude Opus 4.7",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-codex": "GPT-5.2 Codex",
}

# Source dirs to scan: (subdir-under-results, modality, source-label)
SOURCES = [
    ("multi_model", "text (ASCII)", "open-weights/vLLM"),
    ("image_mode", "vision", "open-weights/vLLM"),
    ("copilot_eval", "text (ASCII)", "GitHub Copilot CLI"),
]


def short(name: str) -> str:
    return DISPLAY.get(name, name)


def _aggregate(results_path: Path) -> dict | None:
    """Return aggregated row from a benchmark_results.json, or None if empty.

    Overall = mean of per-env best-view-mode success rates (matches the
    metric used in the CheeseBench paper / analysis pipeline).
    """
    try:
        data = json.loads(results_path.read_text())
    except Exception:
        return None
    per_env_best: dict[str, dict] = {}
    total_trials = 0
    n_views: set[str] = set()
    for r in data.get("results", []):
        if r.get("agent_type") != "LLM":
            continue
        total_trials += r["total_trials"]
        n_views.add(r["view_mode"])
        prev = per_env_best.get(r["env_name"])
        if prev is None or r["success_rate"] > prev["success_rate"]:
            per_env_best[r["env_name"]] = r
    if not per_env_best:
        return None
    per_env = {}
    best_rates: list[float] = []
    for env in ENV_ORDER:
        row = per_env_best.get(env)
        if row is not None:
            best_rates.append(row["success_rate"])
        per_env[env] = {
            "success_rate": row["success_rate"] if row else None,
            "best_view_mode": row["view_mode"] if row else None,
        }
    overall = sum(best_rates) / len(best_rates) if best_rates else 0.0
    return {
        "overall": overall,
        "per_environment": per_env,
        "total_trials": total_trials,
        "view_modes": sorted(n_views),
    }


def _scan_source(subdir: str, modality: str, source: str) -> list[dict]:
    base = RESULTS_ROOT / subdir
    rows: list[dict] = []
    if not base.is_dir():
        return rows
    for sub in sorted(base.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        res = sub / "benchmark_results.json"
        if not res.is_file():
            continue
        agg = _aggregate(res)
        if not agg:
            continue
        rows.append({
            "model": short(sub.name),
            "model_id": sub.name,
            "modality": modality,
            "source": source,
            "overall": agg["overall"],
            "per_environment": agg["per_environment"],
            "total_trials": agg["total_trials"],
            "view_modes": agg["view_modes"],
        })
    return rows


def build() -> dict:
    tasks = json.loads(TASKS.read_text())
    baselines = {e["name"]: e.get("animal_baseline") for e in tasks["environments"]}

    rows: list[dict] = []
    for subdir, modality, source in SOURCES:
        rows.extend(_scan_source(subdir, modality, source))
    rows.sort(key=lambda r: (r["overall"] or 0), reverse=True)

    return {
        "schema_version": "3",
        "envs": ENV_ORDER,
        "rodent_baselines": baselines,
        "entries": rows,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} with {len(payload['entries'])} entries")



if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT} with {len(payload['entries'])} entries")
