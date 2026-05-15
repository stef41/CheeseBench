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

# View modes that count toward each tier's success rate.
# The text (ASCII) tier must exclude TOPDOWN_2D, which is an image view mode
# (cf. benchmark.IMAGE_VIEW_MODES). Without this filter, multi_model runs that
# happened to include TOPDOWN_2D rows would silently inject image-mode wins
# into the headline ASCII numbers.
ALLOWED_VIEW_MODES = {
    "text (ASCII)": {"ASCII_2D", "ASCII_2D_FPV", "ASCII_3D"},
    "vision": {"TOPDOWN_2D", "FPV_3D"},
}

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


def _aggregate(results_path: Path, modality: str) -> dict | None:
    """Return aggregated row from a benchmark_results.json, or None if empty.

    Overall = mean of per-env best-view-mode success rates (matches the
    metric used in the CheeseBench paper / analysis pipeline). Only view
    modes in ALLOWED_VIEW_MODES[modality] are considered.
    """
    try:
        data = json.loads(results_path.read_text())
    except Exception:
        return None
    allowed = ALLOWED_VIEW_MODES.get(modality)
    # per_env_by_view[env][view_mode] = result row
    per_env_by_view: dict[str, dict[str, dict]] = defaultdict(dict)
    total_trials = 0
    n_views: set[str] = set()
    for r in data.get("results", []):
        if r.get("agent_type") != "LLM":
            continue
        vm = r["view_mode"]
        if allowed is not None and vm not in allowed:
            continue
        total_trials += r["total_trials"]
        n_views.add(vm)
        per_env_by_view[r["env_name"]][vm] = r
    if not per_env_by_view:
        return None

    per_env = {}
    best_rates: list[float] = []
    # Per-view-mode overall = mean across envs of that view mode's success rate
    by_view_rates: dict[str, list[float]] = defaultdict(list)
    for env in ENV_ORDER:
        view_rows = per_env_by_view.get(env, {})
        if view_rows:
            best_vm = max(view_rows, key=lambda v: view_rows[v]["success_rate"])
            best = view_rows[best_vm]
            best_rates.append(best["success_rate"])
            for vm, row in view_rows.items():
                by_view_rates[vm].append(row["success_rate"])
            per_env[env] = {
                "success_rate": best["success_rate"],
                "best_view_mode": best_vm,
                "view_mode_scores": {
                    vm: row["success_rate"] for vm, row in view_rows.items()
                },
            }
        else:
            per_env[env] = {
                "success_rate": None,
                "best_view_mode": None,
                "view_mode_scores": {},
            }
    overall = sum(best_rates) / len(best_rates) if best_rates else 0.0
    overall_by_view = {
        vm: (sum(rates) / len(rates)) if rates else None
        for vm, rates in by_view_rates.items()
    }
    return {
        "overall": overall,
        "overall_by_view": overall_by_view,
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
        agg = _aggregate(res, modality)
        if not agg:
            continue
        rows.append({
            "model": short(sub.name),
            "model_id": sub.name,
            "modality": modality,
            "source": source,
            "overall": agg["overall"],
            "overall_by_view": agg["overall_by_view"],
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
        "schema_version": "4",
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
