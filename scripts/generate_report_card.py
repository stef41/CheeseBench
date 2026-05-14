"""Generate a per-model markdown report card from a benchmark_results.json.

Usage:
    python scripts/generate_report_card.py results/copilot_eval/<model>/benchmark_results.json
    python scripts/generate_report_card.py results/copilot_eval/<model>/benchmark_results.json --out cards/<model>.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "task_definitions.json"

ENV_ORDER = [
    "MorrisWaterMaze", "TMaze", "BarnesMaze", "RadialArmMaze",
    "OperantChamber", "ShuttleBox", "PlacePreference", "StarMaze", "DNMSTask",
]


def _wilson(s: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _baseline_pct(env: dict | float | None) -> float | None:
    if isinstance(env, dict):
        curve = env.get("learning_curve") or []
        if curve:
            return curve[-1] * 100.0
    if isinstance(env, (int, float)):
        return float(env) * 100.0
    return None


def build_card(results_path: Path) -> str:
    data = json.loads(results_path.read_text())
    tasks = json.loads(TASKS_PATH.read_text())
    baselines = {e["name"]: e.get("animal_baseline") for e in tasks["environments"]}

    model = data.get("model", results_path.parent.name)
    n_trials = data.get("num_trials")
    max_steps = data.get("max_steps_per_trial")
    seed = data.get("seed")

    # Aggregate LLM rows: per env best across view modes; per env totals
    per_env_rows: dict[str, dict] = {}
    per_env_totals: dict[str, dict] = defaultdict(lambda: {"s": 0, "t": 0})
    overall_s = overall_t = 0
    for r in data["results"]:
        if r.get("agent_type") != "LLM":
            continue
        env = r["env_name"]
        s, t = r["successes"], r["total_trials"]
        per_env_totals[env]["s"] += s
        per_env_totals[env]["t"] += t
        overall_s += s
        overall_t += t
        prev = per_env_rows.get(env)
        if prev is None or r["success_rate"] > prev["success_rate"]:
            per_env_rows[env] = r

    # Headline metric (matches paper / leaderboard): mean of per-env best
    best_rates = [per_env_rows[e]["success_rate"] for e in ENV_ORDER if e in per_env_rows]
    overall_pct = (sum(best_rates) / len(best_rates) * 100) if best_rates else 0.0
    # Pooled (raw) rate for Wilson CI on total trials
    pooled_pct = (overall_s / overall_t * 100) if overall_t else 0.0
    pooled_lo, pooled_hi = _wilson(overall_s, overall_t)

    # Random baseline (overall, pooled)
    rs = rt = 0
    for r in data["results"]:
        if r.get("agent_type") == "Random":
            rs += r["successes"]
            rt += r["total_trials"]
    random_pct = (rs / rt * 100) if rt else 0.0

    lines: list[str] = []
    lines.append(f"# CheeseBench Report Card — `{model}`")
    lines.append("")
    lines.append(f"_Generated {date.today().isoformat()}_  ")
    lines.append(
        f"Trials/env: **{n_trials}** · Max steps/trial: **{max_steps}** · Seed: **{seed}** · "
        f"View modes evaluated: **{len(set(r['view_mode'] for r in data['results']))}**"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **Overall LLM success rate**: **{overall_pct:.1f}%** "
        f"(mean of per-env best view mode — paper metric)"
    )
    lines.append(
        f"- Pooled rate across all view modes: {pooled_pct:.1f}% "
        f"(Wilson 95% CI: {pooled_lo*100:.1f}–{pooled_hi*100:.1f}%, n={overall_t} trials)"
    )
    lines.append(f"- Random baseline (pooled): {random_pct:.1f}%")
    lines.append(
        f"- **Lift over random**: {pooled_pct - random_pct:+.1f} pp"
    )
    lines.append("")

    lines.append("## Per-Environment Results (best view mode)")
    lines.append("")
    lines.append("| Environment | Best View | LLM % | 95% CI | Rodent baseline | Δ vs rodent |")
    lines.append("|---|---|---:|:---:|---:|---:|")
    for env in ENV_ORDER:
        row = per_env_rows.get(env)
        if row is None:
            lines.append(f"| {env} | — | — | — | — | — |")
            continue
        s, t = row["successes"], row["total_trials"]
        pct = row["success_rate"] * 100
        lo, hi = _wilson(s, t)
        bl = _baseline_pct(baselines.get(env))
        bl_str = f"{bl:.0f}%" if bl is not None else "—"
        delta = f"{pct - bl:+.1f} pp" if bl is not None else "—"
        lines.append(
            f"| {env} | `{row['view_mode']}` | {pct:.1f} | "
            f"[{lo*100:.1f}, {hi*100:.1f}] | {bl_str} | {delta} |"
        )
    lines.append("")

    lines.append("## All View Modes")
    lines.append("")
    # Build view-mode matrix
    view_modes = sorted({r["view_mode"] for r in data["results"]})
    header = ["Environment"] + view_modes
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + [":---:"] * len(view_modes)) + "|")
    matrix: dict[tuple[str, str], float] = {}
    for r in data["results"]:
        if r.get("agent_type") == "LLM":
            matrix[(r["env_name"], r["view_mode"])] = r["success_rate"]
    for env in ENV_ORDER:
        cells = [env]
        for v in view_modes:
            sr = matrix.get((env, v))
            cells.append(f"{sr*100:.1f}%" if sr is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install cheesebench")
    view_args = " ".join(view_modes)
    lines.append(
        f"cheesebench --model {model} \\\n"
        f"    --num-trials {n_trials} --max-steps {max_steps} --seed {seed} \\\n"
        f"    --view-modes {view_args} \\\n"
        f"    --api-url <YOUR_OPENAI_COMPATIBLE_ENDPOINT> --api-format openai"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "_Submit the resulting `benchmark_results.json` to "
        "[stef41/CheeseBench](https://github.com/stef41/CheeseBench) "
        "to appear on the [leaderboard](https://huggingface.co/spaces/zachz/cheesebench-leaderboard)._"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="Path to benchmark_results.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    md = build_card(args.results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"Wrote {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
