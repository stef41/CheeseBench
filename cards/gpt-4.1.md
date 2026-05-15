# CheeseBench Report Card — `gpt-4.1`

_Generated 2026-05-15_  
Trials/env: **3** · Max steps/trial: **200** · Seed: **42** · View modes evaluated: **3**

## Headline

- **Overall LLM success rate**: **55.6%** (mean of per-env best view mode — paper metric)
- Pooled rate across all view modes: 33.3% (Wilson 95% CI: 24.0–44.1%, n=81 trials)
- Random baseline (pooled): 33.3%
- **Lift over random**: +0.0 pp

## Per-Environment Results (best view mode)

| Environment | Best View | LLM % | 95% CI | Rodent baseline | Δ vs rodent |
|---|---|---:|:---:|---:|---:|
| MorrisWaterMaze | `ASCII_2D` | 33.3 | [6.1, 79.2] | 85% | -51.7 pp |
| TMaze | `ASCII_2D` | 66.7 | [20.8, 93.9] | 80% | -13.3 pp |
| BarnesMaze | `ASCII_2D` | 66.7 | [20.8, 93.9] | 80% | -13.3 pp |
| RadialArmMaze | `ASCII_2D` | 0.0 | [0.0, 56.2] | 70% | -70.0 pp |
| OperantChamber | `ASCII_2D_FPV` | 100.0 | [43.8, 100.0] | 90% | +10.0 pp |
| ShuttleBox | `ASCII_2D` | 33.3 | [6.1, 79.2] | 70% | -36.7 pp |
| PlacePreference | `ASCII_2D` | 66.7 | [20.8, 93.9] | 75% | -8.3 pp |
| StarMaze | `ASCII_2D` | 66.7 | [20.8, 93.9] | 80% | -13.3 pp |
| DNMSTask | `ASCII_2D` | 66.7 | [20.8, 93.9] | 80% | -13.3 pp |

## All View Modes

| Environment | ASCII_2D | ASCII_2D_FPV | ASCII_3D |
|---|:---:|:---:|:---:|
| MorrisWaterMaze | 33.3% | 0.0% | 0.0% |
| TMaze | 66.7% | 0.0% | 66.7% |
| BarnesMaze | 66.7% | 33.3% | 33.3% |
| RadialArmMaze | 0.0% | 0.0% | 0.0% |
| OperantChamber | 0.0% | 100.0% | 0.0% |
| ShuttleBox | 33.3% | 33.3% | 0.0% |
| PlacePreference | 66.7% | 66.7% | 33.3% |
| StarMaze | 66.7% | 33.3% | 33.3% |
| DNMSTask | 66.7% | 33.3% | 33.3% |

## Reproduce

```bash
pip install cheesebench
cheesebench --model gpt-4.1 \
    --num-trials 3 --max-steps 200 --seed 42 \
    --view-modes ASCII_2D ASCII_2D_FPV ASCII_3D \
    --api-url <YOUR_OPENAI_COMPATIBLE_ENDPOINT> --api-format openai
```

_Submit the resulting `benchmark_results.json` to [stef41/CheeseBench](https://github.com/stef41/CheeseBench) to appear on the [leaderboard](https://huggingface.co/spaces/zachz/cheesebench-leaderboard)._