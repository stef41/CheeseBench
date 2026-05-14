# CheeseBench Report Card — `claude-haiku-4.5`

_Generated 2026-05-14_  
Trials/env: **10** · Max steps/trial: **200** · Seed: **42** · View modes evaluated: **3**

## Headline

- **Overall LLM success rate**: **70.0%** (mean of per-env best view mode — paper metric)
- Pooled rate across all view modes: 43.3% (Wilson 95% CI: 37.6–49.3%, n=270 trials)
- Random baseline (pooled): 30.4%
- **Lift over random**: +13.0 pp

## Per-Environment Results (best view mode)

| Environment | Best View | LLM % | 95% CI | Rodent baseline | Δ vs rodent |
|---|---|---:|:---:|---:|---:|
| MorrisWaterMaze | `ASCII_2D_FPV` | 100.0 | [72.2, 100.0] | 85% | +15.0 pp |
| TMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| BarnesMaze | `ASCII_2D` | 60.0 | [31.3, 83.2] | 80% | -20.0 pp |
| RadialArmMaze | `ASCII_2D` | 10.0 | [1.8, 40.4] | 70% | -60.0 pp |
| OperantChamber | `ASCII_2D` | 70.0 | [39.7, 89.2] | 90% | -20.0 pp |
| ShuttleBox | `ASCII_2D` | 60.0 | [31.3, 83.2] | 70% | -10.0 pp |
| PlacePreference | `ASCII_2D` | 80.0 | [49.0, 94.3] | 75% | +5.0 pp |
| StarMaze | `ASCII_2D` | 70.0 | [39.7, 89.2] | 80% | -10.0 pp |
| DNMSTask | `ASCII_3D` | 80.0 | [49.0, 94.3] | 80% | +0.0 pp |

## All View Modes

| Environment | ASCII_2D | ASCII_2D_FPV | ASCII_3D |
|---|:---:|:---:|:---:|
| MorrisWaterMaze | 30.0% | 100.0% | 100.0% |
| TMaze | 100.0% | 90.0% | 100.0% |
| BarnesMaze | 60.0% | 30.0% | 0.0% |
| RadialArmMaze | 10.0% | 0.0% | 0.0% |
| OperantChamber | 70.0% | 0.0% | 0.0% |
| ShuttleBox | 60.0% | 0.0% | 0.0% |
| PlacePreference | 80.0% | 40.0% | 40.0% |
| StarMaze | 70.0% | 20.0% | 0.0% |
| DNMSTask | 50.0% | 40.0% | 80.0% |

## Reproduce

```bash
pip install cheesebench
cheesebench --model claude-haiku-4.5 \
    --num-trials 10 --max-steps 200 --seed 42 \
    --view-modes ASCII_2D ASCII_2D_FPV ASCII_3D \
    --api-url <YOUR_OPENAI_COMPATIBLE_ENDPOINT> --api-format openai
```

_Submit the resulting `benchmark_results.json` to [stef41/CheeseBench](https://github.com/stef41/CheeseBench) to appear on the [leaderboard](https://huggingface.co/spaces/zachz/cheesebench-leaderboard)._