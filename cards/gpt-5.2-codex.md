# CheeseBench Report Card — `gpt-5.2-codex`

_Generated 2026-05-15_  
Trials/env: **10** · Max steps/trial: **200** · Seed: **42** · View modes evaluated: **3**

## Headline

- **Overall LLM success rate**: **82.2%** (mean of per-env best view mode — paper metric)
- Pooled rate across all view modes: 62.2% (Wilson 95% CI: 56.3–67.8%, n=270 trials)
- Random baseline (pooled): 31.9%
- **Lift over random**: +30.4 pp

## Per-Environment Results (best view mode)

| Environment | Best View | LLM % | 95% CI | Rodent baseline | Δ vs rodent |
|---|---|---:|:---:|---:|---:|
| MorrisWaterMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 85% | +15.0 pp |
| TMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| BarnesMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| RadialArmMaze | `ASCII_2D` | 50.0 | [23.7, 76.3] | 70% | -20.0 pp |
| OperantChamber | `ASCII_2D` | 100.0 | [72.2, 100.0] | 90% | +10.0 pp |
| ShuttleBox | `ASCII_2D_FPV` | 20.0 | [5.7, 51.0] | 70% | -50.0 pp |
| PlacePreference | `ASCII_2D` | 100.0 | [72.2, 100.0] | 75% | +25.0 pp |
| StarMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| DNMSTask | `ASCII_2D` | 70.0 | [39.7, 89.2] | 80% | -10.0 pp |

## All View Modes

| Environment | ASCII_2D | ASCII_2D_FPV | ASCII_3D |
|---|:---:|:---:|:---:|
| MorrisWaterMaze | 100.0% | 100.0% | 0.0% |
| TMaze | 100.0% | 100.0% | 90.0% |
| BarnesMaze | 100.0% | 40.0% | 70.0% |
| RadialArmMaze | 50.0% | 10.0% | 0.0% |
| OperantChamber | 100.0% | 90.0% | 100.0% |
| ShuttleBox | 10.0% | 20.0% | 10.0% |
| PlacePreference | 100.0% | 100.0% | 40.0% |
| StarMaze | 100.0% | 60.0% | 20.0% |
| DNMSTask | 70.0% | 40.0% | 60.0% |

## Reproduce

```bash
pip install cheesebench
cheesebench --model gpt-5.2-codex \
    --num-trials 10 --max-steps 200 --seed 42 \
    --view-modes ASCII_2D ASCII_2D_FPV ASCII_3D \
    --api-url <YOUR_OPENAI_COMPATIBLE_ENDPOINT> --api-format openai
```

_Submit the resulting `benchmark_results.json` to [stef41/CheeseBench](https://github.com/stef41/CheeseBench) to appear on the [leaderboard](https://huggingface.co/spaces/zachz/cheesebench-leaderboard)._