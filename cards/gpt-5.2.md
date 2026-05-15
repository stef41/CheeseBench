# CheeseBench Report Card — `gpt-5.2`

_Generated 2026-05-15_  
Trials/env: **10** · Max steps/trial: **200** · Seed: **42** · View modes evaluated: **3**

## Headline

- **Overall LLM success rate**: **75.6%** (mean of per-env best view mode — paper metric)
- Pooled rate across all view modes: 52.6% (Wilson 95% CI: 46.6–58.5%, n=270 trials)
- Random baseline (pooled): 30.4%
- **Lift over random**: +22.2 pp

## Per-Environment Results (best view mode)

| Environment | Best View | LLM % | 95% CI | Rodent baseline | Δ vs rodent |
|---|---|---:|:---:|---:|---:|
| MorrisWaterMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 85% | +15.0 pp |
| TMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| BarnesMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| RadialArmMaze | `ASCII_2D` | 10.0 | [1.8, 40.4] | 70% | -60.0 pp |
| OperantChamber | `ASCII_2D_FPV` | 100.0 | [72.2, 100.0] | 90% | +10.0 pp |
| ShuttleBox | `ASCII_2D` | 10.0 | [1.8, 40.4] | 70% | -60.0 pp |
| PlacePreference | `ASCII_2D` | 100.0 | [72.2, 100.0] | 75% | +25.0 pp |
| StarMaze | `ASCII_2D` | 100.0 | [72.2, 100.0] | 80% | +20.0 pp |
| DNMSTask | `ASCII_2D` | 60.0 | [31.3, 83.2] | 80% | -20.0 pp |

## All View Modes

| Environment | ASCII_2D | ASCII_2D_FPV | ASCII_3D |
|---|:---:|:---:|:---:|
| MorrisWaterMaze | 100.0% | 40.0% | 0.0% |
| TMaze | 100.0% | 100.0% | 100.0% |
| BarnesMaze | 100.0% | 50.0% | 30.0% |
| RadialArmMaze | 10.0% | 10.0% | 0.0% |
| OperantChamber | 90.0% | 100.0% | 10.0% |
| ShuttleBox | 10.0% | 0.0% | 10.0% |
| PlacePreference | 100.0% | 100.0% | 40.0% |
| StarMaze | 100.0% | 70.0% | 20.0% |
| DNMSTask | 60.0% | 40.0% | 30.0% |

## Reproduce

```bash
pip install cheesebench
cheesebench --model gpt-5.2 \
    --num-trials 10 --max-steps 200 --seed 42 \
    --view-modes ASCII_2D ASCII_2D_FPV ASCII_3D \
    --api-url <YOUR_OPENAI_COMPATIBLE_ENDPOINT> --api-format openai
```

_Submit the resulting `benchmark_results.json` to [stef41/CheeseBench](https://github.com/stef41/CheeseBench) to appear on the [leaderboard](https://huggingface.co/spaces/zachz/cheesebench-leaderboard)._