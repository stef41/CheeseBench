# CheeseBench: Evaluating Large Language Models on Rodent Behavioral Neuroscience Paradigms

A benchmark for evaluating Large Language Models (LLMs) — and Vision-Language Models (VLMs) when run with image observations — on 9 classical behavioral neuroscience paradigms, each grounded in published rodent protocols with quantitative animal baselines. The default protocol is text-only (ASCII renderings of the environment), so any chat-completion model can be evaluated; vision is supported as an optional input mode.

## Key Design Principles

1. **Unified Protocol**: Identical system prompt for ALL tasks — no task-specific hints
2. **Published Baselines**: Every environment maps to a real rodent experiment with peer-reviewed success rates
3. **Cognitive Taxonomy**: 6 cognitive dimensions (spatial learning, navigation, working memory, instrumental conditioning, avoidance learning, associative learning) mapped to neural circuits
4. **Multi-Action**: The model outputs up to 8 actions per call with explicit learnings/working memory

## Quick Start

```bash
# Install from PyPI
pip install cheesebench

# Run against any OpenAI-compatible endpoint
cheesebench --model gpt-oss:120b \
    --api-url http://localhost:11434/v1/chat/completions \
    --api-format openai \
    --num-trials 20

# Quick smoke test (1 trial × 1 view mode)
cheesebench --num-trials 1 --view-modes ASCII_2D

# See all options
cheesebench --help
```

📊 **Live leaderboard**: <https://huggingface.co/spaces/zachz/cheesebench-leaderboard>

### Development install

```bash
git clone https://github.com/stef41/CheeseBench
cd CheeseBench
pip install -e .

# Re-run analysis on your results
python analysis.py results/benchmark_results.json
```

## Project Structure

```
cheesebench/
├── benchmark.py           # Main benchmark runner (CLI)
├── config.py              # Centralized configuration
├── analysis.py            # Cognitive profiling & analysis pipeline
├── task_definitions.json  # Task specs with paper citations & animal baselines
├── visualize.py           # Publication-quality figures
├── environments/          # 9 behavioral paradigms
│   ├── base_env.py        # Shared engine (rendering, sessions, actions)
│   ├── morris_water_maze.py
│   ├── t_maze.py
│   ├── barnes_maze.py
│   ├── radial_arm_maze.py
│   ├── operant_chamber.py
│   ├── shuttle_box.py
│   ├── place_preference.py
│   ├── star_maze.py
│   └── dnms_task.py
└── README.md
```

## Environments & Cognitive Taxonomy

| Environment | Cognitive Dimension | Animal Baseline | Citation |
|---|---|---|---|
| Morris Water Maze | Allocentric Spatial Learning | 85% (session 5) | [PMC2895266](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2895266/) — Vorhees & Williams 2006 |
| Barnes Maze | Allocentric Spatial Learning | 80% (session 5) | [PMC6126525](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6126525/) — Vale et al. 2018 |
| T-Maze | Egocentric Nav + Working Memory | 80% (session 4) | [PMC3399492](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3399492/) — Shoji et al. 2012 |
| Star Maze | Allocentric + Egocentric | 80% (session 10) | [PMC4112136](https://academic.oup.com/ilarjournal/article/55/2/310/643871) — Rondi-Reig et al. 2006 |
| Radial Arm Maze | Working Memory | 70% (session 6) | [PMC4030456](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4030456/) — Penley et al. 2013 |
| Operant Chamber | Instrumental Conditioning | 90% (session 5) | [PMC4598097](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4598097/) — Martin & Iceberg 2015 |
| Shuttle Box | Avoidance Learning | 70% (session 10) | [PMC4692667](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4692667/) — Happel et al. 2015 |
| Place Preference | Associative Learning | 75% (session 6) | [PMC6101638](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6101638/) — Blanco-Gandía et al. 2018 |
| DNMS Task | Working Memory | 80% (session 3) | [PMC3982138](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3982138/) — Oomen et al. 2013 |

## View Modes

| Mode | Description | Information Content |
|---|---|---|
| `ASCII_2D` | Top-down bird's-eye map | Full spatial layout |
| `ASCII_2D_FPV` | Rotated first-person 2D | Egocentric partial view |
| `ASCII_3D` | Pseudo-3D ASCII perspective | Depth cues, limited FOV |

## System Prompt (Unified — Identical for ALL Tasks)

The model receives **no task-specific instructions**. It must discover the goal from observation and reward feedback alone:

```
You are an embodied agent placed in a behavioral experiment.
Your only goal is to maximize cumulative reward.

PERCEPTION:
- ASCII rendering (top-down, FPV, or pseudo-3D)
- Position/orientation shown by arrow: ↑ ↗ → ↘ ↓ ↙ ← ↖
- Walls (#, █) block movement. Open spaces are traversable.

ACTIONS (egocentric):
- FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY

RESPONSE FORMAT:
LEARNINGS: <working memory — position, strategy, hypotheses>
ACTIONS: <1-8 comma-separated actions>
```

## Analysis Pipeline

The analysis module computes:
- **Cognitive profiles** — radar chart scores across 6 dimensions
- **Learning curves** — rolling-window and block-based success rates
- **Strategy metrics** — action entropy, forward ratio, rotation ratio, repetition rate
- **Wilson score CIs** — 95% confidence intervals on all success rates
- **Animal comparison** — model profiles overlaid with rodent baselines

```bash
python analysis.py results/benchmark_results.json
# Outputs: results/benchmark_results_analysis.json
```

## Configuration

All parameters are in `config.py` and overridable via CLI or environment variables:

```bash
export CHEESEBENCH_MODEL=gpt-oss:120b
export CHEESEBENCH_API_URL=http://localhost:11434/api/chat
export CHEESEBENCH_TIMEOUT=120
```

| Parameter | Default | Description |
|---|---|---|
| `--model` | `gpt-oss:120b` | Model name (LLM or VLM) |
| `--num-trials` | 20 | Trials per environment |
| `--max-steps` | 200 | Max steps per trial |
| `--seed` | 42 | Random seed |
| `--output-dir` | `results/` | Output directory |
| `--quiet` | false | Suppress verbose output |

## Citation

If you use CheeseBench in your research, please cite:

```bibtex
@inproceedings{cheesebench2025,
  title={CheeseBench: Evaluating Large Language Models on Rodent Behavioral Neuroscience Paradigms},
  author={},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2025}
}
```

## License

MIT
