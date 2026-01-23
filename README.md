# Minimal VLM Benchmark for Behavioral Neuroscience

Unified evaluation of Vision-Language Models on classical behavioral neuroscience paradigms.

## Key Design Principles

1. **Unified Protocol**: Identical system prompt for ALL tasks - no task-specific hints
2. **Visual-Only Cues**: 3D renders use only visual markers (G, *, ▓) - no cheating text
3. **Fair Comparison**: Same experimental conditions across all environments

## Files

```
minimal_benchmark/
├── benchmark.py           # Main benchmark script
├── task_definitions.json  # Task specs with paper citations
├── environments/          # All 9 behavioral paradigms
│   ├── __init__.py
│   ├── base_env.py
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

## Usage

```bash
# Run benchmark (requires Ollama with qwen2.5vl:32b)
python benchmark.py

# Results saved to benchmark_results.json
```

## System Prompt (Identical for ALL tasks)

```
You are an agent in a behavioral experiment. Your goal is to maximize reward.

ENVIRONMENT:
- You see an ASCII visualization of your current state
- Symbols like ^, v, <, > indicate agent position/direction
- G or green markers indicate goals
- * indicates rewards/food
- Walls block movement

ACTIONS (reply with exactly ONE word):
- FORWARD: Move in facing direction
- TURN_LEFT: Rotate left
- TURN_RIGHT: Rotate right  
- INTERACT: Press lever, check hole, make choice
- STAY: Wait in place

FEEDBACK:
- Positive reward = good action, continue strategy
- Negative reward = bad action (wall hit, wrong choice), try different approach
- Goal: Maximize cumulative reward

Reply with ONLY the action word. No explanations.
```

## Environments

| Environment | Cognitive Domain | Citation |
|------------|-----------------|----------|
| Morris Water Maze | Spatial Learning | [PMC12765891](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12765891/) |
| T-Maze | Working Memory | [PMC3181652](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181652/) |
| Barnes Maze | Spatial Memory | [PMC4480627](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4480627/) |
| Radial Arm Maze | Working/Reference Memory | [PMC2892270](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2892270/) |
| Operant Chamber | Instrumental Learning | [PMC3181652](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181652/) |
| Shuttle Box | Avoidance Learning | [PMC4480627](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4480627/) |
| Place Preference | Reward Association | [PMC2892270](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2892270/) |
| Star Maze | Allocentric Navigation | [PMC3181652](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3181652/) |
| DNMS Task | Recognition Memory | [PMC4480627](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4480627/) |

## View Modes

- **ASCII_2D**: Top-down map (full spatial information)
- **ASCII_2D_FPV**: First-person 2D (egocentric)
- **ASCII_3D**: Pseudo-3D perspective (depth cues)
