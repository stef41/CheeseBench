#!/usr/bin/env python3
"""
Minimal VLM Benchmark for Behavioral Neuroscience Environments

Unified experimental protocol - identical prompt for ALL tasks.
No task-specific hints. LLM must infer goals from observation only.
"""

import sys
import json
import random
import time
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, '.')
from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze,
    OperantChamber, ShuttleBox, PlacePreference, StarMaze, DNMSTask,
    Action, ViewMode
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://g107:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:72b"
MAX_STEPS = 100
NUM_TRIALS = 2
TIMEOUT = 60

# View modes to benchmark
VIEW_MODES = [
    ViewMode.ASCII_2D,
    ViewMode.ASCII_2D_FPV,
    ViewMode.ASCII_3D,
]

# =============================================================================
# UNIVERSAL SYSTEM PROMPT - IDENTICAL FOR ALL TASKS
# =============================================================================

SYSTEM_PROMPT = """You are an agent navigating an environment. Maximize reward through exploration.

SYMBOLS:
- Agent: ↑ ↗ → ↘ ↓ ↙ ← ↖ (arrow = facing direction) or ^ (in 2D grids)
- Goals: G (goal), P (platform), T (target), * (reward)
- Self: S (you, in water views)
- Walls: # █ (solid)
- Water: ~ ≈
- Floors: . ░ ▒ ▓ (textures)
- Objects: [=] (lever), [m] (magazine)
- Holes: O (unknown hole), E (escape hole - only visible when adjacent and facing it)
- Cues: 1-8 or A-D (landmarks for orientation)
- Shapes: ■ ● ▲ ◆ (stimuli)
- Fog: ░ (area outside your field of view)

NAVIGATION:
- Arrow shows YOUR facing direction
- FORWARD moves where arrow points
- TURN_LEFT/TURN_RIGHT rotates 45°
- To reach goal: turn until facing it, then FORWARD

ACTIONS (ONE word):
- FORWARD
- TURN_LEFT
- TURN_RIGHT
- INTERACT (check holes, press levers)
- STAY

Reply with ONLY the action word."""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialResult:
    steps: int = 0
    reward: float = 0.0
    success: bool = False
    actions: List[str] = field(default_factory=list)


@dataclass 
class BenchmarkResult:
    env_name: str
    view_mode: str
    agent_type: str
    trials: List[TrialResult] = field(default_factory=list)
    source_pmc: str = ""
    source_quote: str = ""
    
    @property
    def success_rate(self) -> float:
        return sum(1 for t in self.trials if t.success) / len(self.trials) if self.trials else 0
    
    @property
    def avg_steps(self) -> float:
        return sum(t.steps for t in self.trials) / len(self.trials) if self.trials else 0
    
    @property
    def successes(self) -> int:
        return sum(1 for t in self.trials if t.success)
    
    def to_dict(self) -> Dict:
        return {
            "env_name": self.env_name,
            "view_mode": self.view_mode,
            "agent_type": self.agent_type,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "successes": sum(1 for t in self.trials if t.success),
            "total_trials": len(self.trials),
            "source_pmc": self.source_pmc,
            "source_quote": self.source_quote,
            "trials": [{"steps": t.steps, "reward": t.reward, "success": t.success} for t in self.trials]
        }


# =============================================================================
# AGENTS
# =============================================================================

class RandomAgent:
    """Baseline random agent"""
    
    def __init__(self):
        self.actions = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STAY, Action.INTERACT]
    
    def reset(self):
        pass
    
    def get_action(self, observation: str, reward: float = None) -> Action:
        return random.choice(self.actions)


class LLMAgent:
    """VLM agent with unified protocol"""
    
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.history = []
        self.last_response = ""  # Store last LLM response for logging
    
    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_response = ""
    
    def get_action(self, observation: str, reward: float = None) -> Action:
        # Build user message
        feedback = ""
        if reward is not None:
            if reward > 0:
                feedback = f"[+{reward:.1f} reward]\n"
            elif reward < 0:
                feedback = f"[{reward:.1f} penalty]\n"
        
        user_msg = f"{feedback}{observation}\n\nAction:"
        self.history.append({"role": "user", "content": user_msg})
        
        # Keep history bounded (100 steps = 200 messages + system prompt)
        if len(self.history) > 201:
            self.history = [self.history[0]] + self.history[-200:]
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                msg = response.json().get('message', {})
                # Thinking models use 'thinking' field, others use 'content'
                content = msg.get('content', '') or msg.get('thinking', '') or ''
                self.last_response = content  # Store for logging
                self.history.append({"role": "assistant", "content": content[:100]})
                return self._parse_action(content)
        except Exception as e:
            self.last_response = f"[ERROR: {e}]"
        
        # Remove failed user message
        if self.history and self.history[-1]['role'] == 'user':
            self.history.pop()
        return Action.FORWARD
    
    def _parse_action(self, text: str) -> Action:
        text = text.lower()
        # Check end of response first
        end = text[-80:] if len(text) > 80 else text
        
        if 'forward' in end:
            return Action.FORWARD
        if 'turn_left' in end or 'left' in end:
            return Action.TURN_LEFT
        if 'turn_right' in end or 'right' in end:
            return Action.TURN_RIGHT
        if 'interact' in end:
            return Action.INTERACT
        if 'stay' in end:
            return Action.STAY
        
        # Full text fallback
        if 'forward' in text:
            return Action.FORWARD
        if 'left' in text:
            return Action.TURN_LEFT
        if 'right' in text:
            return Action.TURN_RIGHT
        if 'interact' in text:
            return Action.INTERACT
        
        return Action.FORWARD


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def create_environments(view_mode: ViewMode) -> List[tuple]:
    """Create all benchmark environments with verified source citations from behavioral neuroscience literature"""
    return [
        (MorrisWaterMaze(view_mode=view_mode), 
         "PMC2895266",  # Vorhees & Williams, Nat Protoc 2006
         "The MWM is a test of spatial learning for rodents that relies on distal cues to navigate from start locations around the perimeter of an open swimming arena to locate a submerged escape platform."),
        
        (TMaze(view_mode=view_mode),
         "PMC3399492",  # Shoji et al., J Vis Exp 2012
         "In the forced alternation task, each trial consists of a forced choice run followed by a free choice run. A mouse is subjected to 10 consecutive trials in a session per day."),
        
        (BarnesMaze(view_mode=view_mode),
         "PMC6126525",  # Vale et al., J Vis Exp 2018
         "The Barnes maze consists of a circular platform with 20 equidistant holes, 19 closed with plugs while the remaining hole leads to an escape shelter."),
        
        (RadialArmMaze(view_mode=view_mode),
         "PMC4030456",  # Penley et al., J Vis Exp 2013
         "Subjects are required to avoid arms previously used for escape during each testing day (working memory) as well as avoid fixed arms which never contain escape platforms (reference memory)."),
        
        (OperantChamber(view_mode=view_mode),
         "PMC4598097",  # Martin & Iceberg, J Vis Exp 2015
         "Program the operant responses (lever presses) necessary to obtain reward on a progressive ratio schedule of reinforcement."),
        
        (ShuttleBox(view_mode=view_mode),
         "PMC4692667",  # Happel et al., J Vis Exp 2015
         "A conditioned stimulus (CS) is contingently followed by an aversive unconditioned stimulus (US). Subjects learn to avoid the US by shuttling from one compartment to the other in response to the CS."),
        
        (PlacePreference(view_mode=view_mode),
         "PMC6101638",  # Blanco-Gandía et al., J Vis Exp 2018
         "The procedure consists of three phases: Pre-Conditioning, Conditioning, and Post-Conditioning. Compartments have different floor textures and wall colors."),
        
        (StarMaze(view_mode=view_mode),
         "PMC7866711",  # Zhang et al., BMC Geriatr 2021 (virtual human paradigm adapted for rodent simulation)
         "Star maze assesses navigation strategies using multiple arm configurations radiating from a central hub."),
        
        (DNMSTask(view_mode=view_mode),
         "PMC3982138",  # Oomen et al., Nat Protoc 2013
         "TUNL working memory task requires animals to non-match to a sample location after a delay. A correct response to the novel location leads to reward delivery."),
    ]


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_trial(env, agent, max_steps: int = MAX_STEPS, log_file=None) -> TrialResult:
    """Run single trial with unified protocol"""
    result = TrialResult()
    agent.reset()
    
    obs = env.reset()
    reward = None
    initial_trial = env.session.current_trial
    
    for step in range(max_steps):
        action = agent.get_action(obs, reward)
        
        # Log LLM trace if logging enabled and agent is LLM
        if log_file and hasattr(agent, 'last_response') and agent.last_response:
            log_file.write(f"\n--- Step {step+1} ---\n")
            log_file.write(f"Reward from prev step: {reward}\n")
            log_file.write(f"Observation:\n{obs}\n")
            log_file.write(f"LLM Response: {agent.last_response[:200]}{'...' if len(agent.last_response) > 200 else ''}\n")
            log_file.write(f"Action: {action.name}\n")
            log_file.flush()
        
        obs, reward = env.step(action)
        
        result.steps += 1
        result.reward += reward
        result.actions.append(action.name)
        
        # Check if trial completed (trial number changed or session done)
        if env.is_done or env.session.current_trial != initial_trial:
            # Check last trial result
            if env.session.trial_results:
                result.success = env.session.trial_results[-1].success
            break
    
    return result


def run_benchmark(num_trials: int = NUM_TRIALS, verbose: bool = True) -> Dict:
    """Run full benchmark"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": OLLAMA_MODEL,
        "num_trials": num_trials,
        "system_prompt": SYSTEM_PROMPT,
        "results": []
    }
    
    agents = [
        ("Random", RandomAgent()),
        ("LLM", LLMAgent()),
    ]
    
    # Open log file for LLM traces
    log_file = open("llm_traces.log", "w")
    log_file.write(f"LLM Benchmark Traces - {datetime.now().isoformat()}\n")
    log_file.write(f"Model: {OLLAMA_MODEL}\n")
    log_file.write("="*60 + "\n")
    
    for view_mode in VIEW_MODES:
        mode_name = view_mode.name
        if verbose:
            print(f"\n{'='*60}")
            print(f"VIEW MODE: {mode_name}")
            print('='*60)
        
        envs = create_environments(view_mode)
        
        for env, pmc, quote in envs:
            env_name = env.__class__.__name__
            
            for agent_name, agent in agents:
                if verbose:
                    print(f"\n  {env_name} ({agent_name})...", end=" ", flush=True)
                
                # Log header for this env/agent combo
                if agent_name == "LLM":
                    log_file.write(f"\n{'='*60}\n")
                    log_file.write(f"Environment: {env_name} | Mode: {mode_name}\n")
                    log_file.write(f"{'='*60}\n")
                
                benchmark_result = BenchmarkResult(
                    env_name=env_name,
                    view_mode=mode_name,
                    agent_type=agent_name,
                    source_pmc=pmc,
                    source_quote=quote
                )
                
                for trial in range(num_trials):
                    # Only log LLM trials
                    trial_log = log_file if agent_name == "LLM" else None
                    if trial_log:
                        log_file.write(f"\n--- Trial {trial+1}/{num_trials} ---\n")
                    trial_result = run_trial(env, agent, log_file=trial_log)
                    benchmark_result.trials.append(trial_result)
                    if trial_log:
                        log_file.write(f"Result: {'SUCCESS' if trial_result.success else 'FAIL'} in {trial_result.steps} steps\n")
                
                results["results"].append(benchmark_result.to_dict())
                
                if verbose:
                    sr = benchmark_result.success_rate
                    print(f"{benchmark_result.successes}/{num_trials} ({sr*100:.0f}%)")
    
    log_file.close()
    return results


def print_summary(results: Dict):
    """Print summary table"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Aggregate by view mode and agent
    summary = {}
    for r in results["results"]:
        key = (r["view_mode"], r["agent_type"])
        if key not in summary:
            summary[key] = {"successes": 0, "total": 0}
        summary[key]["successes"] += r["successes"]
        summary[key]["total"] += r["total_trials"]
    
    print(f"\n{'View Mode':<15} {'Agent':<10} {'Success Rate':<15}")
    print("-"*40)
    for (mode, agent), data in sorted(summary.items()):
        rate = data["successes"] / data["total"] if data["total"] > 0 else 0
        print(f"{mode:<15} {agent:<10} {data['successes']}/{data['total']} ({rate*100:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("MINIMAL VLM BENCHMARK - UNIFIED PROTOCOL")
    print("="*60)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Trials per environment: {NUM_TRIALS}")
    print(f"View modes: {[m.name for m in VIEW_MODES]}")
    print("="*60)
    
    results = run_benchmark(NUM_TRIALS, verbose=True)
    
    print_summary(results)
    
    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(f"LLM traces saved to llm_traces.log")
