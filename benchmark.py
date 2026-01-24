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
from typing import List, Dict, Any, Optional, Tuple
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

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gpt-oss:120b"
MAX_STEPS = 200
NUM_TRIALS = 3
TIMEOUT = 120
MAX_ACTIONS_PER_CALL = 8  # Maximum number of actions to generate per LLM call
SAVE_THINKING_IN_HISTORY = False  # Include short thinking in conversation history
SAVE_ACTIONS_IN_HISTORY = False  # Include actions taken in conversation history

# View modes to benchmark
VIEW_MODES = [
    ViewMode.ASCII_2D,
    ViewMode.ASCII_2D_FPV,
    ViewMode.ASCII_3D,
]

# =============================================================================
# UNIVERSAL SYSTEM PROMPT - IDENTICAL FOR ALL TASKS
# =============================================================================

SYSTEM_PROMPT = f"""You are an agent navigating an environment. Maximize cumulative reward.

SYMBOLS:
- Agent: ↑ ↗ → ↘ ↓ ↙ ← ↖ (8 directions, 45° apart)
- Goals: G P * (goal, platform, coin)
- Walls: # █ (BLOCKED)
- Water/Floors: ~ . ░ (TRAVERSABLE)
- Objects: [=] lever, [m] magazine
- Targets: ? E (unknown, revealed)
- Landmarks: 1-4

ACTIONS (egocentric, relative to your facing direction):
- FORWARD: move in facing direction
- ROTATE_LEFT: turn 45° left
- ROTATE_RIGHT: turn 45° right
- STAY: do nothing

You see your last 20 observations. Use [Your learnings: ...] (300 char max) for insights beyond recent history.

Respond in this format:
LEARNINGS: <high-level patterns, rules discovered, or long-term knowledge. Don't repeat recent observations. Write "unchanged" if nothing new>
ACTIONS: <up to {MAX_ACTIONS_PER_CALL} comma-separated actions, e.g. FORWARD, FORWARD, ROTATE_LEFT>"""


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
        self.actions = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.STAY, Action.INTERACT]
    
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
        self.notes = ""  # Working memory: understanding + hypotheses
        self.action_queue = []  # Queue of pending actions
        self.pending_observations = []  # Observations to show on next LLM call
    
    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_response = ""
        self.notes = ""
        self.action_queue = []
        self.pending_observations = []  # List of (observation, reward, action_that_led_here)
    
    def add_observation(self, observation: str, reward: float = None, action: Action = None):
        """Add an observation to be shown on next LLM call."""
        self.pending_observations.append((observation, reward, action))
    
    def get_actions(self, observation: str, reward: float = None, k: int = MAX_ACTIONS_PER_CALL) -> List[Action]:
        """Get k actions from LLM. Returns list of actions."""
        # Add current observation to pending (no action led to first obs)
        self.pending_observations.append((observation, reward, None))
        
        # Build user message from all pending observations
        msg_parts = []
        
        # Include current learnings at the start
        if self.notes:
            msg_parts.append(f"[Your learnings: {self.notes}]")
        
        # Add all pending observations with rewards and actions
        for i, (obs, rew, act) in enumerate(self.pending_observations):
            # Show action that led to this observation (if any)
            if act is not None:
                msg_parts.append(f"[After {act.name}]")
            if rew is not None:
                if rew > 0:
                    msg_parts.append(f"[Step {i+1}: +{rew:.1f} reward - GOOD!]")
                elif rew < 0:
                    msg_parts.append(f"[Step {i+1}: {rew:.1f} penalty]")
                else:
                    msg_parts.append(f"[Step {i+1}: 0 reward]")
            msg_parts.append(obs)
        
        msg_parts.append(f"\nProvide up to {k} actions to execute in sequence.")
        
        user_msg = "\n".join(msg_parts)
        self.history.append({"role": "user", "content": user_msg})
        
        # Clear pending observations
        self.pending_observations = []
        
        # Keep history bounded (20 observations = 40 messages + system prompt)
        if len(self.history) > 41:
            self.history = [self.history[0]] + self.history[-40:]
        
        # Just use history (system prompt already at start)
        messages = self.history
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                msg = response.json().get('message', {})
                content = msg.get('content', '') or ''
                thinking = msg.get('thinking', '') or ''
                
                # Parse and update learnings (working memory)
                import re
                notes_match = re.search(r'LEARNINGS:\s*(.+?)(?:ACTIONS?:|$)', content, re.IGNORECASE | re.DOTALL)
                if notes_match:
                    new_notes = notes_match.group(1).strip()[:300]
                    if new_notes.lower() not in ('unchanged', 'same', 'no change', 'none'):
                        self.notes = new_notes
                
                # For logging
                if thinking:
                    self.last_response = f"[THINKING: {thinking[:500]}...]\n{content}"
                else:
                    self.last_response = content if content else "[EMPTY RESPONSE]"
                
                # For history
                history_parts = []
                if SAVE_THINKING_IN_HISTORY and thinking:
                    short_thinking = thinking[:150].split('.')[0] + '.'
                    history_parts.append(f"Thinking: {short_thinking}")
                history_parts.append(f"LEARNINGS: {self.notes}" if self.notes else "LEARNINGS: (none)")
                if SAVE_ACTIONS_IN_HISTORY:
                    actions_match = re.search(r'ACTIONS?:\s*(.+)', content, re.IGNORECASE)
                    if actions_match:
                        history_parts.append(f"ACTIONS: {actions_match.group(1).strip()}")
                self.history.append({"role": "assistant", "content": "\n".join(history_parts)})
                
                # Parse multiple actions
                return self._parse_actions(content, k)
        except Exception as e:
            self.last_response = f"[ERROR: {e}]"
        
        # Remove failed user message
        if self.history and self.history[-1]['role'] == 'user':
            self.history.pop()
        return [Action.FORWARD] * k
    
    def _parse_actions(self, text: str, k: int) -> List[Action]:
        """Parse up to k actions from response."""
        import re
        actions = []
        
        # Look for ACTIONS: line
        actions_match = re.search(r'ACTIONS?:\s*(.+)', text, re.IGNORECASE)
        if actions_match:
            actions_str = actions_match.group(1)
            # Split by comma or whitespace
            action_words = re.split(r'[,\s]+', actions_str)
            for word in action_words:
                action = self._word_to_action(word.strip())
                if action:
                    actions.append(action)
                if len(actions) >= k:
                    break
        
        # Fallback: look for action words anywhere
        if len(actions) == 0:
            action_pattern = r'\b(forward|rotate_left|rotate_right|stay)\b'
            matches = re.findall(action_pattern, text, re.IGNORECASE)
            for match in matches:
                if len(actions) >= k:
                    break
                action = self._word_to_action(match)
                if action:
                    actions.append(action)
        
        # If still no actions, default to single FORWARD
        if len(actions) == 0:
            actions.append(Action.FORWARD)
        
        return actions[:k]
    
    def _word_to_action(self, word: str) -> Optional[Action]:
        word = word.lower().replace(' ', '_').replace('-', '_')
        if 'forward' in word:
            return Action.FORWARD
        if 'rotate_left' in word or 'left' == word:
            return Action.ROTATE_LEFT
        if 'rotate_right' in word or 'right' == word:
            return Action.ROTATE_RIGHT
        if 'stay' in word:
            return Action.STAY
        return None


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

def run_trial(env, agent, max_steps: int = MAX_STEPS, log_file=None, last_reward=None, k: int = MAX_ACTIONS_PER_CALL) -> Tuple[TrialResult, float]:
    """Run single trial with unified protocol. Returns (result, final_reward) for continuity."""
    result = TrialResult()
    # Don't reset agent - continuous stream from LLM perspective
    
    obs = env.reset()
    reward = last_reward  # Carry over reward from previous trial
    initial_trial = env.session.current_trial
    
    step = 0
    while step < max_steps:
        # For LLM agent, get k actions at once
        if hasattr(agent, 'get_actions'):
            # Collect observations that will be shown to LLM (for logging)
            # pending_observations are 3-tuples (obs, reward, action), add current as (obs, reward, None)
            observations_for_llm = list(agent.pending_observations) + [(obs, reward, None)]
            
            actions = agent.get_actions(obs, reward, k)
            
            # Log the LLM call with all observations it saw
            if log_file and hasattr(agent, 'last_response') and agent.last_response:
                log_file.write(f"\n--- LLM Call (steps {step+1}-{step+len(actions)}) ---\n")
                if hasattr(agent, 'notes') and agent.notes:
                    log_file.write(f"Agent learnings: {agent.notes}\n")
                log_file.write(f"Observations shown to LLM ({len(observations_for_llm)}):\n")
                for idx, (o, r, a) in enumerate(observations_for_llm):
                    r_str = f"reward={r:.2f}" if r is not None else "no reward"
                    a_str = f"after {a.name}" if a is not None else "initial"
                    log_file.write(f"  [Obs {idx+1}, {a_str}, {r_str}]\n{o}\n")
                log_file.write(f"LLM Response: {agent.last_response}\n")
                log_file.write(f"Actions: {[a.name for a in actions]}\n")
                log_file.flush()
            
            # Execute each action, collecting observations for next call
            trial_done = False
            for i, action in enumerate(actions):
                obs, reward = env.step(action)
                
                result.steps += 1
                result.reward += reward
                result.actions.append(action.name)
                step += 1
                
                # Log each step result with observation
                if log_file:
                    log_file.write(f"  Step {step}: {action.name} -> reward={reward:.2f}\n")
                    log_file.write(f"{obs}\n")
                    log_file.flush()
                
                # Check if trial completed
                if env.is_done or env.session.current_trial != initial_trial:
                    if env.session.trial_results:
                        result.success = env.session.trial_results[-1].success
                    if log_file:
                        # Log the final observation that led to trial end
                        log_file.write(f"  Final observation after {action.name}:\n{obs}\n")
                        if result.success:
                            log_file.write(f"Result: SUCCESS in {result.steps} steps\n")
                        else:
                            log_file.write(f"Result: FAILED after {result.steps} steps\n")
                        log_file.flush()
                    trial_done = True
                    break
                
                # Add intermediate observation for next LLM call (except last action)
                if i < len(actions) - 1:
                    agent.add_observation(obs, reward, action)
            
            if trial_done:
                break
        else:
            # Random agent - single action at a time
            action = agent.get_action(obs, reward)
            obs, reward = env.step(action)
            
            result.steps += 1
            result.reward += reward
            result.actions.append(action.name)
            step += 1
            
            # Check if trial completed
            if env.is_done or env.session.current_trial != initial_trial:
                if env.session.trial_results:
                    result.success = env.session.trial_results[-1].success
                break
    
    return result, reward  # Return final reward for next trial


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
    log_file.flush()
    
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
                    # Reset agent only on first trial of each environment
                    if trial == 0:
                        agent.reset()
                        last_reward = None
                    # Only log LLM trials
                    trial_log = log_file if agent_name == "LLM" else None
                    if trial_log:
                        log_file.write(f"\n--- Trial {trial+1}/{num_trials} ---\n")
                    trial_result, last_reward = run_trial(env, agent, log_file=trial_log, last_reward=last_reward)
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
