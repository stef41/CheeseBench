#!/usr/bin/env python3
"""
CheeseBench — LLM Benchmark for Behavioral Neuroscience Environments

Unified experimental protocol — identical prompt for ALL tasks.
No task-specific hints. The model must infer goals from observation only.

Usage:
    python benchmark.py                        # Run with defaults
    python benchmark.py --num-trials 5         # Quick test run
    python benchmark.py --model qwen2.5vl:32b  # Specify model
"""

import sys
import os
import json
import random
import re
import argparse
import time
import base64
import io
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')
from .environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze,
    OperantChamber, ShuttleBox, PlacePreference, StarMaze, DNMSTask,
    Action, ViewMode
)
from .config import BenchmarkConfig

# =============================================================================
# CONFIGURATION — loaded from config.py, overridable via CLI
# =============================================================================

CFG = BenchmarkConfig()

# View modes to benchmark
VIEW_MODES = [
    ViewMode.ASCII_2D,
    ViewMode.ASCII_2D_FPV,
    ViewMode.ASCII_3D,
    ViewMode.TOPDOWN_2D,
]

# Image view modes (return numpy arrays, need base64 encoding)
IMAGE_VIEW_MODES = {ViewMode.TOPDOWN_2D, ViewMode.FPV_3D}


def encode_image(img: np.ndarray) -> str:
    """Encode numpy image array as base64 PNG data URL."""
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# =============================================================================
# UNIVERSAL SYSTEM PROMPT - IDENTICAL FOR ALL TASKS
# =============================================================================

def build_system_prompt(max_actions: int, variant: str = "default") -> str:
    """Build system prompt. Variant controls ablation study prompt formats."""
    if variant == "minimal":
        return f"""You are an agent. Maximize cumulative reward.
You see an ASCII rendering. Your position has an arrow.

ACTIONS: FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY

Respond:
LEARNINGS: <brief notes>
ACTIONS: <1-{max_actions} comma-separated actions>"""

    if variant == "cot":
        return f"""You are an embodied agent in a behavioral experiment. Maximize cumulative reward.

You receive ASCII renderings of your environment. Walls (#, █) block movement.
Your position/orientation is shown by arrow symbols.

Before acting, reason step by step:
1. Where am I? What do I see around me?
2. What reward did I just receive? What does it tell me?
3. What is my hypothesis about the task goal?
4. What should I try next to test this hypothesis?

ACTIONS: FORWARD (move ahead), ROTATE_LEFT (turn left), ROTATE_RIGHT (turn right), STAY (wait)

RESPONSE FORMAT:
LEARNINGS: <Your step-by-step reasoning and updated strategy. Max 500 chars.>
ACTIONS: <1 to {max_actions} comma-separated actions>"""

    if variant == "few_shot":
        return f"""You are an embodied agent in a behavioral experiment. Maximize cumulative reward.

You see an ASCII rendering. Walls (#, █) block movement. Arrow = your position/facing.

ACTIONS: FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY

Example interaction:
Observation:
  # # # # #
  # . . . #
  # . → . #
  # . . G #
  # # # # #
Reward: +0.00

LEARNINGS: I am at center facing east. I see 'G' to my south-east. I should go forward then rotate right toward it.
ACTIONS: FORWARD, ROTATE_RIGHT, FORWARD

Now it's your turn. Respond with LEARNINGS and up to {max_actions} ACTIONS."""

    # Default prompt
    return f"""You are an embodied agent placed in a behavioral experiment. Your only goal is to maximize cumulative reward. You receive NO instructions about what the task is — you must figure it out from observation and reward feedback alone.

PERCEPTION:
- You see a rendering of the environment (top-down map, first-person view, or pseudo-3D)
- In text views: your position/orientation is shown by an arrow: ↑ ↗ → ↘ ↓ ↙ ← ↖
- In image views: you are shown as a colored dot with a darker nose indicating facing direction
- Walls block movement. Open spaces are traversable
- Various symbols or colors represent objects, goals, or interactive elements
- Numbers or colored markers are fixed landmarks for spatial reference

ACTIONS (egocentric — relative to your current facing direction):
- FORWARD: move one cell in the direction you face
- ROTATE_LEFT: turn 45° counterclockwise
- ROTATE_RIGHT: turn 45° clockwise
- STAY: remain in place (useful when waiting is required)

STRATEGY:
1. REWARD SIGNALS: Positive reward → good, repeat. Negative reward → bad, change approach. Zero → neutral.
2. SPATIAL MEMORY: Track where you are, where you've been, what's unexplored.
3. HYPOTHESIS TESTING: Form hypotheses about the goal, test them, revise based on reward.
4. PATTERN RECOGNITION: Rules may involve sequences, timing, or spatial relationships. Infer from feedback.
5. EFFICIENT PLANNING: Visualize the path. Count rotations needed to face your target, then move.

RESPONSE FORMAT (strict):
LEARNINGS: <Brief notes: current position, strategy, hypotheses, what you've learned. Max 500 chars.>
ACTIONS: <1 to {max_actions} comma-separated actions from {{FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY}}>

Example:
LEARNINGS: I am at (3,4) facing north. Goal symbol G is to my east. Need to rotate right twice then move forward.
ACTIONS: ROTATE_RIGHT, ROTATE_RIGHT, FORWARD, FORWARD, FORWARD"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialResult:
    steps: int = 0
    reward: float = 0.0
    success: bool = False
    actions: List[str] = field(default_factory=list)
    parse_failures: int = 0  # how many times action parsing fell back to default


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
    def avg_steps_success(self) -> Optional[float]:
        s = [t.steps for t in self.trials if t.success]
        return sum(s) / len(s) if s else None

    @property
    def avg_steps_failure(self) -> Optional[float]:
        s = [t.steps for t in self.trials if not t.success]
        return sum(s) / len(s) if s else None
    
    @property
    def avg_actions(self) -> float:
        return sum(len(t.actions) for t in self.trials) / len(self.trials) if self.trials else 0
    
    @property
    def successes(self) -> int:
        return sum(1 for t in self.trials if t.success)
    
    def to_dict(self) -> Dict:
        return {
            "env_name": self.env_name,
            "view_mode": self.view_mode,
            "agent_type": self.agent_type,
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 1),
            "avg_steps_success": round(self.avg_steps_success, 1) if self.avg_steps_success is not None else None,
            "avg_steps_failure": round(self.avg_steps_failure, 1) if self.avg_steps_failure is not None else None,
            "avg_actions": round(self.avg_actions, 1),
            "successes": self.successes,
            "total_trials": len(self.trials),
            "source_pmc": self.source_pmc,
            "source_quote": self.source_quote,
            "trials": [
                {
                    "steps": t.steps,
                    "reward": round(t.reward, 4),
                    "success": t.success,
                    "num_actions": len(t.actions),
                    "actions": t.actions,
                    "parse_failures": t.parse_failures,
                }
                for t in self.trials
            ],
        }


# =============================================================================
# AGENTS
# =============================================================================

class RandomAgent:
    """Baseline random agent."""

    def __init__(self, seed: Optional[int] = None):
        self.actions = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.STAY]
        self.rng = random.Random(seed)

    def reset(self):
        pass

    def get_action(self, observation: str, reward: float = None) -> Action:
        return self.rng.choice(self.actions)


class LLMAgent:
    """Model agent with unified protocol."""

    _tokenizer_cache = {}  # class-level cache: model_id → tokenizer

    def __init__(self, model: str = None):
        self.model = model or CFG.model
        self.system_prompt = build_system_prompt(CFG.max_actions_per_call, CFG.prompt_variant)
        self.history = []
        self.last_response = ""
        self.notes = ""
        self.pending_observations = []
        self.parse_failures = 0  # track how often action parsing fails

        # Lazy-load tokenizer for exact token counting
        self._tokenizer = None

        # State tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.recent_rewards = []
        self.negative_reward_streak = 0

    def reset(self):
        self.system_prompt = build_system_prompt(CFG.max_actions_per_call, CFG.prompt_variant)
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.last_response = ""
        self.notes = ""
        self.pending_observations = []
        self.parse_failures = 0

        # Reset state tracking
        self.step_count = 0
        self.total_reward = 0.0
        self.recent_rewards = []
        self.negative_reward_streak = 0

    def _get_tokenizer(self):
        """Lazy-load and cache the tokenizer for this model."""
        if self._tokenizer is None:
            if self.model not in LLMAgent._tokenizer_cache:
                try:
                    from transformers import AutoTokenizer
                    LLMAgent._tokenizer_cache[self.model] = AutoTokenizer.from_pretrained(
                        self.model, trust_remote_code=True
                    )
                except Exception:
                    LLMAgent._tokenizer_cache[self.model] = None
            self._tokenizer = LLMAgent._tokenizer_cache[self.model]
        return self._tokenizer

    def _count_tokens(self, messages: list) -> int:
        """Count tokens using the model's tokenizer. Falls back to char estimate."""
        tok = self._get_tokenizer()
        if tok is None:
            # Fallback: conservative char-based estimate
            total_chars = sum(
                len(m['content']) if isinstance(m['content'], str)
                else sum(len(p.get('text', '')) for p in m['content']
                         if isinstance(p, dict) and p.get('type') == 'text')
                for m in messages
            )
            num_images = sum(
                1 for m in messages if isinstance(m.get('content'), list)
                for p in (m['content'] if isinstance(m['content'], list) else [])
                if isinstance(p, dict) and p.get('type') == 'image_url'
            )
            return int(total_chars / 2.5) + num_images * 300

        # Build text-only version for tokenizer (images counted separately)
        num_images = 0
        text_messages = []
        for m in messages:
            if isinstance(m['content'], str):
                text_messages.append(m)
            else:
                # Multimodal: extract text parts, count images
                text_parts = []
                for p in m['content']:
                    if isinstance(p, dict):
                        if p.get('type') == 'text':
                            text_parts.append(p.get('text', ''))
                        elif p.get('type') == 'image_url':
                            num_images += 1
                text_messages.append({'role': m['role'], 'content': '\n'.join(text_parts)})

        try:
            token_ids = tok.apply_chat_template(text_messages, add_generation_prompt=True)
            return len(token_ids) + num_images * 300
        except Exception:
            # Fallback on any tokenizer error
            total_chars = sum(len(m['content']) for m in text_messages)
            return int(total_chars / 2.5) + num_images * 300

    def add_observation(self, observation: Union[str, np.ndarray], reward: float = None, action: Action = None):
        """Add an observation to be shown on next LLM call."""
        self.pending_observations.append((observation, reward, action))

        if reward is not None:
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 10:
                self.recent_rewards.pop(0)
            self.total_reward += reward
            if reward < -0.05:
                self.negative_reward_streak += 1
            else:
                self.negative_reward_streak = 0

    def get_actions(self, observation: Union[str, np.ndarray], reward: float = None, k: int = None) -> List[Action]:
        """Get k actions from LLM. Returns list of actions."""
        if k is None:
            k = CFG.max_actions_per_call
        # Track this observation's reward
        if reward is not None:
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 10:
                self.recent_rewards.pop(0)
            self.total_reward += reward
            
            if reward < -0.05:
                self.negative_reward_streak += 1
            else:
                self.negative_reward_streak = 0
        
        # Add current observation to pending (no action led to first obs)
        self.pending_observations.append((observation, reward, None))
        
        # Check if any observations are images
        has_images = any(isinstance(obs, np.ndarray) for obs, _, _ in self.pending_observations)
        
        if has_images:
            # Build multi-modal content (text + image parts)
            content_parts = []
            
            if self.notes:
                content_parts.append({"type": "text", "text": f"Your previous learnings: {self.notes}"})
            
            for i, (obs, rew, act) in enumerate(self.pending_observations):
                self.step_count += 1
                text_parts = []
                if act is not None:
                    text_parts.append(f"Action: {act.name}")
                if rew is not None:
                    text_parts.append(f"Reward: {rew:+.2f}")
                if text_parts:
                    content_parts.append({"type": "text", "text": "\n".join(text_parts)})
                
                if isinstance(obs, np.ndarray):
                    content_parts.append({"type": "text", "text": "Observation:"})
                    b64 = encode_image(obs)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                else:
                    content_parts.append({"type": "text", "text": f"Observation:\n{obs}"})
            
            content_parts.append({"type": "text", "text": f"\nProvide LEARNINGS and up to {k} ACTIONS."})
            self.history.append({"role": "user", "content": content_parts})
        else:
            # Text-only path (unchanged)
            msg_parts = []
            
            if self.notes:
                msg_parts.append(f"Your previous learnings: {self.notes}")
            
            for i, (obs, rew, act) in enumerate(self.pending_observations):
                self.step_count += 1
                if act is not None:
                    msg_parts.append(f"Action: {act.name}")
                if rew is not None:
                    msg_parts.append(f"Reward: {rew:+.2f}")
                msg_parts.append(f"Observation:\n{obs}")
            
            msg_parts.append(f"\nProvide LEARNINGS and up to {k} ACTIONS.")
            user_msg = "\n".join(msg_parts)
            self.history.append({"role": "user", "content": user_msg})

        # Clear pending observations
        self.pending_observations = []

        # Keep history bounded — preserve system prompt + recent context
        max_msgs = CFG.max_history_messages * 2 + 1  # pairs + system
        if len(self.history) > max_msgs:
            self.history = [self.history[0]] + self.history[-(max_msgs - 1):]

        # Proactively truncate if token count exceeds context window.
        # Uses actual tokenizer for exact counting; reserve 512 for output.
        max_input_tokens = 32768 - 512
        while len(self.history) > 3:
            if self._count_tokens(self.history) <= max_input_tokens:
                break
            # Drop oldest user+assistant pair (keep system prompt at [0])
            self.history = [self.history[0]] + self.history[3:]

        messages = self.history

        try:
            content, thinking = self._call_api(messages)
                
            # Parse and update learnings (working memory)
            notes_match = re.search(r'LEARNINGS:\s*(.+?)(?:ACTIONS?:|$)', content, re.IGNORECASE | re.DOTALL)
            if notes_match:
                new_notes = notes_match.group(1).strip()[:500]
                if new_notes.lower() not in ('unchanged', 'same', 'no change', 'none', ''):
                    self.notes = new_notes

            # For logging
            if thinking:
                self.last_response = f"[THINKING: {thinking[:500]}...]\n{content}"
            else:
                self.last_response = content if content else "[EMPTY RESPONSE]"

            # For history — ALWAYS save actions for auditability
            history_parts = []
            if CFG.save_thinking_in_history and thinking:
                short_thinking = thinking[:150].split('.')[0] + '.'
                history_parts.append(f"Thinking: {short_thinking}")
            history_parts.append(f"LEARNINGS: {self.notes}" if self.notes else "LEARNINGS: (none)")
            actions_match = re.search(r'ACTIONS?:\s*(.+)', content, re.IGNORECASE)
            if actions_match:
                history_parts.append(f"ACTIONS: {actions_match.group(1).strip()}")
            self.history.append({"role": "assistant", "content": "\n".join(history_parts)})

            # Parse multiple actions
            return self._parse_actions(content, k)
        except requests.exceptions.Timeout:
            self.last_response = "[TIMEOUT]"
        except requests.exceptions.ConnectionError:
            self.last_response = "[CONNECTION ERROR - is the API server running?]"
        except Exception as e:
            self.last_response = f"[ERROR: {e}]"
            # On API errors (likely context too long), aggressively trim history
            if "400" in str(e) or "413" in str(e) or "context" in str(e).lower():
                if len(self.history) > 3:
                    self.history = [self.history[0]] + self.history[-2:]

        # Remove failed user message
        if self.history and self.history[-1]['role'] == 'user':
            self.history.pop()

        self.parse_failures += 1

        # Smart fallback: if we've been getting penalties, try different actions
        if self.negative_reward_streak >= 2:
            return [Action.ROTATE_LEFT, Action.ROTATE_LEFT, Action.FORWARD, Action.FORWARD][:k]
        return [Action.FORWARD] * k

    def _call_api(self, messages: List[Dict]) -> Tuple[str, str]:
        """Call LLM API. Returns (content, thinking). Supports openai and ollama formats."""
        if CFG.api_format == "openai":
            return self._call_openai(messages)
        else:
            return self._call_ollama(messages)

    def _call_openai(self, messages: List[Dict]) -> Tuple[str, str]:
        """OpenAI-compatible API (vLLM, OpenAI, Together, etc.)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        response = requests.post(
            CFG.api_url, json=payload, timeout=CFG.api_timeout,
        )
        if response.status_code == 404:
            body = response.text[:500] if response.text else ""
            raise SystemExit(f"FATAL: Model not found on server. Wrong model served? {body}")
        if response.status_code != 200:
            # Include response body for debugging (e.g. context length errors)
            body = response.text[:500] if response.text else ""
            raise RuntimeError(f"API {response.status_code}: {body}")
        data = response.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        # Some APIs put reasoning in a separate field
        thinking = msg.get("reasoning_content", "") or ""
        return content, thinking

    def _call_ollama(self, messages: List[Dict]) -> Tuple[str, str]:
        """Ollama-native API format."""
        response = requests.post(
            CFG.api_url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
            timeout=CFG.api_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned {response.status_code}")
        msg = response.json().get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        return content, thinking

    def _parse_actions(self, text: str, k: int) -> List[Action]:
        """Parse up to k actions from response. Tracks parse failures."""
        actions = []

        # Look for ACTIONS: line
        actions_match = re.search(r'ACTIONS?:\s*(.+)', text, re.IGNORECASE)
        if actions_match:
            actions_str = actions_match.group(1)
            action_words = re.split(r'[,\s]+', actions_str)
            for word in action_words:
                action = self._word_to_action(word.strip())
                if action:
                    actions.append(action)
                if len(actions) >= k:
                    break

        # Fallback: look for action words anywhere in text
        if not actions:
            action_pattern = r'\b(forward|rotate_left|rotate_right|stay|fwd|turn_left|turn_right|left|right|wait)\b'
            matches = re.findall(action_pattern, text, re.IGNORECASE)
            for match in matches:
                if len(actions) >= k:
                    break
                action = self._word_to_action(match)
                if action:
                    actions.append(action)

        # If still no actions — parse failure
        if not actions:
            self.parse_failures += 1
            if self.negative_reward_streak >= 2:
                actions = [Action.ROTATE_LEFT, Action.FORWARD]
            else:
                actions.append(Action.FORWARD)

        return actions[:k]
    
    def _word_to_action(self, word: str) -> Optional[Action]:
        word = word.lower().replace(' ', '_').replace('-', '_')
        if word in ('forward', 'fwd', 'move_forward'):
            return Action.FORWARD
        if word in ('rotate_left', 'turn_left', 'left'):
            return Action.ROTATE_LEFT
        if word in ('rotate_right', 'turn_right', 'right'):
            return Action.ROTATE_RIGHT
        if word in ('stay', 'wait', 'stop'):
            return Action.STAY
        return None


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

# Per-environment protocol: (max_steps_per_trial, num_trials)
# Derived from maze geometry and rodent literature (see paper_task_descriptions.md).
#   MWM:          circular arena r=9 → ~8 min steps; 5 sessions × 4 trials
#   Barnes:       15×15 grid, center→edge = 6 cells; 4 sessions × 4 trials
#   T-Maze:       stem=3 + arm=2 → 5 min steps; 4 sessions × 10 trials
#   Radial Arm:   25×25, 8 arms of len 8, must visit 4; 5 sessions × 4 trials
#   Star Maze:    25×25, 5 arms of len 8; 8 sessions × 5 trials
#   Operant:      tiny chamber, 1-5 steps (env self-caps at ~100); 5 sessions × 10 trials
#   Shuttle Box:  2-compt, phase-timed ~20 steps (env self-caps); 2 sessions × 20 trials
#   CPP:          2 chambers, statistical measure; 6 sessions × 2 trials
#   DNMS:         phase-based ~14 steps (env self-caps at ~50); 25 sessions × 32 trials
ENV_PROTOCOL = {
    "MorrisWaterMaze":  {"max_steps": 500, "num_trials": 20},
    "BarnesMaze":       {"max_steps": 300, "num_trials": 16},
    "TMaze":            {"max_steps": 200, "num_trials": 40},
    "RadialArmMaze":    {"max_steps": 400, "num_trials": 20},
    "StarMaze":         {"max_steps": 300, "num_trials": 40},
    "OperantChamber":   {"max_steps": 100, "num_trials": 50},
    "ShuttleBox":       {"max_steps":  50, "num_trials": 40},
    "PlacePreference":  {"max_steps": 300, "num_trials": 12},
    "DNMSTask":         {"max_steps":  50, "num_trials": 100},
}


def create_environments(view_mode: ViewMode) -> List[tuple]:
    """Create all benchmark environments with verified source citations from behavioral neuroscience literature.
    
    Returns list of (env, pmc_id, quote, max_steps, num_trials) tuples.
    """
    return [
        (MorrisWaterMaze(view_mode=view_mode), 
         "PMC3259155",  # de Fiebre et al., Age 2006
         "C57BL/6 mice were tested in a Morris water maze over 8 acquisition sessions (5 trials/session). Path length decreased across sessions, with approximately 87% developing spatial search strategies."),
        
        (TMaze(view_mode=view_mode),
         "PMC3399492",  # Shoji et al., J Vis Exp 2012
         "In the forced alternation task, each trial consists of a forced choice run followed by a free choice run. A mouse is subjected to 10 consecutive trials in a session per day."),
        
        (BarnesMaze(view_mode=view_mode),
         "PMC1783636",  # Harrison et al., Learn Mem 2006
         "B6C3F1/J mice were tested on a 12-hole Barnes maze over 5 sessions (4 trials/session). Primary errors decreased across sessions as mice learned to locate the escape hole."),
        
        (RadialArmMaze(view_mode=view_mode),
         "PMC4030456",  # Penley et al., J Vis Exp 2013
         "Subjects are required to avoid arms previously used for escape during each testing day (working memory) as well as avoid fixed arms which never contain escape platforms (reference memory)."),
        
        (OperantChamber(view_mode=view_mode),
         "PMC6619163",  # Jurado-Parras et al., J Neurosci 2013
         "Animals were trained to press the lever to receive pellets from the feeder using a fixed-ratio (1:1) schedule."),
        
        (ShuttleBox(view_mode=view_mode),
         "PMC4633642",  # Lalanza et al., Sci Rep 2015
         "Each trial consisted of 10 sec of conditioned stimulus, immediately followed by a scrambled electric shock. Crossing from one side to the other compartment terminated the CS or UCS presentation."),
        
        (PlacePreference(view_mode=view_mode),
         "PMC6101638",  # Blanco-Gandía et al., J Vis Exp 2018
         "The procedure consists of three phases: Pre-Conditioning, Conditioning, and Post-Conditioning. Compartments have different floor textures and wall colors."),
        
        (StarMaze(view_mode=view_mode),
         "PMC3695082",  # Fouquet et al., PLoS ONE 2013
         "The starmaze task relied on a massed training phase composed of 10 sessions of 4 trials in the presence of all visual cues."),
        
        (DNMSTask(view_mode=view_mode),
         "PMC3982138",  # Oomen et al., Nat Protoc 2013
         "TUNL working memory task requires animals to non-match to a sample location after a delay. A correct response to the novel location leads to reward delivery."),
    ]


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_trial(env, agent, max_steps: int = 200, log_file=None, last_reward=None, k: int = 8, is_first_trial: bool = True) -> Tuple[TrialResult, float]:
    """Run single trial with unified protocol. Returns (result, final_reward) for continuity."""
    result = TrialResult()
    # Don't reset agent - continuous stream from LLM perspective
    
    if is_first_trial:
        obs = env.reset()
    else:
        # If the env session ended (criterion met or max trials reached),
        # extend the session so the benchmark can keep running trials.
        # The benchmark controls how many trials to run, not the env.
        if env.is_done:
            env.session.criterion_met = False
            if env.session.current_trial >= env.session.max_trials:
                env.session.max_trials = env.session.current_trial + 100
            env._start_trial()
        obs = env.render()
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
            
            # Log the LLM call (observations already logged as step outputs)
            if log_file and hasattr(agent, 'last_response') and agent.last_response:
                log_file.write(f"\n--- LLM Call (steps {step+1}-{step+len(actions)}, {len(observations_for_llm)} obs) ---\n")
                if hasattr(agent, 'notes') and agent.notes:
                    log_file.write(f"Agent learnings: {agent.notes}\n")
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
                    if isinstance(obs, np.ndarray):
                        log_file.write(f"[IMAGE {obs.shape[0]}x{obs.shape[1]}]\n")
                    else:
                        log_file.write(f"{obs}\n")
                    log_file.flush()
                
                # Check if trial completed
                if env.is_done or env.session.current_trial != initial_trial:
                    if env.session.trial_results:
                        result.success = env.session.trial_results[-1].success
                    if log_file:
                        # Log the final observation that led to trial end
                        if isinstance(obs, np.ndarray):
                            log_file.write(f"  Final observation after {action.name}:\n[IMAGE {obs.shape[0]}x{obs.shape[1]}]\n")
                        else:
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


def run_benchmark(cfg: BenchmarkConfig = None) -> Dict:
    """Run full benchmark with given configuration."""
    if cfg is None:
        cfg = CFG

    # Seed for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Ensure output directory exists
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(cfg.max_actions_per_call, cfg.prompt_variant)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.model,
        "seed": cfg.seed,
        "num_trials": cfg.num_trials,
        "max_steps_per_trial": cfg.max_steps_per_trial,
        "max_actions_per_call": cfg.max_actions_per_call,
        "max_history_messages": cfg.max_history_messages,
        "prompt_variant": cfg.prompt_variant,
        "system_prompt": system_prompt,
        "env_protocols": ENV_PROTOCOL,
        "results": [],
    }

    agents = [
        ("Random", RandomAgent(seed=cfg.seed)),
        ("LLM", LLMAgent(model=cfg.model)),
    ]

    # Open structured log file for LLM traces
    trace_path = out_dir / "llm_traces.log"
    log_file = open(trace_path, "w")
    log_file.write(f"CheeseBench LLM Traces — {datetime.now().isoformat()}\n")
    log_file.write(f"Model: {cfg.model} | Seed: {cfg.seed}\n")
    log_file.write("=" * 60 + "\n")
    log_file.flush()

    # Determine view modes from config, falling back to VIEW_MODES
    active_view_modes = []
    for vm_name in cfg.view_modes:
        try:
            active_view_modes.append(ViewMode[vm_name])
        except KeyError:
            print(f"WARNING: Unknown view mode '{vm_name}', skipping")

    # Skip image view modes for text-only models
    is_text_only = "text" in cfg.model.lower() or (
        not any(v in cfg.model.lower() for v in ["vl", "vision", "multimodal", "mm"])
        and "Instruct" in cfg.model and "VL" not in cfg.model
    )
    if is_text_only:
        active_view_modes = [vm for vm in active_view_modes if vm not in IMAGE_VIEW_MODES]

    for view_mode in active_view_modes:
        mode_name = view_mode.name
        if cfg.verbose:
            print(f"\n{'=' * 60}")
            print(f"VIEW MODE: {mode_name}")
            print("=" * 60)
        
        envs = create_environments(view_mode)
        
        for env, pmc, quote in envs:
            env_name = env.__class__.__name__

            # Per-environment protocol; fall back to global config
            proto = ENV_PROTOCOL.get(env_name, {})
            env_max_steps = proto.get("max_steps", cfg.max_steps_per_trial)
            env_num_trials = proto.get("num_trials", cfg.num_trials)
            # CLI --num-trials / --max-steps override per-env defaults
            if getattr(cfg, '_cli_max_steps', False):
                env_max_steps = cfg.max_steps_per_trial
            if getattr(cfg, '_cli_num_trials', False):
                env_num_trials = cfg.num_trials

            for agent_name, agent in agents:
                if cfg.verbose:
                    print(f"\n  {env_name} ({agent_name}, {env_num_trials} trials, {env_max_steps} steps)...", end=" ", flush=True)

                if agent_name == "LLM":
                    log_file.write(f"\n{'=' * 60}\n")
                    log_file.write(f"Environment: {env_name} | Mode: {mode_name} | Trials: {env_num_trials} | MaxSteps: {env_max_steps}\n")
                    log_file.write(f"{'=' * 60}\n")

                benchmark_result = BenchmarkResult(
                    env_name=env_name,
                    view_mode=mode_name,
                    agent_type=agent_name,
                    source_pmc=pmc,
                    source_quote=quote,
                )

                for trial in range(env_num_trials):
                    if trial == 0:
                        agent.reset()
                        last_reward = None
                    trial_log = log_file if agent_name == "LLM" else None
                    if trial_log:
                        log_file.write(f"\n--- Trial {trial + 1}/{env_num_trials} ---\n")
                    trial_result, last_reward = run_trial(
                        env, agent,
                        max_steps=env_max_steps,
                        log_file=trial_log,
                        last_reward=last_reward,
                        is_first_trial=(trial == 0),
                    )
                    # Record parse failures from agent
                    if hasattr(agent, 'parse_failures'):
                        trial_result.parse_failures = agent.parse_failures
                        agent.parse_failures = 0
                    benchmark_result.trials.append(trial_result)
                    if trial_log:
                        log_file.write(f"Result: {'SUCCESS' if trial_result.success else 'FAIL'} in {trial_result.steps} steps\n")

                results["results"].append(benchmark_result.to_dict())

                if cfg.verbose:
                    sr = benchmark_result.success_rate
                    print(f"{benchmark_result.successes}/{env_num_trials} ({sr * 100:.0f}%)")
    
    log_file.close()

    # Save results
    output_file = out_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    if cfg.verbose:
        print(f"\nResults saved to {output_file}")
        print(f"LLM traces saved to {trace_path}")

    return results


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("CHEESEBENCH RESULTS SUMMARY")
    print("=" * 80)

    # Per-environment summary
    by_env_agent = {}
    for r in results["results"]:
        key = (r["env_name"], r["agent_type"], r["view_mode"])
        by_env_agent[key] = r

    print(f"\n{'Environment':<20} {'Agent':<8} {'View':<14} {'Success':<12} {'Steps(ok)':<10} {'Steps(fail)':<12}")
    print("-" * 78)
    for key in sorted(by_env_agent.keys()):
        r = by_env_agent[key]
        sr_str = f"{r['successes']}/{r['total_trials']} ({r['success_rate']*100:.0f}%)"
        ok_str = f"{r['avg_steps_success']:.0f}" if r.get('avg_steps_success') else "-"
        fail_str = f"{r['avg_steps_failure']:.0f}" if r.get('avg_steps_failure') else "-"
        print(f"{r['env_name']:<20} {r['agent_type']:<8} {r['view_mode']:<14} {sr_str:<12} {ok_str:<10} {fail_str:<12}")

    # Aggregate by agent
    print(f"\n{'Agent':<10} {'Overall Success Rate':<25}")
    print("-" * 35)
    agent_totals = {}
    for r in results["results"]:
        a = r["agent_type"]
        if a not in agent_totals:
            agent_totals[a] = {"s": 0, "t": 0}
        agent_totals[a]["s"] += r["successes"]
        agent_totals[a]["t"] += r["total_trials"]
    for a, d in sorted(agent_totals.items()):
        rate = d["s"] / d["t"] if d["t"] > 0 else 0
        print(f"{a:<10} {d['s']}/{d['t']} ({rate*100:.1f}%)")


# =============================================================================
# MAIN — with argparse for proper CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="CheeseBench — LLM Benchmark for Behavioral Neuroscience")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: from config/env)")
    parser.add_argument("--num-trials", type=int, default=None,
                        help="Override trials per env (default: per-environment protocol)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per trial (default: per-environment protocol)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: results/)")
    parser.add_argument("--api-url", type=str, default=None, help="API endpoint URL")
    parser.add_argument("--api-format", type=str, default=None, choices=["openai", "ollama"],
                        help="API format: openai (vLLM/OpenAI) or ollama")
    parser.add_argument("--prompt-variant", type=str, default=None,
                        choices=["default", "minimal", "cot", "few_shot"],
                        help="Prompt variant for ablation")
    parser.add_argument("--max-actions", type=int, default=None,
                        help="Max actions per LLM call (default: 8)")
    parser.add_argument("--max-history", type=int, default=None,
                        help="Max history message pairs (default: 5)")
    parser.add_argument("--view-modes", type=str, nargs="+", default=None,
                        help="View modes to run (e.g. ASCII_2D TOPDOWN_2D)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override config from CLI
    if args.model:
        CFG.model = args.model
    if args.num_trials:
        CFG.num_trials = args.num_trials
        CFG._cli_num_trials = True
    if args.max_steps:
        CFG.max_steps_per_trial = args.max_steps
        CFG._cli_max_steps = True
    if args.seed is not None:
        CFG.seed = args.seed
    if args.output_dir:
        CFG.output_dir = args.output_dir
    if args.api_url:
        CFG.api_url = args.api_url
    if args.api_format:
        CFG.api_format = args.api_format
    if args.prompt_variant:
        CFG.prompt_variant = args.prompt_variant
    if args.max_actions:
        CFG.max_actions_per_call = args.max_actions
    if args.max_history:
        CFG.max_history_messages = args.max_history
    if args.view_modes:
        CFG.view_modes = args.view_modes
    if args.quiet:
        CFG.verbose = False

    print("=" * 60)
    print("CHEESEBENCH — LLM Benchmark for Behavioral Neuroscience")
    print("=" * 60)
    print(f"Model:  {CFG.model}")
    print(f"Trials: per-environment (default {CFG.num_trials})")
    print(f"Views:  {CFG.view_modes}")
    print(f"Seed:   {CFG.seed}")
    print("=" * 60)

    results = run_benchmark(CFG)

    print_summary(results)


if __name__ == "__main__":
    main()
