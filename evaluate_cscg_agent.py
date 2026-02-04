#!/usr/bin/env python3
"""
Evaluate the CSCG (Clone-Structured Cognitive Graph) agent on all environments.
Generates benchmark results and demo videos for all view modes.
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import imageio

sys.path.insert(0, '.')
from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze,
    OperantChamber, ShuttleBox, PlacePreference, StarMaze, DNMSTask,
    Action, ViewMode
)

# =============================================================================
# CSCG AGENT (from user's code)
# =============================================================================
import math
from collections import defaultdict, Counter, deque

_AGENT_CHARS = "^>v<"
_GOAL_CHARS = "GP*+"
_WALL_CHARS = "#X"


class ObservationEncoder:
    """Stateless helper to parse ASCII observations."""

    @staticmethod
    def fingerprint(obs: str) -> Tuple[int, str, int]:
        MOD = 2 ** 61 - 1
        h = 0
        for c in obs:
            h = (h * 131 + ord(c)) % MOD

        orient = "^"
        lines = obs.splitlines()
        rows = len(lines)
        ax = ay = None
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                if ch in _AGENT_CHARS:
                    orient = ch
                    ax, ay = x, y
                    break
            if ax is not None:
                break

        mask = 0
        if ax is not None:
            dirs = [(0, -1), (1, -1), (1, 0), (1, 1),
                    (0, 1), (-1, 1), (-1, 0), (-1, -1)]
            for i, (dx, dy) in enumerate(dirs):
                nx, ny = ax + dx, ay + dy
                if 0 <= ny < rows and 0 <= nx < len(lines[ny]):
                    if lines[ny][nx] in _WALL_CHARS:
                        mask |= (1 << i)
                else:
                    mask |= (1 << i)
        return h, orient, mask

    @staticmethod
    def locate_agent_and_goals(obs: str) -> Tuple[Optional[int], Optional[int],
                                                 Optional[str],
                                                 List[Tuple[int, int]]]:
        lines = obs.splitlines()
        ax = ay = None
        orient = None
        goals: List[Tuple[int, int]] = []
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                if ch in _AGENT_CHARS:
                    ax, ay = x, y
                    orient = ch
                if ch in _GOAL_CHARS:
                    goals.append((x, y))
        return ax, ay, orient, goals

    @staticmethod
    def has_goal(obs: str) -> bool:
        return any(ch in _GOAL_CHARS for ch in obs)

    @staticmethod
    def direction_to_goal(ax: int, ay: int, orient: str,
                          goals: List[Tuple[int, int]]) -> Optional[int]:
        if not goals:
            return None
        gx, gy = min(goals, key=lambda g: abs(g[0] - ax) + abs(g[1] - ay))
        dx, dy = gx - ax, gy - ay

        if abs(dx) > abs(dy):
            target_dir = ">" if dx > 0 else "<"
        else:
            target_dir = "v" if dy > 0 else "^"

        order = ["^", ">", "v", "<"]
        cur_idx = order.index(orient)
        targ_idx = order.index(target_dir)
        diff = (targ_idx - cur_idx) % 4

        if diff == 0:
            return 0
        if diff == 1:
            return 2
        if diff == 3:
            return 1
        return 2


@dataclass
class Node:
    nid: int
    fp: Tuple[int, str, int]
    visits: int = 0
    reward: float = 0.0
    is_goal: bool = False
    edges: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    value: float = 0.0


class TransitionGraph:
    def __init__(self, gamma: float = 0.96, eps: float = 1e-4):
        self.nodes: List[Node] = []
        self.fp_index: Dict[Tuple[int, str, int], Node] = {}
        self.gamma = gamma
        self.eps = eps
        self._bellman_queue: deque[int] = deque()

    def get_node(self, fp: Tuple[int, str, int]) -> Node:
        node = self.fp_index.get(fp)
        if node is None:
            nid = len(self.nodes)
            node = Node(nid=nid, fp=fp)
            self.nodes.append(node)
            self.fp_index[fp] = node
        return node

    def add_edge(self, src: Node, action: int, dst: Node, weight: float = 1.0) -> None:
        src.edges[action][dst.nid] += weight

    def _bellman_update(self, nid: int) -> bool:
        node = self.nodes[nid]
        if not node.edges:
            new_val = node.reward
        else:
            best_q = -float('inf')
            for a, cnts in node.edges.items():
                total = sum(cnts.values()) + 1e-9
                q = 0.0
                for dst_nid, cnt in cnts.items():
                    prob = cnt / total
                    dst = self.nodes[dst_nid]
                    q += prob * (node.reward + self.gamma * dst.value)
                if q > best_q:
                    best_q = q
            new_val = best_q
        if abs(new_val - node.value) > self.eps:
            node.value = new_val
            return True
        return False

    def propagate_values(self, changed: set) -> None:
        for nid in changed:
            self._bellman_queue.append(nid)

        while self._bellman_queue:
            nid = self._bellman_queue.popleft()
            if self._bellman_update(nid):
                for pred in self.nodes:
                    for a_counter in pred.edges.values():
                        if nid in a_counter:
                            self._bellman_queue.append(pred.nid)
                            break

    def bfs_to_goals(self, start_nid: int, goal_nids: set,
                     max_depth: int = 20) -> List[int]:
        if not goal_nids:
            return []
        visited = {start_nid}
        queue = deque([(start_nid, [])])
        while queue:
            cur, path = queue.popleft()
            if cur in goal_nids:
                return path
            if len(path) >= max_depth:
                continue
            node = self.nodes[cur]
            for a, cnts in node.edges.items():
                if not cnts:
                    continue
                nb_nid = max(cnts.items(), key=lambda kv: kv[1])[0]
                if nb_nid not in visited:
                    visited.add(nb_nid)
                    queue.append((nb_nid, path + [a]))
        return []


class Policy:
    def __init__(self, graph: TransitionGraph,
                 n_actions: int = 3,
                 ucb_c: float = 0.85,
                 novelty_beta: float = 0.75):
        self.graph = graph
        self.n_actions = n_actions
        self.ucb_c = ucb_c
        self.novelty_beta = novelty_beta
        self.N_sa = defaultdict(int)

    def select(self, node: Node, temperature: float) -> int:
        parent_visits = node.visits + 1e-5
        scores = []
        for a in range(self.n_actions):
            if a in node.edges and node.edges[a]:
                total = sum(node.edges[a].values()) + 1e-9
                q = 0.0
                for dst_nid, cnt in node.edges[a].items():
                    prob = cnt / total
                    dst = self.graph.nodes[dst_nid]
                    q += prob * (node.reward + self.graph.gamma * dst.value)
            else:
                q = node.reward

            n_sa = self.N_sa[(node.nid, a)] + 1
            ucb = self.ucb_c * math.sqrt(math.log(parent_visits) / n_sa)
            novelty = self.novelty_beta / math.sqrt(node.visits + 1e-5)

            scores.append(q + ucb + novelty)

        max_s = max(scores)
        exp_vals = [math.exp((s - max_s) / max(temperature, 1e-3)) for s in scores]
        total = sum(exp_vals)
        probs = [e / total for e in exp_vals]
        chosen = random.choices(range(self.n_actions), probs)[0]
        self.N_sa[(node.nid, chosen)] += 1
        return chosen


class CSCGAgent:
    """Clone-Structured Cognitive Graph agent."""

    ACTIONS = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.STAY]
    ACTION_INDICES = [0, 2, 3, 7]  # FORWARD, LEFT, RIGHT, STAY
    N_ACTIONS = 3  # we never deliberately choose STAY

    def __init__(self,
                 gamma: float = 0.96,
                 ucb_c: float = 0.85,
                 novelty_beta: float = 0.75,
                 seed: int = 123):
        random.seed(seed)
        self.graph = TransitionGraph(gamma=gamma)
        self.policy = Policy(self.graph,
                             n_actions=self.N_ACTIONS,
                             ucb_c=ucb_c,
                             novelty_beta=novelty_beta)

        self.prev_node: Optional[Node] = None
        self.prev_action: Optional[int] = None
        self.prev_fp: Optional[Tuple[int, str, int]] = None
        self.stuck_counter = 0
        self.step_without_new = 0
        self.episode_idx = 0

    def reset(self) -> None:
        self.prev_node = None
        self.prev_action = None
        self.prev_fp = None
        self.stuck_counter = 0
        self.step_without_new = 0
        self.episode_idx += 1

    def _detect_stuck(self, cur_fp: Tuple[int, str, int]) -> bool:
        if cur_fp == self.prev_fp:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.prev_fp = cur_fp

        if self.prev_node and self.prev_node.fp == cur_fp:
            self.step_without_new += 1
        else:
            self.step_without_new = 0

        return self.stuck_counter > 4 or self.step_without_new > 30

    def get_action(self, observation: str, reward: float = 0.0) -> Action:
        """Get action for the given observation."""
        cur_fp = ObservationEncoder.fingerprint(observation)
        has_goal = ObservationEncoder.has_goal(observation)
        ax, ay, orient, goals = ObservationEncoder.locate_agent_and_goals(observation)

        cur_node = self.graph.get_node(cur_fp)
        cur_node.visits += 1
        if has_goal:
            cur_node.is_goal = True
            cur_node.reward = 1.0

        changed_nids = set()
        if self.prev_node is not None and self.prev_action is not None:
            self.graph.add_edge(self.prev_node, self.prev_action, cur_node)
            changed_nids.add(self.prev_node.nid)

        if changed_nids:
            self.graph.propagate_values(changed_nids)

        if self._detect_stuck(cur_fp):
            turn = random.choice([1, 2])
            self.prev_node = cur_node
            self.prev_action = turn
            self.stuck_counter = 0
            self.step_without_new = 0
            return self.ACTIONS[turn]

        if ax is not None and ay is not None and orient is not None and goals:
            dir_idx = ObservationEncoder.direction_to_goal(ax, ay, orient, goals)
            if dir_idx is not None:
                self.prev_node = cur_node
                self.prev_action = dir_idx
                return self.ACTIONS[dir_idx]

        if cur_node.is_goal:
            chosen_idx = 0
        else:
            goal_nids = {n.nid for n in self.graph.nodes if n.is_goal}
            plan = self.graph.bfs_to_goals(cur_node.nid, goal_nids)
            if plan:
                chosen_idx = plan[0]
            else:
                temperature = max(0.2, 1.0 / math.sqrt(self.episode_idx + 1))
                chosen_idx = self.policy.select(cur_node, temperature)

        self.prev_node = cur_node
        self.prev_action = chosen_idx
        return self.ACTIONS[chosen_idx]


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_STEPS = 200
NUM_TRIALS = 10
OUTPUT_DIR = "cscg_evaluation"
FPS = 5

VIEW_MODES = [
    ViewMode.ASCII_2D,
    ViewMode.ASCII_2D_FPV,
    ViewMode.ASCII_3D,
]


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
            "successes": self.successes,
            "total_trials": len(self.trials),
            "trials": [{"steps": t.steps, "reward": t.reward, "success": t.success, 
                       "num_actions": len(t.actions)} for t in self.trials]
        }


# =============================================================================
# RENDERING UTILITIES
# =============================================================================

def get_font(size=12):
    """Get a font, falling back to default if needed."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def render_ascii_to_image(ascii_text: str, width: int = 400, height: int = 400) -> Image.Image:
    """Convert ASCII art to a PIL Image."""
    img = Image.new('RGB', (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    font = get_font(11)
    
    color_map = {
        '#': (100, 100, 120),
        '.': (60, 80, 60),
        ' ': (30, 30, 40),
        'E': (50, 255, 50),
        'G': (50, 255, 50),
        'P': (50, 200, 255),
        '?': (255, 200, 100),
        '*': (255, 215, 0),
        'R': (255, 100, 100),
        '~': (50, 100, 200),
        '1': (255, 100, 100),
        '2': (100, 255, 100),
        '3': (100, 100, 255),
        '4': (255, 255, 100),
        '↑': (255, 80, 80),
        '↓': (255, 80, 80),
        '←': (255, 80, 80),
        '→': (255, 80, 80),
        '↗': (255, 80, 80),
        '↘': (255, 80, 80),
        '↙': (255, 80, 80),
        '↖': (255, 80, 80),
        '^': (255, 80, 80),
        'v': (255, 80, 80),
        '<': (255, 80, 80),
        '>': (255, 80, 80),
        '@': (255, 80, 80),
        'L': (200, 150, 50),
        'S': (255, 100, 100),
        'C': (100, 200, 255),
        '●': (255, 200, 100),
        '■': (200, 100, 200),
        '[': (150, 150, 150),
        ']': (150, 150, 150),
        '=': (200, 200, 100),
        '_': (100, 100, 100),
        'M': (255, 215, 0),
        'm': (100, 100, 100),
    }
    
    lines = ascii_text.split('\n')
    y_offset = 10
    
    for line in lines:
        x_offset = 10
        for char in line:
            color = color_map.get(char, (180, 180, 180))
            draw.text((x_offset, y_offset), char, fill=color, font=font)
            x_offset += 8
        y_offset += 14
    
    return img


def create_combined_frame(observations: Dict[ViewMode, str], env_name: str, step: int,
                          reward: float, total_reward: float, action: str = "",
                          trial: int = 1, success_count: int = 0) -> Image.Image:
    """Create a single frame with all 3 view modes side by side."""
    
    view_width = 400
    view_height = 400
    padding = 10
    header_height = 70
    footer_height = 40
    
    total_width = view_width * 3 + padding * 4
    total_height = view_height + header_height + footer_height + padding * 2
    
    img = Image.new('RGB', (total_width, total_height), color=(15, 15, 20))
    draw = ImageDraw.Draw(img)
    
    font_large = get_font(20)
    font_medium = get_font(14)
    font_small = get_font(12)
    
    # Header
    draw.text((padding, 8), f"{env_name} - CSCG Agent", fill=(100, 200, 255), font=font_large)
    
    # Trial and step info
    info_x = total_width - 400
    draw.text((info_x, 8), f"Trial: {trial}/10", fill=(200, 200, 100), font=font_medium)
    draw.text((info_x + 100, 8), f"Step: {step}", fill=(150, 150, 150), font=font_medium)
    if action:
        draw.text((info_x + 180, 8), f"Action: {action}", fill=(180, 180, 180), font=font_medium)
    
    # Reward info
    reward_color = (100, 255, 100) if reward >= 0 else (255, 100, 100)
    draw.text((info_x, 28), f"Reward: {reward:+.2f}", fill=reward_color, font=font_medium)
    draw.text((info_x + 130, 28), f"Total: {total_reward:.2f}", fill=(150, 150, 150), font=font_medium)
    draw.text((info_x + 260, 28), f"Successes: {success_count}", fill=(100, 255, 100), font=font_medium)
    
    # Render each view mode
    view_labels = ["ASCII_2D (Top-down)", "ASCII_2D_FPV (Cropped)", "ASCII_3D (First-person)"]
    view_keys = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
    
    for i, (label, view_key) in enumerate(zip(view_labels, view_keys)):
        x = padding + i * (view_width + padding)
        y = header_height
        
        draw.text((x + 10, y - 18), label, fill=(180, 180, 180), font=font_small)
        
        obs_text = observations.get(view_key, "No observation")
        ascii_img = render_ascii_to_image(obs_text, view_width, view_height)
        img.paste(ascii_img, (x, y))
        draw.rectangle([x-1, y-1, x + view_width, y + view_height], outline=(60, 60, 80))
    
    return img


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def create_environments(view_mode: ViewMode) -> List[Tuple]:
    """Create all benchmark environments."""
    return [
        (MorrisWaterMaze(view_mode=view_mode), "MorrisWaterMaze"),
        (TMaze(view_mode=view_mode), "TMaze"),
        (BarnesMaze(view_mode=view_mode), "BarnesMaze"),
        (RadialArmMaze(view_mode=view_mode), "RadialArmMaze"),
        (OperantChamber(view_mode=view_mode), "OperantChamber"),
        (ShuttleBox(view_mode=view_mode), "ShuttleBox"),
        (PlacePreference(view_mode=view_mode), "PlacePreference"),
        (StarMaze(view_mode=view_mode), "StarMaze"),
        (DNMSTask(view_mode=view_mode), "DNMSTask"),
    ]


# =============================================================================
# BENCHMARK RUNNER WITH VIDEO GENERATION
# =============================================================================

def run_trial_with_video(env, agent, max_steps: int, 
                         record_video: bool = False,
                         trial_num: int = 1,
                         success_count: int = 0,
                         env_name: str = "") -> Tuple[TrialResult, List[np.ndarray]]:
    """Run single trial and optionally record frames."""
    result = TrialResult()
    frames = []
    
    obs = env.reset()
    reward = 0.0
    total_reward = 0.0
    initial_trial = env.session.current_trial
    
    def get_all_observations():
        """Render the same env state in all view modes."""
        observations = {}
        original_mode = env.view_mode
        for vm in VIEW_MODES:
            env.view_mode = vm
            observations[vm] = env.render()
        env.view_mode = original_mode
        return observations
    
    # Initial frame
    if record_video:
        observations = get_all_observations()
        frame = create_combined_frame(observations, env_name, 0, 0.0, total_reward, 
                                      "START", trial_num, success_count)
        frames.append(np.array(frame))
    
    step = 0
    while step < max_steps:
        action = agent.get_action(obs, reward)
        obs, reward = env.step(action)
        
        result.steps += 1
        result.reward += reward
        total_reward += reward
        result.actions.append(action.name)
        step += 1
        
        if record_video:
            observations = get_all_observations()
            frame = create_combined_frame(observations, env_name, step, reward, total_reward,
                                          action.name, trial_num, success_count)
            frames.append(np.array(frame))
        
        # Check if trial completed
        if env.is_done or env.session.current_trial != initial_trial:
            if env.session.trial_results:
                result.success = env.session.trial_results[-1].success
            break
    
    # Add pause at end for video
    if record_video and frames:
        for _ in range(3):
            frames.append(frames[-1])
    
    return result, frames


def run_benchmark_with_videos(num_trials: int = NUM_TRIALS, verbose: bool = True) -> Dict:
    """Run full benchmark with video generation."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/videos", exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "agent_type": "CSCG",
        "num_trials": num_trials,
        "results": []
    }
    
    for view_mode in VIEW_MODES:
        mode_name = view_mode.name
        if verbose:
            print(f"\n{'='*60}")
            print(f"VIEW MODE: {mode_name}")
            print('='*60)
        
        envs = create_environments(view_mode)
        
        for env, env_name in envs:
            if verbose:
                print(f"\n  {env_name}...", end=" ", flush=True)
            
            # Create fresh agent for each env/view combination
            agent = CSCGAgent(gamma=0.96, ucb_c=0.85, novelty_beta=0.75, seed=42)
            
            benchmark_result = BenchmarkResult(
                env_name=env_name,
                view_mode=mode_name,
                agent_type="CSCG"
            )
            
            all_frames = []
            success_count = 0
            
            for trial in range(num_trials):
                # Record video for all trials
                record = True
                
                # Reset agent for new trial
                agent.reset()
                
                trial_result, frames = run_trial_with_video(
                    env, agent, MAX_STEPS,
                    record_video=record,
                    trial_num=trial + 1,
                    success_count=success_count,
                    env_name=env_name
                )
                
                benchmark_result.trials.append(trial_result)
                
                if trial_result.success:
                    success_count += 1
                
                if record:
                    all_frames.extend(frames)
                    # Add transition frame between trials
                    if trial < num_trials - 1 and frames:
                        for _ in range(2):
                            all_frames.append(frames[-1])
            
            # Save video for this env/mode combination
            if all_frames:
                video_path = f"{OUTPUT_DIR}/videos/{env_name}_{mode_name}.mp4"
                imageio.mimsave(video_path, all_frames, fps=FPS)
                if verbose:
                    print(f"Video saved: {video_path}")
            
            results["results"].append(benchmark_result.to_dict())
            
            if verbose:
                sr = benchmark_result.success_rate
                print(f"  -> {benchmark_result.successes}/{num_trials} ({sr*100:.0f}%)")
    
    return results


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "="*80)
    print("CSCG AGENT BENCHMARK SUMMARY")
    print("="*80)
    
    # Aggregate by view mode
    summary = {}
    for r in results["results"]:
        mode = r["view_mode"]
        if mode not in summary:
            summary[mode] = {"successes": 0, "total": 0, "envs": {}}
        summary[mode]["successes"] += r["successes"]
        summary[mode]["total"] += r["total_trials"]
        summary[mode]["envs"][r["env_name"]] = r["success_rate"]
    
    print(f"\n{'View Mode':<20} {'Success Rate':<20} {'Successes/Total':<15}")
    print("-"*60)
    for mode, data in sorted(summary.items()):
        rate = data["successes"] / data["total"] if data["total"] > 0 else 0
        print(f"{mode:<20} {rate*100:.1f}%{'':<15} {data['successes']}/{data['total']}")
    
    # Per-environment breakdown
    print("\n" + "="*80)
    print("PER-ENVIRONMENT BREAKDOWN")
    print("="*80)
    
    env_results = {}
    for r in results["results"]:
        env_name = r["env_name"]
        if env_name not in env_results:
            env_results[env_name] = {}
        env_results[env_name][r["view_mode"]] = r["success_rate"]
    
    print(f"\n{'Environment':<20}", end="")
    for mode in ["ASCII_2D", "ASCII_2D_FPV", "ASCII_3D"]:
        print(f"{mode:<15}", end="")
    print()
    print("-"*65)
    
    for env_name in sorted(env_results.keys()):
        print(f"{env_name:<20}", end="")
        for mode in ["ASCII_2D", "ASCII_2D_FPV", "ASCII_3D"]:
            rate = env_results[env_name].get(mode, 0)
            print(f"{rate*100:.0f}%{'':<12}", end="")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CSCG AGENT EVALUATION")
    print("="*60)
    print(f"Trials per environment: {NUM_TRIALS}")
    print(f"View modes: {[m.name for m in VIEW_MODES]}")
    print("="*60)
    
    results = run_benchmark_with_videos(NUM_TRIALS, verbose=True)
    
    print_summary(results)
    
    # Save results
    output_file = f"{OUTPUT_DIR}/cscg_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print(f"Videos saved to {OUTPUT_DIR}/videos/")
