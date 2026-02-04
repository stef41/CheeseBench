"""
CSCG-QL (Clone-Structured Cognitive Graph + Q-Learning) Agent.
"""

import random
import numpy as np
from typing import Dict, Tuple, List, Set
from collections import defaultdict, deque

from environments import Action


def _hash_obs(obs: str) -> int:
    """Deterministic 32-bit hash of the raw ASCII observation."""
    return hash(obs) & 0xffffffff


class CSCGQLAgent:
    """
    Hybrid CSCG + Q-learning agent.
    """

    # Action codes used by the environment
    ACTIONS = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.STAY]
    ACTION_CODES = [0, 2, 3, 7]  # FORWARD, ROTATE_LEFT, ROTATE_RIGHT, STAY
    N_ACTIONS = len(ACTIONS)

    # Symbols that typically indicate a goal / reward in CheeseBench
    GOAL_SYMBOLS = set("GP*+")

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.98,
        min_epsilon: float = 0.05,
        seed: int = 42,
        stuck_window: int = 5,
        bfs_depth_limit: int = 50,
    ):
        random.seed(seed)
        np.random.seed(seed)

        # Q-table: state → action-value vector
        self.Q: Dict[Tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.N_ACTIONS, dtype=np.float32)
        )

        # CSCG transition graph: (state, action_idx) → next_state
        self.graph: Dict[Tuple[Tuple[int, int], int], Tuple[int, int]] = {}
        # Reverse lookup for BFS from goal (optional, built on-the-fly)
        self.rev_graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int]]] = defaultdict(list)

        # Set of states that have been observed to contain a goal symbol
        self.goal_states: Set[Tuple[int, int]] = set()

        # Hyper-parameters
        self.base_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Stuck detection
        self.stuck_window = stuck_window
        self.recent_states: deque = deque(maxlen=stuck_window)

        # Planning limits
        self.bfs_depth_limit = bfs_depth_limit

        # Episode bookkeeping
        self.episode_num = 0
        self.heading = 0
        self.last_state: Tuple[int, int] = None
        self.last_action_idx: int = None

    def reset(self) -> None:
        """Reset episode-level state and decay exploration."""
        self.episode_num += 1
        self.heading = 0
        self.last_state = None
        self.last_action_idx = None
        self.recent_states.clear()
        # decay epsilon but keep above min
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_q(self, prev_state, prev_action_idx, reward, cur_state):
        """Standard Q-learning update."""
        alpha = self.base_alpha / np.sqrt(self.episode_num)
        max_next = np.max(self.Q[cur_state])
        td_target = reward + self.gamma * max_next
        td_error = td_target - self.Q[prev_state][prev_action_idx]
        self.Q[prev_state][prev_action_idx] += alpha * td_error

    def _store_transition(self, prev_state, action_idx, next_state):
        """Add transition to CSCG graph (both forward and reverse)."""
        key = (prev_state, action_idx)
        self.graph[key] = next_state
        self.rev_graph[next_state].append((prev_state, action_idx))

    def _detect_goal(self, observation: str, cur_state: Tuple[int, int]) -> bool:
        """If a goal symbol appears, register the state as a goal."""
        if any(sym in observation for sym in self.GOAL_SYMBOLS):
            self.goal_states.add(cur_state)
            return True
        return False

    def _bfs_to_goal(self, start: Tuple[int, int]) -> List[int]:
        """
        Breadth-first search from `start` to any known goal state.
        Returns a list of action indices leading to the goal (empty if none).
        """
        if not self.goal_states:
            return []

        visited = {start}
        queue = deque()
        queue.append((start, []))

        while queue:
            node, path = queue.popleft()
            if node in self.goal_states:
                return path

            if len(path) >= self.bfs_depth_limit:
                continue

            for a_idx in range(self.N_ACTIONS):
                nxt = self.graph.get((node, a_idx))
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [a_idx]))

        return []

    def get_action(self, observation: str, reward: float = 0.0) -> Action:
        """
        Choose the next action based on current observation and received reward.
        """
        # Build current state (hash + heading)
        obs_hash = _hash_obs(observation)
        cur_state = (obs_hash, self.heading)

        # Q-learning update from previous step (if any)
        if self.last_state is not None and self.last_action_idx is not None:
            self._update_q(self.last_state, self.last_action_idx, reward, cur_state)
            self._store_transition(self.last_state, self.last_action_idx, cur_state)

        # Goal detection
        goal_seen = self._detect_goal(observation, cur_state)

        # Stuck detection
        self.recent_states.append(cur_state)
        is_stuck = self.recent_states.count(cur_state) == self.stuck_window

        # Planning: try to follow a known path to any goal
        planned_action_idx = None
        if self.goal_states:
            path = self._bfs_to_goal(cur_state)
            if path:
                planned_action_idx = path[0]

        # Decide which action to take
        if goal_seen:
            chosen_action_idx = 0  # FORWARD
        elif planned_action_idx is not None:
            chosen_action_idx = planned_action_idx
        elif is_stuck:
            chosen_action_idx = random.choice([1, 2])  # LEFT or RIGHT
        else:
            if random.random() < self.epsilon:
                chosen_action_idx = random.choice([0, 1, 2])
            else:
                q_vals = self.Q[cur_state]
                chosen_action_idx = int(np.argmax(q_vals))

        # Update heading
        action = self.ACTIONS[chosen_action_idx]
        if action == Action.ROTATE_LEFT:
            self.heading = (self.heading - 1) % 8
        elif action == Action.ROTATE_RIGHT:
            self.heading = (self.heading + 1) % 8

        # Store for next step
        self.last_state = cur_state
        self.last_action_idx = chosen_action_idx

        return action
