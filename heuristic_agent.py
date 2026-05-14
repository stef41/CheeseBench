"""
Simple goal-seeking heuristic agent for CheeseBench.
Parses ASCII_2D observations, finds goal markers, and uses BFS navigation.
Falls back to systematic exploration when no goal is visible.
"""
from collections import deque
from environments import Action


GOAL_CHARS = {'G', 'P', '*', '+', 'E'}
WALL_CHARS = {'#'}
AGENT_CHARS = {'^', 'v', '<', '>', '↑', '↓', '←', '→', '↗', '↘', '↖', '↙'}
HEADING_MAP = {
    '^': 0, '↑': 0,
    '>': 1, '→': 1, '↗': 1,
    'v': 2, '↓': 2, '↘': 2,
    '<': 3, '←': 3, '↖': 3, '↙': 3,
}
# Direction deltas: 0=up, 1=right, 2=down, 3=left
DY = [-1, 0, 1, 0]
DX = [0, 1, 0, -1]


class HeuristicAgent:
    """BFS goal-seeking agent with systematic exploration fallback."""

    def __init__(self):
        self.explore_dir = 0  # rotation counter for exploration

    def _parse_grid(self, obs: str):
        lines = obs.strip().split('\n')
        grid = [list(line) for line in lines]
        agent_pos = None
        agent_heading = None
        goals = []
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch in AGENT_CHARS:
                    agent_pos = (r, c)
                    agent_heading = HEADING_MAP.get(ch, 0)
                if ch in GOAL_CHARS:
                    goals.append((r, c))
        return grid, agent_pos, agent_heading, goals

    def _bfs(self, grid, start, goals):
        if not goals or not start:
            return None
        goal_set = set(goals)
        rows, cols = len(grid), max(len(r) for r in grid)
        visited = set()
        queue = deque([(start, [])])
        visited.add(start)
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) in goal_set:
                return path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < len(grid[nr]) and (nr, nc) not in visited:
                    ch = grid[nr][nc]
                    if ch not in WALL_CHARS:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(dr, dc)]))
        return None

    def _step_to_action(self, dr, dc, heading):
        """Convert a grid step (dr, dc) into egocentric action given current heading."""
        # Target absolute direction
        for d in range(4):
            if DY[d] == dr and DX[d] == dc:
                target_dir = d
                break
        else:
            return Action.FORWARD
        # How many right turns from current heading to target
        diff = (target_dir - heading) % 4
        if diff == 0:
            return Action.FORWARD
        elif diff == 1:
            return Action.ROTATE_RIGHT
        elif diff == 3:
            return Action.ROTATE_LEFT
        else:  # diff == 2, turn around
            return Action.ROTATE_RIGHT

    def get_action(self, obs: str, reward: float = None) -> Action:
        grid, agent_pos, heading, goals = self._parse_grid(obs)
        if agent_pos is None or heading is None:
            # Can't parse, explore
            self.explore_dir = (self.explore_dir + 1) % 8
            if self.explore_dir < 4:
                return Action.FORWARD
            else:
                return Action.ROTATE_RIGHT

        if goals:
            path = self._bfs(grid, agent_pos, goals)
            if path and len(path) > 0:
                dr, dc = path[0]
                return self._step_to_action(dr, dc, heading)

        # No goal visible: systematic exploration (rotate + forward)
        self.explore_dir += 1
        if self.explore_dir % 3 == 0:
            return Action.ROTATE_RIGHT
        elif self.explore_dir % 5 == 0:
            return Action.ROTATE_LEFT
        else:
            return Action.FORWARD
