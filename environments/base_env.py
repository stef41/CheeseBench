"""
Base environment class for VLM evaluation.

Interface:
    next_view, reward = env.step(action)

The environment internally handles:
    - Trial counting
    - Timeout detection  
    - Success criteria checking
    - Agent teleportation between trials
    - Session management
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import time


class ViewMode(Enum):
    """Observation format options."""
    FPV_3D = auto()         # First-person 3D view (default for VLM)
    TOPDOWN_2D = auto()     # Top-down 2D view
    ASCII_2D = auto()       # ASCII top-down (full map)
    ASCII_3D = auto()       # ASCII pseudo-3D FPV
    ASCII_2D_FPV = auto()   # ASCII top-down cropped around agent (partial map)


class Action(Enum):
    """Standard discrete actions."""
    FORWARD = 0
    TURN_LEFT = 2
    TURN_RIGHT = 3
    # Context-specific actions
    INTERACT = 6      # Press lever, enter hole, etc.
    STAY = 7          # Do nothing


@dataclass
class TrialResult:
    """Result of a single trial."""
    trial_number: int
    success: bool
    time_steps: int
    timeout: bool
    path_length: float
    reward_collected: float
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SessionState:
    """Current session state."""
    current_trial: int = 0
    total_trials: int = 0
    trials_to_criterion: int = 0
    max_trials: int = 100
    max_steps_per_trial: int = 1000
    criterion_met: bool = False
    consecutive_successes: int = 0
    required_consecutive: int = 3
    trial_results: List[TrialResult] = field(default_factory=list)


# Direction constants (0-7, clockwise from East)
# 0=E(→), 1=NE(↗), 2=N(↑), 3=NW(↖), 4=W(←), 5=SW(↙), 6=S(↓), 7=SE(↘)
# Integer grid movement - each FORWARD moves exactly 1 cell
DIR_VECTORS = [
    (1, 0),    # 0: East
    (1, 1),    # 1: Northeast
    (0, 1),    # 2: North
    (-1, 1),   # 3: Northwest
    (-1, 0),   # 4: West
    (-1, -1),  # 5: Southwest
    (0, -1),   # 6: South
    (1, -1),   # 7: Southeast
]

@dataclass
class AgentState:
    """Agent position and orientation on integer grid."""
    x: int = 0
    y: int = 0
    angle: int = 0  # Direction index 0-7 (0=East, 2=North, 4=West, 6=South)


@dataclass
class EnvironmentConfig:
    """Environment configuration from protocol."""
    name: str
    task_type: str
    
    # From verified protocol
    trials_to_criterion: int
    sessions_to_criterion: int
    trials_per_session: int
    
    # Timing
    max_trial_steps: int = 500
    trial_timeout_seconds: float = 60.0
    inter_trial_interval: float = 1.0
    
    # Success criteria
    success_criterion: str = "reach_goal"  # or "avoid_zone", "collect_reward", etc.
    success_threshold: float = 0.5  # distance to goal
    
    # From paper
    source_pmc: str = ""
    source_quote: str = ""
    
    # Environment specifics
    arena_size: float = 10.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """
    Base class for all VLM evaluation environments.
    
    Key principle: The environment controls everything internally.
    The agent only sees observations and takes actions.
    The environment counts trials, handles timeouts, teleports agent.
    """
    
    def __init__(self, config: EnvironmentConfig, view_mode: ViewMode = ViewMode.FPV_3D):
        self.config = config
        self.view_mode = view_mode
        
        # Session state
        self.session = SessionState(
            max_trials=config.trials_to_criterion * 2,  # Allow extra trials
            max_steps_per_trial=config.max_trial_steps
        )
        
        # Agent state
        self.agent = AgentState()
        
        # Current trial
        self._current_step = 0
        self._trial_start_time = 0.0
        self._trial_path_length = 0.0
        self._trial_reward = 0.0
        
        # Rendering
        self._renderer = None
        self._last_observation = None
        
        # Action mapping (can be overridden)
        self.action_names = {
            Action.FORWARD: "move forward",
            Action.TURN_LEFT: "turn left",
            Action.TURN_RIGHT: "turn right",
            Action.INTERACT: "interact",
            Action.STAY: "stay"
        }
        
        # Valid actions for this environment (subset)
        self.valid_actions = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT, Action.STAY]
        
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of observation array."""
        if self.view_mode in [ViewMode.FPV_3D, ViewMode.TOPDOWN_2D]:
            return (224, 224, 3)
        else:  # ASCII modes
            return (40, 80)  # rows, cols
    
    @property
    def action_space(self) -> List[Action]:
        """Available actions."""
        return self.valid_actions
    
    @property
    def is_done(self) -> bool:
        """Whether session is complete (criterion met or max trials reached)."""
        return (self.session.criterion_met or 
                self.session.current_trial >= self.session.max_trials)
    
    @property
    def trial_complete(self) -> bool:
        """Whether current trial is complete (success or timeout)."""
        if self._current_step >= self.session.max_steps_per_trial:
            return True
        if self._check_success():
            return True
        return False
    
    def start_new_trial(self):
        """Manually start a new trial (after trial_complete)."""
        if not self.is_done:
            self._start_trial()
    
    @property
    def trial_info(self) -> Dict[str, Any]:
        """Current trial information (for display, not for agent)."""
        return {
            "trial": self.session.current_trial,
            "step": self._current_step,
            "max_steps": self.session.max_steps_per_trial,
            "successes": len([t for t in self.session.trial_results if t.success]),
            "consecutive": self.session.consecutive_successes,
            "criterion_met": self.session.criterion_met
        }
    
    # ==================== Core Interface ====================
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new session.
        Returns initial observation.
        """
        # Reset session
        self.session = SessionState(
            max_trials=self.config.trials_to_criterion * 2,
            max_steps_per_trial=self.config.max_trial_steps
        )
        
        # Start first trial
        self._start_trial()
        
        return self._get_observation()
    
    def step(self, action: Union[Action, int, str]) -> Tuple[np.ndarray, float]:
        """
        Take action, return (next_observation, reward).
        
        This is the main interface for VLM agents.
        The environment internally handles trial management.
        """
        # Parse action
        if isinstance(action, int):
            action = Action(action)
        elif isinstance(action, str):
            action = self._parse_action_string(action)
        
        # Check if session is done
        if self.is_done:
            return self._get_observation(), 0.0
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update step count
        self._current_step += 1
        
        # Check trial end conditions
        trial_done, trial_success = self._check_trial_end()
        
        if trial_done:
            # Record trial result
            self._end_trial(trial_success)
            
            # Start next trial (if not done)
            if not self.is_done:
                self._start_trial()
        
        # Get observation
        observation = self._get_observation()
        
        return observation, reward
    
    # ==================== Trial Management (Internal) ====================
    
    def _start_trial(self):
        """Start a new trial. Teleports agent to start position."""
        self.session.current_trial += 1
        self._current_step = 0
        self._trial_start_time = time.time()
        self._trial_path_length = 0.0
        self._trial_reward = 0.0
        
        # Reset agent to start position (subclass defines this)
        self._reset_agent_position()
        
        # Setup trial (subclass can override)
        self._setup_trial()
    
    def _end_trial(self, success: bool):
        """End current trial and record result."""
        result = TrialResult(
            trial_number=self.session.current_trial,
            success=success,
            time_steps=self._current_step,
            timeout=self._current_step >= self.session.max_steps_per_trial,
            path_length=self._trial_path_length,
            reward_collected=self._trial_reward,
            extra_info=self._get_trial_extra_info()
        )
        self.session.trial_results.append(result)
        
        # Update consecutive successes
        if success:
            self.session.consecutive_successes += 1
            if self.session.consecutive_successes >= self.session.required_consecutive:
                self.session.criterion_met = True
                self.session.trials_to_criterion = self.session.current_trial
        else:
            self.session.consecutive_successes = 0
    
    def _check_trial_end(self) -> Tuple[bool, bool]:
        """
        Check if trial should end.
        Returns (trial_done, trial_success).
        """
        # Timeout
        if self._current_step >= self.session.max_steps_per_trial:
            return True, False
        
        # Check success condition (subclass implements)
        if self._check_success():
            return True, True
        
        # Check failure condition (subclass implements)
        if self._check_failure():
            return True, False
        
        return False, False
    
    # ==================== Abstract Methods (Subclass Must Implement) ====================
    
    @abstractmethod
    def _reset_agent_position(self):
        """Reset agent to trial start position."""
        pass
    
    @abstractmethod
    def _setup_trial(self):
        """Setup for new trial (e.g., randomize goal)."""
        pass
    
    @abstractmethod
    def _execute_action(self, action: Action) -> float:
        """Execute action and return immediate reward."""
        pass
    
    @abstractmethod
    def _check_success(self) -> bool:
        """Check if success condition is met."""
        pass
    
    @abstractmethod
    def _check_failure(self) -> bool:
        """Check if failure condition is met (besides timeout)."""
        pass
    
    @abstractmethod
    def _render_fpv(self) -> np.ndarray:
        """Render first-person 3D view."""
        pass
    
    @abstractmethod
    def _render_topdown(self) -> np.ndarray:
        """Render top-down 2D view."""
        pass
    
    @abstractmethod
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view (full map)."""
        pass
    
    @abstractmethod
    def _render_ascii_3d(self) -> str:
        """Render ASCII pseudo-3D FPV."""
        pass
    
    def _render_ascii_2d_fpv(self, view_width: int = 35, view_height: int = 23, 
                              view_distance: float = 5.0, fov_degrees: float = 120.0) -> str:
        """
        Render ASCII 2D FPV using the Template Method Pattern.
        
        This method defines the overall structure. Environments can customize
        behavior by overriding the hook methods:
        - _fpv_get_cell_content(ray_angle, distance) -> char
        - _fpv_get_wall_distance(ray_angle) -> float  
        - _fpv_get_visible_landmarks(fov_half) -> list of (screen_col, char)
        - _fpv_get_goal_marker() -> (row, col, char) or None
        - _fpv_get_max_view_distance() -> float
        
        For grid-based mazes, the default uses map rotation.
        For continuous/circular environments, override the hooks.
        """
        import numpy as np
        
        # Check if this environment uses continuous coordinates (has _fpv_get_wall_distance)
        if hasattr(self, '_fpv_get_wall_distance'):
            return self._render_ascii_2d_fpv_continuous(view_width, view_height, fov_degrees)
        else:
            return self._render_ascii_2d_fpv_grid(view_width, view_height, view_distance, fov_degrees)
    
    def _render_ascii_2d_fpv_continuous(self, view_width: int, view_height: int, 
                                         fov_degrees: float) -> str:
        """
        FPV rendering for continuous/circular environments.
        Uses raycasting to walls and world-coordinate landmark visibility.
        """
        import math
        import numpy as np
        
        # Initialize grid with fog
        grid = [['░' for _ in range(view_width)] for _ in range(view_height)]
        
        # Agent position in view
        agent_col = view_width // 2
        agent_row = view_height - 2
        
        # FOV parameters
        fov = np.radians(fov_degrees)
        half_fov = fov / 2
        
        # Get max view distance from environment
        max_dist = self._fpv_get_max_view_distance() if hasattr(self, '_fpv_get_max_view_distance') else 10.0
        
        # Track wall row for each column (for landmark placement)
        wall_row_per_col = {}
        
        # For each cell, determine content based on raycasting
        for row in range(view_height - 2):
            for col in range(view_width):
                dx = col - agent_col
                dy = agent_row - row
                
                if dx == 0 and dy == 0:
                    continue
                
                # Calculate angle from forward direction
                cell_angle = math.atan2(dx, dy)
                
                # Check FOV
                if abs(cell_angle) > half_fov:
                    continue
                
                # Get wall distance in this direction
                ray_world_angle = self.agent.angle + cell_angle
                wall_dist = self._fpv_get_wall_distance(ray_world_angle)
                
                # Map distance to row
                dist_ratio = min(1.0, wall_dist / max_dist)
                wall_row = int(agent_row - dist_ratio * (agent_row - 1))
                
                if col not in wall_row_per_col or wall_row > wall_row_per_col[col]:
                    wall_row_per_col[col] = wall_row
                
                # Get cell content
                if row < wall_row:
                    grid[row][col] = '░'  # Beyond wall
                elif row == wall_row:
                    grid[row][col] = '#'  # Wall
                else:
                    # Get floor content from environment
                    grid[row][col] = self._fpv_get_floor_char() if hasattr(self, '_fpv_get_floor_char') else ' '
        
        # Add landmarks
        if hasattr(self, '_fpv_get_visible_landmarks'):
            for screen_col, char in self._fpv_get_visible_landmarks(half_fov):
                if 0 <= screen_col < view_width and screen_col in wall_row_per_col:
                    wall_row = wall_row_per_col[screen_col]
                    if 0 <= wall_row < view_height - 2:
                        grid[wall_row][screen_col] = char
        
        # Add goal marker
        if hasattr(self, '_fpv_get_goal_marker'):
            goal_info = self._fpv_get_goal_marker(grid, agent_row, agent_col, view_width, view_height)
            if goal_info:
                row, col, char = goal_info
                if 0 <= row < view_height - 2 and 0 <= col < view_width:
                    grid[row][col] = char
        
        # Agent marker
        grid[agent_row][agent_col] = '↑'
        
        return '\n'.join([''.join(row) for row in grid])
    
    def _render_ascii_2d_fpv_grid(self, view_width: int = 35, view_height: int = 23, 
                                   view_distance: float = 5.0, fov_degrees: float = 120.0) -> str:
        """
        FPV rendering for grid-based environments.
        Uses map rotation with raycasting for visibility.
        """
        import numpy as np
        
        # Get full map and parse into grid
        full_map = self._render_ascii_2d()
        all_lines = full_map.split('\n')
        
        # Find dimensions and create grid
        if not all_lines or not all_lines[0]:
            return "No map available"
        
        # Filter out status/info lines
        status_indicators = [':', '%', 'Trial', 'Phase', 'Reward', 'Score', 'Step', 'Error', '|']
        
        lines = []
        for line in all_lines:
            is_status = any(ind in line for ind in status_indicators)
            if not is_status and len(line) > 0:
                lines.append(line)
        
        if not lines:
            lines = all_lines
        
        full_height = len(lines)
        full_width = max(len(line) for line in lines) if lines else 1
        
        # Pad lines to uniform width  
        padded = [line.ljust(full_width) for line in lines]
        
        # Define characters that block vision completely (solid walls)
        # Landmarks 1-4 are ON the wall but should be VISIBLE (not block rays)
        wall_chars_blocking = set('#│|─═╔╗╚╝╠╣╦╩╬┌┐└┘├┤┬┴┼█▓')
        # All wall chars including landmarks (for determining what IS a wall)
        wall_chars = set('#│|─═╔╗╚╝╠╣╦╩╬┌┐└┘├┤┬┴┼█▓1234')
        # Holes are visible but don't block seeing landmarks/walls behind them
        # (holes are at floor level, landmarks are on walls above)
        
        # Find agent in full map
        agent_markers = set('^v<>↑↓←→↖↗↙↘@')
        agent_row, agent_col = None, None
        for r, line in enumerate(padded):
            for c, ch in enumerate(line):
                if ch in agent_markers:
                    agent_row, agent_col = r, c
                    break
            if agent_row is not None:
                break
        
        # If no agent found, place at center
        if agent_row is None:
            agent_row = full_height // 2
            agent_col = full_width // 2
        
        # Determine what character represents floor in this map
        # Check neighbors of agent position - floor is what's around the agent
        floor_chars = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = agent_row + dr, agent_col + dc
                if 0 <= r < full_height and 0 <= c < full_width:
                    ch = padded[r][c]
                    if ch not in wall_chars and ch not in agent_markers:
                        floor_chars.add(ch)
        # If space is used as floor near agent, don't treat it as void
        space_is_floor = ' ' in floor_chars
        
        # Get agent's facing angle
        # Handle both integer direction index (0-7) and radians
        if hasattr(self, 'agent') and hasattr(self.agent, 'angle'):
            angle_val = self.agent.angle
            # Check if it's an integer direction (0-7) or radians
            if isinstance(angle_val, (int, np.integer)) or (isinstance(angle_val, float) and angle_val == int(angle_val) and 0 <= angle_val <= 7):
                # Integer direction: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
                # Convert to radians: direction * (π/4)
                agent_angle = int(angle_val) * (np.pi / 4)
            else:
                # Already in radians
                agent_angle = angle_val
        else:
            agent_angle = np.pi / 2  # Default North
        
        # Rotation angle to make agent's forward point UP on screen
        # Screen UP corresponds to angle π/2 in math coords
        # So we rotate by (agent_angle - π/2)
        rotation_angle = agent_angle - np.pi / 2
        cos_rot = np.cos(rotation_angle)
        sin_rot = np.sin(rotation_angle)
        
        # For diagonal rotations, we need to sample more densely
        # because a 45° rotation maps grid cells to non-grid positions
        # Check if angle is close to a diagonal (45°, 135°, 225°, 315°)
        angle_mod = agent_angle % (np.pi / 2)
        is_diagonal = np.pi / 8 < angle_mod < 3 * np.pi / 8
        
        # FOV half-angle
        fov_half = np.radians(fov_degrees / 2)
        
        # Agent position in view coordinates
        # Place agent at BOTTOM center so entire view shows what's ahead
        half_w = view_width // 2
        agent_view_y = view_height - 2  # Near bottom
        
        # Create output grid filled with fog
        output = [['░' for _ in range(view_width)] for _ in range(view_height)]
        
        def view_to_map(vx: float, vy: float) -> tuple:
            """Convert view coordinates to map coordinates with rotation.
            
            View coords: (0,0) top-left, x right, y down, agent at bottom center
            Map coords: (row, col), row down, col right
            """
            # Offset from agent position
            dx = vx - half_w
            dy = vy - agent_view_y
            
            # In view space: +Y is down (toward agent's back), -Y is up (forward)
            # In map space after rotation: need to transform
            # Rotate by -rotation_angle (inverse) to go from view to map
            map_dx = dx * cos_rot + dy * sin_rot
            map_dy = -dx * sin_rot + dy * cos_rot
            
            # Map position (float for proper sampling)
            map_col = agent_col + map_dx
            map_row = agent_row + map_dy
            return map_row, map_col
        
        def get_char_at_map(row: float, col: float) -> str:
            """Get character at map position, preferring important chars over walls.
            
            When sampling rotated coordinates, we check a small neighborhood
            and prefer meaningful characters (holes O, goals G/P/*, landmarks 1234 ABCD)
            over structural characters (walls #) to avoid losing important info.
            """
            r = int(round(row))
            c = int(round(col))
            
            if not (0 <= r < full_height and 0 <= c < full_width):
                return ' '
            
            ch = padded[r][c]
            return ch
        
        def get_display_char_at_map(row: float, col: float) -> str:
            """Get character for DISPLAY at map position, preferring important chars.
            
            This is used for rendering only, NOT for wall detection.
            When sampling rotated coordinates, we check a small neighborhood
            and prefer meaningful characters (holes O, goals G/P/*, landmarks 1234 ABCD)
            over structural characters (walls #) to avoid losing important info.
            """
            r = int(round(row))
            c = int(round(col))
            
            if not (0 <= r < full_height and 0 <= c < full_width):
                return ' '
            
            ch = padded[r][c]
            
            # Landmark letters/numbers are important - return them directly
            if ch in '1234ABCD':
                return ch
            
            # If we got an important character, return it
            important_chars = set('OoGP*[]=-~EABCDabcd')
            if ch in important_chars or ch not in wall_chars:
                return ch
            
            # We got a wall - check immediate neighbors for important chars
            # Use radius of 0.8 to find landmarks at diagonally rotated positions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < full_height and 0 <= nc < full_width:
                    neighbor = padded[nr][nc]
                    # Check for landmarks and other important chars
                    if neighbor in '1234ABCD' or neighbor in important_chars:
                        # Accept if reasonably close (handles rotation sampling)
                        if abs(row - nr) < 0.9 and abs(col - nc) < 0.9:
                            return neighbor
            
            return ch
        
        def is_in_fov(vx: int, vy: int) -> bool:
            """Check if view position is within field of view (FOV centered on UP)."""
            dx = vx - half_w
            dy = vy - agent_view_y
            
            if dx == 0 and dy == 0:
                return True
            
            # Angle from agent position, with UP being π/2
            target_angle = np.arctan2(-dy, dx)  # -dy because UP is forward
            diff = target_angle - np.pi / 2
            
            # Normalize to [-π, π]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            
            return abs(diff) <= fov_half
        
        # Create visibility map using ray casting
        visible = [[False for _ in range(view_width)] for _ in range(view_height)]
        
        def is_blocking(vx: int, vy: int) -> bool:
            """Check if a view cell blocks vision."""
            if not (0 <= vx < view_width and 0 <= vy < view_height):
                return True  # Out of bounds blocks
            map_row, map_col = view_to_map(vx, vy)
            ch = get_char_at_map(map_row, map_col)
            if ch in wall_chars_blocking:
                return True
            if ch == ' ' and not space_is_floor:
                return True
            return False
        
        def cast_ray_dda(target_vx: int, target_vy: int):
            """Cast ray using DDA with diagonal blocking checks."""
            x0, y0 = float(half_w), float(agent_view_y)
            x1, y1 = float(target_vx), float(target_vy)
            
            dx = x1 - x0
            dy = y1 - y0
            
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                return
            
            # Use more steps to ensure we don't skip cells
            length = max(abs(dx), abs(dy))
            steps = int(length * 2) + 1  # Oversample
            
            x_inc = dx / steps
            y_inc = dy / steps
            
            x, y = x0, y0
            prev_ix, prev_iy = int(round(x)), int(round(y))
            
            for _ in range(steps + 1):
                ix, iy = int(round(x)), int(round(y))
                
                if 0 <= ix < view_width and 0 <= iy < view_height:
                    # Check for diagonal movement through wall corners
                    # If we moved diagonally, check if both adjacent cells are walls
                    if ix != prev_ix and iy != prev_iy:
                        # Diagonal move - check the two cells we're cutting between
                        cell1_blocking = is_blocking(prev_ix, iy)
                        cell2_blocking = is_blocking(ix, prev_iy)
                        if cell1_blocking and cell2_blocking:
                            # Can't see through diagonal wall corner
                            return
                    
                    # Check what's at this position BEFORE marking visible
                    if ix != half_w or iy != agent_view_y:
                        map_row, map_col = view_to_map(ix, iy)
                        ch = get_char_at_map(map_row, map_col)
                        
                        # Stop at walls
                        if ch in wall_chars_blocking:
                            visible[iy][ix] = True  # Mark wall as visible
                            return  # Stop ray
                        
                        # Stop at void (outside arena) - but only if space isn't floor
                        if ch == ' ' and not space_is_floor:
                            return  # Stop ray, don't mark as visible
                    
                    # Mark cell as visible
                    visible[iy][ix] = True
                    prev_ix, prev_iy = ix, iy
                
                x += x_inc
                y += y_inc
        
        def is_in_fov(target_x: int, target_y: int) -> bool:
            """Check if target is within field of view."""
            dx = target_x - half_w
            dy = agent_view_y - target_y  # Negative because UP is forward (decreasing y)
            
            if dx == 0 and dy == 0:
                return True
            
            # Angle from forward (up) direction
            angle = np.arctan2(abs(dx), dy)  # angle from forward
            return angle <= fov_half
        
        # Mark agent position as visible
        visible[agent_view_y][half_w] = True
        
        # Cast rays to edge cells WITHIN the FOV
        # Top edge - always in FOV
        for x in range(view_width):
            if is_in_fov(x, 0):
                cast_ray_dda(x, 0)
        # Left edge - only cells within FOV
        for y in range(view_height):
            if is_in_fov(0, y):
                cast_ray_dda(0, y)
        # Right edge - only cells within FOV
        for y in range(view_height):
            if is_in_fov(view_width - 1, y):
                cast_ray_dda(view_width - 1, y)
        
        # Fill output grid from visibility map
        for vy in range(view_height):
            for vx in range(view_width):
                if visible[vy][vx]:
                    map_row, map_col = view_to_map(vx, vy)
                    # Use display char for rendering (prefers important chars)
                    ch = get_display_char_at_map(map_row, map_col)
                    # Don't copy the agent marker from the source - we'll place our own
                    if ch not in agent_markers:
                        # Handle space character
                        if ch == ' ':
                            if space_is_floor:
                                # Space is floor in this map - render as floor
                                output[vy][vx] = '.'
                            # else: leave as fog (void outside arena)
                        else:
                            output[vy][vx] = ch
                    else:
                        output[vy][vx] = '.'  # Clear agent's old position, show floor
        
        # Place agent marker at bottom center - always pointing UP since view is rotated
        output[agent_view_y][half_w] = '↑'
        
        return '\n'.join(''.join(row) for row in output)
    
    # ==================== Rendering ====================
    
    def _get_observation(self) -> Union[np.ndarray, str]:
        """Get observation in current view mode."""
        if self.view_mode == ViewMode.FPV_3D:
            return self._render_fpv()
        elif self.view_mode == ViewMode.TOPDOWN_2D:
            return self._render_topdown()
        elif self.view_mode == ViewMode.ASCII_2D:
            return self._render_ascii_2d()
        elif self.view_mode == ViewMode.ASCII_3D:
            return self._render_ascii_3d()
        elif self.view_mode == ViewMode.ASCII_2D_FPV:
            return self._render_ascii_2d_fpv()
        else:
            return self._render_fpv()
    
    def render(self, mode: Optional[ViewMode] = None) -> Union[np.ndarray, str]:
        """Render environment (for display/debugging)."""
        old_mode = self.view_mode
        if mode:
            self.view_mode = mode
        obs = self._get_observation()
        self.view_mode = old_mode
        return obs
    
    # ==================== Utility ====================
    
    def _parse_action_string(self, action_str: str) -> Action:
        """Parse action from string (for VLM text output)."""
        action_str = action_str.lower().strip()
        
        mappings = {
            "forward": Action.FORWARD,
            "move forward": Action.FORWARD,
            "go forward": Action.FORWARD,
            "ahead": Action.FORWARD,
            "left": Action.TURN_LEFT,
            "turn left": Action.TURN_LEFT,
            "rotate left": Action.TURN_LEFT,
            "right": Action.TURN_RIGHT,
            "turn right": Action.TURN_RIGHT,
            "rotate right": Action.TURN_RIGHT,
            "interact": Action.INTERACT,
            "press": Action.INTERACT,
            "enter": Action.INTERACT,
            "stay": Action.STAY,
            "wait": Action.STAY,
            "stop": Action.STAY
        }
        
        for key, value in mappings.items():
            if key in action_str:
                return value
        
        # Default to stay if can't parse
        return Action.STAY
    
    def _get_trial_extra_info(self) -> Dict[str, Any]:
        """Get extra info for trial result (subclass can override)."""
        return {}
    
    def parse_action(self, action_str: str) -> Action:
        """Public interface to parse action from string (for VLM text output)."""
        return self._parse_action_string(action_str)
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state information (for debugging/display)."""
        return {
            'current_trial': self.session.current_trial,
            'trial_step': self._current_step,
            'max_steps': self.session.max_steps_per_trial,
            'total_trials': self.session.max_trials,
            'trials_completed': len(self.session.trial_results),
            'trials_successful': sum(1 for t in self.session.trial_results if t.success),
            'trial_complete': self.trial_complete,
            'criterion_met': self.session.criterion_met,
            'path_length': self._trial_path_length,
            'trial_reward': self._trial_reward,
        }
    
    def get_observation(self) -> Union[np.ndarray, str]:
        """Public interface to get current observation."""
        return self._get_observation()
    
    def get_action_prompt(self) -> str:
        """Get prompt describing available actions (for VLM)."""
        actions = [f"- {self.action_names[a]}" for a in self.valid_actions]
        return "Available actions:\n" + "\n".join(actions)
    
    def get_task_description(self) -> str:
        """Get task description (for VLM context)."""
        return f"""Task: {self.config.name}
Type: {self.config.task_type}
Objective: {self.config.success_criterion}
Source: {self.config.source_pmc}"""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of completed session."""
        successes = [t for t in self.session.trial_results if t.success]
        return {
            "task": self.config.name,
            "total_trials": self.session.current_trial,
            "successful_trials": len(successes),
            "criterion_met": self.session.criterion_met,
            "trials_to_criterion": self.session.trials_to_criterion if self.session.criterion_met else None,
            "expected_trials": self.config.trials_to_criterion,
            "average_steps": np.mean([t.time_steps for t in self.session.trial_results]) if self.session.trial_results else 0,
            "average_path_length": np.mean([t.path_length for t in self.session.trial_results]) if self.session.trial_results else 0
        }


# ==================== Navigation Environment Base ====================

class NavigationEnvironment(BaseEnvironment):
    """
    Base class for navigation-based environments.
    Handles movement, collision detection, goal reaching.
    """
    
    def __init__(self, config: EnvironmentConfig, view_mode: ViewMode = ViewMode.FPV_3D):
        super().__init__(config, view_mode)
        
        # Arena
        self.arena_size = config.arena_size
        self.walls = []  # List of wall segments
        
        # Goal (integer grid coordinates)
        self.goal_x = 0
        self.goal_y = 0
        self.goal_radius = 1  # Must be on exact goal cell
        self.goal_visible = True
        
        # Landmarks (for spatial cues)
        self.landmarks = []
        
    def _execute_action(self, action: Action) -> float:
        """Execute movement action. Integer grid movement - always exactly 1 cell."""
        old_x, old_y = self.agent.x, self.agent.y
        
        if action == Action.FORWARD:
            dx, dy = DIR_VECTORS[self.agent.angle]
            new_x = self.agent.x + dx
            new_y = self.agent.y + dy
            
            # Check collision at destination
            if self._check_collision_at(new_x, new_y):
                return -0.1  # Hit wall
            
            # For diagonal moves, also check that we don't clip through corners
            # Can't move diagonally if EITHER adjacent cell is a wall
            if dx != 0 and dy != 0:  # Diagonal move
                side1_blocked = self._check_collision_at(self.agent.x + dx, self.agent.y)
                side2_blocked = self._check_collision_at(self.agent.x, self.agent.y + dy)
                if side1_blocked or side2_blocked:
                    # Can't cut through corner - must go around
                    return -0.1
            
            self.agent.x, self.agent.y = new_x, new_y
                        
        elif action == Action.TURN_LEFT:
            self.agent.angle = (self.agent.angle + 1) % 8
        elif action == Action.TURN_RIGHT:
            self.agent.angle = (self.agent.angle - 1) % 8
        
        # Update path length (Euclidean distance for diagonals)
        dx_moved = abs(self.agent.x - old_x)
        dy_moved = abs(self.agent.y - old_y)
        if dx_moved > 0 and dy_moved > 0:
            # Diagonal move: sqrt(2) ≈ 1.414
            moved = 1.414
        else:
            moved = dx_moved + dy_moved
        self._trial_path_length += moved
        
        # Check if reached goal (exact cell match)
        if self.agent.x == self.goal_x and self.agent.y == self.goal_y:
            self._trial_reward += 1.0
            return 1.0  # Reward for reaching goal
        
        return -0.01  # Small time penalty
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if position would collide. Override in subclass."""
        return False
    
    def _check_collision(self) -> bool:
        """Check if current position collides."""
        return self._check_collision_at(self.agent.x, self.agent.y)
    
    def _check_success(self) -> bool:
        """Success = reached goal cell."""
        return self.agent.x == self.goal_x and self.agent.y == self.goal_y
    
    def _check_failure(self) -> bool:
        """No automatic failure besides timeout."""
        return False
    
    def _distance_to_goal(self) -> float:
        """Get current distance to goal."""
        return np.sqrt((self.agent.x - self.goal_x)**2 + (self.agent.y - self.goal_y)**2)
