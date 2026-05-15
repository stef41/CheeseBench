"""
Base environment class for LLM evaluation.

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
    FPV_3D = auto()         # First-person 3D view (default for image-based models)
    TOPDOWN_2D = auto()     # Top-down 2D view
    ASCII_2D = auto()       # ASCII top-down (full map)
    ASCII_3D = auto()       # ASCII pseudo-3D FPV
    ASCII_2D_FPV = auto()   # ASCII top-down cropped around agent (partial map)
    FRONT_BLOCK = auto()    # Shows only the first block directly in front of the agent


class Action(Enum):
    """Standard discrete actions."""
    FORWARD = 0
    ROTATE_LEFT = 2
    ROTATE_RIGHT = 3
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


DIRECTION_ARROWS = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']


class AsciiCanvas:
    """Lightweight character grid that renders to a string.
    
    Centralizes the repeated pattern of:
      1. Create 2D char grid
      2. Place walls, floor, markers
      3. Place agent arrow
      4. Convert to string
    """
    
    def __init__(self, width: int, height: int, fill: str = ' '):
        self.width = width
        self.height = height
        self.grid = [[fill] * width for _ in range(height)]
    
    def put(self, x: int, y: int, ch: str):
        """Place a single character at (x, y) with bounds checking."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = ch
    
    def get(self, x: int, y: int) -> str:
        """Get char at (x, y), or ' ' if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return ' '
    
    def fill_rect(self, x1: int, y1: int, x2: int, y2: int, ch: str):
        """Fill a rectangle (inclusive) with a character."""
        for y in range(max(0, y1), min(self.height, y2 + 1)):
            for x in range(max(0, x1), min(self.width, x2 + 1)):
                self.grid[y][x] = ch
    
    def hline(self, x1: int, x2: int, y: int, ch: str):
        """Draw a horizontal line."""
        if 0 <= y < self.height:
            for x in range(max(0, x1), min(self.width, x2 + 1)):
                self.grid[y][x] = ch
    
    def vline(self, x: int, y1: int, y2: int, ch: str):
        """Draw a vertical line."""
        if 0 <= x < self.width:
            for y in range(max(0, y1), min(self.height, y2 + 1)):
                self.grid[y][x] = ch
    
    def blit(self, x: int, y: int, text: str):
        """Place a multi-char string horizontally starting at (x, y)."""
        for i, c in enumerate(text):
            self.put(x + i, y, c)
    
    def put_agent(self, x: int, y: int, angle: int):
        """Place an agent direction arrow at (x, y) using integer direction 0-7."""
        if 0 <= angle < len(DIRECTION_ARROWS):
            self.put(x, y, DIRECTION_ARROWS[angle])
    
    def to_string(self) -> str:
        """Convert grid to newline-separated string."""
        return '\n'.join(''.join(row) for row in self.grid)


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
    Base class for all LLM evaluation environments.
    
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
        self._last_observation = None
        
        # Action mapping (can be overridden)
        self.action_names = {
            Action.FORWARD: "move forward",
            Action.ROTATE_LEFT: "rotate left",
            Action.ROTATE_RIGHT: "rotate right",
            Action.STAY: "stay"
        }
        
        # Valid actions for this environment (subset)
        self.valid_actions = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT, Action.STAY]
        
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
        
        This is the main interface for model agents.
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
    
    # ==================== Optional Override Methods ====================
    
    def _get_char_under_agent(self) -> str:
        """
        Return the character that should appear at the agent's position.
        
        Override in subclasses where the agent can stand on meaningful content
        (e.g., landmarks in RadialArmMaze). Default returns '.' (floor).
        
        This is used by FPV rendering to show what's under the agent when
        the 2D map would show an agent marker.
        """
        return '.'
    
    # ==================== Shared 3D Rendering Helpers ====================
    
    @staticmethod
    def wall_char(dist: float) -> str:
        """Get wall character based on distance (shared shading scheme)."""
        if dist < 1.5: return '█'
        if dist < 3.0: return '▓'
        if dist < 6.0: return '▒'
        if dist < 10.0: return '░'
        return '·'
    
    @staticmethod
    def ceiling_char(depth: int, view_height: int) -> str:
        """Get ceiling character based on depth from horizon."""
        if depth > view_height // 3:
            return ' '  # Far ceiling (dark)
        elif depth > view_height // 6:
            return '░'  # Mid ceiling
        return '▒'  # Near ceiling
    
    @staticmethod
    def floor_char(depth: int, view_height: int) -> str:
        """Get floor character based on depth from horizon."""
        if depth > view_height // 3:
            return '░'  # Far floor
        elif depth > view_height // 6:
            return '▒'  # Mid floor
        return '▓'  # Near floor
    
    def _render_3d_raycasting(self, width: int = 60, height: int = 28,
                               fov: float = None, agent_angle: float = None,
                               cast_ray_func=None, 
                               overlay_func=None,
                               ceiling_char_override: str = None,
                               floor_char_override: str = None) -> str:
        """
        Shared raycasting-based 3D renderer.
        
        Args:
            width: View width in chars
            height: View height in chars  
            fov: Field of view in radians (default π/2 = 90°)
            agent_angle: Agent facing angle in radians
            cast_ray_func: Function(ray_angle) -> distance to wall
            overlay_func: Optional function(row, col, char, dist, wall_top, wall_bottom) -> char
                          Called for each cell to allow custom overlays
            ceiling_char_override: Override default ceiling character
            floor_char_override: Override default floor character
            
        Returns:
            ASCII string with frame border
        """
        import numpy as np
        
        if fov is None:
            fov = np.pi / 2  # 90 degrees
        if agent_angle is None:
            agent_angle = self.agent.angle * (np.pi / 4) if isinstance(self.agent.angle, int) else self.agent.angle
        if cast_ray_func is None:
            cast_ray_func = getattr(self, '_cast_ray', lambda a: 10.0)
        
        lines = []
        view_width = width - 2
        view_height = height - 2
        horizon = view_height // 2
        
        lines.append("╔" + "═" * view_width + "╗")
        
        # Cast rays for each column
        column_data = []
        for col in range(view_width):
            ray_offset = (col / max(1, view_width - 1)) - 0.5
            ray_angle = agent_angle - ray_offset * fov
            
            raw_dist = cast_ray_func(ray_angle)
            corrected_dist = raw_dist * np.cos(ray_offset * fov)  # Fish-eye correction
            
            if corrected_dist < 0.5:
                wall_height = view_height
            else:
                wall_height = min(view_height, int(view_height * 1.5 / (corrected_dist + 0.5)))
            
            wall_top = max(0, horizon - wall_height // 2)
            wall_bottom = min(view_height, horizon + wall_height // 2)
            
            column_data.append((wall_top, wall_bottom, corrected_dist, raw_dist))
        
        # Render rows
        for row in range(view_height):
            row_chars = []
            for col, (wall_top, wall_bottom, dist, raw_dist) in enumerate(column_data):
                if row < wall_top:
                    ceiling_depth = wall_top - row
                    if ceiling_char_override:
                        char = ceiling_char_override
                    else:
                        char = self.ceiling_char(ceiling_depth, view_height)
                elif row >= wall_bottom:
                    floor_depth = row - wall_bottom
                    if floor_char_override:
                        char = floor_char_override
                    else:
                        char = self.floor_char(floor_depth, view_height)
                else:
                    char = self.wall_char(dist)
                
                # Allow overlay customization (raw_dist for depth comparison)
                if overlay_func:
                    char = overlay_func(row, col, char, raw_dist, wall_top, wall_bottom)
                
                row_chars.append(char)
            
            lines.append("║" + ''.join(row_chars) + "║")
        
        lines.append("╚" + "═" * view_width + "╝")
        
        return '\n'.join(lines)

    def _cast_ray(self, angle: float, max_dist: float = 20.0, step_size: float = 0.2) -> float:
        """
        Shared raycasting method for 3D rendering.
        
        Cast a ray from agent position at given angle and return distance to wall.
        Uses _check_collision_at() for collision detection.
        
        Args:
            angle: Ray angle in radians
            max_dist: Maximum ray distance
            step_size: Step size for ray marching
            
        Returns:
            Distance to first collision (or max_dist if none)
        """
        import numpy as np
        
        dx = np.cos(angle)
        dy = np.sin(angle)
        x, y = float(self.agent.x), float(self.agent.y)
        
        steps = int(max_dist / step_size)
        for step in range(steps):
            t = step * step_size
            rx = x + dx * t
            ry = y + dy * t
            
            ix, iy = int(round(rx)), int(round(ry))
            
            if self._check_collision_at(ix, iy):
                return max(0.1, t)
        
        return max_dist

    # ==================== Shared 2D Topdown Rendering Helpers ====================

    @staticmethod
    def _draw_disk(img: np.ndarray, cx: int, cy: int, radius: int, 
                   color: tuple, img_size: int = 224):
        """Draw a filled circle (disk) on an image at (cx, cy)."""
        r2 = radius * radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= r2:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < img_size and 0 <= py < img_size:
                        img[py, px] = color

    @staticmethod
    def _draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                   color: tuple, thickness: int = 1, img_size: int = 224):
        """Draw a line on the image using Bresenham-style stepping."""
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            half = thickness // 2
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < img_size and 0 <= py < img_size:
                        img[py, px] = color

    @staticmethod
    def _draw_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                   color: tuple, filled: bool = True):
        """Draw a rectangle on the image."""
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = max(0, x1), min(img.shape[1], x2)
        y1, y2 = max(0, y1), min(img.shape[0], y2)
        if filled:
            img[y1:y2, x1:x2] = color
        else:
            img[y1:y1+1, x1:x2] = color  # Top
            img[y2-1:y2, x1:x2] = color  # Bottom
            img[y1:y2, x1:x1+1] = color  # Left
            img[y1:y2, x2-1:x2] = color  # Right

    # ==================== Shared Movement Helpers ====================

    def _move_continuous(self, action: Action, speed: float = 0.2, 
                         turn_rate: float = np.pi / 4,
                         x_bounds: tuple = None, y_bounds: tuple = None) -> bool:
        """
        Execute continuous (float-based) movement.
        
        Args:
            action: The action to execute
            speed: Forward movement speed per step
            turn_rate: Rotation amount per turn (radians)
            x_bounds: Optional (min, max) tuple for x clamping
            y_bounds: Optional (min, max) tuple for y clamping
            
        Returns:
            True if movement occurred, False otherwise
        """
        moved = False
        
        if action == Action.FORWARD:
            self.agent.x += speed * np.cos(self.agent.angle)
            self.agent.y += speed * np.sin(self.agent.angle)
            moved = True
            
            # Apply bounds if specified
            if x_bounds is not None:
                self.agent.x = np.clip(self.agent.x, x_bounds[0], x_bounds[1])
            if y_bounds is not None:
                self.agent.y = np.clip(self.agent.y, y_bounds[0], y_bounds[1])
                
        elif action == Action.ROTATE_LEFT:
            self.agent.angle += turn_rate
            
        elif action == Action.ROTATE_RIGHT:
            self.agent.angle -= turn_rate
        
        # Normalize angle to [0, 2π)
        self.agent.angle = self.agent.angle % (2 * np.pi)
        
        return moved

    def _get_chamber(self, threshold: float = 0.0) -> int:
        """
        Get current chamber based on x position.
        
        Args:
            threshold: X value that separates chambers (default 0)
            
        Returns:
            0 for left chamber (x < threshold), 1 for right chamber
        """
        return 0 if self.agent.x < threshold else 1

    def _distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance from agent to a point."""
        return np.sqrt((self.agent.x - x)**2 + (self.agent.y - y)**2)

    # ==================== Shared FPV Rendering Helpers ====================

    @staticmethod
    def _draw_shape(img: np.ndarray, shape: str, cx: int, cy: int, size: int, 
                    color: tuple, y_min: int = 0, y_max: int = 224):
        """
        Draw a shape at position for FPV rendering.
        
        Args:
            img: Image array to draw on
            shape: 'circle', 'square', 'triangle', 'diamond', 'star'
            cx, cy: Center position
            size: Shape size in pixels
            color: RGB color tuple
            y_min, y_max: Y bounds for clipping
        """
        if shape == 'circle':
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    if dx * dx + dy * dy <= size * size:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < 224 and y_min <= py < y_max:
                            img[py, px] = color
        elif shape == 'square':
            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 224 and y_min <= py < y_max:
                        img[py, px] = color
        elif shape == 'triangle':
            for dy in range(-size, size + 1):
                width = int(size * (1 - abs(dy) / size)) if size > 0 else 0
                for dx in range(-width, width + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 224 and y_min <= py < y_max:
                        img[py, px] = color
        elif shape == 'diamond':
            for dy in range(-size, size + 1):
                width = size - abs(dy)
                for dx in range(-width, width + 1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 224 and y_min <= py < y_max:
                        img[py, px] = color
        elif shape == 'star':
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    if abs(dx) <= 2 or abs(dy) <= 2 or abs(abs(dx) - abs(dy)) <= 2:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < 224 and y_min <= py < y_max:
                            img[py, px] = color

    def _render_fpv_raycasting(self, img_size: int = 224,
                                fov: float = None,
                                num_rays: int = 224,
                                ceiling_color: tuple = (150, 150, 150),
                                floor_color: tuple = None,
                                wall_color_func=None,
                                max_dist: float = 15.0,
                                horizon: int = 112,
                                overlay_func=None) -> np.ndarray:
        """
        Shared raycasting-based FPV renderer for maze environments.
        
        Args:
            img_size: Output image size (default 224)
            fov: Field of view in radians (default π/2)
            num_rays: Number of rays to cast (default 224)
            ceiling_color: RGB tuple for ceiling
            floor_color: RGB tuple for floor (uses self.floor_color if None)
            wall_color_func: Optional function(distance) -> RGB for wall shading
            max_dist: Maximum ray distance for wall rendering
            horizon: Y position of horizon line
            overlay_func: Optional function(img, agent_angle, fov) for goals/landmarks
            
        Returns:
            numpy array of shape (img_size, img_size, 3)
        """
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        if fov is None:
            fov = np.pi / 2
        if floor_color is None:
            floor_color = getattr(self, 'floor_color', (180, 180, 180))
        
        # Draw ceiling and floor
        img[:horizon - 42, :] = ceiling_color
        img[horizon + 42:, :] = floor_color
        
        # Get agent angle
        if isinstance(self.agent.angle, int):
            agent_angle = self.agent.angle * (np.pi / 4)
        else:
            agent_angle = self.agent.angle
        
        # Cast rays for walls
        for i in range(num_rays):
            ray_angle = agent_angle - fov / 2 + (i / num_rays) * fov
            dist = self._cast_ray(ray_angle)
            
            if dist < max_dist:
                wall_height = min(84, int(150 / (dist + 0.5)))
                y_start = horizon - wall_height
                y_end = horizon + wall_height
                
                if wall_color_func:
                    wall_color = wall_color_func(dist)
                else:
                    # Default wall shading
                    shade = max(50, 255 - int(dist * 20))
                    wall_color = (int(shade * 0.4), int(shade * 0.3), int(shade * 0.25))
                
                img[y_start:y_end, i] = wall_color
        
        # Allow custom overlays (goals, landmarks)
        if overlay_func:
            overlay_func(img, agent_angle, fov)
        
        return img

    def _render_goal_in_fpv(self, img: np.ndarray, goal_x: float, goal_y: float,
                             goal_color: tuple, fov: float = None,
                             horizon: int = 130, y_min: int = 70, y_max: int = 154):
        """
        Render a goal marker in FPV based on relative position.
        Only renders if goal is visible (not blocked by walls).
        
        Args:
            img: Image array to draw on
            goal_x, goal_y: Goal world coordinates
            goal_color: RGB color for goal
            fov: Field of view in radians (default π/2)
            horizon: Y position to draw goal at
            y_min, y_max: Y bounds for clipping
        """
        if fov is None:
            fov = np.pi / 2
            
        dx = goal_x - self.agent.x
        dy = goal_y - self.agent.y
        dist_to_goal = np.sqrt(dx * dx + dy * dy)
        
        # Get agent angle
        if isinstance(self.agent.angle, int):
            agent_angle = self.agent.angle * (np.pi / 4)
        else:
            agent_angle = self.agent.angle
            
        angle_to_goal = np.arctan2(dy, dx) - agent_angle
        
        # Normalize to [-π, π]
        while angle_to_goal > np.pi:
            angle_to_goal -= 2 * np.pi
        while angle_to_goal < -np.pi:
            angle_to_goal += 2 * np.pi
        
        if abs(angle_to_goal) < fov / 2:
            # Check line-of-sight: cast ray toward goal and see if wall is closer
            ray_angle = np.arctan2(dy, dx)  # Absolute angle to goal
            wall_dist = self._cast_ray(ray_angle)  # Use default max_dist
            
            # Only render goal if no wall is blocking it
            if wall_dist >= dist_to_goal - 0.5:
                # Screen position: positive angle (left) -> smaller x (left on screen)
                screen_x = int(112 - angle_to_goal / (fov / 2) * 100)
                size = max(3, int(30 / (dist_to_goal + 1)))
                
                self._draw_shape(img, 'circle', screen_x, horizon, size, goal_color, y_min, y_max)

    def _render_ascii_2d_fpv(self, view_width: int = 35, view_height: int = 23, 
                              view_distance: float = 5.0, fov_degrees: float = 120.0) -> str:
        """
        Render ASCII 2D FPV for grid-based environments.
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
        
        # Adapt viewport to maze size to avoid excessive fog on small maps
        view_width = min(view_width, full_width + 4)
        view_height = min(view_height, full_height + 4)
        # Ensure odd widths for centering
        if view_width % 2 == 0:
            view_width += 1
        
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
        # Determine if spaces inside the arena are floor (not void outside boundary)
        floor_chars = set()
        
        # First check: if the agent is surrounded by mostly spaces AND walls,
        # then space IS the floor character (e.g., OperantChamber)
        space_neighbors = 0
        wall_neighbors = 0
        total_neighbors = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip agent position
                r, c = agent_row + dr, agent_col + dc
                if 0 <= r < full_height and 0 <= c < full_width:
                    total_neighbors += 1
                    ch = padded[r][c]
                    if ch == ' ':
                        space_neighbors += 1
                    elif ch in wall_chars:
                        wall_neighbors += 1
                    elif ch not in agent_markers:
                        floor_chars.add(ch)
        
        # If agent is surrounded by spaces and walls (no other floor chars found),
        # then space IS the floor
        if space_neighbors > 0 and len(floor_chars) == 0:
            # Agent is in a room where space = floor
            space_is_floor = True
        else:
            # Check for explicit floor characters or mixed floor
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = agent_row + dr, agent_col + dc
                    if 0 <= r < full_height and 0 <= c < full_width:
                        ch = padded[r][c]
                        if ch not in wall_chars and ch not in agent_markers:
                            if ch != ' ':
                                floor_chars.add(ch)
            space_is_floor = ' ' in floor_chars or (space_neighbors > 0 and len(floor_chars) == 0)
        
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
        
        # Scale factor: view cells per map cell
        # Keep scale at 1.0 for all angles - diagonal corridors naturally
        # appear longer because you're viewing them at an angle
        scale = 1.0
        
        # Create output grid filled with fog
        output = [['░' for _ in range(view_width)] for _ in range(view_height)]
        
        def view_to_map(vx: float, vy: float) -> tuple:
            """Convert view coordinates to map coordinates with rotation and scaling.
            
            View coords: (0,0) top-left, x right, y down, agent at bottom center
            Map coords: (row, col), row down, col right
            """
            # Offset from agent position
            dx = vx - half_w
            dy = vy - agent_view_y
            
            # Apply scale factor
            dx *= scale
            dy *= scale
            
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
            """Get character at map position for WALL DETECTION (raycasting).
            
            Uses simple rounding. Bottleneck detection is done in raycasting.
            """
            r = int(round(row))
            c = int(round(col))
            
            if not (0 <= r < full_height and 0 <= c < full_width):
                return ' '
            
            return padded[r][c]
        
        def get_display_char_at_map(row: float, col: float) -> str:
            """Get character for DISPLAY at map position.
            """
            r = int(round(row))
            c = int(round(col))
            
            if not (0 <= r < full_height and 0 <= c < full_width):
                return ' '
            
            return padded[r][c]
        
        # Create visibility map using ray casting
        visible = [[False for _ in range(view_width)] for _ in range(view_height)]
        
        def is_blocking(vx: int, vy: int) -> bool:
            """Check if a view cell blocks vision for diagonal corner checks.
            
            Only WALLS physically block diagonal vision. Void (empty space outside arena)
            should NOT block diagonal vision - you can see around void to walls beyond.
            """
            if not (0 <= vx < view_width and 0 <= vy < view_height):
                return True  # Out of bounds blocks
            map_row, map_col = view_to_map(vx, vy)
            ch = get_char_at_map(map_row, map_col)
            # Only walls block diagonal vision, not void
            return ch in wall_chars_blocking
        
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
            
            for step_num in range(steps + 1):
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
                        
                        # Stop at void (outside arena) - render as boundary wall
                        if ch == ' ' and not space_is_floor:
                            visible[iy][ix] = True  # Mark boundary as visible
                            return  # Stop ray
                    
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
        
        # Cast rays to ALL cells in view (not just edges) to ensure full coverage
        # This ensures no cells are missed between ray paths
        for vy in range(view_height):
            for vx in range(view_width):
                if is_in_fov(vx, vy):
                    cast_ray_dda(vx, vy)
        
        # Fill output grid from visibility map
        # Only visible cells get content; non-visible cells remain as fog
        # Track which (map_row, map_col) have already been rendered with important chars
        # to avoid duplicating landmarks/goals when multiple view cells map to same map cell
        rendered_important_cells = set()
        important_unique_chars = set('0123456789ABCDabcdGP*OoE!X')  # Landmarks, goals, special objects
        
        for vy in range(view_height):
            for vx in range(view_width):
                if visible[vy][vx]:
                    map_row, map_col = view_to_map(vx, vy)
                    # Round to get the actual map cell being sampled
                    map_cell = (int(round(map_row)), int(round(map_col)))
                    # Use display char for rendering (prefers important chars)
                    ch = get_display_char_at_map(map_row, map_col)
                    # Don't copy the agent marker from the source - we'll place our own
                    if ch not in agent_markers:
                        # Handle space character
                        if ch == ' ':
                            if space_is_floor:
                                # Space is floor in this map - render as floor
                                output[vy][vx] = '.'
                            else:
                                # Void outside arena - render as boundary wall
                                output[vy][vx] = '#'
                        elif ch in important_unique_chars:
                            # Important character - only render once per map cell
                            if map_cell not in rendered_important_cells:
                                output[vy][vx] = ch
                                rendered_important_cells.add(map_cell)
                            else:
                                # Already rendered this important char, show floor instead
                                output[vy][vx] = '.'
                        else:
                            output[vy][vx] = ch
                    else:
                        # Agent marker - show what's actually under the agent
                        output[vy][vx] = self._get_char_under_agent()
        
        # Place agent marker at bottom center - always pointing UP since view is rotated
        output[agent_view_y][half_w] = '↑'
        
        # Post-processing: fix bottleneck artifacts
        floor_like_chars = set('.0123456789*')
        
        # Pass 1: Fix .#. pattern (wall between floor) - always an artifact
        for vy in range(view_height):
            for vx in range(1, view_width - 1):
                ch = output[vy][vx]
                left = output[vy][vx - 1]
                right = output[vy][vx + 1]
                
                if ch == '#' and left in floor_like_chars and right in floor_like_chars:
                    output[vy][vx] = '.'
        
        # Pass 1b: Fix #░# pattern (fog between walls) - artifact from alternating rays
        # This creates ugly #░#░# patterns that should be solid wall
        for vy in range(view_height):
            for vx in range(1, view_width - 1):
                ch = output[vy][vx]
                left = output[vy][vx - 1]
                right = output[vy][vx + 1]
                
                if ch == '░' and left == '#' and right == '#':
                    output[vy][vx] = '#'
        
        # Pass 2: Fix #.# pattern (floor between walls) when it's clearly an artifact
        # Artifact detection: if the center floor continues vertically AND at least one 
        # adjacent row has floor at the wall positions, AND the row below is NOT also #.#
        # continuing as a real narrow corridor, it's likely an artifact
        for vy in range(1, view_height - 1):
            for vx in range(1, view_width - 1):
                ch = output[vy][vx]
                left = output[vy][vx - 1]
                right = output[vy][vx + 1]
                
                if ch in floor_like_chars and left == '#' and right == '#':
                    # Check if center continues vertically (corridor pattern)
                    above_center = output[vy-1][vx]
                    below_center = output[vy+1][vx]
                    center_continues = above_center in floor_like_chars and below_center in floor_like_chars
                    
                    if center_continues:
                        # Check if this is a real narrow corridor by looking multiple rows below
                        # A real narrow corridor has #.# pattern continuing for several rows
                        # An artifact may have ##.## (thick walls) below but then widen
                        below_is_narrow = (output[vy+1][vx-1] == '#' and output[vy+1][vx+1] == '#')
                        
                        if below_is_narrow:
                            # Row below is narrow - but check if it's really a narrow corridor
                            # or just thick walls (##.## pattern) that open up further down
                            # Check row+2 - if it's wider there, this row is still an artifact
                            if vy + 2 < view_height:
                                row_plus_2_wider = (
                                    output[vy+2][vx-1] in floor_like_chars or 
                                    output[vy+2][vx+1] in floor_like_chars
                                )
                                if not row_plus_2_wider:
                                    # Real narrow corridor continues
                                    continue
                            else:
                                # Can't check row+2, assume it's a real corridor
                                continue
                        
                        # Check if adjacent rows have floor at the wall positions
                        above_left = output[vy-1][vx-1]
                        above_right = output[vy-1][vx+1]
                        below_left = output[vy+1][vx-1]
                        below_right = output[vy+1][vx+1]
                        
                        # Widen if either adjacent row has floor at wall positions
                        above_wider = above_left in floor_like_chars or above_right in floor_like_chars
                        below_wider = below_left in floor_like_chars or below_right in floor_like_chars
                        
                        if above_wider or below_wider:
                            output[vy][vx - 1] = '.'
                            output[vy][vx + 1] = '.'
        
        # Pass 3: Fix horizontal fog gaps between floor cells
        for vy in range(view_height):
            for vx in range(1, view_width - 1):
                if output[vy][vx] == '░':
                    left_is_floor = output[vy][vx - 1] in floor_like_chars
                    right_is_floor = output[vy][vx + 1] in floor_like_chars
                    
                    if left_is_floor and right_is_floor:
                        map_row, map_col = view_to_map(vx, vy)
                        ch = get_display_char_at_map(map_row, map_col)
                        
                        if ch in wall_chars:
                            output[vy][vx] = '#'
                        elif ch == ' ':
                            output[vy][vx] = '.' if space_is_floor else '#'
                        elif ch not in agent_markers:
                            output[vy][vx] = ch if ch not in important_unique_chars else '.'
                        else:
                            output[vy][vx] = '.'
        
        # Pass 4: Fix vertical fog gaps between floor cells
        for vx in range(view_width):
            for vy in range(1, view_height - 1):
                if output[vy][vx] == '░':
                    above_is_floor = output[vy - 1][vx] in floor_like_chars
                    below_is_floor = output[vy + 1][vx] in floor_like_chars
                    
                    if above_is_floor and below_is_floor:
                        map_row, map_col = view_to_map(vx, vy)
                        ch = get_display_char_at_map(map_row, map_col)
                        
                        if ch in wall_chars:
                            output[vy][vx] = '#'
                        elif ch == ' ':
                            output[vy][vx] = '.' if space_is_floor else '#'
                        elif ch not in agent_markers:
                            output[vy][vx] = ch if ch not in important_unique_chars else '.'
                        else:
                            output[vy][vx] = '.'
        
        # Pass 5-6: Fix fog between wall and floor (iterative)
        # Run multiple times to propagate fills through fog gaps
        for _ in range(max(view_width, view_height)):
            changed = False
            
            # Horizontal: wall on one side, floor on the other
            for vy in range(view_height):
                for vx in range(1, view_width - 1):
                    if output[vy][vx] == '░':
                        left = output[vy][vx - 1]
                        right = output[vy][vx + 1]
                        
                        if (left == '#' and right in floor_like_chars) or \
                           (left in floor_like_chars and right == '#'):
                            map_row, map_col = view_to_map(vx, vy)
                            ch = get_display_char_at_map(map_row, map_col)
                            
                            if ch in wall_chars:
                                output[vy][vx] = '#'
                            elif ch == ' ':
                                output[vy][vx] = '.' if space_is_floor else '#'
                            elif ch not in agent_markers:
                                output[vy][vx] = ch if ch not in important_unique_chars else '.'
                            else:
                                output[vy][vx] = '.'
                            changed = True
            
            # Vertical: wall above/below, floor on the other side
            for vx in range(view_width):
                for vy in range(1, view_height - 1):
                    if output[vy][vx] == '░':
                        above = output[vy - 1][vx]
                        below = output[vy + 1][vx]
                        
                        if (above == '#' and below in floor_like_chars) or \
                           (above in floor_like_chars and below == '#'):
                            map_row, map_col = view_to_map(vx, vy)
                            ch = get_display_char_at_map(map_row, map_col)
                            
                            if ch in wall_chars:
                                output[vy][vx] = '#'
                            elif ch == ' ':
                                output[vy][vx] = '.' if space_is_floor else '#'
                            elif ch not in agent_markers:
                                output[vy][vx] = ch if ch not in important_unique_chars else '.'
                            else:
                                output[vy][vx] = '.'
                            changed = True
            
            if not changed:
                break
        
        # Pass 7: Limit walls to 1 cell thick next to floor (horizontal)
        # You can't see through a wall, so only show the wall immediately adjacent to floor
        # BUT: don't thin walls that connect to another floor section
        for vy in range(view_height):
            row = output[vy]
            # Left-to-right: for floor->wall, keep only 1 wall IF walls extend into fog
            for vx in range(view_width - 1):
                if row[vx] in floor_like_chars and row[vx + 1] == '#':
                    # Find end of wall run
                    wall_end = vx + 1
                    for wx in range(vx + 2, view_width):
                        if row[wx] == '#':
                            wall_end = wx
                        else:
                            break
                    # Check what's after walls - if fog, thin; if floor, keep
                    after = row[wall_end + 1] if wall_end + 1 < view_width else '░'
                    if after == '░':
                        for wx in range(vx + 2, wall_end + 1):
                            row[wx] = '░'
            
            # Right-to-left: same logic
            for vx in range(view_width - 1, 0, -1):
                if row[vx] in floor_like_chars and row[vx - 1] == '#':
                    wall_start = vx - 1
                    for wx in range(vx - 2, -1, -1):
                        if row[wx] == '#':
                            wall_start = wx
                        else:
                            break
                    before = row[wall_start - 1] if wall_start > 0 else '░'
                    if before == '░':
                        for wx in range(vx - 2, wall_start - 1, -1):
                            row[wx] = '░'
        
        # Pass 8: Limit walls to 1 cell thick (vertical direction)
        for vx in range(view_width):
            # Top-to-bottom
            for vy in range(view_height - 1):
                if output[vy][vx] in floor_like_chars and output[vy + 1][vx] == '#':
                    wall_end = vy + 1
                    for wy in range(vy + 2, view_height):
                        if output[wy][vx] == '#':
                            wall_end = wy
                        else:
                            break
                    after = output[wall_end + 1][vx] if wall_end + 1 < view_height else '░'
                    if after == '░':
                        for wy in range(vy + 2, wall_end + 1):
                            output[wy][vx] = '░'
            
            # Bottom-to-top
            for vy in range(view_height - 1, 0, -1):
                if output[vy][vx] in floor_like_chars and output[vy - 1][vx] == '#':
                    wall_start = vy - 1
                    for wy in range(vy - 2, -1, -1):
                        if output[wy][vx] == '#':
                            wall_start = wy
                        else:
                            break
                    before = output[wall_start - 1][vx] if wall_start > 0 else '░'
                    if before == '░':
                        for wy in range(vy - 2, wall_start - 1, -1):
                            output[wy][vx] = '░'
        
        return '\n'.join(''.join(row) for row in output)

    def _render_front_block(self) -> str:
        """
        Render only the first block directly in front of the agent.
        
        This is a minimal view showing just what's immediately ahead,
        useful for testing very limited perception scenarios.
        
        Returns:
            ASCII string showing the single block in front of the agent
        """
        import numpy as np
        
        # Get the full ASCII 2D map to extract positions
        full_map = self._render_ascii_2d()
        lines = full_map.split('\n')
        
        # Filter out status/info lines
        status_indicators = [':', '%', 'Trial', 'Phase', 'Reward', 'Score', 'Step', 'Error', '|']
        map_lines = []
        for line in lines:
            is_status = any(ind in line for ind in status_indicators)
            if not is_status and len(line) > 0:
                map_lines.append(line)
        
        if not map_lines:
            map_lines = lines
        
        full_height = len(map_lines)
        full_width = max(len(line) for line in map_lines) if map_lines else 1
        
        # Pad lines to uniform width
        padded = [line.ljust(full_width) for line in map_lines]
        
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
        
        # If no agent found, return unknown
        if agent_row is None:
            return '?'
        
        # Get agent's facing direction
        # Map row increases downward, direction 2 (North/↑) should decrease row
        if hasattr(self, 'agent') and hasattr(self.agent, 'angle'):
            angle_val = self.agent.angle
            if isinstance(angle_val, (int, np.integer)) or (isinstance(angle_val, float) and angle_val == int(angle_val) and 0 <= angle_val <= 7):
                # Integer direction: 0=E, 2=N, 4=W, 6=S
                dir_idx = int(angle_val)
            else:
                # Radians to direction index
                dir_idx = int(round(angle_val / (np.pi / 4))) % 8
        else:
            dir_idx = 2  # Default North
        
        # Direction vectors in map coordinates (row, col)
        # Note: In map coords, row increases downward, col increases rightward
        # Direction 0=E (col+1), 2=N (row-1), 4=W (col-1), 6=S (row+1)
        dir_vectors = [
            (0, 1),    # 0: East (col+)
            (-1, 1),   # 1: Northeast
            (-1, 0),   # 2: North (row-)
            (-1, -1),  # 3: Northwest
            (0, -1),   # 4: West (col-)
            (1, -1),   # 5: Southwest
            (1, 0),    # 6: South (row+)
            (1, 1),    # 7: Southeast
        ]
        
        dr, dc = dir_vectors[dir_idx]
        front_row = agent_row + dr
        front_col = agent_col + dc
        
        # Get the character at the front position
        if 0 <= front_row < full_height and 0 <= front_col < full_width:
            front_char = padded[front_row][front_col]
        else:
            front_char = '#'  # Out of bounds = wall
        
        return front_char
    
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
        elif self.view_mode == ViewMode.FRONT_BLOCK:
            return self._render_front_block()
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
        """Parse action from string (for model text output)."""
        action_str = action_str.lower().strip()
        
        mappings = {
            "forward": Action.FORWARD,
            "move forward": Action.FORWARD,
            "go forward": Action.FORWARD,
            "ahead": Action.FORWARD,
            "left": Action.ROTATE_LEFT,
            "rotate left": Action.ROTATE_LEFT,
            "rotate left": Action.ROTATE_LEFT,
            "right": Action.ROTATE_RIGHT,
            "rotate right": Action.ROTATE_RIGHT,
            "rotate right": Action.ROTATE_RIGHT,
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
        """Public interface to parse action from string (for model text output)."""
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
        """Get prompt describing available actions (for the model)."""
        actions = [f"- {self.action_names[a]}" for a in self.valid_actions]
        return "Available actions:\n" + "\n".join(actions)
    
    def get_task_description(self) -> str:
        """Get task description (for model context)."""
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
            
            # For diagonal moves, check corner-clipping but allow diagonal corridors
            # Block only if BOTH adjacent cells are walls (true corner)
            # Allow if at least one adjacent cell is walkable (diagonal corridor)
            if dx != 0 and dy != 0:  # Diagonal move
                side1_blocked = self._check_collision_at(self.agent.x + dx, self.agent.y)
                side2_blocked = self._check_collision_at(self.agent.x, self.agent.y + dy)
                if side1_blocked and side2_blocked:
                    # Both sides blocked = true corner, can't cut through
                    return -0.1
            
            self.agent.x, self.agent.y = new_x, new_y
                        
        elif action == Action.ROTATE_LEFT:
            self.agent.angle = (self.agent.angle + 1) % 8
        elif action == Action.ROTATE_RIGHT:
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
