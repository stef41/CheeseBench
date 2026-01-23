"""
Radial Arm Maze environment for VLM evaluation.

Central platform with multiple arms radiating outward.
Tests working memory and reference memory.

Uses integer grid coordinates like other environments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
import math

from .base_env import (
    NavigationEnvironment, 
    EnvironmentConfig, 
    ViewMode, 
    Action,
    AgentState,
    DIR_VECTORS
)


@dataclass
class RadialArmMazeConfig(EnvironmentConfig):
    """Radial Arm Maze specific configuration."""
    num_arms: int = 8
    arm_length: int = 6  # Integer cells
    rewarded_arms: List[int] = field(default_factory=lambda: [0, 2, 4, 6])


class RadialArmMaze(NavigationEnvironment):
    """
    Radial Arm Maze environment with integer grid coordinates.
    
    Protocol: Central platform with radiating arms.
    Some arms contain rewards. Tests both:
    - Working memory (don't revisit within trial)
    - Reference memory (remember which arms are rewarded)
    
    Grid Layout (for 8 arms):
    - Center at (10, 10) with radius ~2 cells
    - Arms extend outward in 8 directions (matching the 8-direction movement)
    - Each arm is arm_length cells long
    
    From verified protocols (PMC4030456 - Penley et al., J Vis Exp 2013):
    - "Subjects are required to avoid arms previously used for escape during each 
       testing day (working memory) as well as avoid fixed arms which never contain 
       escape platforms (reference memory)."
    """
    
    def __init__(self, 
                 config: Optional[RadialArmMazeConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = RadialArmMazeConfig(
                name="Radial Arm Maze",
                task_type="navigation",
                trials_to_criterion=20,
                sessions_to_criterion=5,
                trials_per_session=4,
                max_trial_steps=400,
                success_criterion="collect_all_rewards",
                arena_size=21.0,
                source_pmc="PMC4030456",
                source_quote="Subjects are required to avoid arms previously used for escape during each testing day (working memory) as well as avoid fixed arms which never contain escape platforms (reference memory)."
            )
        
        super().__init__(config, view_mode)
        
        # Maze geometry (integer grid)
        self.num_arms = int(config.extra_params.get('num_arms', 8))
        self.arm_length = int(config.extra_params.get('arm_length', 6))
        self.center_radius = 2  # Integer radius for center
        
        # Grid size (needs to fit center + arms in all directions)
        self.grid_size = 2 * (self.center_radius + self.arm_length) + 5
        self.center_x = self.grid_size // 2
        self.center_y = self.grid_size // 2
        
        # Create valid positions
        self.valid_positions = set()
        self._create_maze()
        
        # Rewards
        self.rewarded_arms = list(config.extra_params.get('rewarded_arms', [0, 2, 4, 6]))
        self.rewards_collected = [False] * self.num_arms
        self.arms_visited = [False] * self.num_arms
        
        # Arm end positions (where rewards are)
        self.arm_ends = self._compute_arm_ends()
        
        # Goal tracking (first rewarded arm)
        if self.rewarded_arms:
            first_arm = self.rewarded_arms[0]
            self.goal_x, self.goal_y = self.arm_ends[first_arm]
        else:
            self.goal_x, self.goal_y = self.center_x, self.center_y
        self.goal_radius = 1
        self.goal_visible = True
        
        # Visual cues (landmarks at arm ends)
        self.landmarks = self._create_landmarks()
        
        # Colors
        self.floor_color = (180, 160, 140)
        self.wall_color = (120, 100, 80)
        self.reward_color = (255, 215, 0)
        
        # Actions (rewards collected automatically at arm ends)
        self.valid_actions = [
            Action.FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
            Action.STAY
        ]
        
        # Error tracking
        self.working_memory_errors = 0
        self.reference_memory_errors = 0
    
    def _create_maze(self):
        """Create the maze as a set of valid positions.
        
        Layout: Central hub with 8 equal-width arms radiating outward.
        All arms are 3 cells wide for consistency.
        """
        # Create center platform (large enough to connect all arms)
        # Use a square center that spans all arm connections
        for dx in range(-self.center_radius, self.center_radius + 1):
            for dy in range(-self.center_radius, self.center_radius + 1):
                self.valid_positions.add((self.center_x + dx, self.center_y + dy))
        
        # Create 8 arms - all 3 cells wide
        for arm_idx in range(self.num_arms):
            dx, dy = DIR_VECTORS[arm_idx]
            is_diagonal = (arm_idx % 2 == 1)
            
            for dist in range(1, self.arm_length + 1):
                # Center line of arm
                cx = self.center_x + dx * (self.center_radius + dist)
                cy = self.center_y + dy * (self.center_radius + dist)
                self.valid_positions.add((cx, cy))
                
                if not is_diagonal:
                    # Cardinal arms (E, N, W, S): add perpendicular width
                    if dx == 0:  # N/S arm - add width in x
                        self.valid_positions.add((cx - 1, cy))
                        self.valid_positions.add((cx + 1, cy))
                    else:  # E/W arm - add width in y
                        self.valid_positions.add((cx, cy - 1))
                        self.valid_positions.add((cx, cy + 1))
                else:
                    # Diagonal arms: create 3-cell width perpendicular to diagonal
                    # For NE/SW diagonals (dx*dy > 0), perpendicular is NW-SE direction
                    # For NW/SE diagonals (dx*dy < 0), perpendicular is NE-SW direction
                    if dx * dy > 0:  # NE (1,1) or SW (-1,-1)
                        self.valid_positions.add((cx - 1, cy))
                        self.valid_positions.add((cx, cy - 1))
                        self.valid_positions.add((cx + 1, cy))
                        self.valid_positions.add((cx, cy + 1))
                    else:  # NW (-1,1) or SE (1,-1)
                        self.valid_positions.add((cx + 1, cy))
                        self.valid_positions.add((cx, cy - 1))
                        self.valid_positions.add((cx - 1, cy))
                        self.valid_positions.add((cx, cy + 1))
    
    def _compute_arm_ends(self) -> List[Tuple[int, int]]:
        """Compute the end position of each arm."""
        ends = []
        for arm_idx in range(self.num_arms):
            dx, dy = DIR_VECTORS[arm_idx]
            x = self.center_x + dx * (self.center_radius + self.arm_length)
            y = self.center_y + dy * (self.center_radius + self.arm_length)
            ends.append((x, y))
        return ends
    
    def _create_landmarks(self) -> List[Dict[str, Any]]:
        """Create visual cues at arm ends."""
        landmarks = []
        # Use numbers 1-8 for the 8 arms
        for arm_idx in range(self.num_arms):
            x, y = self.arm_ends[arm_idx]
            landmarks.append({
                'arm_index': arm_idx,
                'x': x,
                'y': y,
                'char': str((arm_idx + 1) % 10),  # 1-8, then 0
                'has_reward': arm_idx in self.rewarded_arms
            })
        return landmarks
    
    def _reset_agent_position(self):
        """Start at center, random orientation."""
        self.agent.x = self.center_x
        self.agent.y = self.center_y
        self.agent.angle = np.random.randint(0, 8)
        
        # Reset tracking
        self.rewards_collected = [False] * self.num_arms
        self.arms_visited = [False] * self.num_arms
        self.working_memory_errors = 0
        self.reference_memory_errors = 0
    
    def _setup_trial(self):
        """Setup for new trial."""
        pass
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if position is outside valid maze area."""
        return (x, y) not in self.valid_positions
    
    def _get_current_arm(self) -> Optional[int]:
        """Determine which arm agent is in (or None if in center)."""
        # Check if in center
        dx = self.agent.x - self.center_x
        dy = self.agent.y - self.center_y
        if dx*dx + dy*dy <= self.center_radius * self.center_radius + 1:
            return None
        
        # Find which arm based on direction from center
        if dx == 0 and dy == 0:
            return None
        
        # Compute angle and map to arm index
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        
        # Each arm covers 45 degrees (π/4)
        arm_idx = int((angle + math.pi/8) / (math.pi/4)) % 8
        return arm_idx
    
    def _check_at_arm_end(self) -> Optional[int]:
        """Check if at end of an arm (within 1 cell)."""
        for arm_idx, (ex, ey) in enumerate(self.arm_ends):
            if abs(self.agent.x - ex) <= 1 and abs(self.agent.y - ey) <= 1:
                return arm_idx
        return None
    
    def _execute_action(self, action: Action) -> float:
        """Execute action using grid-based movement."""
        old_x, old_y = self.agent.x, self.agent.y
        old_arm = self._get_current_arm()
        
        # Use parent class for basic movement
        reward = super()._execute_action(action)
        
        # If hit wall, return the penalty
        if reward == -0.1:
            return reward
        
        # Track arm visits
        new_arm = self._get_current_arm()
        if new_arm is not None and new_arm != old_arm:
            if self.arms_visited[new_arm]:
                self.working_memory_errors += 1
            self.arms_visited[new_arm] = True
        
        # Check for reward collection at arm end
        arm_end = self._check_at_arm_end()
        if arm_end is not None:
            if arm_end in self.rewarded_arms and not self.rewards_collected[arm_end]:
                self.rewards_collected[arm_end] = True
                self._trial_reward += 0.25
                if self._check_success():
                    return 1.0
                return 0.5
            elif arm_end not in self.rewarded_arms:
                if not self.arms_visited[arm_end]:
                    self.reference_memory_errors += 1
                    self.arms_visited[arm_end] = True
        
        return reward
    
    def _check_success(self) -> bool:
        """Success = collected all rewards from baited arms."""
        for i in self.rewarded_arms:
            if not self.rewards_collected[i]:
                return False
        return True
    
    def _check_failure(self) -> bool:
        """No automatic failure."""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info including memory errors."""
        base_info = super().get_info()
        base_info.update({
            'working_memory_errors': self.working_memory_errors,
            'reference_memory_errors': self.reference_memory_errors,
            'arms_visited': self.arms_visited.copy(),
            'rewards_collected': self.rewards_collected.copy(),
            'rewarded_arms': self.rewarded_arms.copy()
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view using raycasting."""
        def wall_color(dist):
            shade = max(40, 255 - int(dist * 12))
            return (int(shade * 0.47), int(shade * 0.39), int(shade * 0.31))
        
        def overlay(img, agent_angle, fov):
            # Render goal (first rewarded arm end)
            self._render_goal_in_fpv(img, self.goal_x, self.goal_y, self.reward_color,
                                     fov, horizon=112, y_min=70, y_max=154)
            # Render uncollected rewards
            for arm_idx in self.rewarded_arms:
                if not self.rewards_collected[arm_idx]:
                    ex, ey = self.arm_ends[arm_idx]
                    if (ex, ey) != (self.goal_x, self.goal_y):
                        self._render_goal_in_fpv(img, ex, ey, (255, 200, 0),
                                                 fov, horizon=112, y_min=70, y_max=154)
        
        return self._render_fpv_raycasting(
            ceiling_color=(150, 150, 150),
            floor_color=self.floor_color,
            wall_color_func=wall_color,
            max_dist=20.0,
            overlay_func=overlay
        )
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view.
        
        Y-axis is flipped so that North (increasing Y) appears as UP on screen.
        Screen row = 224 - y * scale (higher Y = lower row index = higher on screen)
        """
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)  # Background (void)
        
        scale = 224 // self.grid_size
        max_screen_y = 224 - scale  # Highest valid screen Y
        
        # Helper to convert world Y to screen Y (flip Y-axis)
        def screen_y(y):
            return max_screen_y - y * scale
        
        # Draw walls around valid positions first
        for (x, y) in self.valid_positions:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.valid_positions and 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    px, py = nx * scale, screen_y(ny)
                    if 0 <= px < 224 - scale and 0 <= py < 224 - scale:
                        img[py:py+scale, px:px+scale] = self.wall_color
        
        # Draw valid positions
        for (x, y) in self.valid_positions:
            px, py = x * scale, screen_y(y)
            img[py:py+scale, px:px+scale] = self.floor_color
        
        # Draw rewards (uncollected)
        for arm_idx in self.rewarded_arms:
            if not self.rewards_collected[arm_idx]:
                ex, ey = self.arm_ends[arm_idx]
                px, py = ex * scale, screen_y(ey)
                img[py:py+scale, px:px+scale] = self.reward_color
        
        # Draw agent
        ax, ay = self.agent.x * scale, screen_y(self.agent.y)
        img[ay:ay+scale, ax:ax+scale] = (255, 100, 100)
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view.
        
        Y-axis is flipped so that North (increasing Y) appears as UP on screen.
        Screen row = grid_size - y (higher Y = lower row index = higher on screen)
        """
        grid = [[' ' for _ in range(self.grid_size + 2)] for _ in range(self.grid_size + 2)]
        
        # Helper to convert world Y to screen row (flip Y-axis)
        def screen_row(y):
            return self.grid_size - y
        
        # Draw walls around valid positions
        for (x, y) in self.valid_positions:
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.valid_positions:
                    grid[screen_row(ny)][nx + 1] = '#'
        
        # Draw floor
        for (x, y) in self.valid_positions:
            grid[screen_row(y)][x + 1] = '.'
        
        # Draw rewards (uncollected) with *
        for arm_idx in self.rewarded_arms:
            if not self.rewards_collected[arm_idx]:
                ex, ey = self.arm_ends[arm_idx]
                grid[screen_row(ey)][ex + 1] = '*'
        
        # Draw arm numbers at ends
        for lm in self.landmarks:
            x, y = lm['x'], lm['y']
            if not (lm['arm_index'] in self.rewarded_arms and not self.rewards_collected[lm['arm_index']]):
                grid[screen_row(y)][x + 1] = lm['char']
        
        # Draw agent
        arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
        ax, ay = self.agent.x, self.agent.y
        grid[screen_row(ay)][ax + 1] = arrows[self.agent.angle]
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII pseudo-3D view with proper wall continuity."""
        agent_angle = self.agent.angle * math.pi / 4
        fov = math.pi / 2
        view_height = height - 2
        horizon = view_height // 2
        
        # Calculate visible landmarks
        landmark_cols = {}
        for lm in self.landmarks:
            dx = lm['x'] - self.agent.x
            dy = lm['y'] - self.agent.y
            if dx == 0 and dy == 0:
                continue
            lm_dist = math.sqrt(dx**2 + dy**2)
            lm_angle = math.atan2(dy, dx)
            lm_rel_angle = lm_angle - agent_angle
            while lm_rel_angle > math.pi: lm_rel_angle -= 2*math.pi
            while lm_rel_angle < -math.pi: lm_rel_angle += 2*math.pi
            
            if abs(lm_rel_angle) < fov / 2:
                lm_col = int((0.5 - lm_rel_angle / fov) * (width - 3))
                if 0 <= lm_col < width - 2:
                    arm_idx = lm['arm_index']
                    if arm_idx in self.rewarded_arms and not self.rewards_collected[arm_idx]:
                        landmark_cols[lm_col] = ('*', lm_dist)
                    else:
                        landmark_cols[lm_col] = (lm['char'], lm_dist)
        
        def overlay(row, col, char, dist, wall_top, wall_bottom):
            if col in landmark_cols and row == horizon:
                lm_char, lm_dist = landmark_cols[col]
                if lm_dist < dist:
                    return lm_char
            return char
        
        return self._render_3d_raycasting(
            width=width, height=height,
            fov=fov, agent_angle=agent_angle,
            cast_ray_func=self._cast_ray_simple,
            overlay_func=overlay
        )
    
    def _cast_ray_simple(self, angle: float) -> float:
        """Cast ray using shared base implementation."""
        return self._cast_ray(angle, max_dist=10.0, step_size=0.5)


def create_radial_arm_maze(
    num_arms: int = 8,
    arm_length: int = 6,
    rewarded_arms: List[int] = None,
    trials_to_criterion: int = 20,
    trials_per_session: int = 4,
    view_mode: ViewMode = ViewMode.FPV_3D,
    source_pmc: str = "",
    source_quote: str = ""
) -> RadialArmMaze:
    """Factory function to create Radial Arm Maze."""
    
    if rewarded_arms is None:
        rewarded_arms = [0, 2, 4, 6]
    
    config = RadialArmMazeConfig(
        name="Radial Arm Maze",
        task_type="navigation",
        trials_to_criterion=trials_to_criterion,
        sessions_to_criterion=trials_to_criterion // trials_per_session,
        trials_per_session=trials_per_session,
        max_trial_steps=400,
        success_criterion="collect_all_rewards",
        arena_size=21.0,
        source_pmc=source_pmc,
        source_quote=source_quote,
        extra_params={
            'num_arms': num_arms,
            'arm_length': arm_length,
            'rewarded_arms': rewarded_arms
        }
    )
    
    return RadialArmMaze(config, view_mode)
