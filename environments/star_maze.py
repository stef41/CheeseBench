"""
Star Maze environment for VLM evaluation.

Complex multi-arm maze for spatial navigation and shortcut learning.
Covers Starmaze, Sunburst maze, and similar complex branching mazes.

Uses integer grid coordinates like other environments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Set
from dataclasses import dataclass
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
class StarMazeConfig(EnvironmentConfig):
    """Star Maze specific configuration."""
    num_arms: int = 5  # Number of arms radiating from center
    arm_length: int = 8  # Integer cells
    maze_type: str = "star"  # "star", "sunburst"


class StarMaze(NavigationEnvironment):
    """
    Star/Complex Maze environment with integer grid coordinates.
    
    Protocol: Multi-arm maze radiating from central hub. Agent must learn
    to navigate to goal arm.
    
    Grid Layout (for 5 arms):
    - Center hub at grid center
    - Arms extend outward at evenly-spaced angles
    - Uses 8-direction movement system (0=E, 1=NE, 2=N, etc.)
    
    From reference (PMC3399492 - Vorhees & Williams 2014 - rodent maze protocols):
    - "Complex multi-arm mazes assess spatial navigation strategies and shortcut 
       learning through multiple arm configurations radiating from a central hub."
    """
    
    def __init__(self, 
                 config: Optional[StarMazeConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = StarMazeConfig(
                name="Star Maze",
                task_type="navigation",
                trials_to_criterion=40,
                sessions_to_criterion=8,
                trials_per_session=5,
                max_trial_steps=300,
                success_criterion="reach_goal",
                arena_size=25.0,
                source_pmc="PMC3399492",
                source_quote="Complex multi-arm mazes assess spatial navigation strategies and shortcut learning through multiple arm configurations."
            )
        
        super().__init__(config, view_mode)
        
        # Maze parameters (integer grid)
        self.num_arms = int(config.extra_params.get('num_arms', 5))
        self.arm_length = int(config.extra_params.get('arm_length', 8))
        self.maze_type = config.extra_params.get('maze_type', 'star')
        self.center_radius = 2  # Integer radius for hub
        
        # Grid size
        self.grid_size = 2 * (self.center_radius + self.arm_length) + 5
        self.center_x = self.grid_size // 2
        self.center_y = self.grid_size // 2
        
        # Map arm indices to direction indices (0-7)
        # For 5 arms at 0°, 72°, 144°, 216°, 288°:
        # Arm 0 (0°) -> direction 0 (E)
        # Arm 1 (72°) -> closest direction 1 (NE at 45°) or 2 (N at 90°)
        # We'll map to the 8 directions as best we can
        self._arm_to_direction = self._compute_arm_directions()
        
        # Create valid positions
        self.valid_positions: Set[Tuple[int, int]] = set()
        self._create_maze()
        
        # Goal and start arms
        self.goal_arm = 0
        self.start_arm = self.num_arms // 2
        
        # Compute arm end positions
        self.arm_ends = self._compute_arm_ends()
        
        # Goal position
        self.goal_x, self.goal_y = self.arm_ends[self.goal_arm]
        self.goal_radius = 1
        
        # Visited arms tracking
        self.visited_arms: Set[int] = set()
        self.current_arm = -1  # -1 = hub
        
        # Colors
        self.floor_color = (180, 160, 140)
        self.wall_color = (100, 80, 60)
        self.goal_color = (0, 255, 0)
        self.hub_color = (160, 140, 120)
        
        # Valid actions (8-direction turning)
        self.valid_actions = [
            Action.FORWARD,
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.STAY
        ]
    
    def _compute_arm_directions(self) -> List[int]:
        """Map arm indices to the closest 8-direction index."""
        directions = []
        for i in range(self.num_arms):
            # Angle of this arm in radians (0 = East, counter-clockwise)
            arm_angle = 2 * np.pi * i / self.num_arms
            
            # Convert to direction index (0-7)
            # Direction 0 = 0°, 1 = 45°, 2 = 90°, etc.
            dir_idx = int(round(arm_angle / (np.pi / 4))) % 8
            directions.append(dir_idx)
        
        return directions
    
    def _create_maze(self):
        """Create the maze as a set of valid integer positions."""
        # Create center hub (large enough to connect all arms)
        for dx in range(-self.center_radius - 1, self.center_radius + 2):
            for dy in range(-self.center_radius - 1, self.center_radius + 2):
                if dx*dx + dy*dy <= (self.center_radius + 1.5) ** 2:
                    x, y = self.center_x + dx, self.center_y + dy
                    # Only add if within bounds (leave 1-cell border for walls)
                    if 1 <= x < self.grid_size - 1 and 1 <= y < self.grid_size - 1:
                        self.valid_positions.add((x, y))
        
        # Create arms using their mapped directions
        for arm_idx in range(self.num_arms):
            dir_idx = self._arm_to_direction[arm_idx]
            dx, dy = DIR_VECTORS[dir_idx]
            is_diagonal = (dir_idx % 2 == 1)
            
            # Extend arm from center
            for dist in range(0, self.arm_length + 2):
                cx = self.center_x + dx * (self.center_radius + dist)
                cy = self.center_y + dy * (self.center_radius + dist)
                
                # Only add if within bounds (leave 1-cell border for walls)
                if 1 <= cx < self.grid_size - 1 and 1 <= cy < self.grid_size - 1:
                    self.valid_positions.add((cx, cy))
                
                # Add width to arm - wider for diagonals
                if not is_diagonal:
                    # Cardinal: perpendicular width (3 cells)
                    if dx == 0:  # N or S
                        for wx in [cx - 1, cx + 1]:
                            if 1 <= wx < self.grid_size - 1 and 1 <= cy < self.grid_size - 1:
                                self.valid_positions.add((wx, cy))
                    else:  # E or W
                        for wy in [cy - 1, cy + 1]:
                            if 1 <= cx < self.grid_size - 1 and 1 <= wy < self.grid_size - 1:
                                self.valid_positions.add((cx, wy))
                else:
                    # Diagonal: wider corridor (5 cells cross-section)
                    # Add all neighbors including diagonals for smooth movement
                    for ddx in [-1, 0, 1]:
                        for ddy in [-1, 0, 1]:
                            nx, ny = cx + ddx, cy + ddy
                            if 1 <= nx < self.grid_size - 1 and 1 <= ny < self.grid_size - 1:
                                self.valid_positions.add((nx, ny))
    
    def _compute_arm_ends(self) -> List[Tuple[int, int]]:
        """Compute the end position of each arm."""
        ends = []
        for arm_idx in range(self.num_arms):
            dir_idx = self._arm_to_direction[arm_idx]
            dx, dy = DIR_VECTORS[dir_idx]
            x = self.center_x + dx * (self.center_radius + self.arm_length)
            y = self.center_y + dy * (self.center_radius + self.arm_length)
            ends.append((x, y))
        return ends
    
    def _reset_agent_position(self):
        """Start agent at end of start arm, facing hub."""
        dir_idx = self._arm_to_direction[self.start_arm]
        dx, dy = DIR_VECTORS[dir_idx]
        
        # Position at end of start arm
        self.agent.x = self.center_x + dx * (self.center_radius + self.arm_length - 1)
        self.agent.y = self.center_y + dy * (self.center_radius + self.arm_length - 1)
        
        # Face toward hub (opposite direction)
        self.agent.angle = (dir_idx + 4) % 8
        
        self.visited_arms = {self.start_arm}
        self.current_arm = self.start_arm
    
    def _setup_trial(self):
        """Setup for new trial - randomize goal."""
        available = [i for i in range(self.num_arms) if i != self.start_arm]
        self.goal_arm = np.random.choice(available)
        self.goal_x, self.goal_y = self.arm_ends[self.goal_arm]
    
    def _get_arm_at_position(self, x: int, y: int) -> int:
        """Determine which arm the position is in (-1 for hub)."""
        dx = x - self.center_x
        dy = y - self.center_y
        
        # In hub?
        if dx*dx + dy*dy <= (self.center_radius + 0.5) ** 2:
            return -1
        
        # Find which arm by checking which direction this is from center
        # and if that arm exists
        if dx == 0 and dy == 0:
            return -1
        
        # Compute angle from center
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * np.pi
        
        # Find closest arm
        min_diff = float('inf')
        closest_arm = -1
        for arm_idx in range(self.num_arms):
            arm_angle = 2 * np.pi * arm_idx / self.num_arms
            diff = abs(angle - arm_angle)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff < min_diff:
                min_diff = diff
                closest_arm = arm_idx
        
        # Verify we're actually in the corridor of this arm
        # by checking if we're on the arm's axis (within width)
        return closest_arm
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if position is outside valid maze area."""
        return (x, y) not in self.valid_positions
    
    def _check_success(self) -> bool:
        """Success = reached goal arm end."""
        dist_sq = (self.agent.x - self.goal_x)**2 + (self.agent.y - self.goal_y)**2
        return dist_sq <= self.goal_radius ** 2
    
    def _check_failure(self) -> bool:
        """No automatic failure."""
        return False
    
    def _execute_action(self, action: Action) -> float:
        """Execute action using integer grid movement."""
        old_x, old_y = self.agent.x, self.agent.y
        
        if action == Action.FORWARD:
            # Move one cell in current direction
            dx, dy = DIR_VECTORS[self.agent.angle]
            new_x = self.agent.x + dx
            new_y = self.agent.y + dy
            
            if not self._check_collision_at(new_x, new_y):
                self.agent.x = new_x
                self.agent.y = new_y
            else:
                return -0.1  # Hit wall
                
        elif action == Action.ROTATE_LEFT:
            # Turn 45° counter-clockwise
            self.agent.angle = (self.agent.angle + 1) % 8
            
        elif action == Action.ROTATE_RIGHT:
            # Turn 45° clockwise
            self.agent.angle = (self.agent.angle - 1) % 8
        
        # Update current arm
        new_arm = self._get_arm_at_position(self.agent.x, self.agent.y)
        if new_arm >= 0 and new_arm not in self.visited_arms:
            self.visited_arms.add(new_arm)
        self.current_arm = new_arm
        
        # Track path
        self._trial_path_length += math.sqrt(
            (self.agent.x - old_x)**2 + (self.agent.y - old_y)**2
        )
        
        # Success check
        if self._check_success():
            self._trial_reward += 1.0
            return 1.0
        
        return -0.01
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info."""
        base_info = super().get_info()
        base_info.update({
            'current_arm': self.current_arm,
            'goal_arm': self.goal_arm,
            'visited_arms': list(self.visited_arms),
            'num_arms': self.num_arms,
            'maze_type': self.maze_type
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view using shared raycasting."""
        def wall_color(dist):
            shade = max(40, 255 - int(dist * 15))
            return (int(shade * 0.4), int(shade * 0.32), int(shade * 0.24))
        
        def overlay(img, agent_angle, fov):
            self._render_goal_in_fpv(img, self.goal_x, self.goal_y, self.goal_color, 
                                     fov, horizon=120, y_min=70, y_max=154)
        
        return self._render_fpv_raycasting(
            ceiling_color=(150, 150, 160),
            floor_color=self.floor_color,
            wall_color_func=wall_color,
            max_dist=15.0,
            overlay_func=overlay
        )
    
    def _cast_ray(self, angle: float) -> float:
        """Cast ray using shared base implementation."""
        return super()._cast_ray(angle, max_dist=30.0, step_size=1.0)
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view using shared helpers."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Background (void)
        
        scale = 224 // self.grid_size
        offset = (224 - self.grid_size * scale) // 2
        
        # Draw walls around valid positions first
        for (x, y) in self.valid_positions:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.valid_positions and 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    px = offset + nx * scale
                    py = offset + (self.grid_size - 1 - ny) * scale
                    if 0 <= px < 224 - scale and 0 <= py < 224 - scale:
                        img[py:py+scale, px:px+scale] = self.wall_color
        
        # Draw floor (valid positions)
        for (x, y) in self.valid_positions:
            px = offset + x * scale
            py = offset + (self.grid_size - 1 - y) * scale  # Flip y
            if 0 <= px < 224 - scale and 0 <= py < 224 - scale:
                img[py:py+scale, px:px+scale] = self.floor_color
        
        # Draw goal (using shared _draw_disk)
        gx = offset + self.goal_x * scale + scale // 2
        gy = offset + (self.grid_size - 1 - self.goal_y) * scale + scale // 2
        NavigationEnvironment._draw_disk(img, gx, gy, 4, self.goal_color)
        
        # Draw agent (using shared _draw_disk)
        ax = offset + self.agent.x * scale + scale // 2
        ay = offset + (self.grid_size - 1 - self.agent.y) * scale + scale // 2
        NavigationEnvironment._draw_disk(img, ax, ay, 3, (0, 150, 255))
        
        # Direction indicator
        dir_dx, dir_dy = DIR_VECTORS[self.agent.angle]
        dir_x = int(ax + dir_dx * 6)
        dir_y = int(ay - dir_dy * 6)  # Flip y
        NavigationEnvironment._draw_disk(img, dir_x, dir_y, 2, (255, 255, 255))
        
        return img
    
    def _render_ascii_2d(self, width: int = None, height: int = None) -> str:
        """Render ASCII top-down view at 1:1 scale."""
        # Use grid size directly (no scaling)
        width = self.grid_size
        height = self.grid_size
        
        # Create grid filled with walls
        grid = [['#' for _ in range(width)] for _ in range(height)]
        
        # Draw floor (valid positions)
        for (x, y) in self.valid_positions:
            # Flip y for display (y=0 at bottom in world, top in display)
            dy = height - 1 - y
            if 0 <= x < width and 0 <= dy < height:
                grid[dy][x] = '.'
        
        # Draw goal
        gy = height - 1 - self.goal_y
        if 0 <= self.goal_x < width and 0 <= gy < height:
            grid[gy][self.goal_x] = 'G'
        
        # Draw agent
        ay = height - 1 - self.agent.y
        if 0 <= self.agent.x < width and 0 <= ay < height:
            dirs = {0: '→', 1: '↗', 2: '↑', 3: '↖', 4: '←', 5: '↙', 6: '↓', 7: '↘'}
            grid[ay][self.agent.x] = dirs.get(self.agent.angle, '@')
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII 3D view with proper wall continuity using raycasting."""
        agent_angle = self.agent.angle * (np.pi / 4)
        fov = np.pi / 2
        view_height = height - 2
        horizon = view_height // 2
        
        # Calculate goal visibility
        dx = self.goal_x - self.agent.x
        dy = self.goal_y - self.agent.y
        goal_dist = np.sqrt(dx**2 + dy**2)
        goal_angle = math.atan2(dy, dx)
        goal_rel_angle = goal_angle - agent_angle
        while goal_rel_angle > np.pi: goal_rel_angle -= 2*np.pi
        while goal_rel_angle < -np.pi: goal_rel_angle += 2*np.pi
        
        goal_col = None
        if abs(goal_rel_angle) < fov / 2:
            goal_col = int((0.5 - goal_rel_angle / fov) * (width - 3))
            if goal_col < 0 or goal_col >= width - 2:
                goal_col = None
        
        def overlay(row, col, char, dist, wall_top, wall_bottom):
            if goal_col is not None and col == goal_col and row == horizon:
                if goal_dist < dist:
                    return 'G'
            return char
        
        return self._render_3d_raycasting(
            width=width, height=height,
            fov=fov, agent_angle=agent_angle,
            cast_ray_func=self._cast_ray,
            overlay_func=overlay
        )
    
    def _get_facing_arm(self) -> int:
        """Determine which arm the agent is facing."""
        # Agent angle is 0-7, map to arm index
        agent_dir = self.agent.angle
        
        # Find closest arm that matches this direction
        min_diff = float('inf')
        facing_arm = -1
        for arm_idx in range(self.num_arms):
            arm_dir = self._arm_to_direction[arm_idx]
            diff = abs(agent_dir - arm_dir)
            if diff > 4:
                diff = 8 - diff
            if diff < min_diff:
                min_diff = diff
                facing_arm = arm_idx
        
        return facing_arm
