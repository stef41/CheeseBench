"""
T-Maze environment for LLM evaluation.

T-shaped maze for testing spatial working memory and navigation.
Agent must choose correct arm based on task rules.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base_env import (
    NavigationEnvironment, 
    EnvironmentConfig, 
    ViewMode, 
    Action,
    AsciiCanvas,
)


@dataclass
class TMazeConfig(EnvironmentConfig):
    """T-Maze specific configuration."""
    stem_length: int = 3
    arm_length: int = 2
    corridor_width: int = 1
    reward_arm: str = "left"  # "left", "right", or "alternating"


class TMaze(NavigationEnvironment):
    """
    T-Maze environment.
    
    Protocol: Agent starts at base of stem, navigates to T-junction,
    then must choose correct arm to find reward.
    
    From verified protocols (PMC3399492 - Shoji et al., J Vis Exp 2012):
    - "In the forced alternation task, each trial consists of a forced choice run 
       followed by a free choice run."
    - "A mouse is subjected to 10 consecutive trials in a session per day."
    - Stem composed of area leading to arms
    - Arms at T-junction for left/right choice
    """
    
    def __init__(self, 
                 config: Optional[TMazeConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = TMazeConfig(
                name="T-Maze",
                task_type="navigation",
                trials_to_criterion=40,  # 4 sessions × 10 trials (per Shoji protocol)
                sessions_to_criterion=4,
                trials_per_session=10,   # Standard protocol
                max_trial_steps=200,
                success_criterion="reach_correct_arm",
                arena_size=8.0,
                source_pmc="PMC3399492",
                source_quote="In the forced alternation task, each trial consists of a forced choice run followed by a free choice run. A mouse is subjected to 10 consecutive trials in a session per day."
            )
        
        super().__init__(config, view_mode)
        
        # Maze geometry (integer grid)
        self.stem_length = int(config.extra_params.get('stem_length', 3))
        self.arm_length = int(config.extra_params.get('arm_length', 2))
        self.corridor_width = int(config.extra_params.get('corridor_width', 1))
        
        # Reward arm
        self.reward_arm_type = config.extra_params.get('reward_arm', 'left')
        self.current_reward_arm = 'left'
        
        # Reversal learning support
        self.reversal_learning = config.extra_params.get('reversal_learning', False)
        self.reversal_criterion = config.extra_params.get('reversal_criterion', 8)  # Correct trials to reverse
        self.consecutive_correct = 0
        self.reversal_count = 0
        self.max_reversals = config.extra_params.get('max_reversals', 10)
        
        # Goal positions (integer grid cells at arm ends)
        # Left arm: x = -arm_length = -2, y = stem_length = 3
        # Right arm: x = +arm_length = 2, y = stem_length = 3
        self.left_goal = (-self.arm_length, self.stem_length)
        self.right_goal = (self.arm_length, self.stem_length)
        
        # Set initial goal
        self._set_goal_position()
        
        # Build walls
        self.walls = self._build_walls()
        
        # Colors
        self.floor_color = (200, 180, 160)
        self.wall_color = (100, 80, 60)
        self.goal_color = (0, 255, 0)
        
        # Valid actions
        self.valid_actions = [
            Action.FORWARD,
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.STAY
        ]
    
    def _set_goal_position(self):
        """Set goal based on reward arm."""
        if self.reward_arm_type == 'alternating':
            # Alternate based on trial number
            if self.session.current_trial % 2 == 0:
                self.current_reward_arm = 'left'
            else:
                self.current_reward_arm = 'right'
        elif self.reward_arm_type == 'reversal' or self.reversal_learning:
            # Reversal learning: keep current arm until criterion then switch
            pass  # current_reward_arm already set, switches on criterion
        else:
            self.current_reward_arm = self.reward_arm_type
        
        if self.current_reward_arm == 'left':
            self.goal_x, self.goal_y = self.left_goal
        else:
            self.goal_x, self.goal_y = self.right_goal
        
        self.goal_visible = True
    
    def _trigger_reversal(self):
        """Switch the rewarded arm (reversal learning)."""
        if self.current_reward_arm == 'left':
            self.current_reward_arm = 'right'
        else:
            self.current_reward_arm = 'left'
        self.reversal_count += 1
        self.consecutive_correct = 0
        self._set_goal_position()
    
    def _build_walls(self) -> List[Dict[str, float]]:
        """Build T-maze walls."""
        walls = []
        w = self.corridor_width / 2
        
        # Stem walls (left and right)
        # Left wall of stem
        walls.append({'x1': -w, 'y1': 0, 'x2': -w, 'y2': self.stem_length})
        # Right wall of stem  
        walls.append({'x1': w, 'y1': 0, 'x2': w, 'y2': self.stem_length})
        
        # Bottom wall (start)
        walls.append({'x1': -w, 'y1': 0, 'x2': w, 'y2': 0})
        
        # Top of T (arms)
        arm_y_bottom = self.stem_length
        arm_y_top = self.stem_length + self.corridor_width
        
        # Left arm walls
        walls.append({'x1': -self.arm_length, 'y1': arm_y_bottom, 
                      'x2': -self.arm_length, 'y2': arm_y_top})  # End cap
        walls.append({'x1': -self.arm_length, 'y1': arm_y_top, 
                      'x2': -w, 'y2': arm_y_top})  # Top of left arm
        walls.append({'x1': -self.arm_length, 'y1': arm_y_bottom, 
                      'x2': -w, 'y2': arm_y_bottom})  # Bottom of left arm
        
        # Right arm walls
        walls.append({'x1': self.arm_length, 'y1': arm_y_bottom, 
                      'x2': self.arm_length, 'y2': arm_y_top})  # End cap
        walls.append({'x1': w, 'y1': arm_y_top, 
                      'x2': self.arm_length, 'y2': arm_y_top})  # Top of right arm
        walls.append({'x1': w, 'y1': arm_y_bottom, 
                      'x2': self.arm_length, 'y2': arm_y_bottom})  # Bottom of right arm
        
        # Top wall connecting arms
        walls.append({'x1': -w, 'y1': arm_y_top, 'x2': w, 'y2': arm_y_top})
        
        return walls
    
    def _reset_agent_position(self):
        """Start agent at bottom of stem, facing up."""
        self.agent.x = 0
        self.agent.y = 0
        self.agent.angle = 2  # Direction 2 = North (facing up toward T)
    
    def _setup_trial(self):
        """Setup for new trial."""
        # Check if reversal should happen (from previous trial success)
        if self.reversal_learning and hasattr(self, '_last_trial_correct'):
            if self._last_trial_correct:
                self.consecutive_correct += 1
                if self.consecutive_correct >= self.reversal_criterion:
                    if self.reversal_count < self.max_reversals:
                        self._trigger_reversal()
            else:
                self.consecutive_correct = 0
        
        self._last_trial_correct = False
        self._set_goal_position()
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if integer grid position is valid (returns True if blocked)."""
        # T-maze integer grid layout:
        # Stem: x=0, y=0 to y=stem_length-1 (0,1,2 for stem_length=3)
        # Junction: x=0, y=stem_length (0,3)
        # Left arm: x=-1 to x=-arm_length, y=stem_length (-1,3 and -2,3)
        # Right arm: x=1 to x=arm_length, y=stem_length (1,3 and 2,3)
        
        # In stem (x must be 0, y from 0 to stem_length-1)
        in_stem = (x == 0 and 0 <= y < self.stem_length)
        
        # At junction or in arms (y must equal stem_length, x from -arm_length to +arm_length)
        in_arms = (y == self.stem_length and -self.arm_length <= x <= self.arm_length)
        
        return not (in_stem or in_arms)
    
    def _check_collision(self) -> bool:
        """Check current position."""
        return self._check_collision_at(self.agent.x, self.agent.y)

    def _check_success(self) -> bool:
        """Success = reached correct arm end (exact grid cell)."""
        success = (self.agent.x == self.goal_x and self.agent.y == self.goal_y)
        if success:
            self._last_trial_correct = True
        return success
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info including reversal learning stats."""
        base_info = super().get_info()
        base_info.update({
            'current_reward_arm': self.current_reward_arm,
            'consecutive_correct': self.consecutive_correct,
            'reversal_count': self.reversal_count,
            'reversal_learning': self.reversal_learning,
            'reversal_criterion': self.reversal_criterion
        })
        return base_info
    
    def _execute_action(self, action: Action) -> float:
        """Execute movement action with T-maze specific rewards."""
        # Use base class for movement and collision
        reward = super()._execute_action(action)
        
        # If base returned success reward, we're done
        if reward >= 1.0:
            return reward
        
        # Check wrong arm penalty (T-maze specific)
        wrong_goal = self.right_goal if self.current_reward_arm == 'left' else self.left_goal
        if self.agent.x == wrong_goal[0] and self.agent.y == wrong_goal[1]:
            return -0.5  # Wrong choice penalty
        
        return reward

    # ==================== Rendering ====================
    
    def _dir_to_radians(self) -> float:
        """Convert direction index (0-7) to radians."""
        # 0=E(0), 1=NE(π/4), 2=N(π/2), 3=NW(3π/4), 4=W(π), 5=SW(5π/4), 6=S(3π/2), 7=SE(7π/4)
        return self.agent.angle * np.pi / 4
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view using shared raycasting."""
        def overlay(img, agent_angle, fov):
            self._render_goal_in_fpv(img, self.goal_x, self.goal_y, self.goal_color, fov)
        
        return self._render_fpv_raycasting(
            ceiling_color=(150, 150, 150),
            floor_color=self.floor_color,
            max_dist=10.0,
            overlay_func=overlay
        )
    
    def _cast_ray(self, angle: float) -> float:
        """Cast ray using shared base implementation."""
        return super()._cast_ray(angle, max_dist=10.0, step_size=0.2)
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)  # Background
        
        scale = 25
        cx, cy = 112, 150  # Offset to show maze
        
        # Draw maze floor
        # Stem
        x1 = int(cx - self.corridor_width/2 * scale)
        x2 = int(cx + self.corridor_width/2 * scale)
        y1 = int(cy - 0 * scale)
        y2 = int(cy - self.stem_length * scale)
        self._draw_rect(img, x1, y2, x2, y1, self.floor_color)
        
        # Arms
        arm_y1 = int(cy - self.stem_length * scale)
        arm_y2 = int(cy - (self.stem_length + self.corridor_width) * scale)
        arm_x1 = int(cx - self.arm_length * scale)
        arm_x2 = int(cx + self.arm_length * scale)
        self._draw_rect(img, arm_x1, arm_y2, arm_x2, arm_y1, self.floor_color)
        
        # Draw walls (using shared _draw_line from base class)
        for wall in self.walls:
            wx1 = int(cx + wall['x1'] * scale)
            wy1 = int(cy - wall['y1'] * scale)
            wx2 = int(cx + wall['x2'] * scale)
            wy2 = int(cy - wall['y2'] * scale)
            NavigationEnvironment._draw_line(img, wx1, wy1, wx2, wy2, self.wall_color, 3)
        
        # Draw goal (using shared _draw_disk from base class)
        # Offset y by +0.5 to center grid cells within their continuous floor bands
        gx = int(cx + self.goal_x * scale)
        gy = int(cy - (self.goal_y + 0.5) * scale)
        NavigationEnvironment._draw_disk(img, gx, gy, 8, self.goal_color)
        
        # Draw agent
        ax = int(cx + self.agent.x * scale)
        ay = int(cy - (self.agent.y + 0.5) * scale)
        NavigationEnvironment._draw_disk(img, ax, ay, 5, (255, 100, 100))
        
        # Direction indicator
        agent_angle_rad = self._dir_to_radians()
        nose_x = int(ax + 8 * np.cos(agent_angle_rad))
        nose_y = int(ay - 8 * np.sin(agent_angle_rad))
        NavigationEnvironment._draw_disk(img, nose_x, nose_y, 3, (200, 50, 50))
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view - 1 world unit = 1 ASCII char."""
        pad = 2
        world_min_x = -self.arm_length
        world_max_x = self.arm_length
        world_min_y = 0
        world_max_y = self.stem_length
        
        grid_width = (world_max_x - world_min_x + 1) + 2 + 2 * pad
        grid_height = (world_max_y - world_min_y + 1) + 2 + 2 * pad
        
        c = AsciiCanvas(grid_width, grid_height)
        
        def world_to_grid(wx, wy):
            gx = pad + 1 + (wx - world_min_x)
            gy = pad + 1 + (world_max_y - wy)
            return int(gx), int(gy)
        
        # Draw walls around valid positions
        for y in range(self.stem_length):
            gx, gy = world_to_grid(0, y)
            c.put(gx - 1, gy, '#')
            c.put(gx + 1, gy, '#')
        
        # Bottom wall of stem
        gx, gy = world_to_grid(0, -1)
        for dx in [-1, 0, 1]:
            c.put(gx + dx, gy, '#')
        
        # Arms
        arm_y = self.stem_length
        for x in range(-self.arm_length, self.arm_length + 1):
            gx, gy = world_to_grid(x, arm_y)
            c.put(gx, gy - 1, '#')
            if x != 0:
                c.put(gx, gy + 1, '#')
        
        # End walls of arms
        for x in [-self.arm_length - 1, self.arm_length + 1]:
            gx, gy = world_to_grid(x, arm_y)
            c.put(gx, gy - 1, '#')
            c.put(gx, gy, '#')
            c.put(gx, gy + 1, '#')
        
        # Fill floor
        for y in range(self.stem_length):
            gx, gy = world_to_grid(0, y)
            if c.get(gx, gy) == ' ':
                c.put(gx, gy, '.')
        for x in range(-self.arm_length, self.arm_length + 1):
            gx, gy = world_to_grid(x, arm_y)
            if c.get(gx, gy) == ' ':
                c.put(gx, gy, '.')
        
        # Mark goal
        gx, gy = world_to_grid(self.goal_x, self.goal_y)
        c.put(gx, gy, 'G')
        
        # Draw agent
        gx, gy = world_to_grid(self.agent.x, self.agent.y)
        c.put_agent(gx, gy, self.agent.angle)
        
        return c.to_string()
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII pseudo-3D view with proper wall continuity."""
        agent_angle_rad = self._dir_to_radians()
        fov = np.pi / 2
        view_height = height - 2
        horizon = view_height // 2
        
        # Calculate goal visibility
        dx = self.goal_x - self.agent.x
        dy = self.goal_y - self.agent.y
        goal_dist = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx)
        goal_rel_angle = goal_angle - agent_angle_rad
        while goal_rel_angle > np.pi: goal_rel_angle -= 2*np.pi
        while goal_rel_angle < -np.pi: goal_rel_angle += 2*np.pi
        
        goal_col = None
        if abs(goal_rel_angle) < fov / 2:
            goal_col = int((0.5 - goal_rel_angle / fov) * (width - 3))
            if goal_col < 0 or goal_col >= width - 2:
                goal_col = None
        
        def overlay(row, col, char, dist, wall_top, wall_bottom):
            # Show goal marker if visible
            if goal_col is not None and col == goal_col and row == horizon:
                if goal_dist < dist:
                    return 'G'
            return char
        
        return self._render_3d_raycasting(
            width=width, height=height,
            fov=fov, agent_angle=agent_angle_rad,
            cast_ray_func=self._cast_ray,
            overlay_func=overlay
        )


def create_t_maze(
    trials_to_criterion: int = 20,
    trials_per_session: int = 5,
    reward_arm: str = "alternating",
    view_mode: ViewMode = ViewMode.FPV_3D,
    source_pmc: str = "",
    source_quote: str = ""
) -> TMaze:
    """Factory function to create T-Maze."""
    
    config = TMazeConfig(
        name="T-Maze",
        task_type="navigation",
        trials_to_criterion=trials_to_criterion,
        sessions_to_criterion=trials_to_criterion // trials_per_session,
        trials_per_session=trials_per_session,
        max_trial_steps=200,
        success_criterion="reach_correct_arm",
        arena_size=8.0,
        source_pmc=source_pmc,
        source_quote=source_quote,
        extra_params={
            'stem_length': 3.0,
            'arm_length': 2.0,
            'corridor_width': 1.0,
            'reward_arm': reward_arm
        }
    )
    
    return TMaze(config, view_mode)
