"""
Barnes Maze environment for VLM evaluation.

Circular platform with multiple holes, only one leads to escape box.
Tests spatial learning and memory using visual cues.

Uses INTEGER GRID coordinates for consistent movement.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

from .base_env import (
    BaseEnvironment,
    EnvironmentConfig, 
    ViewMode, 
    Action,
    AgentState,
    DIR_VECTORS
)


@dataclass  
class BarnesMazeConfig(EnvironmentConfig):
    """Barnes Maze specific configuration."""
    platform_radius: int = 6  # Grid cells from center to edge
    num_holes: int = 12  # Number of holes around perimeter
    escape_hole_index: int = 0
    
    def __post_init__(self):
        # Ensure extra_params has our config values
        if not self.extra_params:
            self.extra_params = {}
        self.extra_params.setdefault('platform_radius', self.platform_radius)
        self.extra_params.setdefault('num_holes', self.num_holes)
        self.extra_params.setdefault('escape_hole_index', self.escape_hole_index)


class BarnesMaze(BaseEnvironment):
    """
    Barnes Maze environment - INTEGER GRID based.
    
    Protocol: Circular platform with holes around perimeter.
    One hole leads to escape box. Uses aversive stimuli (bright light)
    and visual cues to learn escape location.
    
    Grid layout:
    - Platform is a discrete circle of walkable cells
    - Holes are at specific grid positions on the edge
    - Agent moves 1 cell per FORWARD action
    - 8 directions (N/NE/E/SE/S/SW/W/NW)
    """
    
    def __init__(self, 
                 config: Optional[BarnesMazeConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = BarnesMazeConfig(
                name="Barnes Maze",
                task_type="navigation",
                trials_to_criterion=16,
                sessions_to_criterion=4,
                trials_per_session=4,
                max_trial_steps=300,
                success_criterion="find_escape_hole",
                arena_size=14,
                source_pmc="PMC6126525",
                source_quote="The Barnes maze consists of a circular platform with 20 equidistant holes, 19 closed with plugs while the remaining hole leads to an escape shelter."
            )
        
        super().__init__(config, view_mode)
        
        # Platform setup - INTEGER grid
        self.platform_radius = config.extra_params.get('platform_radius', 6)
        
        # Create the circular platform grid
        self.grid_size = self.platform_radius * 2 + 3  # Extra space for walls
        self._create_platform()
        
        # Holes around perimeter
        self.num_holes = config.extra_params.get('num_holes', 12)
        self.escape_hole_index = config.extra_params.get('escape_hole_index', 0)
        self.holes = self._create_holes()
        
        # Goal = escape hole position
        escape_hole = self.holes[self.escape_hole_index]
        self.goal_x = escape_hole['x']
        self.goal_y = escape_hole['y']
        
        # Visual cues (landmarks) - placed at cardinal directions
        self.landmarks = self._create_landmarks()
        
        # Tracking
        self.hole_visits = [0] * self.num_holes
        self.holes_checked = [False] * self.num_holes
        self.found_escape = False
        
        # Valid actions
        self.valid_actions = [
            Action.FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
            Action.INTERACT,  # Check hole
            Action.STAY
        ]
        
        self.action_names[Action.INTERACT] = "check_hole"
    
    def _create_platform(self):
        """Create the circular platform as a set of valid positions."""
        self.valid_positions = set()
        center = self.platform_radius + 1  # Center of grid
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Distance from center
                dx = x - center
                dy = y - center
                dist = np.sqrt(dx*dx + dy*dy)
                # Inside platform circle
                if dist <= self.platform_radius:
                    self.valid_positions.add((x, y))
        
        self.center_x = center
        self.center_y = center
    
    def _create_holes(self) -> List[Dict[str, Any]]:
        """Create holes at grid positions around platform perimeter."""
        holes = []
        
        # Place holes evenly around the circle at radius-1 from center
        hole_radius = self.platform_radius - 1
        
        for i in range(self.num_holes):
            angle = 2 * np.pi * i / self.num_holes
            # Round to nearest grid cell
            x = self.center_x + int(round(hole_radius * np.cos(angle)))
            y = self.center_y + int(round(hole_radius * np.sin(angle)))
            
            holes.append({
                'index': i,
                'x': x,
                'y': y,
                'angle': angle,
                'is_escape': i == self.escape_hole_index
            })
        
        return holes
    
    def _create_landmarks(self) -> List[Dict[str, Any]]:
        """Create visual cues at cardinal directions on the wall.
        
        Landmarks are placed at platform_radius + 1, which is the INNER edge
        of the wall. This ensures they're visible in FPV before rays hit the
        wall characters behind them.
        """
        wall_dist = self.platform_radius + 1  # Inner edge of wall
        
        landmarks = [
            {'name': 'A', 'x': self.center_x + wall_dist, 'y': self.center_y, 'angle': 0},           # East
            {'name': 'B', 'x': self.center_x, 'y': self.center_y + wall_dist, 'angle': np.pi/2},    # North
            {'name': 'C', 'x': self.center_x - wall_dist, 'y': self.center_y, 'angle': np.pi},      # West
            {'name': 'D', 'x': self.center_x, 'y': self.center_y - wall_dist, 'angle': 3*np.pi/2},  # South
        ]
        return landmarks
    
    def _reset_agent_position(self):
        """Start agent at platform center."""
        self.agent.x = self.center_x
        self.agent.y = self.center_y
        self.agent.angle = np.random.randint(0, 8)  # Random direction 0-7
        
        # Reset tracking
        self.hole_visits = [0] * self.num_holes
        self.holes_checked = [False] * self.num_holes
        self.found_escape = False
    
    def _setup_trial(self):
        """Setup for new trial."""
        pass
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if position is outside platform or would fall into hole."""
        if (x, y) not in self.valid_positions:
            return True
        # Can walk onto holes (need to check them), but not off edge
        return False
    
    def _get_hole_at(self, x: int, y: int) -> Optional[int]:
        """Get hole index at position, or None."""
        for hole in self.holes:
            if hole['x'] == x and hole['y'] == y:
                return hole['index']
        return None
    
    def _is_hole_position(self, x: int, y: int) -> bool:
        """Check if position is a hole."""
        return self._get_hole_at(x, y) is not None
    
    def _can_see_escape_hole(self, hole_x: int, hole_y: int) -> bool:
        """
        Check if agent can see the escape hole identity.
        Only visible when adjacent (touching the hole) AND looking toward it.
        """
        dx = hole_x - self.agent.x
        dy = hole_y - self.agent.y
        
        # Must be adjacent (1 cell away - touching the border)
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        
        # Must be looking toward it (within 90 degree FOV)
        agent_rad = self.agent.angle * np.pi / 4
        hole_angle = np.arctan2(dy, dx)
        rel_angle = hole_angle - agent_rad
        while rel_angle > np.pi: rel_angle -= 2*np.pi
        while rel_angle < -np.pi: rel_angle += 2*np.pi
        
        # Within ~90 degree field of view
        return abs(rel_angle) < np.pi / 2
    
    def _execute_action(self, action: Action) -> float:
        """Execute action using integer grid movement."""
        old_x, old_y = self.agent.x, self.agent.y
        
        if action == Action.FORWARD:
            # Move exactly 1 cell in current direction
            dx, dy = DIR_VECTORS[self.agent.angle]
            new_x = self.agent.x + dx
            new_y = self.agent.y + dy
            
            # Check collision
            if not self._check_collision_at(new_x, new_y):
                self.agent.x = new_x
                self.agent.y = new_y
            else:
                return -0.1  # Hit wall/edge
                
        elif action == Action.TURN_LEFT:
            self.agent.angle = (self.agent.angle + 1) % 8
            
        elif action == Action.TURN_RIGHT:
            self.agent.angle = (self.agent.angle - 1) % 8
            
        elif action == Action.INTERACT:
            # Check if on a hole
            hole_idx = self._get_hole_at(self.agent.x, self.agent.y)
            if hole_idx is not None:
                self.hole_visits[hole_idx] += 1
                self.holes_checked[hole_idx] = True
                
                if hole_idx == self.escape_hole_index:
                    self.found_escape = True
                    self._trial_reward += 1.0
                    return 1.0  # Found escape!
                else:
                    return -0.2  # Wrong hole
            return -0.05  # No hole here
            
        elif action == Action.STAY:
            return -0.02  # Time in bright light
        
        # Update path length
        moved = abs(self.agent.x - old_x) + abs(self.agent.y - old_y)
        self._trial_path_length += moved
        
        # Check if stepped onto escape hole
        hole_idx = self._get_hole_at(self.agent.x, self.agent.y)
        if hole_idx == self.escape_hole_index and not self.found_escape:
            # Give hint reward for finding it (still need INTERACT to escape)
            return 0.1
        
        return -0.01  # Time penalty (aversive light)
    
    def _check_success(self) -> bool:
        """Success = found escape hole via INTERACT."""
        return self.found_escape
    
    def _check_failure(self) -> bool:
        """No automatic failure."""
        return False
    
    def _get_trial_extra_info(self) -> Dict[str, Any]:
        """Extra info for trial result."""
        return {
            'holes_checked': sum(self.holes_checked),
            'wrong_holes': sum(self.hole_visits) - self.hole_visits[self.escape_hole_index]
        }
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:100, :] = (255, 255, 240)  # Bright ceiling
        img[124:, :] = (220, 220, 220)  # Platform floor
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view.
        
        Y-axis is flipped so that North (increasing Y) appears as UP on screen.
        Screen row = 224 - y * scale (higher Y = lower row index = higher on screen)
        """
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (180, 180, 180)
        
        scale = 224 // self.grid_size
        max_screen_y = 224 - scale  # Highest valid screen Y
        
        # Helper to convert world Y to screen Y (flip Y-axis)
        def screen_y(y):
            return max_screen_y - y * scale
        
        # Draw platform
        for (x, y) in self.valid_positions:
            px, py = x * scale, screen_y(y)
            img[py:py+scale, px:px+scale] = (220, 220, 220)
        
        # Draw holes
        for hole in self.holes:
            px, py = hole['x'] * scale, screen_y(hole['y'])
            color = (0, 100, 0) if hole['is_escape'] else (50, 50, 50)
            img[py:py+scale, px:px+scale] = color
        
        # Draw agent
        ax, ay = self.agent.x * scale, screen_y(self.agent.y)
        img[ay:ay+scale, ax:ax+scale] = (255, 100, 100)
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view with integer grid."""
        # Grid dimensions for display (extra space for thick walls)
        width = self.grid_size + 6
        height = self.grid_size + 6
        
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Map grid coords to display coords
        def to_display(gx, gy):
            # Direct mapping, flip Y for visual
            dx = gx - self.center_x + width // 2
            dy = self.center_y - gy + height // 2
            return dx, dy
        
        # Draw thick walls (multiple rings around platform)
        for wall_offset in [0.5, 1.5, 2.5]:  # 3 layers of wall
            for angle_i in range(72):  # More points for smoother circle
                angle = 2 * np.pi * angle_i / 72
                wall_r = self.platform_radius + wall_offset
                wx = self.center_x + wall_r * np.cos(angle)
                wy = self.center_y + wall_r * np.sin(angle)
                dx, dy = to_display(int(round(wx)), int(round(wy)))
                if 0 <= dx < width and 0 <= dy < height:
                    grid[dy][dx] = '#'
        
        # Draw platform floor (needed for FPV to distinguish inside from outside)
        for (gx, gy) in self.valid_positions:
            dx, dy = to_display(gx, gy)
            if 0 <= dx < width and 0 <= dy < height:
                if grid[dy][dx] == ' ':
                    grid[dy][dx] = '.'
        
        # Draw holes - escape hole only revealed when close AND looking at it
        for hole in self.holes:
            dx, dy = to_display(hole['x'], hole['y'])
            if 0 <= dx < width and 0 <= dy < height:
                if hole['is_escape'] and self._can_see_escape_hole(hole['x'], hole['y']):
                    grid[dy][dx] = 'E'  # Escape hole visible!
                else:
                    grid[dy][dx] = 'O'  # All holes look the same from afar
        
        # Draw landmarks with wall directly behind them (no gap)
        landmark_chars = ['A', 'B', 'C', 'D']
        # Directions from center for each landmark: E, N, W, S
        landmark_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, lm in enumerate(self.landmarks):
            dx, dy = to_display(lm['x'], lm['y'])
            if 0 <= dx < width and 0 <= dy < height:
                grid[dy][dx] = landmark_chars[i]
            # Fill wall behind landmark (outward direction)
            dir_x, dir_y = landmark_dirs[i]
            for offset in range(1, 3):  # 2 layers of wall behind
                wall_x = lm['x'] + dir_x * offset
                wall_y = lm['y'] + dir_y * offset
                wx, wy = to_display(wall_x, wall_y)
                if 0 <= wx < width and 0 <= wy < height:
                    grid[wy][wx] = '#'
        
        # Draw agent
        dx, dy = to_display(self.agent.x, self.agent.y)
        if 0 <= dx < width and 0 <= dy < height:
            # Direction arrows: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
            arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
            grid[dy][dx] = arrows[self.agent.angle]
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self) -> str:
        """Render ASCII pseudo-3D view from agent perspective."""
        width, height = 40, 12
        lines = []
        
        # Sky/ceiling (bright light)
        lines.append('~' * width)
        lines.append('~' * width)
        
        # Horizon line with landmarks and walls
        horizon = [' '] * width
        
        # Agent direction in radians
        agent_rad = self.agent.angle * np.pi / 4
        fov = np.pi / 2  # 90 degree field of view
        
        # Draw landmarks on horizon
        for lm in self.landmarks:
            dx = lm['x'] - self.agent.x
            dy = lm['y'] - self.agent.y
            angle = np.arctan2(dy, dx) - agent_rad
            while angle > np.pi: angle -= 2*np.pi
            while angle < -np.pi: angle += 2*np.pi
            
            if abs(angle) < fov/2:
                screen_x = int(width/2 - angle / (fov/2) * (width/2 - 2))
                if 0 <= screen_x < width:
                    horizon[screen_x] = lm['name']
        
        # Draw walls on horizon (platform edge)
        for i in range(36):
            wall_angle = 2 * np.pi * i / 36
            wall_r = self.platform_radius + 0.5
            wx = self.center_x + wall_r * np.cos(wall_angle)
            wy = self.center_y + wall_r * np.sin(wall_angle)
            
            dx = wx - self.agent.x
            dy = wy - self.agent.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            rel_angle = np.arctan2(dy, dx) - agent_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov/2 and dist < 8:
                screen_x = int(width/2 - rel_angle / (fov/2) * (width/2 - 2))
                if 0 <= screen_x < width and horizon[screen_x] == ' ':
                    # Wall height based on distance
                    horizon[screen_x] = '#' if dist < 4 else '.'
        
        lines.append(''.join(horizon))
        
        # Ground with visible holes
        ground = ['.'] * width
        for hole in self.holes:
            dx = hole['x'] - self.agent.x
            dy = hole['y'] - self.agent.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            rel_angle = np.arctan2(dy, dx) - agent_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov/2 and dist < 6:
                screen_x = int(width/2 - rel_angle / (fov/2) * (width/2 - 2))
                if 0 <= screen_x < width:
                    # Only reveal escape hole when adjacent (touching)
                    if hole['is_escape'] and dist <= 1.5:
                        ground[screen_x] = 'E'
                    else:
                        ground[screen_x] = 'O'
        
        # Multiple ground rows
        for row in range(6):
            lines.append(''.join(ground))
        
        # Direction indicator at bottom
        arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
        indicator = f"Facing: {arrows[self.agent.angle]}"
        lines.append(indicator.center(width))
        
        return '\n'.join(lines)


def create_barnes_maze(
    trials_to_criterion: int = 16,
    trials_per_session: int = 4,
    num_holes: int = 12,
    platform_radius: int = 6,
    escape_hole_index: int = 0,
    view_mode: ViewMode = ViewMode.FPV_3D,
    source_pmc: str = "",
    source_quote: str = ""
) -> BarnesMaze:
    """Factory function to create Barnes Maze."""
    
    config = BarnesMazeConfig(
        name="Barnes Maze",
        task_type="navigation",
        trials_to_criterion=trials_to_criterion,
        sessions_to_criterion=trials_to_criterion // trials_per_session,
        trials_per_session=trials_per_session,
        max_trial_steps=300,
        success_criterion="find_escape_hole",
        arena_size=platform_radius * 2 + 2,
        source_pmc=source_pmc,
        source_quote=source_quote,
        extra_params={
            'platform_radius': platform_radius,
            'num_holes': num_holes,
            'escape_hole_index': escape_hole_index
        }
    )
    
    return BarnesMaze(config, view_mode)
