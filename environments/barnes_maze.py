"""
Barnes Maze environment for VLM evaluation.

Circular platform with multiple holes, only one leads to escape box.
Tests spatial learning and memory using visual cues.

Uses INTEGER GRID coordinates for consistent movement.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base_env import (
    BaseEnvironment,
    EnvironmentConfig, 
    ViewMode, 
    Action,
    DIR_VECTORS,
    AsciiCanvas,
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
                source_pmc="PMC1783636",
                source_quote="B6C3F1/J mice were tested on a 12-hole Barnes maze over 5 sessions (4 trials/session). Primary errors decreased across sessions as mice learned to locate the escape hole."
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
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.STAY
        ]
    
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
            {'name': '1', 'x': self.center_x + wall_dist, 'y': self.center_y, 'angle': 0},           # East
            {'name': '2', 'x': self.center_x, 'y': self.center_y + wall_dist, 'angle': np.pi/2},    # North
            {'name': '3', 'x': self.center_x - wall_dist, 'y': self.center_y, 'angle': np.pi},      # West
            {'name': '4', 'x': self.center_x, 'y': self.center_y - wall_dist, 'angle': 3*np.pi/2},  # South
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
                
        elif action == Action.ROTATE_LEFT:
            self.agent.angle = (self.agent.angle + 1) % 8
            
        elif action == Action.ROTATE_RIGHT:
            self.agent.angle = (self.agent.angle - 1) % 8
        
        # Update path length
        moved = abs(self.agent.x - old_x) + abs(self.agent.y - old_y)
        self._trial_path_length += moved
        
        # Auto-enter hole when on it (check every step including STAY)
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
        
        return -0.01  # Time penalty (aversive light)
    
    def _check_success(self) -> bool:
        """Success = stepped on escape hole."""
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
        """Render first-person view using raycasting."""
        def wall_color(dist):
            # Barnes maze is open - walls are just the edge
            shade = max(80, 255 - int(dist * 10))
            return (shade, shade, int(shade * 0.9))
        
        def overlay(img, agent_angle, fov):
            # Render escape hole (goal)
            escape_hole = self.holes[self.escape_hole_index]
            self._render_goal_in_fpv(img, escape_hole['x'], escape_hole['y'], (50, 200, 50),
                                     fov, horizon=112, y_min=70, y_max=154)
            # Render other holes as darker circles
            for hole in self.holes:
                if hole['index'] != self.escape_hole_index:
                    self._render_goal_in_fpv(img, hole['x'], hole['y'], (80, 80, 80),
                                             fov, horizon=112, y_min=70, y_max=154)
        
        return self._render_fpv_raycasting(
            ceiling_color=(255, 255, 240),  # Bright (aversive light)
            floor_color=(220, 220, 220),
            wall_color_func=wall_color,
            max_dist=15.0,
            overlay_func=overlay
        )
    
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
        
        # Direction indicator (nose)
        dx, dy = DIR_VECTORS[self.agent.angle]
        half = scale // 2
        nose_cx = ax + half + dx * half
        nose_cy = ay + half - dy * half  # screen Y is flipped
        BaseEnvironment._draw_disk(img, nose_cx, nose_cy, max(2, scale // 4), (200, 50, 50))
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view with integer grid."""
        width = self.grid_size + 6
        height = self.grid_size + 6
        
        c = AsciiCanvas(width, height)
        
        def to_display(gx, gy):
            dx = gx - self.center_x + width // 2
            dy = self.center_y - gy + height // 2
            return dx, dy
        
        # Draw walls (rings around platform)
        for wall_offset in [0.5, 1.5]:
            for angle_i in range(72):
                angle = 2 * np.pi * angle_i / 72
                wall_r = self.platform_radius + wall_offset
                wx = self.center_x + wall_r * np.cos(angle)
                wy = self.center_y + wall_r * np.sin(angle)
                dx, dy = to_display(int(round(wx)), int(round(wy)))
                c.put(dx, dy, '#')
        
        # Draw platform floor
        for (gx, gy) in self.valid_positions:
            dx, dy = to_display(gx, gy)
            if c.get(dx, dy) == ' ':
                c.put(dx, dy, '.')
        
        # Draw holes
        for hole in self.holes:
            dx, dy = to_display(hole['x'], hole['y'])
            if hole['is_escape'] and self._can_see_escape_hole(hole['x'], hole['y']):
                c.put(dx, dy, 'E')
            else:
                c.put(dx, dy, '?')
        
        # Draw landmarks with wall behind them
        landmark_chars = ['1', '2', '3', '4']
        landmark_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, lm in enumerate(self.landmarks):
            dx, dy = to_display(lm['x'], lm['y'])
            c.put(dx, dy, landmark_chars[i])
            dir_x, dir_y = landmark_dirs[i]
            for offset in range(1, 3):
                wall_x = lm['x'] + dir_x * offset
                wall_y = lm['y'] + dir_y * offset
                wx, wy = to_display(wall_x, wall_y)
                c.put(wx, wy, '#')
        
        # Draw agent
        dx, dy = to_display(self.agent.x, self.agent.y)
        c.put_agent(dx, dy, self.agent.angle)
        
        return c.to_string()
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII pseudo-3D view from agent perspective on open platform."""
        lines = []
        
        # Header
        lines.append(f"╔{'═' * (width-2)}╗")
        
        view_height = height - 2  # Leave room for frame
        horizon = view_height // 2  # Middle of screen
        
        # Agent direction in radians
        agent_rad = self.agent.angle * np.pi / 4
        fov = np.pi / 2  # 90 degree field of view
        
        # Calculate distance to edge for each column
        edge_data = []
        for col in range(width - 2):
            ray_offset = (col / (width - 3)) - 0.5  # -0.5 to 0.5
            ray_angle = agent_rad - ray_offset * fov
            
            # Find where ray hits platform edge
            cos_a, sin_a = np.cos(ray_angle), np.sin(ray_angle)
            
            # Ray: (agent.x + t*cos_a, agent.y + t*sin_a)
            # Platform: (x - center_x)^2 + (y - center_y)^2 = r^2
            # Solve for t
            ax = self.agent.x - self.center_x
            ay = self.agent.y - self.center_y
            
            a = 1  # cos^2 + sin^2
            b = 2 * (ax * cos_a + ay * sin_a)
            c = ax*ax + ay*ay - self.platform_radius**2
            
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                # We want the positive intersection in front of us
                dist = max(0.5, min(t1, t2) if min(t1, t2) > 0 else max(t1, t2))
            else:
                dist = 15  # No intersection
            
            edge_data.append((dist * np.cos(ray_offset * fov), dist))  # (corrected, raw)
        
        # Calculate visible objects (holes and landmarks)
        visible_objects = {}  # col -> (char, distance)
        
        # Check holes
        for hole in self.holes:
            dx = hole['x'] - self.agent.x
            dy = hole['y'] - self.agent.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            rel_angle = np.arctan2(dy, dx) - agent_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov/2 and dist < 12:
                col = int((0.5 - rel_angle / fov) * (width - 3))
                if 0 <= col < width - 2:
                    # Show 'E' for escape hole only when close
                    if hole['is_escape'] and dist <= 2.0:
                        char = 'E'
                    else:
                        char = '?'
                    if col not in visible_objects or dist < visible_objects[col][1]:
                        visible_objects[col] = (char, dist)
        
        # Check landmarks
        for lm in self.landmarks:
            dx = lm['x'] - self.agent.x
            dy = lm['y'] - self.agent.y
            dist = np.sqrt(dx*dx + dy*dy)
            
            rel_angle = np.arctan2(dy, dx) - agent_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov/2:
                col = int((0.5 - rel_angle / fov) * (width - 3))
                if 0 <= col < width - 2:
                    if col not in visible_objects or dist < visible_objects[col][1]:
                        visible_objects[col] = (lm['name'], dist)
        
        # Render each row
        for row in range(view_height):
            line = '║'
            for col in range(width - 2):
                edge_dist, raw_edge_dist = edge_data[col]
                
                # Check for visible objects at ground level (horizon + a bit)
                if col in visible_objects and horizon <= row <= horizon + 2:
                    char, obj_dist = visible_objects[col]
                    if obj_dist < raw_edge_dist:
                        line += char
                        continue
                
                if row < horizon - 2:
                    # Sky - bright/light
                    sky_height = horizon - 2 - row
                    if sky_height > view_height // 4:
                        line += '~'  # High sky
                    else:
                        line += '░'  # Near horizon sky
                elif row == horizon - 2 or row == horizon - 1:
                    # Horizon line with edge markers
                    if edge_dist < 3:
                        line += '█'  # Close edge
                    elif edge_dist < 6:
                        line += '▓'  # Medium edge
                    elif edge_dist < 10:
                        line += '▒'  # Far edge
                    else:
                        line += '░'  # Very far/no edge
                else:
                    # Ground/floor
                    floor_depth = row - horizon
                    if floor_depth < view_height // 6:
                        line += '·'  # Near floor
                    elif floor_depth < view_height // 3:
                        line += '.'  # Mid floor
                    else:
                        line += ','  # Far floor
            
            line += '║'
            lines.append(line)
        
        # Footer
        lines.append(f"╚{'═' * (width-2)}╝")
        
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
