"""
Morris Water Maze environment for VLM evaluation.

Based on verified protocols from neuroscience literature.
Animal must navigate circular pool to find hidden escape platform.

Grid-based implementation: Uses integer grid coordinates like other environments.
The pool is represented as a circular region within a square grid.
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
class MorrisWaterMazeConfig(EnvironmentConfig):
    """Morris Water Maze specific configuration."""
    pool_radius: int = 9  # Grid cells from center to edge
    platform_visible: bool = False  # Hidden platform
    platform_quadrant: int = 0  # 0-3, which quadrant
    num_landmarks: int = 4
    start_positions: str = "random"  # "random", "fixed", "quadrant"
    

class MorrisWaterMaze(NavigationEnvironment):
    """
    Morris Water Maze environment (Grid-based).
    
    Protocol: Agent starts at pool edge, must find hidden platform.
    Uses distal visual cues (landmarks) for spatial navigation.
    
    Grid representation:
    - Pool is a circular region of radius pool_radius cells
    - Center is at (0, 0) in grid coordinates
    - Landmarks are placed ON the wall at cardinal directions
    - Platform is 1 cell
    
    From verified protocols (PMC2895266 - Vorhees & Williams, Nat Protoc 2006):
    - "The MWM is a test of spatial learning for rodents that relies on distal cues 
       to navigate from start locations around the perimeter of an open swimming arena 
       to locate a submerged escape platform."
    """
    
    def __init__(self, 
                 config: Optional[MorrisWaterMazeConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = MorrisWaterMazeConfig(
                name="Morris Water Maze",
                task_type="navigation",
                trials_to_criterion=20,  # 5 days × 4 trials (per Vorhees protocol)
                sessions_to_criterion=5,
                trials_per_session=4,    # Standard protocol
                max_trial_steps=500,
                success_criterion="reach_hidden_platform",
                arena_size=20,  # Grid size (diameter)
                source_pmc="PMC2895266",
                source_quote="The MWM is a test of spatial learning for rodents that relies on distal cues to navigate from start locations around the perimeter of an open swimming arena to locate a submerged escape platform."
            )
        
        super().__init__(config, view_mode)
        
        # Pool setup - INTEGER grid
        self.pool_radius = config.extra_params.get('pool_radius', 9)  # Grid cells
        self.pool_center = (0, 0)  # Grid center
        
        # Platform (hidden goal) - INTEGER position
        self.platform_quadrant = config.extra_params.get('platform_quadrant', 0)
        self.goal_visible = config.extra_params.get('platform_visible', False)
        self._set_platform_position()
        
        # Visual cues (landmarks on walls) - INTEGER positions
        self.landmarks = self._create_landmarks()
        
        # Water appearance (for pixel rendering)
        self.water_color = (100, 150, 200)
        self.wall_color = (180, 180, 180)
        
        # Start positions (pool edge)
        self.start_quadrants = [0, 1, 2, 3]
        self._current_start_quadrant = 0
        
        # Build the pool map
        self._build_pool_map()
        
        # Actions - use 8-direction system
        self.valid_actions = [
            Action.FORWARD,
            Action.ROTATE_LEFT, 
            Action.ROTATE_RIGHT,
            Action.STAY
        ]
        self.action_names[Action.FORWARD] = "swim forward"
        
    def _build_pool_map(self):
        """Build the circular pool as a grid map."""
        # Map size: diameter + margin for walls
        self.map_size = self.pool_radius * 2 + 3
        self.map_offset = self.map_size // 2  # Offset to convert grid coords to map indices
        
        # Initialize map: ' ' = outside, '~' = water, '#' = wall
        self.pool_map = [[' ' for _ in range(self.map_size)] for _ in range(self.map_size)]
        
        # Fill pool and walls
        for y in range(self.map_size):
            for x in range(self.map_size):
                # Convert to grid coords (centered at 0,0)
                gx = x - self.map_offset
                gy = y - self.map_offset
                dist = math.sqrt(gx*gx + gy*gy)
                
                if dist < self.pool_radius - 0.5:
                    self.pool_map[y][x] = '~'  # Water
                elif dist < self.pool_radius + 0.5:
                    self.pool_map[y][x] = '#'  # Wall
        
    def _set_platform_position(self):
        """Set hidden platform position based on quadrant (INTEGER coords)."""
        # Platform at ~60% radius from center in specified quadrant
        # Quadrant 0: NE (+x, +y), 1: NW (-x, +y), 2: SW (-x, -y), 3: SE (+x, -y)
        distance = int(self.pool_radius * 0.6)
        
        # Position based on quadrant (diagonal directions)
        quadrant_offsets = [
            (1, 1),   # 0: NE
            (-1, 1),  # 1: NW
            (-1, -1), # 2: SW
            (1, -1),  # 3: SE
        ]
        dx, dy = quadrant_offsets[self.platform_quadrant % 4]
        
        # Place platform at diagonal position
        diag_dist = int(distance / math.sqrt(2))
        self.goal_x = dx * diag_dist
        self.goal_y = dy * diag_dist
        self.goal_radius = 1  # Must be on exact cell
        
    def _create_landmarks(self) -> List[Dict[str, Any]]:
        """Create distal visual cues on pool wall at cardinal directions."""
        # Landmarks are ON the wall at cardinal directions
        # Position at pool_radius (on the wall circle)
        landmarks = [
            {"name": "1", "char": "1", "gx": self.pool_radius, "gy": 0, 
             "color": (255, 0, 0), "shape": "triangle"},      # East
            {"name": "2", "char": "2", "gx": 0, "gy": self.pool_radius,
             "color": (0, 255, 0), "shape": "circle"},        # North
            {"name": "3", "char": "3", "gx": -self.pool_radius, "gy": 0,
             "color": (0, 0, 255), "shape": "square"},        # West
            {"name": "4", "char": "4", "gx": 0, "gy": -self.pool_radius,
             "color": (255, 255, 0), "shape": "star"},        # South
        ]
        return landmarks
    
    def _reset_agent_position(self):
        """Start agent at pool edge in rotating quadrant (INTEGER coords)."""
        # Select quadrant: rotate through quadrants
        if hasattr(self, '_reset_count'):
            self._reset_count += 1
        else:
            self._reset_count = 0
        
        idx = (self.session.current_trial + self._reset_count) % len(self.start_quadrants)
        quadrant = self.start_quadrants[idx]
        
        # Start positions at pool edge (inside the wall)
        # Quadrants: 0=NE, 1=NW, 2=SW, 3=SE (diagonal positions)
        edge_dist = self.pool_radius - 2  # Inside the wall
        diag_dist = int(edge_dist / math.sqrt(2))
        
        start_positions = [
            (diag_dist, diag_dist, 5),      # 0: NE position, face SW (direction 5)
            (-diag_dist, diag_dist, 7),     # 1: NW position, face SE (direction 7)
            (-diag_dist, -diag_dist, 1),    # 2: SW position, face NE (direction 1)
            (diag_dist, -diag_dist, 3),     # 3: SE position, face NW (direction 3)
        ]
        
        self.agent.x, self.agent.y, self.agent.angle = start_positions[quadrant]
        self._current_start_quadrant = quadrant
    
    def _setup_trial(self):
        """Setup for new trial. Platform stays in same location."""
        pass  # Platform position is fixed during acquisition
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info including platform quadrant."""
        info = super().get_info()
        info['platform_quadrant'] = self.platform_quadrant
        info['goal_position'] = (self.goal_x, self.goal_y)
        info['distance_to_goal'] = self._distance_to(self.goal_x, self.goal_y)
        return info
    
    def _is_in_pool(self, x: int, y: int) -> bool:
        """Check if grid position is inside the pool (not wall or outside)."""
        dist = math.sqrt(x*x + y*y)
        return dist < self.pool_radius - 0.5
    
    def _check_collision_at(self, x: int, y: int) -> bool:
        """Check if position would collide with wall or outside pool."""
        return not self._is_in_pool(x, y)
    
    def _check_collision(self) -> bool:
        """Check if current position is in collision."""
        return self._check_collision_at(self.agent.x, self.agent.y)
    
    def _check_success(self) -> bool:
        """Success = on platform cell."""
        return self.agent.x == self.goal_x and self.agent.y == self.goal_y
    
    def _execute_action(self, action: Action) -> float:
        """Execute swimming action on grid (8-direction movement)."""
        old_x, old_y = self.agent.x, self.agent.y
        
        if action == Action.FORWARD:
            # Move one cell in current direction
            dx, dy = DIR_VECTORS[self.agent.angle]
            new_x = self.agent.x + dx
            new_y = self.agent.y + dy
            
            # Check collision before moving
            if not self._check_collision_at(new_x, new_y):
                self.agent.x, self.agent.y = new_x, new_y
            else:
                return -0.05  # Hit wall
                
        elif action == Action.ROTATE_LEFT:
            # Turn 45° counter-clockwise (increment direction index)
            self.agent.angle = (self.agent.angle + 1) % 8
        elif action == Action.ROTATE_RIGHT:
            # Turn 45° clockwise (decrement direction index)
            self.agent.angle = (self.agent.angle - 1) % 8
        elif action == Action.STAY:
            pass  # Treading water
        
        # Update path length
        moved = abs(self.agent.x - old_x) + abs(self.agent.y - old_y)
        self._trial_path_length += moved
        
        # Check platform
        if self._check_success():
            self._trial_reward += 1.0
            return 1.0  # Escape reward
        
        return -0.01  # Time in water penalty

    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view from water level."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Sky (top half)
        img[:112, :] = [135, 206, 235]  # Light blue
        
        # Water (bottom half)
        img[112:, :] = self.water_color
        
        # Convert grid direction to radians for rendering
        # Direction: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        agent_angle_rad = self.agent.angle * (math.pi / 4)
        
        # Render landmarks
        for lm in self.landmarks:
            # Calculate relative angle to landmark
            dx = lm["gx"] - self.agent.x
            dy = lm["gy"] - self.agent.y
            angle_to_lm = math.atan2(dy, dx)
            rel_angle = angle_to_lm - agent_angle_rad
            
            # Normalize to [-pi, pi]
            while rel_angle > np.pi:
                rel_angle -= 2 * np.pi
            while rel_angle < -np.pi:
                rel_angle += 2 * np.pi
            
            # Check if in FOV (90 degrees)
            fov = np.pi / 2
            if abs(rel_angle) < fov / 2:
                # Calculate screen position
                screen_x = int(112 - (rel_angle / (fov/2)) * 100)
                
                # Distance affects size
                dist = math.sqrt(dx**2 + dy**2)
                size = max(5, int(40 / (dist + 1)))
                
                # Draw landmark using shared helper
                self._draw_shape(img, lm["shape"], screen_x, 60, size, lm["color"], 0, 112)
        
        # Pool wall on horizon
        for x in range(224):
            screen_angle = (x - 112) / 100 * (np.pi / 4)
            world_angle = agent_angle_rad + screen_angle
            
            # Distance to wall in this direction
            wall_dist = self._distance_to_wall_from_angle(world_angle)
            if wall_dist < 20:
                wall_height = min(50, int(100 / (wall_dist + 1)))
                y_start = 112 - wall_height
                img[y_start:112, x] = self.wall_color
        
        return img
    
    def _distance_to_wall_from_angle(self, angle: float) -> float:
        """Calculate distance to pool wall in given direction (radians)."""
        # Ray-circle intersection from agent position
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Agent position relative to pool center
        ax = self.agent.x - self.pool_center[0]
        ay = self.agent.y - self.pool_center[1]
        
        # Quadratic formula for ray-circle
        a = dx**2 + dy**2
        b = 2 * (ax*dx + ay*dy)
        c = ax**2 + ay**2 - self.pool_radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 20.0
        
        t = (-b + math.sqrt(discriminant)) / (2*a)
        return max(0.1, t)
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view using shared helpers."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:] = (240, 240, 240)  # Background
        
        scale = 10  # pixels per grid cell
        cx, cy = 112, 112  # center of image
        
        # Draw pool (circle) - this needs pixel-level iteration for the circle edge
        for x in range(224):
            for y in range(224):
                gx = (x - cx) / scale
                gy = (y - cy) / scale
                dist = math.sqrt(gx**2 + gy**2)
                if dist < self.pool_radius - 0.5:
                    img[y, x] = self.water_color
                elif dist < self.pool_radius + 0.5:
                    img[y, x] = self.wall_color
        
        # Draw platform (using shared _draw_disk)
        px = int(cx + self.goal_x * scale)
        py = int(cy + self.goal_y * scale)
        pr = int(scale // 2)
        platform_color = (0, 200, 0) if self.goal_visible else (200, 200, 200)
        self._draw_disk(img, px, py, pr, platform_color)
        
        # Draw landmarks (using shared _draw_disk)
        for lm in self.landmarks:
            lx = int(cx + lm["gx"] * scale)
            ly = int(cy + lm["gy"] * scale)
            self._draw_disk(img, lx, ly, 5, lm["color"])
        
        # Draw agent (using shared _draw_disk)
        ax = int(cx + self.agent.x * scale)
        ay = int(cy + self.agent.y * scale)
        self._draw_disk(img, ax, ay, 4, (255, 100, 100))
        
        # Direction indicator
        agent_angle_rad = self.agent.angle * (math.pi / 4)
        nose_x = int(ax + 6 * math.cos(agent_angle_rad))
        nose_y = int(ay - 6 * math.sin(agent_angle_rad))
        self._draw_disk(img, nose_x, nose_y, 2, (200, 50, 50))
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """
        Render ASCII top-down view of circular pool.
        This is the SOURCE for _render_ascii_2d_fpv_grid in base class.
        
        Convention: Screen Y increases DOWN, but grid Y is UP (North positive).
        So we NEGATE Y when converting grid→screen.
        """
        # Fixed dimensions matching other environments
        width, height = 25, 25
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        cx, cy = width // 2, height // 2
        
        # Scale: map grid coordinates to ASCII coordinates
        # Pool radius is 9 grid cells, we want it to fit in ~10 char radius
        scale = 1.0  # 1 grid cell = 1 character
        
        # Draw pool: water inside, wall on edge
        for y in range(height):
            for x in range(width):
                # Convert ASCII position to grid coordinates
                # Screen y increases DOWN, grid y increases UP (North)
                gx = (x - cx) * scale
                gy = -(y - cy) * scale  # NEGATE: screen down = grid south
                dist = math.sqrt(gx*gx + gy*gy)
                
                if dist < self.pool_radius - 0.5:
                    grid[y][x] = '~'  # Water
                elif dist < self.pool_radius + 0.5:
                    grid[y][x] = '#'  # Wall
        
        # Draw landmarks ON the wall at cardinal positions
        for lm in self.landmarks:
            lx = int(cx + lm["gx"] / scale)
            ly = int(cy - lm["gy"] / scale)  # NEGATE: grid north = screen up
            if 0 <= lx < width and 0 <= ly < height:
                grid[ly][lx] = lm["char"]
        
        # Draw platform
        px = int(cx + self.goal_x / scale)
        py = int(cy - self.goal_y / scale)  # NEGATE
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = 'P'
        
        # Draw agent LAST (always visible)
        ax = int(cx + self.agent.x / scale)
        ay = int(cy - self.agent.y / scale)  # NEGATE
        if 0 <= ax < width and 0 <= ay < height:
            # 8-direction arrows matching DIR_VECTORS
            # 0=E(→), 1=NE(↗), 2=N(↑), 3=NW(↖), 4=W(←), 5=SW(↙), 6=S(↓), 7=SE(↘)
            dirs = {0: '→', 1: '↗', 2: '↑', 3: '↖', 4: '←', 5: '↙', 6: '↓', 7: '↘'}
            grid[ay][ax] = dirs.get(self.agent.angle, '@')
        
        return '\n'.join([''.join(row) for row in grid])
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII pseudo-3D first-person view of open water pool."""
        lines = []
        
        # Header
        lines.append(f"╔{'═' * (width-2)}╗")
        
        view_height = height - 2  # Leave room for frame
        horizon = view_height // 2  # Middle of screen
        
        # Calculate distance to platform
        dist_to_goal = self._distance_to_goal()
        near_platform = dist_to_goal < 3.0  # Within 3 cells
        
        # Convert direction to radians
        agent_angle_rad = self.agent.angle * (math.pi / 4)
        fov = np.pi / 2  # 90 degree FOV
        
        # Pool center
        center_x, center_y = self.pool_center
        
        # Calculate distance to pool edge for each column
        edge_data = []
        for col in range(width - 2):
            ray_offset = (col / (width - 3)) - 0.5
            ray_angle = agent_angle_rad - ray_offset * fov
            
            cos_a, sin_a = math.cos(ray_angle), math.sin(ray_angle)
            
            # Calculate intersection with pool edge
            ax = self.agent.x - center_x
            ay = self.agent.y - center_y
            
            a = 1
            b = 2 * (ax * cos_a + ay * sin_a)
            c = ax*ax + ay*ay - self.pool_radius**2
            
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                t1 = (-b + math.sqrt(discriminant)) / (2*a)
                t2 = (-b - math.sqrt(discriminant)) / (2*a)
                dist = max(0.5, min(t1, t2) if min(t1, t2) > 0 else max(t1, t2))
            else:
                dist = 15
            
            edge_data.append(dist * math.cos(ray_offset * fov))
        
        # Calculate visible landmarks
        landmark_cols = {}
        for lm in self.landmarks:
            dx = lm["gx"] - self.agent.x
            dy = lm["gy"] - self.agent.y
            dist = math.sqrt(dx**2 + dy**2)
            
            rel_angle = math.atan2(dy, dx) - agent_angle_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov / 2:
                col = int((0.5 - rel_angle / fov) * (width - 3))
                if 0 <= col < width - 2:
                    landmark_cols[col] = (lm["char"], dist)
        
        # Calculate platform visibility (only if near)
        platform_col = None
        if near_platform:
            dx = self.goal_x - self.agent.x
            dy = self.goal_y - self.agent.y
            rel_angle = math.atan2(dy, dx) - agent_angle_rad
            while rel_angle > np.pi: rel_angle -= 2*np.pi
            while rel_angle < -np.pi: rel_angle += 2*np.pi
            
            if abs(rel_angle) < fov / 2:
                platform_col = int((0.5 - rel_angle / fov) * (width - 3))
        
        # Render each row
        for row in range(view_height):
            line = '║'
            for col in range(width - 2):
                edge_dist = edge_data[col]
                
                # Check for landmarks at horizon
                if col in landmark_cols and row >= horizon - 3 and row <= horizon - 1:
                    line += landmark_cols[col][0]
                    continue
                
                # Check for platform visibility (at water level)
                if platform_col is not None and col >= platform_col - 1 and col <= platform_col + 1:
                    if row >= horizon and row <= horizon + 2:
                        line += '▓'
                        continue
                
                if row < horizon - 4:
                    # Sky
                    line += ' '
                elif row < horizon - 1:
                    # Near-horizon sky
                    line += '░'
                elif row == horizon - 1 or row == horizon:
                    # Horizon with pool edge
                    if edge_dist < 4:
                        line += '█'  # Close wall
                    elif edge_dist < 8:
                        line += '▓'  # Medium wall
                    elif edge_dist < 12:
                        line += '▒'  # Far wall
                    else:
                        line += '~'  # Very far (water/horizon)
                else:
                    # Water
                    water_depth = row - horizon
                    if water_depth < view_height // 6:
                        line += '≈' if (col + row) % 2 == 0 else '~'
                    elif water_depth < view_height // 3:
                        line += '~'
                    else:
                        line += '≈'
            
            line += '║'
            lines.append(line)
        
        # Footer
        lines.append(f"╚{'═' * (width-2)}╝")
        
        return '\n'.join(lines)


def create_morris_water_maze(
    trials_to_criterion: int = 15,
    trials_per_session: int = 3,
    platform_quadrant: int = 0,
    view_mode: ViewMode = ViewMode.FPV_3D,
    source_pmc: str = "PMC12765891",
    source_quote: str = "Training was conducted for five consecutive days, consisting of three trials per day."
) -> MorrisWaterMaze:
    """Factory function to create Morris Water Maze with verified parameters."""
    
    config = MorrisWaterMazeConfig(
        name="Morris Water Maze",
        task_type="navigation",
        trials_to_criterion=trials_to_criterion,
        sessions_to_criterion=trials_to_criterion // trials_per_session,
        trials_per_session=trials_per_session,
        max_trial_steps=500,
        success_criterion="reach_hidden_platform",
        arena_size=20,  # Grid size
        source_pmc=source_pmc,
        source_quote=source_quote,
        extra_params={
            'pool_radius': 9,  # Grid cells
            'platform_quadrant': platform_quadrant,
            'platform_visible': False
        }
    )
    
    return MorrisWaterMaze(config, view_mode)
