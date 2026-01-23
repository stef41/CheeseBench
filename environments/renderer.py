"""
Simple 2D/3D rendering utilities for VLM environments.
Renders FPV (first-person view) and top-down views.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class Wall:
    """A wall segment."""
    x1: float
    y1: float
    x2: float
    y2: float
    color: Tuple[int, int, int] = (128, 128, 128)
    height: float = 1.0


@dataclass
class Object:
    """An object in the environment."""
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int]
    name: str
    height: float = 0.5


@dataclass
class Goal:
    """Goal location."""
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int] = (0, 255, 0)
    visible: bool = True  # Hidden platform = False


class SimpleRenderer:
    """
    Simple renderer for 2D environments with pseudo-3D FPV.
    """
    
    def __init__(self, 
                 width: int = 224, 
                 height: int = 224,
                 world_size: float = 10.0):
        self.width = width
        self.height = height
        self.world_size = world_size
        
        # Environment elements
        self.walls: List[Wall] = []
        self.objects: List[Object] = []
        self.goals: List[Goal] = []
        self.floor_color = (200, 200, 200)
        self.sky_color = (135, 206, 235)
        
        # FPV settings
        self.fov = math.pi / 2  # 90 degrees
        self.view_distance = 15.0
        
    def clear(self):
        """Clear all elements."""
        self.walls = []
        self.objects = []
        self.goals = []
    
    def add_wall(self, x1: float, y1: float, x2: float, y2: float, 
                 color: Tuple[int, int, int] = (128, 128, 128)):
        """Add a wall segment."""
        self.walls.append(Wall(x1, y1, x2, y2, color))
    
    def add_rect_room(self, x: float, y: float, w: float, h: float,
                      color: Tuple[int, int, int] = (128, 128, 128)):
        """Add rectangular room walls."""
        self.add_wall(x, y, x + w, y, color)
        self.add_wall(x + w, y, x + w, y + h, color)
        self.add_wall(x + w, y + h, x, y + h, color)
        self.add_wall(x, y + h, x, y, color)
    
    def add_circular_arena(self, cx: float, cy: float, radius: float, 
                           segments: int = 32,
                           color: Tuple[int, int, int] = (128, 128, 128)):
        """Add circular arena walls."""
        for i in range(segments):
            a1 = 2 * math.pi * i / segments
            a2 = 2 * math.pi * (i + 1) / segments
            x1 = cx + radius * math.cos(a1)
            y1 = cy + radius * math.sin(a1)
            x2 = cx + radius * math.cos(a2)
            y2 = cy + radius * math.sin(a2)
            self.add_wall(x1, y1, x2, y2, color)
    
    def add_circular_boundary(self, cx: float, cy: float, radius: float,
                              segments: int = 32,
                              color: Tuple[int, int, int] = (100, 100, 100)):
        """Add circular boundary (alias for add_circular_arena)."""
        self.add_circular_arena(cx, cy, radius, segments, color)
    
    def add_object(self, x: float, y: float, radius: float,
                   color: Tuple[int, int, int], name: str):
        """Add an object."""
        self.objects.append(Object(x, y, radius, color, name))
    
    def add_goal(self, x: float, y: float, radius: float = 0.5,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 visible: bool = True):
        """Add a goal location."""
        self.goals.append(Goal(x, y, radius, color, visible))
    
    def render_topdown(self, agent_x: float, agent_y: float, 
                       agent_heading: float,
                       trail: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Render top-down 2D view.
        """
        img = np.full((self.height, self.width, 3), self.floor_color, dtype=np.uint8)
        
        # Scale factor
        scale = self.width / self.world_size
        
        def to_pixel(x, y):
            px = int(x * scale)
            py = int(self.height - y * scale)  # Flip Y
            return px, py
        
        # Draw trail if provided
        if trail and len(trail) > 1:
            for i in range(1, len(trail)):
                p1 = to_pixel(*trail[i-1])
                p2 = to_pixel(*trail[i])
                self._draw_line(img, p1, p2, (200, 200, 255), 1)
        
        # Draw walls
        for wall in self.walls:
            p1 = to_pixel(wall.x1, wall.y1)
            p2 = to_pixel(wall.x2, wall.y2)
            self._draw_line(img, p1, p2, wall.color, 2)
        
        # Draw goals
        for goal in self.goals:
            px, py = to_pixel(goal.x, goal.y)
            r = int(goal.radius * scale)
            if goal.visible:
                self._draw_circle(img, px, py, r, goal.color, filled=True)
            else:
                # Hidden goal - draw dashed circle
                self._draw_circle(img, px, py, r, (100, 100, 100), filled=False)
        
        # Draw objects
        for obj in self.objects:
            px, py = to_pixel(obj.x, obj.y)
            r = int(obj.radius * scale)
            self._draw_circle(img, px, py, r, obj.color, filled=True)
        
        # Draw agent
        ax, ay = to_pixel(agent_x, agent_y)
        self._draw_circle(img, ax, ay, 5, (255, 0, 0), filled=True)
        
        # Draw heading indicator
        hx = int(ax + 10 * math.cos(agent_heading))
        hy = int(ay - 10 * math.sin(agent_heading))  # Flip Y
        self._draw_line(img, (ax, ay), (hx, hy), (255, 0, 0), 2)
        
        return img
    
    def render_fpv(self, agent_x: float, agent_y: float, 
                   agent_heading: float) -> np.ndarray:
        """
        Render first-person 3D view using raycasting.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky and floor
        img[:self.height//2] = self.sky_color
        img[self.height//2:] = self.floor_color
        
        # Cast rays for each column
        for col in range(self.width):
            # Ray angle
            ray_offset = (col - self.width / 2) / self.width * self.fov
            ray_angle = agent_heading + ray_offset
            
            # Find closest wall intersection
            min_dist = self.view_distance
            wall_color = (128, 128, 128)
            
            for wall in self.walls:
                dist = self._ray_wall_intersection(
                    agent_x, agent_y, ray_angle,
                    wall.x1, wall.y1, wall.x2, wall.y2
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    wall_color = wall.color
            
            # Draw wall column
            if min_dist < self.view_distance:
                # Correct fisheye
                dist_corrected = min_dist * math.cos(ray_offset)
                
                # Wall height based on distance
                wall_height = min(self.height, int(self.height * 0.8 / max(0.1, dist_corrected)))
                
                # Shade based on distance
                shade = max(0.3, 1.0 - dist_corrected / self.view_distance)
                color = tuple(int(c * shade) for c in wall_color)
                
                # Draw column
                top = self.height // 2 - wall_height // 2
                bottom = self.height // 2 + wall_height // 2
                img[max(0, top):min(self.height, bottom), col] = color
        
        # Render objects and goals in view
        self._render_objects_fpv(img, agent_x, agent_y, agent_heading)
        
        return img
    
    def _render_objects_fpv(self, img: np.ndarray, 
                            agent_x: float, agent_y: float,
                            agent_heading: float):
        """Render objects in first-person view."""
        # Combine objects and visible goals
        items = [(o.x, o.y, o.radius, o.color, o.name) for o in self.objects]
        items += [(g.x, g.y, g.radius, g.color, "goal") for g in self.goals if g.visible]
        
        for x, y, radius, color, name in items:
            # Relative position
            dx = x - agent_x
            dy = y - agent_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 0.5 or dist > self.view_distance:
                continue
            
            # Angle to object
            obj_angle = math.atan2(dy, dx)
            rel_angle = obj_angle - agent_heading
            
            # Normalize to [-pi, pi]
            while rel_angle > math.pi:
                rel_angle -= 2 * math.pi
            while rel_angle < -math.pi:
                rel_angle += 2 * math.pi
            
            # Check if in FOV
            if abs(rel_angle) > self.fov / 2:
                continue
            
            # Screen position
            # Negate: positive rel_angle (left) -> smaller screen_x (left on screen)
            screen_x = int(self.width / 2 - rel_angle / self.fov * self.width)
            
            # Size based on distance
            apparent_size = int(self.width * radius / dist * 0.5)
            
            # Vertical position (on ground)
            screen_y = self.height // 2 + int(self.height * 0.2 / dist)
            
            # Draw object
            self._draw_circle(img, screen_x, screen_y, apparent_size, color, filled=True)
    
    def _ray_wall_intersection(self, rx: float, ry: float, angle: float,
                               x1: float, y1: float, x2: float, y2: float) -> Optional[float]:
        """Calculate ray-wall intersection distance."""
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Wall vector
        wx = x2 - x1
        wy = y2 - y1
        
        # Denominator (cross product of ray direction and wall vector)
        denom = dx * wy - dy * wx
        if abs(denom) < 1e-10:
            return None
        
        # Vector from ray origin to wall start
        qx = x1 - rx
        qy = y1 - ry
        
        # Parameters
        # t = distance along ray to intersection
        t = (qx * wy - qy * wx) / denom
        # u = position along wall segment (0 = start, 1 = end)
        u = (qx * dy - qy * dx) / denom
        
        # Valid intersection: ray goes forward (t > 0) and hits the wall segment (0 <= u <= 1)
        if t > 0 and 0 <= u <= 1:
            return t
        return None
    
    def _draw_line(self, img: np.ndarray, p1: Tuple[int, int], 
                   p2: Tuple[int, int], color: Tuple[int, int, int], 
                   thickness: int = 1):
        """Draw a line using Bresenham's algorithm."""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            for tx in range(-thickness//2, thickness//2 + 1):
                for ty in range(-thickness//2, thickness//2 + 1):
                    px, py = x1 + tx, y1 + ty
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        img[py, px] = color
            
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
    
    def _draw_circle(self, img: np.ndarray, cx: int, cy: int, 
                     radius: int, color: Tuple[int, int, int],
                     filled: bool = True):
        """Draw a circle."""
        for y in range(max(0, cy - radius), min(img.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(img.shape[1], cx + radius + 1)):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if filled:
                    if dist <= radius:
                        img[y, x] = color
                else:
                    if abs(dist - radius) < 1.5:
                        img[y, x] = color
