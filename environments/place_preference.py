"""
Conditioned Place Preference (CPP) environment for VLM evaluation.

Two-chamber apparatus where agent learns to associate one chamber with reward.
Tests reward learning and preference formation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import math

from .base_env import (
    BaseEnvironment, 
    EnvironmentConfig, 
    ViewMode, 
    Action,
    AgentState,
    SessionState,
    TrialResult
)


@dataclass
class PlacePreferenceConfig(EnvironmentConfig):
    """CPP specific configuration."""
    conditioning_chamber: int = 0  # Chamber paired with reward (0=left, 1=right)
    conditioning_trials: int = 6  # Number of conditioning sessions
    test_duration: int = 300  # Steps for preference test


class PlacePreference(BaseEnvironment):
    """
    Conditioned Place Preference (CPP) apparatus.
    
    Protocol: Two distinct chambers. During conditioning, one chamber is 
    paired with reward. Test measures time spent in each chamber.
    
    From verified protocols (PMC6101638 - Blanco-Gandía et al., J Vis Exp 2018):
    - "For place conditioning, use identical boxes made with two identical 
       compartments separated by a smaller central grey area."
    - "Compartments have different floor textures and wall colors (a smooth 
       floor in the black compartment and a rough floor in the white one)."
    - "The procedure consists of three phases: Pre-Conditioning (3 days), 
       Conditioning (4 days), and Post-Conditioning."
    """
    
    def __init__(self, 
                 config: Optional[PlacePreferenceConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = PlacePreferenceConfig(
                name="Place Preference",
                task_type="cpp",
                trials_to_criterion=12,
                sessions_to_criterion=6,
                trials_per_session=2,
                max_trial_steps=300,
                success_criterion="show_preference",
                arena_size=4.0,
                source_pmc="PMC6101638",
                source_quote="The procedure consists of three phases: Pre-Conditioning, Conditioning, and Post-Conditioning. Compartments have different floor textures and wall colors."
            )
        
        super().__init__(config, view_mode)
        
        # Task parameters
        self.conditioning_chamber = config.extra_params.get('conditioning_chamber', 0)
        self.conditioning_trials = config.extra_params.get('conditioning_trials', 6)
        self.test_duration = config.extra_params.get('test_duration', 300)
        
        # Phase: 'conditioning' or 'test'
        self.phase = 'conditioning'
        self.conditioning_count = 0
        
        # Chamber properties - distinct visual features
        self.chambers = [
            {
                'floor_pattern': 'stripes',
                'wall_color': (100, 80, 60),
                'floor_color': (180, 160, 140),
            },
            {
                'floor_pattern': 'dots',
                'wall_color': (60, 80, 100),
                'floor_color': (140, 160, 180),
            }
        ]
        
        # Time tracking for preference
        self.time_in_chamber = [0, 0]
        self.current_chamber = 0
        
        # Agent state
        self.agent = AgentState(x=0.0, y=0.0, angle=0.0)
        
        # Layout
        self.chamber_width = 2.0
        self.chamber_depth = 1.5
        
        # Actions
        self.valid_actions = [
            Action.FORWARD,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
            Action.STAY
        ]
    
    def _reset_agent_position(self):
        """Place agent at center (doorway between chambers)."""
        self.agent.x = 0.0
        self.agent.y = 0.0
        self.agent.angle = np.random.choice([0, np.pi])  # Face random direction
        
        # Reset time tracking for this trial
        self.time_in_chamber = [0, 0]
        
        # Update chamber based on position
        self.current_chamber = 0 if self.agent.x < 0 else 1
    
    def _setup_trial(self):
        """Setup for new trial."""
        # Check if entering test phase
        if self.phase == 'conditioning':
            self.conditioning_count += 1
            if self.conditioning_count > self.conditioning_trials:
                self.phase = 'test'
    
    def _execute_action(self, action: Action) -> float:
        """Execute action and return reward."""
        reward = 0.0
        
        # Movement using shared helper
        old_chamber = self.current_chamber
        self._move_continuous(action, speed=0.2, x_bounds=(-1.8, 1.8), y_bounds=(-0.7, 0.7))
        self.current_chamber = self._get_chamber()
        
        # Track time
        self.time_in_chamber[self.current_chamber] += 1
        
        # Reward in conditioning phase
        if self.phase == 'conditioning':
            if self.current_chamber == self.conditioning_chamber:
                reward = 0.1  # Small continuous reward in paired chamber
                self._trial_reward += 0.1
        
        return reward
    
    def _check_success(self) -> bool:
        """Success in test = preference for conditioned chamber."""
        if self.phase == 'test':
            total_time = sum(self.time_in_chamber)
            if total_time > 0:
                pref_ratio = self.time_in_chamber[self.conditioning_chamber] / total_time
                return pref_ratio > 0.6  # >60% time in conditioned chamber
        return False
    
    def _check_failure(self) -> bool:
        """No explicit failure."""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info."""
        base_info = super().get_info()
        total = sum(self.time_in_chamber) or 1
        base_info.update({
            'current_chamber': self.current_chamber,
            'conditioning_chamber': self.conditioning_chamber,
            'phase': self.phase,
            'time_in_left': self.time_in_chamber[0],
            'time_in_right': self.time_in_chamber[1],
            'preference_ratio': self.time_in_chamber[self.conditioning_chamber] / total,
            'conditioning_count': self.conditioning_count
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view with distinct chamber features."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        chamber = self.chambers[self.current_chamber]
        
        # Ceiling
        img[:70, :] = (100, 100, 100)
        
        # Walls
        img[70:154, :] = chamber['wall_color']
        
        # Floor with pattern
        floor_base = np.array(chamber['floor_color'])
        if chamber['floor_pattern'] == 'stripes':
            for y in range(154, 224):
                for x in range(224):
                    if (x // 20) % 2 == 0:
                        img[y, x] = floor_base
                    else:
                        img[y, x] = (floor_base * 0.7).astype(np.uint8)
        else:  # dots
            img[154:, :] = floor_base
            dot_color = tuple((floor_base * 0.6).astype(np.uint8))
            for y in range(160, 220, 15):
                for x in range(10, 220, 15):
                    self._draw_disk(img, x, y, 3, dot_color)
        
        # Door to other chamber (center)
        self._draw_rect(img, 87, 90, 137, 154, (40, 40, 40))
        
        # Phase indicator (using shared _draw_disk)
        if self.phase == 'conditioning' and self.current_chamber == self.conditioning_chamber:
            self._draw_disk(img, 112, 40, 15, (255, 215, 0))  # Gold for reward
        
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view using shared helpers."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Left/Right chambers
        ch0, ch1 = self.chambers[0], self.chambers[1]
        self._draw_rect(img, 20, 40, 107, 184, ch0['floor_color'])
        self._draw_rect(img, 117, 40, 204, 184, ch1['floor_color'])
        
        # Patterns - stripes in left chamber
        stripe_color = (int(ch0['floor_color'][0]*0.7), int(ch0['floor_color'][1]*0.7), int(ch0['floor_color'][2]*0.7))
        for x in range(20, 107, 12):
            self._draw_rect(img, x, 40, x+6, 184, stripe_color)
        
        # Dots in right chamber
        dot_color = (int(ch1['floor_color'][0]*0.6), int(ch1['floor_color'][1]*0.6), int(ch1['floor_color'][2]*0.6))
        for y in range(50, 180, 20):
            for x in range(127, 200, 20):
                self._draw_disk(img, x, y, 4, dot_color)
        
        # Door and walls
        self._draw_rect(img, 107, 90, 117, 134, (60, 60, 60))
        self._draw_rect(img, 18, 38, 206, 40, (80, 80, 80))  # Top
        self._draw_rect(img, 18, 184, 206, 186, (80, 80, 80))  # Bottom
        self._draw_rect(img, 18, 38, 20, 186, (80, 80, 80))  # Left
        self._draw_rect(img, 204, 38, 206, 186, (80, 80, 80))  # Right
        
        # Conditioning chamber marker
        if self.phase == 'conditioning':
            marker_x = 63 if self.conditioning_chamber == 0 else 160
            self._draw_disk(img, marker_x, 55, 8, (255, 215, 0))
        
        # Agent
        agent_x = int(112 + self.agent.x * 50)
        agent_y = int(112 - self.agent.y * 50)  # Flip Y
        self._draw_disk(img, agent_x, agent_y, 6, (0, 150, 255))
        
        return img
    
    def _render_ascii_2d(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII view."""
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        ch_top, ch_bot = 2, height - 3
        ch_left, ch_mid, ch_right = 2, width // 2, width - 3
        
        # Walls (all #)
        for x in range(ch_left, ch_right + 1):
            grid[ch_top][x] = '#'
            grid[ch_bot][x] = '#'
        for y in range(ch_top, ch_bot + 1):
            grid[y][ch_left] = '#'
            grid[y][ch_right] = '#'
        
        # Center divider with door
        door_y = height // 2
        for y in range(ch_top + 1, ch_bot):
            if abs(y - door_y) > 2:
                grid[y][ch_mid] = '#'
        
        # Goal marker in conditioning chamber
        if self.conditioning_chamber == 0:
            grid[ch_top + 2][ch_left + 2] = 'G'
        else:
            grid[ch_top + 2][ch_mid + 2] = 'G'
        
        # Agent
        agent_x = int(ch_left + 1 + (self.agent.x + 1.8) / 3.6 * (ch_right - ch_left - 2))
        agent_y = int(height // 2 - self.agent.y * 5)  # Flip Y: positive Y = UP on screen (lower row)
        agent_x = max(ch_left + 1, min(ch_right - 1, agent_x))
        agent_y = max(ch_top + 1, min(ch_bot - 1, agent_y))
        
        # Agent direction - 8 directions for finer angle gradation
        # Standard mapping: 0=E(→), 1=NE(↗), 2=N(↑), 3=NW(↖), 4=W(←), 5=SW(↙), 6=S(↓), 7=SE(↘)
        dirs = {0: '→', 1: '↗', 2: '↑', 3: '↖', 4: '←', 5: '↙', 6: '↓', 7: '↘'}
        dir_idx = int((self.agent.angle + np.pi/8) / (np.pi/4)) % 8
        agent_char = dirs.get(dir_idx, '@')
        grid[agent_y][agent_x] = agent_char
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII 3D view with raycasting-based perspective."""
        lines = []
        view_width = width - 2
        view_height = height - 2
        
        # Wall characters by distance
        def wall_char(dist: float) -> str:
            if dist < 1.5: return '█'
            if dist < 3.0: return '▓'
            if dist < 5.0: return '▒'
            if dist < 8.0: return '░'
            return '·'
        
        chamber = self.chambers[self.current_chamber]
        pattern = chamber['floor_pattern']
        
        # Different ceiling texture per chamber (visual cue)
        ceiling_char = '░' if self.current_chamber == 0 else '▒'
        
        # Chamber boundaries
        if self.current_chamber == 0:  # Left chamber
            wall_left = -self.chamber_width
            wall_right = 0
        else:  # Right chamber
            wall_left = 0
            wall_right = self.chamber_width
        
        wall_front = self.chamber_depth / 2
        wall_back = -self.chamber_depth / 2
        
        fov = np.pi / 2
        agent_angle = self.agent.angle
        
        # Cast rays
        ray_distances = []
        ray_hit_door = []
        for col in range(view_width):
            ray_offset = (col / view_width - 0.5) * fov
            ray_angle = agent_angle + ray_offset
            
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Distance to front/back walls
            if abs(dy) > 0.01:
                if dy > 0:
                    dist_y = (wall_front - self.agent.y) / dy
                else:
                    dist_y = (wall_back - self.agent.y) / dy
                dist_y = max(0.1, abs(dist_y))
            else:
                dist_y = 20
            
            # Distance to side walls / door
            hit_door = False
            if abs(dx) > 0.01:
                if dx > 0:
                    dist_x = (wall_right - self.agent.x) / dx
                else:
                    dist_x = (wall_left - self.agent.x) / dx
                dist_x = max(0.1, abs(dist_x))
                
                # Check if hitting door at x=0
                if (self.current_chamber == 0 and dx > 0) or (self.current_chamber == 1 and dx < 0):
                    hit_y = self.agent.y + dy * dist_x
                    if abs(hit_y) < 0.4:  # Door opening
                        hit_door = True
                        dist_x = 12
            else:
                dist_x = 20
            
            dist = min(dist_x, dist_y)
            dist *= np.cos(ray_offset)  # Fish-eye correction
            ray_distances.append(max(0.5, dist))
            ray_hit_door.append(hit_door)
        
        lines.append("╔" + "═" * view_width + "╗")
        
        for row in range(view_height):
            row_chars = []
            for col in range(view_width):
                dist = ray_distances[col]
                wall_height = int(view_height * 1.3 / (dist + 0.3))
                half_wall = wall_height // 2
                center = view_height // 2
                
                if row < center - half_wall:
                    char = ceiling_char
                elif row > center + half_wall:
                    # Floor with distinct pattern
                    if pattern == 'stripes':
                        char = '│' if (col % 2 == 0) else ' '
                    else:
                        char = '.' if (col + row) % 3 == 0 else ' '
                else:
                    if ray_hit_door[col]:
                        char = '░'  # Door opening
                    else:
                        char = wall_char(dist)
                
                row_chars.append(char)
            
            lines.append("║" + ''.join(row_chars) + "║")
        
        lines.append("╚" + "═" * view_width + "╝")
        
        return '\n'.join(lines)
