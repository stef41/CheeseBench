"""
Shuttle Box environment for VLM evaluation.

Two-chamber apparatus for fear conditioning and active/passive avoidance tasks.
Agent learns to shuttle between chambers to avoid aversive stimuli.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .base_env import (
    BaseEnvironment, 
    EnvironmentConfig, 
    ViewMode, 
    Action,
    AgentState,
    AsciiCanvas,
)


@dataclass
class ShuttleBoxConfig(EnvironmentConfig):
    """Shuttle Box specific configuration."""
    task_mode: str = "active_avoidance"  # "active_avoidance", "fear_conditioning", "escapable_shock"
    cue_duration: int = 5  # Steps of warning cue before shock
    shock_duration: int = 10  # Steps of shock if not escaped
    iti_duration: int = 5  # Inter-trial interval (reduced for LLM testing)
    escape_window: int = 15  # Steps allowed to escape after cue


class ShuttleBox(BaseEnvironment):
    """
    Shuttle Box / Two-Way Avoidance Apparatus.
    
    Protocol: Agent in two-chamber box. Warning cue signals upcoming shock.
    Agent must shuttle to other chamber to avoid/escape shock.
    
    From verified protocols (PMC4692667 - Happel et al., J Vis Exp 2015):
    - "A shuttle-box consists of 2 compartments separated by a hurdle or doorway."
    - "A conditioned stimulus (CS) is contingently followed by an aversive 
       unconditioned stimulus (US), as for instance a foot shock over a metal 
       grid floor."
    - "Subjects can learn to avoid the US by shuttling from one compartment to 
       the other in response to the CS."
    - "Classify a compartment change after CS+ onset within a critical time 
       window of 4 sec (CR) as a hit response."
    """
    
    def __init__(self, 
                 config: Optional[ShuttleBoxConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = ShuttleBoxConfig(
                name="Shuttle Box",
                task_type="avoidance",
                trials_to_criterion=40,
                sessions_to_criterion=2,
                trials_per_session=20,
                max_trial_steps=50,
                success_criterion="avoid_shock",
                arena_size=4.0,
                source_pmc="PMC4633642",
                source_quote="A conditioned stimulus (CS) is contingently followed by an aversive unconditioned stimulus (US). Subjects can learn to avoid the US by shuttling from one compartment to the other in response to the CS."
            )
        
        super().__init__(config, view_mode)
        
        # Task parameters
        self.task_mode = config.extra_params.get('task_mode', 'active_avoidance')
        self.cue_duration = config.extra_params.get('cue_duration', 5)
        self.shock_duration = config.extra_params.get('shock_duration', 10)
        self.iti_duration = config.extra_params.get('iti_duration', 5)  # Reduced for LLM testing
        self.escape_window = config.extra_params.get('escape_window', 15)
        
        # Chamber layout
        self.chamber_width = 2.0
        self.chamber_depth = 1.5
        self.door_width = 0.6
        
        # Current chamber (0 = left, 1 = right)
        self.current_chamber = 0
        self.shock_chamber = 0  # Chamber that will receive shock
        
        # Trial state
        self.trial_phase = 'iti'  # 'iti', 'cue', 'shock', 'escaped'
        self.phase_timer = 0
        self.shock_active = False
        self.cue_active = False
        
        # Performance tracking
        self.avoidances = 0  # Shuttled during cue (avoided shock)
        self.escapes = 0  # Shuttled during shock (escaped)
        self.failures = 0  # Did not shuttle (received full shock)
        
        # Agent state
        self.agent = AgentState(x=-1.0, y=0.0, angle=0.0)
        
        # Colors
        self.wall_color = (80, 80, 80)
        self.floor_color = (60, 60, 60)
        self.safe_floor_color = (60, 80, 60)  # Greenish for safe
        self.shock_floor_color = (100, 60, 60)  # Reddish when shock
        self.cue_light_color = (255, 255, 100)
        self.door_color = (40, 40, 40)
        
        # Actions
        self.valid_actions = [
            Action.FORWARD,  # Move toward door
            Action.ROTATE_LEFT,
            Action.ROTATE_RIGHT,
            Action.STAY
        ]
    
    def _reset_agent_position(self):
        """Place agent in random chamber at trial start."""
        self.current_chamber = np.random.randint(0, 2)
        if self.current_chamber == 0:
            self.agent.x = -1.0
        else:
            self.agent.x = 1.0
        self.agent.y = 0.0
        self.agent.angle = 0.0 if self.current_chamber == 0 else np.pi
        
        # Shock will come in current chamber
        self.shock_chamber = self.current_chamber
        
        # Reset trial state
        self.trial_phase = 'iti'
        self.phase_timer = self.iti_duration
        self.shock_active = False
        self.cue_active = False
        self._trial_shuttled = False  # Track if agent shuttled this trial
    
    def _setup_trial(self):
        """Setup for new trial."""
        pass
    
    def _update_trial_phase(self):
        """Update trial phase based on timer."""
        self.phase_timer -= 1
        
        if self.trial_phase == 'iti':
            if self.phase_timer <= 0:
                self.trial_phase = 'cue'
                self.phase_timer = self.cue_duration
                self.cue_active = True
        
        elif self.trial_phase == 'cue':
            if self.phase_timer <= 0:
                self.trial_phase = 'shock'
                self.phase_timer = self.shock_duration
                self.shock_active = True
                self.cue_active = False
        
        elif self.trial_phase == 'shock':
            if self.phase_timer <= 0:
                self.failures += 1
                self.trial_phase = 'escaped'  # Trial ends
                self.shock_active = False
    
    def _execute_action(self, action: Action) -> float:
        """Execute action and return reward."""
        reward = 0.0
        
        # Update phase
        self._update_trial_phase()
        
        # Movement using shared helper
        # y_bounds added to constrain movement within chamber depth
        old_chamber = self.current_chamber
        self._move_continuous(action, speed=0.3, x_bounds=(-1.8, 1.8), y_bounds=(-0.6, 0.6))
        self.current_chamber = self._get_chamber()
        
        # Did agent shuttle?
        if old_chamber != self.current_chamber:
            if self.current_chamber != self.shock_chamber:
                # Successfully moved to safe chamber
                if self.trial_phase == 'cue':
                    # Avoidance - shuttled during cue
                    self.avoidances += 1
                    self._trial_shuttled = True
                    reward = 1.0
                    self._trial_reward += 1.0  # Track for success criterion
                    self.trial_phase = 'escaped'
                    self.cue_active = False
                elif self.trial_phase == 'shock':
                    # Escape - shuttled during shock
                    self.escapes += 1
                    self._trial_shuttled = True
                    reward = 0.5
                    self._trial_reward += 0.5  # Track for success criterion
                    self.trial_phase = 'escaped'
                    self.shock_active = False
        
        # Shock penalty
        if self.shock_active and self.current_chamber == self.shock_chamber:
            reward = -0.5
            self._trial_reward -= 0.5
        
        return reward
    
    def _check_success(self) -> bool:
        """Success = avoided or escaped shock (shuttled to safe chamber)."""
        return self.trial_phase == 'escaped' and self._trial_shuttled
    
    def _check_failure(self) -> bool:
        """Failure = received full shock without escaping."""
        return self.trial_phase == 'escaped' and not self._trial_shuttled
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info."""
        base_info = super().get_info()
        base_info.update({
            'current_chamber': self.current_chamber,
            'shock_chamber': self.shock_chamber,
            'trial_phase': self.trial_phase,
            'cue_active': self.cue_active,
            'shock_active': self.shock_active,
            'avoidances': self.avoidances,
            'escapes': self.escapes,
            'failures': self.failures
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view with raycasting based on agent position and angle."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Chamber boundaries
        chamber_half_depth = self.chamber_depth / 2
        if self.current_chamber == 0:
            wall_left = -self.chamber_width
            wall_right = 0
        else:
            wall_left = 0
            wall_right = self.chamber_width
        
        fov = np.pi / 2
        agent_angle = self.agent.angle
        horizon = 112
        num_rays = 224
        
        # Floor color based on state
        if self.shock_active and self.current_chamber == self.shock_chamber:
            floor_col = self.shock_floor_color
        elif self.current_chamber != self.shock_chamber:
            floor_col = self.safe_floor_color
        else:
            floor_col = self.floor_color
        
        # Ceiling color differs per chamber
        ceiling_col = (100, 100, 110) if self.current_chamber == 0 else (90, 90, 80)
        
        # Draw ceiling and floor gradients
        for y in range(horizon):
            t = y / horizon
            shade = 0.4 + 0.6 * t
            img[y, :] = tuple(int(c * shade) for c in ceiling_col)
        for y in range(horizon, 224):
            t = (y - horizon) / (224 - horizon)
            shade = 0.4 + 0.6 * t
            img[y, :] = tuple(int(c * shade) for c in floor_col)
        
        # Cast rays
        for col in range(num_rays):
            ray_offset = (col / max(1, num_rays - 1) - 0.5) * fov
            ray_angle = agent_angle - ray_offset
            
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Distance to front/back walls
            if abs(dy) > 0.01:
                dist_y = ((chamber_half_depth if dy > 0 else -chamber_half_depth) - self.agent.y) / dy
                dist_y = max(0.1, dist_y)
            else:
                dist_y = 20
            
            # Distance to side walls / door
            hit_door = False
            if abs(dx) > 0.01:
                dist_x = ((wall_right if dx > 0 else wall_left) - self.agent.x) / dx
                dist_x = max(0.1, dist_x)
                
                if (self.current_chamber == 0 and dx > 0) or (self.current_chamber == 1 and dx < 0):
                    hit_y = self.agent.y + dy * dist_x
                    if abs(hit_y) < self.door_width / 2:
                        hit_door = True
                        dist_x = 15
            else:
                dist_x = 20
            
            dist = min(dist_x, dist_y)
            dist *= np.cos(ray_offset)  # Fish-eye correction
            dist = max(0.5, dist)
            
            wall_height = min(100, int(150 / (dist + 0.3)))
            y_top = horizon - wall_height
            y_bot = horizon + wall_height
            
            if hit_door:
                wall_col = self.door_color
            else:
                shade = max(40, min(255, int(200 / (dist + 0.3))))
                wall_col = (shade // 2, shade // 2, int(shade * 0.6))
            
            img[y_top:y_bot, col] = wall_col
        
        # Cue light (top center)
        if self.cue_active:
            self._draw_disk(img, 112, 30, 17, self.cue_light_color)
        
        # Shock indicator (red flash on edges)
        if self.shock_active and self.current_chamber == self.shock_chamber:
            img[:, :15] = (200, 50, 50)
            img[:, 209:] = (200, 50, 50)
        
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view using shared helpers."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Two chambers
        left_color = self.shock_floor_color if (self.shock_active and self.shock_chamber == 0) else self.floor_color
        right_color = self.shock_floor_color if (self.shock_active and self.shock_chamber == 1) else self.floor_color
        
        # Left/Right chambers and door (using _draw_rect)
        self._draw_rect(img, 20, 40, 105, 184, left_color)
        self._draw_rect(img, 119, 40, 204, 184, right_color)
        self._draw_rect(img, 105, 90, 119, 134, self.door_color)
        
        # Walls (outlines)
        self._draw_rect(img, 18, 38, 206, 40, self.wall_color)  # Top
        self._draw_rect(img, 18, 184, 206, 186, self.wall_color)  # Bottom
        self._draw_rect(img, 18, 38, 20, 186, self.wall_color)  # Left
        self._draw_rect(img, 204, 38, 206, 186, self.wall_color)  # Right
        # Center divider (with door gap)
        self._draw_rect(img, 105, 38, 107, 90, self.wall_color)
        self._draw_rect(img, 105, 134, 107, 186, self.wall_color)
        self._draw_rect(img, 117, 38, 119, 90, self.wall_color)
        self._draw_rect(img, 117, 134, 119, 186, self.wall_color)
        
        # Agent (using _draw_disk)
        agent_screen_x = int(62 + (self.agent.x + 1.8) / 3.6 * 140)
        agent_screen_y = int(112 - self.agent.y * 120)  # Scale for y_bounds ±0.6
        self._draw_disk(img, agent_screen_x, agent_screen_y, 6, (0, 150, 255))
        
        # Direction indicator
        dir_x = int(agent_screen_x + 10 * np.cos(self.agent.angle))
        dir_y = int(agent_screen_y - 10 * np.sin(self.agent.angle))
        self._draw_disk(img, dir_x, dir_y, 2, (255, 255, 255))
        
        # Cue light indicator
        if self.cue_active:
            shock_x = 62 if self.shock_chamber == 0 else 161
            self._draw_disk(img, shock_x, 55, 8, self.cue_light_color)
        
        return img
    
    def _render_ascii_2d(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII view."""
        c = AsciiCanvas(width, height)
        
        ch_top, ch_bot = 2, height - 3
        ch_left, ch_mid, ch_right = 2, width // 2, width - 3
        
        # Fill interior floor with dots so LLMs can distinguish floor from void
        c.fill_rect(ch_left + 1, ch_top + 1, ch_right - 1, ch_bot - 1, '.')
        
        # Walls
        c.hline(ch_left, ch_right, ch_top, '#')
        c.hline(ch_left, ch_right, ch_bot, '#')
        c.vline(ch_left, ch_top, ch_bot, '#')
        c.vline(ch_right, ch_top, ch_bot, '#')
        
        # Center divider with door
        door_top = height // 2 - 2
        door_bot = height // 2 + 2
        for y in range(ch_top + 1, ch_bot):
            if y < door_top or y > door_bot:
                c.put(ch_mid, y, '#')
        
        # Cue indicator and goal marker
        if self.cue_active or self.shock_active:
            shock_col = ch_left + 2 if self.shock_chamber == 0 else ch_mid + 2
            for row in range(ch_top + 1, ch_bot):
                c.put(shock_col, row, '!')
            
            safe_chamber = 1 - self.shock_chamber
            safe_x = ch_left + (ch_mid - ch_left) // 2 if safe_chamber == 0 else ch_mid + (ch_right - ch_mid) // 2
            c.put(safe_x, height // 2 - 1, 'G')
        
        # Agent position
        agent_x = int((self.agent.x + 1.8) / 3.6 * (ch_right - ch_left - 2)) + ch_left + 1
        usable_height = ch_bot - ch_top - 2
        agent_y = int(height // 2 - self.agent.y / 0.6 * (usable_height // 2))
        agent_x = max(ch_left + 1, min(ch_right - 1, agent_x))
        agent_y = max(ch_top + 1, min(ch_bot - 1, agent_y))
        
        dirs = {0: '→', 1: '↗', 2: '↑', 3: '↖', 4: '←', 5: '↙', 6: '↓', 7: '↘'}
        dir_idx = int((self.agent.angle + np.pi/8) / (np.pi/4)) % 8
        c.put(agent_x, agent_y, dirs.get(dir_idx, '@'))
        
        return c.to_string()
    
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
        
        # Chamber properties
        chamber_half_width = self.chamber_width / 2
        chamber_half_depth = self.chamber_depth / 2
        
        # Agent in left chamber (x<0) or right chamber (x>0)
        # Door passage at x=0
        if self.current_chamber == 0:
            wall_left = -self.chamber_width
            wall_right = 0
        else:
            wall_left = 0
            wall_right = self.chamber_width
        
        fov = np.pi / 2
        agent_angle = self.agent.angle
        
        # Cast rays
        ray_distances = []
        ray_hit_door = []
        for col in range(view_width):
            ray_offset = (col / max(1, view_width - 1) - 0.5) * fov
            ray_angle = agent_angle - ray_offset
            
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Distance to front/back walls
            if abs(dy) > 0.01:
                if dy > 0:
                    dist_y = (chamber_half_depth - self.agent.y) / dy
                else:
                    dist_y = (-chamber_half_depth - self.agent.y) / dy
                dist_y = max(0.1, dist_y)
            else:
                dist_y = 20
            
            # Distance to side walls / door
            hit_door = False
            if abs(dx) > 0.01:
                if dx > 0:
                    dist_x = (wall_right - self.agent.x) / dx
                else:
                    dist_x = (wall_left - self.agent.x) / dx
                dist_x = max(0.1, dist_x)
                
                # Check if hitting door (x=0 with door opening)
                if (self.current_chamber == 0 and dx > 0) or (self.current_chamber == 1 and dx < 0):
                    # Could be hitting door
                    hit_y = self.agent.y + dy * dist_x
                    if abs(hit_y) < self.door_width / 2:
                        hit_door = True
                        dist_x = 15  # See through door
            else:
                dist_x = 20
            
            dist = min(dist_x, dist_y)
            dist *= np.cos(ray_offset)  # Fish-eye correction
            ray_distances.append(max(0.5, dist))
            ray_hit_door.append(hit_door)
        
        # Visual cues
        ceiling_char = '░' if self.current_chamber == 0 else '▒'
        floor_char = '!' if self.shock_active and self.current_chamber == self.shock_chamber else '▓'
        
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
                    # Cue light warning
                    if self.cue_active and row < 3:
                        char = '*'
                elif row > center + half_wall:
                    char = floor_char
                else:
                    # Wall or door
                    if ray_hit_door[col]:
                        char = '▒'  # Door opening visible
                    else:
                        char = wall_char(dist)
                
                row_chars.append(char)
            
            lines.append("║" + ''.join(row_chars) + "║")
        
        lines.append("╚" + "═" * view_width + "╝")
        
        return '\n'.join(lines)

