"""
Operant Chamber (Skinner Box) environment for VLM evaluation.

Lever press / nose poke task with various reinforcement schedules.
Tests instrumental conditioning and learning.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
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
class OperantChamberConfig(EnvironmentConfig):
    """Operant Chamber specific configuration."""
    num_levers: int = 1
    reward_schedule: str = "FR1"  # FR1, FR5, VR5, FI30, VI30
    schedule_parameter: int = 1
    has_cue_light: bool = True
    has_house_light: bool = True


class OperantChamber(BaseEnvironment):
    """
    Operant Chamber / Skinner Box environment.
    
    Protocol: Agent learns to press lever(s) for reward.
    Various reinforcement schedules: FR, VR, FI, VI.
    
    From verified protocols (PMC2895266 - Vorhees & Williams 2006 - behavioral testing):
    - "Operant conditioning tasks use lever presses or nose pokes to assess 
       instrumental learning and response-outcome associations."
    - "Various reinforcement schedules (fixed ratio, variable ratio, fixed interval, 
       variable interval) test different aspects of learning and motivation."
    """
    
    def __init__(self, 
                 config: Optional[OperantChamberConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = OperantChamberConfig(
                name="Operant Chamber",
                task_type="operant",
                trials_to_criterion=50,
                sessions_to_criterion=5,
                trials_per_session=10,
                max_trial_steps=100,
                success_criterion="earn_criterion_rewards",
                arena_size=2.0,
                source_pmc="PMC2895266",
                source_quote="Operant conditioning tasks use lever presses to assess instrumental learning with various reinforcement schedules."
            )
        
        super().__init__(config, view_mode)
        
        # Chamber setup
        self.chamber_width = 1.5
        self.chamber_depth = 1.0
        self.chamber_height = 1.2
        
        # Levers/response devices - protocol uses 2 levers with only one active
        # "Place mouse levers in the test chamber" - discrimination between active/inactive
        self.num_levers = config.extra_params.get('num_levers', 2)  # Default 2 levers per protocol
        self.levers = self._create_levers()
        self.active_lever = 0  # Index of rewarded lever (randomized on reset)
        
        # Reinforcement schedule
        self.schedule_type = config.extra_params.get('reward_schedule', 'FR1')
        self.schedule_param = config.extra_params.get('schedule_parameter', 1)
        self._parse_schedule()
        
        # Reward tracking
        self.lever_presses = [0] * self.num_levers
        self.responses_since_reward = 0
        self.time_since_reward = 0
        self.rewards_earned = 0
        
        # Cue lights - disabled for realistic learning (animals learn by trial and error)
        self.has_cue_light = False  # No discriminative stimulus - must learn through feedback
        self.has_house_light = config.extra_params.get('has_house_light', True)
        self.cue_light_on = False
        self.house_light_on = True
        
        # Food magazine
        self.magazine_x = 0.0
        self.magazine_lit = False
        
        # Agent position (limited movement in chamber)
        self.agent = AgentState(x=0.0, y=-0.3, angle=np.pi/2)
        
        # Colors
        self.wall_color = (100, 100, 100)
        self.lever_color = (180, 180, 200)
        self.lever_pressed_color = (120, 120, 140)
        self.light_on_color = (255, 255, 100)
        self.light_off_color = (80, 80, 60)
        
        # Actions - simpler for operant box
        self.valid_actions = [
            Action.FORWARD,   # Press currently faced lever
            Action.ROTATE_LEFT,  # Face left lever
            Action.ROTATE_RIGHT,  # Face right lever
            Action.STAY
        ]
        self.action_names[Action.FORWARD] = "press_lever"
        self.action_names[Action.ROTATE_LEFT] = "look_left"
        self.action_names[Action.ROTATE_RIGHT] = "look_right"
        
        # Current lever being looked at
        self.facing_lever = 0
        
        # Lever press animation
        self.lever_press_frames = [0] * self.num_levers
    
    def _parse_schedule(self):
        """Parse reinforcement schedule string."""
        if self.schedule_type.startswith('FR'):
            self.schedule_mode = 'fixed_ratio'
            self.ratio_requirement = int(self.schedule_type[2:]) if len(self.schedule_type) > 2 else self.schedule_param
        elif self.schedule_type.startswith('VR'):
            self.schedule_mode = 'variable_ratio'
            self.ratio_mean = int(self.schedule_type[2:]) if len(self.schedule_type) > 2 else self.schedule_param
            self._set_next_vr_requirement()
        elif self.schedule_type.startswith('FI'):
            self.schedule_mode = 'fixed_interval'
            self.interval_requirement = int(self.schedule_type[2:]) if len(self.schedule_type) > 2 else self.schedule_param
        elif self.schedule_type.startswith('VI'):
            self.schedule_mode = 'variable_interval'
            self.interval_mean = int(self.schedule_type[2:]) if len(self.schedule_type) > 2 else self.schedule_param
            self._set_next_vi_requirement()
        else:
            self.schedule_mode = 'fixed_ratio'
            self.ratio_requirement = 1
    
    def _set_next_vr_requirement(self):
        """Set next variable ratio requirement."""
        self.current_ratio_requirement = max(1, int(np.random.exponential(self.ratio_mean)))
    
    def _set_next_vi_requirement(self):
        """Set next variable interval requirement."""
        self.current_interval_requirement = max(1, int(np.random.exponential(self.interval_mean)))
    
    def _create_levers(self) -> List[Dict[str, Any]]:
        """Create lever positions."""
        levers = []
        if self.num_levers == 1:
            levers.append({
                'index': 0,
                'x': 0.0,
                'y': 0.4,
                'active': True
            })
        else:
            # Two levers, left and right
            levers.append({'index': 0, 'x': -0.4, 'y': 0.4, 'active': True})
            levers.append({'index': 1, 'x': 0.4, 'y': 0.4, 'active': False})
        return levers
    
    def _check_reward_criterion(self) -> bool:
        """Check if reward should be delivered."""
        if self.schedule_mode == 'fixed_ratio':
            return self.responses_since_reward >= self.ratio_requirement
        elif self.schedule_mode == 'variable_ratio':
            return self.responses_since_reward >= self.current_ratio_requirement
        elif self.schedule_mode == 'fixed_interval':
            return self.time_since_reward >= self.interval_requirement and self.responses_since_reward > 0
        elif self.schedule_mode == 'variable_interval':
            return self.time_since_reward >= self.current_interval_requirement and self.responses_since_reward > 0
        return False
    
    def _deliver_reward(self):
        """Deliver reward."""
        self.rewards_earned += 1
        self.responses_since_reward = 0
        self.time_since_reward = 0
        self.magazine_lit = True
        
        # Reset variable schedules
        if self.schedule_mode == 'variable_ratio':
            self._set_next_vr_requirement()
        elif self.schedule_mode == 'variable_interval':
            self._set_next_vi_requirement()
    
    # ==================== Abstract Method Implementations ====================
    
    def _reset_agent_position(self):
        """Reset agent (facing center, must choose lever)."""
        # Randomize which lever is active (discrimination learning)
        self.active_lever = np.random.randint(0, self.num_levers)
        
        # Agent starts facing center (between levers) - must choose direction
        self.facing_lever = -1  # -1 = facing center/magazine, not a lever
        
        # Set agent position in center of chamber (for rendering)
        self.agent.x = self.chamber_width / 2
        self.agent.y = self.chamber_height / 2
        self.agent.angle = np.pi / 2  # Face forward (toward levers)
        
        # Reset tracking
        self.lever_presses = [0] * self.num_levers
        self.responses_since_reward = 0
        self.time_since_reward = 0
        self.rewards_earned = 0
        
        # Reset lights - cue light indicates which lever is active
        self.house_light_on = True
        self.cue_light_on = self.has_cue_light  # Cue light near active lever
        self.magazine_lit = False
        self.lever_press_frames = [0] * self.num_levers
    
    def _setup_trial(self):
        """Setup for new trial."""
        pass  # Operant chamber doesn't need per-trial setup
    
    def _execute_action(self, action: Action) -> float:
        """Execute action and return reward."""
        self.time_since_reward += 1
        
        # Update lever animations
        for i in range(self.num_levers):
            if self.lever_press_frames[i] > 0:
                self.lever_press_frames[i] -= 1
        
        # Turn off magazine light after a step
        if self.magazine_lit and self._current_step % 3 == 0:
            self.magazine_lit = False
        
        reward = 0.0
        
        if action == Action.ROTATE_LEFT:
            # Move toward left lever (index 0)
            if self.facing_lever == -1:
                self.facing_lever = 0  # From center to left lever
            elif self.facing_lever > 0:
                self.facing_lever -= 1
        elif action == Action.ROTATE_RIGHT:
            # Move toward right lever (index 1 for 2-lever setup)
            if self.facing_lever == -1:
                self.facing_lever = self.num_levers - 1  # From center to right lever
            elif self.facing_lever < self.num_levers - 1:
                self.facing_lever += 1
        elif action == Action.FORWARD:
            # Press currently faced lever - only works if facing a lever
            if self.facing_lever == -1:
                # Facing magazine/center - pressing does nothing useful
                reward = -0.05  # Small penalty for pressing nothing
            elif 0 <= self.facing_lever < self.num_levers:
                self.lever_presses[self.facing_lever] += 1
                self.lever_press_frames[self.facing_lever] = 3
                
                if self.facing_lever == self.active_lever:
                    self.responses_since_reward += 1
                    
                    if self._check_reward_criterion():
                        self._deliver_reward()
                        self._trial_reward += 1.0
                        reward = 1.0
                else:
                    reward = -0.2  # Inactive lever - wrong choice
        elif action == Action.STAY:
            reward = -0.01
        
        return reward
    
    def _check_success(self) -> bool:
        """Success = earned enough rewards."""
        criterion_rewards = self.config.extra_params.get('criterion_rewards', 10)
        return self.rewards_earned >= criterion_rewards
    
    def _check_failure(self) -> bool:
        """No failure condition besides timeout."""
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info."""
        base_info = super().get_info()
        base_info.update({
            'lever_presses': self.lever_presses.copy(),
            'rewards_earned': self.rewards_earned,
            'schedule': self.schedule_type,
            'responses_since_reward': self.responses_since_reward,
            'time_since_reward': self.time_since_reward,
            'facing_lever': self.facing_lever
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view of operant chamber."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Chamber walls
        img[:] = self.wall_color
        
        # Floor
        img[160:, :] = (80, 80, 80)
        
        # House light effect
        if self.house_light_on:
            img[:50, :] = (150, 150, 140)  # Lit ceiling
        
        # Draw lever(s) based on facing direction
        lever = self.levers[self.facing_lever] if self.facing_lever < len(self.levers) else None
        
        if lever:
            # Lever
            lever_y = 100
            lever_width = 60
            lever_height = 40
            lever_x = 112 - lever_width // 2
            
            # Lever pressed animation
            is_pressed = self.lever_press_frames[self.facing_lever] > 0
            color = self.lever_pressed_color if is_pressed else self.lever_color
            y_offset = 5 if is_pressed else 0
            
            self._draw_rect(img, lever_x, lever_y + y_offset, 
                           lever_x + lever_width, lever_y + lever_height + y_offset, color)
            
            # Cue light above lever (using shared _draw_disk)
            if self.has_cue_light:
                light_color = self.light_on_color if self.cue_light_on else self.light_off_color
                self._draw_disk(img, 112, 70, 10, light_color)
        
        # Food magazine (bottom center)
        mag_color = self.light_on_color if self.magazine_lit else (60, 60, 60)
        self._draw_rect(img, 90, 170, 134, 200, mag_color)
        
        # Arrow indicators for other levers
        if self.num_levers > 1:
            if self.facing_lever > 0:
                # Left arrow
                for i in range(15):
                    img[112-i:112+i+1, 10+i] = (200, 200, 200)
            if self.facing_lever < self.num_levers - 1:
                # Right arrow
                for i in range(15):
                    img[112-i:112+i+1, 213-i] = (200, 200, 200)
        
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view using shared helpers."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        margin = 30
        
        # Chamber floor and walls
        self._draw_rect(img, margin, margin, 224-margin, 224-margin, (80, 80, 80))
        self._draw_rect(img, margin, margin, 224-margin, margin+5, self.wall_color)  # Top
        self._draw_rect(img, margin, 224-margin-5, 224-margin, 224-margin, self.wall_color)  # Bottom
        self._draw_rect(img, margin, margin, margin+5, 224-margin, self.wall_color)  # Left
        self._draw_rect(img, 224-margin-5, margin, 224-margin, 224-margin, self.wall_color)  # Right
        
        # Levers (on front wall)
        for lever in self.levers:
            lx = int(112 + lever['x'] * 80)
            ly = margin + 20
            is_pressed = self.lever_press_frames[lever['index']] > 0
            color = self.lever_pressed_color if is_pressed else self.lever_color
            self._draw_rect(img, lx-15, ly-8, lx+16, ly+9, color)
        
        # Food magazine
        mag_color = self.light_on_color if self.magazine_lit else (60, 60, 60)
        self._draw_rect(img, 100, margin+30, 124, margin+50, mag_color)
        
        # Agent (rat shape) - body as ellipse, head as circle
        ax, ay = 112, 150
        # Body (ellipse approximation)
        for dx in range(-12, 13):
            for dy in range(-8, 9):
                if (dx/12)**2 + (dy/8)**2 <= 1:
                    img[ay + dy, ax + dx] = (200, 180, 160)
        
        # Head (pointing at faced lever)
        head_x = ax
        if self.facing_lever == 0 and self.num_levers > 1:
            head_x = ax - 15
        elif self.facing_lever == 1:
            head_x = ax + 15
        self._draw_disk(img, head_x, ay - 10, 6, (180, 160, 140))
        
        return img
    
    def _render_ascii_2d(self) -> str:
        """Render ASCII top-down view."""
        width = 25
        lines = []
        lines.append("#" * (width + 2))
        
        # Lever row
        lever_line = "#"
        for i, lever in enumerate(self.levers):
            pos = 5 + i * 12
            lever_line += " " * (pos - len(lever_line) + 1)
            is_pressed = self.lever_press_frames[i] > 0
            lever_line += "[=]" if not is_pressed else "[_]"
        lever_line = lever_line.ljust(width + 1) + "#"
        lines.append(lever_line)
        
        # Cue lights - only active lever has light on (discriminative stimulus)
        if self.has_cue_light:
            light_line = "#"
            for i in range(self.num_levers):
                pos = 5 + i * 12
                light_line += " " * (pos - len(light_line) + 1)
                # Cue light indicates which lever is active (rewarded)
                is_active = (i == self.active_lever)
                light_line += "(*)" if is_active else "( )"
            light_line = light_line.ljust(width + 1) + "#"
            lines.append(light_line)
        
        # Magazine
        mag_str = "[M]" if self.magazine_lit else "[m]"
        lines.append("#" + mag_str.center(width) + "#")
        
        # Empty space
        for _ in range(4):
            lines.append("#" + " " * width + "#")
        
        # Agent - use standard direction symbols
        # In operant chamber, agent faces the lever (north/up)
        agent_str = "↑"
        lines.append("#" + agent_str.center(width) + "#")
        
        lines.append("#" * (width + 2))
        
        return '\n'.join(lines)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 28) -> str:
        """Render ASCII 3D view of operant chamber with raycasting."""
        lines = []
        view_width = width - 2  # Account for frame
        view_height = height - 2
        
        # Wall characters by distance
        def wall_char(dist: float) -> str:
            if dist < 1.5: return '█'
            if dist < 3.0: return '▓'
            if dist < 5.0: return '▒'
            if dist < 8.0: return '░'
            return '·'
        
        # Chamber dimensions for raycasting
        chamber_depth = 3.0
        chamber_width = 2.0
        
        # Agent is looking at front wall with levers
        # facing_lever: -1 = center view, 0 = left lever, 1 = right lever
        base_angle = np.pi / 2  # Looking forward
        if self.facing_lever == 0:
            base_angle = np.pi / 2 + 0.4  # Slightly left
        elif self.facing_lever == 1:
            base_angle = np.pi / 2 - 0.4  # Slightly right
        
        # Build frame with raycasting
        fov = np.pi / 2  # 90 degree FOV
        
        lines.append("╔" + "═" * view_width + "╗")
        
        # Cast rays for each column
        ray_distances = []
        for col in range(view_width):
            ray_offset = (col / view_width - 0.5) * fov
            ray_angle = base_angle + ray_offset
            
            # Simple box raycasting
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            # Distance to front wall (y = chamber_depth)
            if abs(dy) > 0.01:
                dist_front = chamber_depth / abs(dy)
            else:
                dist_front = 20
            
            # Distance to side walls
            if abs(dx) > 0.01:
                dist_side = chamber_width / abs(dx)
            else:
                dist_side = 20
            
            dist = min(dist_front, dist_side)
            # Fish-eye correction
            dist *= np.cos(ray_offset)
            ray_distances.append(max(0.5, dist))
        
        # Render view rows
        for row in range(view_height):
            row_chars = []
            for col in range(view_width):
                dist = ray_distances[col]
                wall_height = int(view_height * 1.2 / (dist + 0.3))
                half_wall = wall_height // 2
                center = view_height // 2
                
                if row < center - half_wall:
                    # Ceiling
                    char = '*' if self.house_light_on else '░'
                elif row > center + half_wall:
                    # Floor
                    char = '▓'
                else:
                    # Wall with levers/magazine
                    char = wall_char(dist)
                
                row_chars.append(char)
            
            row_str = ''.join(row_chars)
            
            # Add visual elements at specific rows
            center_row = view_height // 2
            if abs(row - center_row + 3) <= 1:  # Upper panel area - levers
                if self.facing_lever == -1:
                    # Show both levers
                    left_pos = view_width // 4
                    right_pos = 3 * view_width // 4
                    row_str = row_str[:left_pos-2] + '[=]' + row_str[left_pos+1:right_pos-2] + '[=]' + row_str[right_pos+1:]
                elif self.facing_lever >= 0:
                    # Show single lever
                    lever_pos = view_width // 2
                    is_pressed = self.lever_press_frames[self.facing_lever] > 0
                    lever_sym = '[▼]' if is_pressed else '[=]'
                    row_str = row_str[:lever_pos-1] + lever_sym + row_str[lever_pos+2:]
            
            if abs(row - center_row - 2) <= 1:  # Lower panel area - magazine
                mag_pos = view_width // 2
                mag_sym = '[*]' if self.magazine_lit else '[m]'
                row_str = row_str[:mag_pos-1] + mag_sym + row_str[mag_pos+2:]
            
            lines.append("║" + row_str + "║")
        
        lines.append("╚" + "═" * view_width + "╝")
        
        return '\n'.join(lines)


def create_operant_chamber(
    schedule: str = "FR1",
    num_levers: int = 1,
    trials_to_criterion: int = 50,
    view_mode: ViewMode = ViewMode.FPV_3D,
    source_pmc: str = "",
    source_quote: str = ""
) -> OperantChamber:
    """Factory function to create Operant Chamber."""
    
    # Parse schedule parameter
    schedule_param = 1
    if len(schedule) > 2:
        try:
            schedule_param = int(schedule[2:])
        except ValueError:
            pass
    
    config = OperantChamberConfig(
        name="Operant Chamber",
        task_type="operant",
        trials_to_criterion=trials_to_criterion,
        sessions_to_criterion=5,
        trials_per_session=10,
        max_trial_steps=100,
        success_criterion="earn_criterion_rewards",
        arena_size=2.0,
        source_pmc=source_pmc,
        source_quote=source_quote,
        extra_params={
            'num_levers': num_levers,
            'reward_schedule': schedule,
            'schedule_parameter': schedule_param,
            'has_cue_light': True,
            'has_house_light': True,
            'criterion_rewards': 10
        }
    )
    
    return OperantChamber(config, view_mode)
