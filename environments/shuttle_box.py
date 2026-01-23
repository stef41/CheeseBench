"""
Shuttle Box environment for VLM evaluation.

Two-chamber apparatus for fear conditioning and active/passive avoidance tasks.
Agent learns to shuttle between chambers to avoid aversive stimuli.
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
                source_pmc="PMC4692667",
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
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
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
        
        # Movement
        old_x = self.agent.x
        
        if action == Action.FORWARD:
            dx = 0.3 * np.cos(self.agent.angle)
            self.agent.x += dx
            # Clamp to chambers
            self.agent.x = np.clip(self.agent.x, -1.8, 1.8)
        
        elif action == Action.TURN_LEFT:
            self.agent.angle += np.pi / 4  # 45° per turn
        elif action == Action.TURN_RIGHT:
            self.agent.angle -= np.pi / 4  # 45° per turn
        
        self.agent.angle = self.agent.angle % (2 * np.pi)
        
        # Check chamber transition
        old_chamber = self.current_chamber
        if self.agent.x < 0:
            self.current_chamber = 0
        else:
            self.current_chamber = 1
        
        # Did agent shuttle?
        if old_chamber != self.current_chamber:
            if self.current_chamber != self.shock_chamber:
                # Successfully moved to safe chamber
                if self.trial_phase == 'cue':
                    # Avoidance - shuttled during cue
                    self.avoidances += 1
                    reward = 1.0
                    self._trial_reward += 1.0  # Track for success criterion
                    self.trial_phase = 'escaped'
                    self.cue_active = False
                elif self.trial_phase == 'shock':
                    # Escape - shuttled during shock
                    self.escapes += 1
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
        """Success = avoided or escaped shock."""
        return self.trial_phase == 'escaped' and self._trial_reward > 0
    
    def _check_failure(self) -> bool:
        """Failure = received full shock."""
        return self.trial_phase == 'escaped' and self._trial_reward <= 0
    
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
        """Render first-person view."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Ceiling
        img[:70, :] = self.wall_color
        
        # Floor - different color based on chamber state
        if self.shock_active and self.current_chamber == self.shock_chamber:
            floor_col = self.shock_floor_color
        elif self.current_chamber != self.shock_chamber:
            floor_col = self.safe_floor_color
        else:
            floor_col = self.floor_color
        img[154:, :] = floor_col
        
        # Back wall
        img[70:154, :] = self.wall_color
        
        # Door/passage to other chamber
        door_center = 112
        door_half_width = 40
        img[90:154, door_center-door_half_width:door_center+door_half_width] = self.door_color
        
        # Cue light (top center)
        if self.cue_active:
            for dx in range(-20, 21):
                for dy in range(-15, 16):
                    if dx**2 + dy**2 <= 300:
                        px, py = 112 + dx, 40 + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[py, px] = self.cue_light_color
        
        # Shock indicator (red flash on edges)
        if self.shock_active and self.current_chamber == self.shock_chamber:
            img[:, :15] = (200, 50, 50)
            img[:, 209:] = (200, 50, 50)
        
        # Chamber indicator
        text_y = 200
        if self.current_chamber == 0:
            img[text_y:text_y+10, 10:50] = (150, 150, 150)  # Left indicator
        else:
            img[text_y:text_y+10, 174:214] = (150, 150, 150)  # Right indicator
        
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Two chambers
        left_color = self.shock_floor_color if (self.shock_active and self.shock_chamber == 0) else self.floor_color
        right_color = self.shock_floor_color if (self.shock_active and self.shock_chamber == 1) else self.floor_color
        
        # Left chamber
        img[40:184, 20:105] = left_color
        # Right chamber  
        img[40:184, 119:204] = right_color
        
        # Door/passage
        img[90:134, 105:119] = self.door_color
        
        # Walls (outlines)
        img[38:40, 18:206] = self.wall_color  # Top
        img[184:186, 18:206] = self.wall_color  # Bottom
        img[38:186, 18:20] = self.wall_color  # Left
        img[38:186, 204:206] = self.wall_color  # Right
        # Center divider (with door gap)
        img[38:90, 105:107] = self.wall_color
        img[134:186, 105:107] = self.wall_color
        img[38:90, 117:119] = self.wall_color
        img[134:186, 117:119] = self.wall_color
        
        # Agent
        agent_screen_x = int(62 + (self.agent.x + 1.8) / 3.6 * 140)
        agent_screen_y = int(112 - self.agent.y * 40)  # Flip Y: positive Y = UP on screen
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                if dx**2 + dy**2 <= 36:
                    px, py = agent_screen_x + dx, agent_screen_y + dy
                    if 0 <= px < 224 and 0 <= py < 224:
                        img[py, px] = (0, 150, 255)
        
        # Direction indicator (flip Y for sin component)
        dir_x = int(agent_screen_x + 10 * np.cos(self.agent.angle))
        dir_y = int(agent_screen_y - 10 * np.sin(self.agent.angle))  # Flip Y
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px, py = dir_x + dx, dir_y + dy
                if 0 <= px < 224 and 0 <= py < 224:
                    img[py, px] = (255, 255, 255)
        
        # Cue light indicator
        if self.cue_active:
            shock_x = 62 if self.shock_chamber == 0 else 161
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx**2 + dy**2 <= 64:
                        px, py = shock_x + dx, 55 + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[py, px] = self.cue_light_color
        
        return img
    
    def _render_ascii_2d(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII view."""
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw chambers
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
        door_top = height // 2 - 2
        door_bot = height // 2 + 2
        for y in range(ch_top + 1, ch_bot):
            if y < door_top or y > door_bot:
                grid[y][ch_mid] = '#'
        
        # Cue indicator and goal marker (visual only - no text)
        if self.cue_active or self.shock_active:
            # Show ! in shock chamber as warning
            if self.shock_chamber == 0:
                for row in range(ch_top + 1, ch_bot):
                    grid[row][ch_left + 2] = '!'
            else:
                for row in range(ch_top + 1, ch_bot):
                    grid[row][ch_mid + 2] = '!'
            
            # Show goal 'G' in SAFE chamber
            safe_chamber = 1 - self.shock_chamber
            safe_x = ch_left + (ch_mid - ch_left) // 2 if safe_chamber == 0 else ch_mid + (ch_right - ch_mid) // 2
            safe_y = height // 2 - 1
            grid[safe_y][safe_x] = 'G'
        
        # Agent position
        agent_x = int((self.agent.x + 1.8) / 3.6 * (ch_right - ch_left - 2)) + ch_left + 1
        agent_y = int(height / 2)
        agent_x = max(ch_left + 1, min(ch_right - 1, agent_x))
        
        # Agent direction - 8 directions for finer angle gradation
        # Standard mapping: 0=E(→), 1=NE(↗), 2=N(↑), 3=NW(↖), 4=W(←), 5=SW(↙), 6=S(↓), 7=SE(↘)
        dirs = {0: '→', 1: '↗', 2: '↑', 3: '↖', 4: '←', 5: '↙', 6: '↓', 7: '↘'}
        dir_idx = int((self.agent.angle + np.pi/8) / (np.pi/4)) % 8
        agent_char = dirs.get(dir_idx, '@')
        
        grid[agent_y][agent_x] = agent_char
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 30) -> str:
        """Render ASCII 3D view."""
        lines = []
        
        # Different visual cues per chamber - no text labels
        # Left chamber: ░ ceiling, Right chamber: ▒ ceiling
        ceiling_char = '░' if self.current_chamber == 0 else '▒'
        
        lines.append(f"╔{'═' * (width-2)}╗")
        
        # Chamber view - visual only
        for i in range(height - 6):
            if i < 5:
                # Ceiling - different per chamber
                lines.append(f"║{ceiling_char * (width-2)}║")
            elif i < 10:
                # Door area
                door_chars = "▓▓▓▓▓▓▓▓▓▓"
                padding = (width - 2 - len(door_chars)) // 2
                lines.append(f"║{' ' * padding}{door_chars}{' ' * (width - 2 - padding - len(door_chars))}║")
            else:
                # Floor - shows shock visually with ▓▓▓ pattern
                if self.shock_active and self.current_chamber == self.shock_chamber:
                    floor_char = '▓'  # Electrified floor visual
                else:
                    floor_char = '░'
                lines.append(f"║{floor_char * (width-2)}║")
        
        lines.append(f"╚{'═' * (width-2)}╝")
        
        return '\n'.join(lines)
