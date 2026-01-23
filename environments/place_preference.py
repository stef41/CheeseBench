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
        
        # Movement
        if action == Action.FORWARD:
            self.agent.x += 0.2 * np.cos(self.agent.angle)
            self.agent.y += 0.2 * np.sin(self.agent.angle)
            # Clamp to arena
            self.agent.x = np.clip(self.agent.x, -1.8, 1.8)
            self.agent.y = np.clip(self.agent.y, -0.7, 0.7)
        elif action == Action.TURN_LEFT:
            self.agent.angle += np.pi / 4  # 45° per turn
        elif action == Action.TURN_RIGHT:
            self.agent.angle -= np.pi / 4  # 45° per turn
        
        self.agent.angle = self.agent.angle % (2 * np.pi)
        
        # Update current chamber
        old_chamber = self.current_chamber
        self.current_chamber = 0 if self.agent.x < 0 else 1
        
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
            for y in range(160, 220, 15):
                for x in range(10, 220, 15):
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx**2 + dy**2 <= 9:
                                px, py = x + dx, y + dy
                                if 154 <= py < 224 and 0 <= px < 224:
                                    img[py, px] = (floor_base * 0.6).astype(np.uint8)
        
        # Door to other chamber (center)
        door_width = 50
        img[90:154, 112-door_width//2:112+door_width//2] = (40, 40, 40)
        
        # Phase indicator
        if self.phase == 'conditioning' and self.current_chamber == self.conditioning_chamber:
            # Reward indicator
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    if dx**2 + dy**2 <= 225:
                        px, py = 112 + dx, 40 + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[py, px] = (255, 215, 0)  # Gold for reward
        
        return img
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Left chamber
        ch0 = self.chambers[0]
        img[40:184, 20:107] = ch0['floor_color']
        
        # Right chamber
        ch1 = self.chambers[1]
        img[40:184, 117:204] = ch1['floor_color']
        
        # Patterns
        # Stripes in left chamber
        for x in range(20, 107, 12):
            img[40:184, x:x+6] = (ch0['floor_color'][0]*0.7, ch0['floor_color'][1]*0.7, ch0['floor_color'][2]*0.7)
        
        # Dots in right chamber
        for y in range(50, 180, 20):
            for x in range(127, 200, 20):
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx**2 + dy**2 <= 16:
                            px, py = x + dx, y + dy
                            if 40 <= py < 184 and 117 <= px < 204:
                                img[py, px] = (ch1['floor_color'][0]*0.6, ch1['floor_color'][1]*0.6, ch1['floor_color'][2]*0.6)
        
        # Door
        img[90:134, 107:117] = (60, 60, 60)
        
        # Walls
        img[38:40, 18:206] = (80, 80, 80)
        img[184:186, 18:206] = (80, 80, 80)
        img[38:186, 18:20] = (80, 80, 80)
        img[38:186, 204:206] = (80, 80, 80)
        
        # Conditioning chamber marker
        if self.phase == 'conditioning':
            marker_x = 63 if self.conditioning_chamber == 0 else 160
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx**2 + dy**2 <= 64:
                        px, py = marker_x + dx, 55 + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[py, px] = (255, 215, 0)
        
        # Agent
        agent_x = int(112 + self.agent.x * 50)
        agent_y = int(112 - self.agent.y * 50)  # Flip Y: positive Y = UP on screen (lower pixel row)
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                if dx**2 + dy**2 <= 36:
                    px, py = agent_x + dx, agent_y + dy
                    if 0 <= px < 224 and 0 <= py < 224:
                        img[py, px] = (0, 150, 255)
        
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
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 30) -> str:
        """Render ASCII 3D view."""
        lines = []
        chamber = self.chambers[self.current_chamber]
        pattern = chamber['floor_pattern']
        
        # Different ceiling texture per chamber (visual cue)
        ceiling_char = '░' if self.current_chamber == 0 else '▒'
        
        lines.append(f"╔{'═' * (width-2)}╗")
        
        # 3D chamber view - no labels, just visual differences
        for i in range(height - 5):
            if i < 5:
                lines.append(f"║{ceiling_char * (width-2)}║")
            elif i < 10:
                # Door
                door = "░░░░░░░░░░"
                pad = (width - 2 - len(door)) // 2
                lines.append(f"║{' ' * pad}{door}{' ' * (width - 2 - pad - len(door))}║")
            else:
                # Floor with distinct pattern per chamber
                if pattern == 'stripes':
                    floor_line = '│ ' * ((width-2) // 2)  # Striped floor
                else:
                    floor_line = '. ' * ((width-2) // 2)  # Dotted floor
                lines.append(f"║{floor_line[:width-2]}║")
        
        lines.append(f"╚{'═' * (width-2)}╝")
        
        return '\n'.join(lines)
