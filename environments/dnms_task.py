"""
Delayed Non-Match-to-Sample (DNMS) Working Memory environment for VLM evaluation.

Tests working memory by requiring agent to remember a sample stimulus
and choose the non-matching stimulus after a delay.
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
class DNMSConfig(EnvironmentConfig):
    """DNMS specific configuration."""
    delay_steps: int = 10  # Delay between sample and choice
    num_stimuli: int = 2  # Number of choice options
    stimulus_type: str = "visual"  # "visual", "olfactory", "spatial"


class DNMSTask(BaseEnvironment):
    """
    Delayed Non-Match-to-Sample Working Memory Task.
    
    Protocol: Agent sees sample stimulus, waits through delay, then must
    choose the stimulus that does NOT match the sample.
    
    From verified protocols (PMC3982138 - Oomen et al., Nat Protoc 2013):
    - "TUNL, a working memory task, requires animals to 'non-match' to a sample 
       location after a delay."
    - "A sample location is presented the moment the animal exits the magazine. 
       During the sample phase, the animal is required to respond by touching the 
       sample location."
    - "At the end of the delay, the incorrect sample location and a different, 
       correct choice location are simultaneously presented on the screen."
    - "In the case of a correct response to the novel location, touching the 
       screen leads to reward delivery accompanied by a tone."
    """
    
    def __init__(self, 
                 config: Optional[DNMSConfig] = None,
                 view_mode: ViewMode = ViewMode.FPV_3D):
        
        if config is None:
            config = DNMSConfig(
                name="DNMS Task",
                task_type="working_memory",
                trials_to_criterion=800,
                sessions_to_criterion=25,
                trials_per_session=32,
                max_trial_steps=50,
                success_criterion="correct_choice",
                arena_size=2.0,
                source_pmc="PMC3982138",
                source_quote="TUNL working memory task requires animals to non-match to a sample location after a delay. A correct response to the novel location leads to reward delivery."
            )
        
        super().__init__(config, view_mode)
        
        # Task parameters
        self.delay_steps = config.extra_params.get('delay_steps', 10)
        self.num_stimuli = config.extra_params.get('num_stimuli', 2)
        self.stimulus_type = config.extra_params.get('stimulus_type', 'visual')
        
        # Stimulus properties
        self.stimuli = self._create_stimuli()
        
        # Trial state
        self.phase = 'sample'  # 'sample', 'delay', 'choice', 'response'
        self.sample_stimulus = 0
        self.choice_stimuli = [0, 1]
        self.correct_choice = 1  # Index of non-matching stimulus
        self.delay_counter = 0
        
        # Response tracking
        self.correct_responses = 0
        self.total_responses = 0
        
        # Agent position (simplified)
        self.agent = AgentState(x=0.0, y=0.0, angle=np.pi/2)
        
        # Actions
        self.valid_actions = [
            Action.TURN_LEFT,   # Choose left option
            Action.TURN_RIGHT,  # Choose right option
            Action.INTERACT,    # Confirm choice / advance phase
            Action.STAY         # Wait
        ]
        
        # Colors for visual stimuli
        self.stimulus_colors = [
            (255, 100, 100),  # Red
            (100, 100, 255),  # Blue
            (100, 255, 100),  # Green
            (255, 255, 100),  # Yellow
        ]
        
        self.stimulus_shapes = ['circle', 'square', 'triangle', 'diamond']
        
        # Choice position
        self.choice_position = 0  # 0 = center, -1 = left, 1 = right
    
    def _create_stimuli(self) -> List[Dict[str, Any]]:
        """Create stimulus definitions."""
        stimuli = []
        for i in range(max(4, self.num_stimuli)):
            stimuli.append({
                'id': i,
                'color_idx': i % len(self.stimulus_colors) if hasattr(self, 'stimulus_colors') else i,
                'shape_idx': i % 4,
            })
        return stimuli
    
    def _reset_agent_position(self):
        """Reset for new trial."""
        self.agent.x = 0.0
        self.agent.y = 0.0
        self.agent.angle = np.pi / 2
        self.choice_position = 0
        
        # Reset trial state
        self.phase = 'sample'
        self.delay_counter = 0
        
        # Randomize sample
        self.sample_stimulus = np.random.randint(0, self.num_stimuli)
        
        # Randomize which position has the non-match
        self.choice_stimuli = list(range(self.num_stimuli))
        np.random.shuffle(self.choice_stimuli)
        
        # Correct choice is the one that doesn't match sample
        for i, stim in enumerate(self.choice_stimuli):
            if stim != self.sample_stimulus:
                self.correct_choice = i  # Position of correct choice
                break
    
    def _setup_trial(self):
        """Setup for new trial."""
        pass
    
    def _execute_action(self, action: Action) -> float:
        """Execute action and return reward."""
        reward = 0.0
        
        if self.phase == 'sample':
            # Show sample, wait for interaction to proceed
            if action == Action.INTERACT:
                self.phase = 'delay'
                self.delay_counter = self.delay_steps
        
        elif self.phase == 'delay':
            # Delay period - just wait
            self.delay_counter -= 1
            if self.delay_counter <= 0:
                self.phase = 'choice'
        
        elif self.phase == 'choice':
            # Choose which stimulus (left/right)
            if action == Action.TURN_LEFT:
                self.choice_position = 0
            elif action == Action.TURN_RIGHT:
                self.choice_position = 1
            elif action == Action.INTERACT:
                # Make the choice
                self.phase = 'response'
                self.total_responses += 1
                
                if self.choice_position == self.correct_choice:
                    self.correct_responses += 1
                    self._trial_reward += 1.0
                    reward = 1.0
                else:
                    reward = -0.5
        
        return reward
    
    def _check_success(self) -> bool:
        """Success = correct choice made."""
        return self.phase == 'response' and self._trial_reward > 0
    
    def _check_failure(self) -> bool:
        """Failure = incorrect choice."""
        return self.phase == 'response' and self._trial_reward <= 0
    
    def get_info(self) -> Dict[str, Any]:
        """Get current state info."""
        base_info = super().get_info()
        base_info.update({
            'phase': self.phase,
            'sample_stimulus': self.sample_stimulus,
            'choice_stimuli': self.choice_stimuli,
            'correct_choice': self.correct_choice,
            'choice_position': self.choice_position,
            'delay_remaining': self.delay_counter,
            'correct_responses': self.correct_responses,
            'total_responses': self.total_responses,
            'accuracy': self.correct_responses / max(1, self.total_responses)
        })
        return base_info
    
    # ==================== Rendering ====================
    
    def _render_fpv(self) -> np.ndarray:
        """Render first-person view based on phase."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Background
        img[:] = (60, 60, 60)
        
        if self.phase == 'sample':
            # Show sample stimulus in center
            self._draw_stimulus(img, self.sample_stimulus, 112, 112, 60)
            # Instruction text area
            img[180:200, 50:174] = (40, 40, 40)
            
        elif self.phase == 'delay':
            # Blank screen during delay
            # Show delay progress
            progress = 1 - (self.delay_counter / self.delay_steps)
            bar_width = int(150 * progress)
            img[100:120, 37:37+bar_width] = (100, 100, 200)
            img[100:120, 37:187] = np.maximum(img[100:120, 37:187], 30)
            
        elif self.phase == 'choice':
            # Show choice stimuli
            left_stim = self.choice_stimuli[0]
            right_stim = self.choice_stimuli[1]
            
            self._draw_stimulus(img, left_stim, 60, 112, 40)
            self._draw_stimulus(img, right_stim, 164, 112, 40)
            
            # Highlight selected
            if self.choice_position == 0:
                img[70:154, 18:102] = np.clip(img[70:154, 18:102].astype(int) + 30, 0, 255).astype(np.uint8)
            else:
                img[70:154, 122:206] = np.clip(img[70:154, 122:206].astype(int) + 30, 0, 255).astype(np.uint8)
        
        elif self.phase == 'response':
            # Show result
            if self._trial_reward > 0:
                img[:] = (50, 100, 50)  # Green tint for correct
            else:
                img[:] = (100, 50, 50)  # Red tint for incorrect
        
        return img
    
    def _draw_stimulus(self, img: np.ndarray, stim_id: int, cx: int, cy: int, size: int):
        """Draw a stimulus at position."""
        color = self.stimulus_colors[stim_id % len(self.stimulus_colors)]
        shape = self.stimulus_shapes[stim_id % len(self.stimulus_shapes)]
        
        if shape == 'circle':
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if dx**2 + dy**2 <= size**2:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < 224 and 0 <= py < 224:
                            img[py, px] = color
                            
        elif shape == 'square':
            x1, y1 = cx - size, cy - size
            x2, y2 = cx + size, cy + size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(223, x2), min(223, y2)
            img[y1:y2, x1:x2] = color
            
        elif shape == 'triangle':
            for dy in range(-size, size+1):
                width = int(size * (1 - abs(dy) / size))
                for dx in range(-width, width+1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 224 and 0 <= py < 224:
                        img[py, px] = color
                        
        elif shape == 'diamond':
            for dy in range(-size, size+1):
                width = size - abs(dy)
                for dx in range(-width, width+1):
                    px, py = cx + dx, cy + dy
                    if 0 <= px < 224 and 0 <= py < 224:
                        img[py, px] = color
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view (same as FPV for this task)."""
        return self._render_fpv()
    
    def _render_ascii_2d(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII view with phase indicator."""
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Border (all #)
        for x in range(width):
            grid[0][x] = '#'
            grid[height-1][x] = '#'
        for y in range(height):
            grid[y][0] = '#'
            grid[y][width-1] = '#'
        
        center_y = height // 2
        center_x = width // 2
        
        # Use visual symbols only - no text
        symbols = ['●', '■', '▲', '◆']
        
        if self.phase == 'sample':
            # Show sample stimulus as large symbol
            stim = symbols[self.sample_stimulus % len(symbols)]
            # Draw stimulus pattern
            for dy in [-1, 0, 1]:
                for dx in [-2, -1, 0, 1, 2]:
                    if 0 < center_x + dx < width - 1 and 0 < center_y + dy < height - 1:
                        grid[center_y + dy][center_x + dx] = stim
            
        elif self.phase == 'delay':
            # Show progress bar visually
            progress = int((1 - self.delay_counter / self.delay_steps) * 20)
            bar_y = center_y
            for i in range(22):
                bx = center_x - 11 + i
                if 0 < bx < width - 1:
                    if i == 0:
                        grid[bar_y][bx] = '['
                    elif i == 21:
                        grid[bar_y][bx] = ']'
                    elif i - 1 < progress:
                        grid[bar_y][bx] = '█'
                    else:
                        grid[bar_y][bx] = '░'
            
        elif self.phase == 'choice':
            left_x = width // 4
            right_x = 3 * width // 4
            
            # Show both stimuli
            left_stim = symbols[self.choice_stimuli[0] % len(symbols)]
            right_stim = symbols[self.choice_stimuli[1] % len(symbols)]
            
            # Draw left stimulus
            for dy in [-1, 0, 1]:
                grid[center_y + dy][left_x] = left_stim
            
            # Draw right stimulus
            for dy in [-1, 0, 1]:
                grid[center_y + dy][right_x] = right_stim
            
            # No hint - agent must remember sample and choose non-matching
            # (Real DNMS task tests working memory)
            
            # Show selection with arrow
            if self.choice_position == 0:
                grid[center_y + 2][left_x] = '^'
            else:
                grid[center_y + 2][right_x] = '^'
        
        elif self.phase == 'response':
            # Show result with symbol
            if self._trial_reward > 0:
                grid[center_y][center_x] = '*'  # Reward
            else:
                grid[center_y][center_x] = 'X'  # Wrong
        
        return '\n'.join(''.join(row) for row in grid)
    
    def _render_ascii_3d(self, width: int = 60, height: int = 30) -> str:
        """Render ASCII 3D view."""
        lines = []
        
        # Pure visual render - no text hints (like animal experiments)
        symbols = ['●', '■', '▲', '◆']
        
        # Simple border
        lines.append("╔" + "═" * (width-2) + "╗")
        
        if self.phase == 'sample':
            # Show sample stimulus centered
            stim_sym = symbols[self.sample_stimulus % len(symbols)] * 7
            empty_rows = (height - 6) // 2
            for _ in range(empty_rows):
                lines.append(f"║{' ' * (width-2)}║")
            lines.append(f"║{stim_sym:^{width-2}}║")
            lines.append(f"║{stim_sym:^{width-2}}║")
            lines.append(f"║{stim_sym:^{width-2}}║")
            for _ in range(height - empty_rows - 5):
                lines.append(f"║{' ' * (width-2)}║")
            
        elif self.phase == 'delay':
            # Show progress bar only
            progress = 1 - (self.delay_counter / self.delay_steps)
            bar_len = width - 10
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            empty_rows = (height - 4) // 2
            for _ in range(empty_rows):
                lines.append(f"║{' ' * (width-2)}║")
            lines.append(f"║{'[' + bar + ']':^{width-2}}║")
            for _ in range(height - empty_rows - 3):
                lines.append(f"║{' ' * (width-2)}║")
                
        elif self.phase == 'choice':
            # Show both stimuli side by side
            left_sym = symbols[self.choice_stimuli[0] % len(symbols)] * 5
            right_sym = symbols[self.choice_stimuli[1] % len(symbols)] * 5
            
            empty_rows = (height - 6) // 2
            for _ in range(empty_rows):
                lines.append(f"║{' ' * (width-2)}║")
            
            # Stimuli
            half = (width - 2) // 2
            lines.append(f"║{left_sym:^{half}}{right_sym:^{half}}║")
            lines.append(f"║{left_sym:^{half}}{right_sym:^{half}}║")
            lines.append(f"║{left_sym:^{half}}{right_sym:^{half}}║")
            
            # Selection indicator (arrow under selected)
            left_arrow = "^" if self.choice_position == 0 else " "
            right_arrow = "^" if self.choice_position == 1 else " "
            lines.append(f"║{left_arrow:^{half}}{right_arrow:^{half}}║")
            
            for _ in range(height - empty_rows - 6):
                lines.append(f"║{' ' * (width-2)}║")
        
        elif self.phase == 'response':
            # Show result symbol only
            result_sym = "★" if self._trial_reward > 0 else "✗"
            empty_rows = (height - 4) // 2
            for _ in range(empty_rows):
                lines.append(f"║{' ' * (width-2)}║")
            lines.append(f"║{result_sym:^{width-2}}║")
            for _ in range(height - empty_rows - 3):
                lines.append(f"║{' ' * (width-2)}║")
        
        lines.append("╚" + "═" * (width-2) + "╝")
        
        return '\n'.join(lines)
        
        return '\n'.join(lines)
