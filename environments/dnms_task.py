"""
Delayed Non-Match-to-Sample (DNMS) Working Memory environment for VLM evaluation.

Tests working memory by requiring agent to remember a sample stimulus
and choose the non-matching stimulus after a delay.
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
            Action.FORWARD,     # Confirm choice
            Action.ROTATE_LEFT,   # Choose left option
            Action.ROTATE_RIGHT,  # Choose right option
            Action.STAY         # Wait (auto-advances in sample phase)
        ]
        
        # Colors for visual stimuli
        self.stimulus_colors = [
            (255, 100, 100),  # Red
            (100, 100, 255),  # Blue
            (100, 255, 100),  # Green
            (255, 255, 100),  # Yellow
        ]
        
        self.stimulus_shapes = ['circle', 'square', 'triangle', 'diamond']
        
        # Stimulus properties (must be after stimulus_colors/shapes are defined)
        self.stimuli = self._create_stimuli()
        
        # Choice position
        self.choice_position = 0  # 0 = center, -1 = left, 1 = right
    
    def _create_stimuli(self) -> List[Dict[str, Any]]:
        """Create stimulus definitions."""
        stimuli = []
        for i in range(max(4, self.num_stimuli)):
            stimuli.append({
                'id': i,
                'color_idx': i % len(self.stimulus_colors),
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
            # Show sample, any action advances to delay
            self.phase = 'delay'
            self.delay_counter = self.delay_steps
        
        elif self.phase == 'delay':
            # Delay period - just wait
            self.delay_counter -= 1
            if self.delay_counter <= 0:
                self.phase = 'choice'
        
        elif self.phase == 'choice':
            # Choose which stimulus (left/right) then confirm with FORWARD
            if action == Action.ROTATE_LEFT:
                self.choice_position = 0
            elif action == Action.ROTATE_RIGHT:
                self.choice_position = 1
            elif action == Action.FORWARD:
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
            # Blank/dark screen during delay (realistic - no visual timer)
            # Animals just wait in darkness during retention interval
            pass  # Screen stays dark gray (background color)
            
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
        """Draw a stimulus at position using shared shape helper."""
        color = self.stimulus_colors[stim_id % len(self.stimulus_colors)]
        shape = self.stimulus_shapes[stim_id % len(self.stimulus_shapes)]
        self._draw_shape(img, shape, cx, cy, size, color)
    
    def _render_topdown(self) -> np.ndarray:
        """Render top-down view (same as FPV for this task)."""
        return self._render_fpv()
    
    def _render_ascii_2d(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII view with phase indicator."""
        c = AsciiCanvas(width, height)
        
        # Border
        c.hline(0, width - 1, 0, '#')
        c.hline(0, width - 1, height - 1, '#')
        c.vline(0, 0, height - 1, '#')
        c.vline(width - 1, 0, height - 1, '#')
        
        center_y = height // 2
        center_x = width // 2
        
        symbols = ['●', '■', '▲', '◆']
        
        if self.phase == 'sample':
            stim = symbols[self.sample_stimulus % len(symbols)]
            for dy in [-1, 0, 1]:
                for dx in [-2, -1, 0, 1, 2]:
                    if 0 < center_x + dx < width - 1 and 0 < center_y + dy < height - 1:
                        c.put(center_x + dx, center_y + dy, stim)
            
        elif self.phase == 'delay':
            pass
            
        elif self.phase == 'choice':
            left_x = width // 4
            right_x = 3 * width // 4
            
            left_stim = symbols[self.choice_stimuli[0] % len(symbols)]
            right_stim = symbols[self.choice_stimuli[1] % len(symbols)]
            
            for dy in [-1, 0, 1]:
                c.put(left_x, center_y + dy, left_stim)
                c.put(right_x, center_y + dy, right_stim)
            
            # Show selection pointer under chosen stimulus (use ▼ to avoid FPV agent detection)
            if self.choice_position == 0:
                c.put(left_x, center_y + 2, '▼')
            else:
                c.put(right_x, center_y + 2, '▼')
        
        elif self.phase == 'response':
            if self._trial_reward > 0:
                c.put(center_x, center_y, '*')
            else:
                c.put(center_x, center_y, 'X')
        
        # Agent marker at bottom center (all phases)
        c.put(center_x, height - 2, '↑')
        
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
        
        # Symbols for stimuli
        symbols = ['●', '■', '▲', '◆']
        
        # Chamber raycasting
        chamber_depth = 4.0
        chamber_width = 3.0
        fov = np.pi / 2
        
        # Cast rays
        ray_distances = []
        for col in range(view_width):
            ray_offset = (col / view_width - 0.5) * fov
            ray_angle = np.pi / 2 + ray_offset  # Looking forward
            
            dx = np.cos(ray_angle)
            dy = np.sin(ray_angle)
            
            if abs(dy) > 0.01:
                dist_front = chamber_depth / abs(dy)
            else:
                dist_front = 20
            
            if abs(dx) > 0.01:
                dist_side = chamber_width / abs(dx)
            else:
                dist_side = 20
            
            dist = min(dist_front, dist_side)
            dist *= np.cos(ray_offset)  # Fish-eye correction
            ray_distances.append(max(0.5, dist))
        
        lines.append("╔" + "═" * view_width + "╗")
        
        # Render view
        for row in range(view_height):
            row_chars = []
            for col in range(view_width):
                dist = ray_distances[col]
                wall_height = int(view_height * 1.3 / (dist + 0.3))
                half_wall = wall_height // 2
                center = view_height // 2
                
                if row < center - half_wall:
                    char = '░'  # Ceiling
                elif row > center + half_wall:
                    char = '▓'  # Floor
                else:
                    char = wall_char(dist)  # Wall
                
                row_chars.append(char)
            
            row_str = ''.join(row_chars)
            
            # Add phase-specific content
            center_col = view_width // 2
            center_row = view_height // 2
            
            if self.phase == 'sample':
                # Show sample stimulus centered
                stim_sym = symbols[self.sample_stimulus % len(symbols)]
                if abs(row - center_row) <= 3:
                    size = 5 - abs(row - center_row)
                    stim_str = stim_sym * size
                    start = center_col - len(stim_str) // 2
                    if start >= 0 and start + len(stim_str) <= view_width:
                        row_str = row_str[:start] + stim_str + row_str[start + len(stim_str):]
            
            elif self.phase == 'delay':
                # Blank/dark screen during delay (realistic - no visual timer)
                # Animals just wait in darkness during retention interval
                pass  # Keep the basic chamber view, no progress indicator
            
            elif self.phase == 'choice':
                # Show two stimuli side by side
                left_sym = symbols[self.choice_stimuli[0] % len(symbols)]
                right_sym = symbols[self.choice_stimuli[1] % len(symbols)]
                
                if abs(row - center_row) <= 2:
                    size = 3 - abs(row - center_row)
                    left_pos = view_width // 4
                    right_pos = 3 * view_width // 4
                    
                    # Highlight selected
                    left_str = left_sym * size
                    right_str = right_sym * size
                    
                    row_str = row_str[:left_pos - size//2] + left_str + row_str[left_pos + size//2 + size%2:]
                    row_str = row_str[:right_pos - size//2] + right_str + row_str[right_pos + size//2 + size%2:]
                
                # Selection indicator
                if row == center_row + 4:
                    if self.choice_position == 0:
                        row_str = row_str[:view_width//4 - 1] + '▼▼▼' + row_str[view_width//4 + 2:]
                    elif self.choice_position == 1:
                        row_str = row_str[:3*view_width//4 - 1] + '▼▼▼' + row_str[3*view_width//4 + 2:]
            
            elif self.phase == 'response':
                # Show result
                result_sym = '★' if self._trial_reward > 0 else '✗'
                if abs(row - center_row) <= 2:
                    size = 5 - abs(row - center_row)
                    result_str = result_sym * size
                    start = center_col - len(result_str) // 2
                    row_str = row_str[:start] + result_str + row_str[start + len(result_str):]
            
            lines.append("║" + row_str + "║")
        
        lines.append("╚" + "═" * view_width + "╝")
        
        return '\n'.join(lines)
