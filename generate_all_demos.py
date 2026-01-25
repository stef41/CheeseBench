"""
Generate demo videos for all environments showing optimal solutions.
Creates a single MP4 video per environment with all 3 view modes side by side.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import os
from typing import List, Dict, Tuple

from environments import (
    TMaze, BarnesMaze, MorrisWaterMaze, RadialArmMaze, StarMaze,
    OperantChamber, ShuttleBox, PlacePreference, DNMSTask
)
from environments.base_env import ViewMode, Action

# Configuration
FPS = 5
OUTPUT_DIR = "demo_gifs"


def get_font(size=12):
    """Get a font, falling back to default if needed."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def render_ascii_to_image(ascii_text: str, width: int = 400, height: int = 400) -> Image.Image:
    """Convert ASCII art to a PIL Image."""
    img = Image.new('RGB', (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    font = get_font(11)
    
    # Color mapping
    color_map = {
        '#': (100, 100, 120),   # Walls
        '.': (60, 80, 60),      # Floor
        ' ': (30, 30, 40),      # Empty
        'E': (50, 255, 50),     # Exit/End
        'G': (50, 255, 50),     # Goal
        'P': (50, 200, 255),    # Platform
        '?': (255, 200, 100),   # Unknown
        '*': (255, 215, 0),     # Reward
        'R': (255, 100, 100),   # Reward marker
        '~': (50, 100, 200),    # Water
        '1': (255, 100, 100),   # Landmarks
        '2': (100, 255, 100),
        '3': (100, 100, 255),  
        '4': (255, 255, 100),
        '↑': (255, 80, 80),     # Agent
        '↓': (255, 80, 80),
        '←': (255, 80, 80),
        '→': (255, 80, 80),
        '↗': (255, 80, 80),
        '↘': (255, 80, 80),
        '↙': (255, 80, 80),
        '↖': (255, 80, 80),
        '@': (255, 80, 80),
        'L': (200, 150, 50),    # Lever
        'S': (255, 100, 100),   # Stimulus
        'C': (100, 200, 255),   # Cue
        '●': (255, 200, 100),   # Circle/stimulus
        '■': (200, 100, 200),   # Square/stimulus
        '[': (150, 150, 150),   # Brackets
        ']': (150, 150, 150),
        '=': (200, 200, 100),   # Lever up
        '_': (100, 100, 100),   # Lever down
        'M': (255, 215, 0),     # Magazine lit
        'm': (100, 100, 100),   # Magazine unlit
    }
    
    lines = ascii_text.split('\n')
    y_offset = 10
    
    for line in lines:
        x_offset = 10
        for char in line:
            color = color_map.get(char, (180, 180, 180))
            draw.text((x_offset, y_offset), char, fill=color, font=font)
            x_offset += 8
        y_offset += 14
    
    return img


def create_combined_frame(observations: Dict[str, str], env_name: str, step: int, 
                          reward: float, total_reward: float, action: str = "") -> Image.Image:
    """Create a single frame with all 3 view modes side by side."""
    
    view_width = 400
    view_height = 400
    padding = 10
    header_height = 60
    footer_height = 40
    
    total_width = view_width * 3 + padding * 4
    total_height = view_height + header_height + footer_height + padding * 2
    
    # Create main canvas
    img = Image.new('RGB', (total_width, total_height), color=(15, 15, 20))
    draw = ImageDraw.Draw(img)
    
    font_large = get_font(20)
    font_medium = get_font(14)
    font_small = get_font(12)
    
    # Header with env name and step info
    draw.text((padding, 10), env_name, fill=(100, 200, 255), font=font_large)
    
    # Step and action info on the right side of header
    info_x = total_width - 350
    draw.text((info_x, 10), f"Step: {step}", fill=(150, 150, 150), font=font_medium)
    if action:
        draw.text((info_x + 100, 10), f"Action: {action}", fill=(200, 200, 100), font=font_medium)
    
    # Reward info
    reward_color = (100, 255, 100) if reward >= 0 else (255, 100, 100)
    draw.text((info_x, 32), f"Reward: {reward:+.2f}", fill=reward_color, font=font_medium)
    draw.text((info_x + 130, 32), f"Total: {total_reward:.2f}", fill=(150, 150, 150), font=font_medium)
    
    # Render each view mode
    view_labels = ["ASCII_2D (Top-down)", "ASCII_2D_FPV (Cropped)", "ASCII_3D (First-person)"]
    view_keys = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
    
    for i, (label, view_key) in enumerate(zip(view_labels, view_keys)):
        x = padding + i * (view_width + padding)
        y = header_height
        
        # View label
        draw.text((x + 10, y - 18), label, fill=(180, 180, 180), font=font_small)
        
        # Render ASCII to image
        obs_text = observations.get(view_key, "No observation")
        ascii_img = render_ascii_to_image(obs_text, view_width, view_height)
        
        # Paste onto main canvas
        img.paste(ascii_img, (x, y))
        
        # Border around view
        draw.rectangle([x-1, y-1, x + view_width, y + view_height], outline=(60, 60, 80))
    
    return img


def generate_optimal_actions(env_name: str, env) -> List[Action]:
    """Generate optimal action sequence for each environment.
    
    These are adaptive solutions that account for the environment's
    actual starting state (agent position, angle, goals, etc.)
    """
    
    def rotate_to_angle(current: int, target: int) -> List[Action]:
        """Generate rotations to turn from current to target angle (0-7).
        
        Angles: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        ROTATE_LEFT increases angle (0->1->2...)
        ROTATE_RIGHT decreases angle (0->7->6...)
        """
        actions = []
        diff = (target - current) % 8
        if diff == 0:
            return actions
        if diff <= 4:
            # Need to increase angle -> ROTATE_LEFT
            for _ in range(diff):
                actions.append(Action.ROTATE_LEFT)
        else:
            # Need to decrease angle -> ROTATE_RIGHT  
            for _ in range(8 - diff):
                actions.append(Action.ROTATE_RIGHT)
        return actions
    
    if env_name == "TMaze":
        # TMaze: Go forward to junction, turn to rewarded arm, go to end
        # Rewarded arm is West (left from start facing North)
        actions = [
            Action.FORWARD, Action.FORWARD, Action.FORWARD,  # To junction
            Action.ROTATE_LEFT, Action.ROTATE_LEFT,           # Turn West
            Action.FORWARD, Action.FORWARD                    # To reward
        ]
        return actions
    
    elif env_name == "BarnesMaze":
        # BarnesMaze: Rotate to face the exit hole (angle 0 = East), walk to it
        actions = rotate_to_angle(env.agent.angle, 0)
        actions.extend([Action.FORWARD] * 8)  # Walk to edge
        return actions
    
    elif env_name == "MorrisWaterMaze":
        # MorrisWaterMaze: Navigate to hidden platform
        # Two-step approach: align x, then align y
        import math
        
        px, py = env.goal_x, env.goal_y
        ax, ay = env.agent.x, env.agent.y
        
        actions = []
        
        # Step 1: Align x-coordinate
        dx = px - ax
        if dx > 0:
            # Go East
            actions.extend(rotate_to_angle(env.agent.angle, 0))
            actions.extend([Action.FORWARD] * abs(dx))
        elif dx < 0:
            # Go West
            actions.extend(rotate_to_angle(env.agent.angle, 4))
            actions.extend([Action.FORWARD] * abs(dx))
        
        # After walking, agent's angle is still the same
        current_angle = 0 if dx > 0 else (4 if dx < 0 else env.agent.angle)
        
        # Step 2: Align y-coordinate
        dy = py - ay
        if dy > 0:
            # Go North
            actions.extend(rotate_to_angle(current_angle, 2))
            actions.extend([Action.FORWARD] * (abs(dy) + 2))  # Extra to ensure hit
        elif dy < 0:
            # Go South
            actions.extend(rotate_to_angle(current_angle, 6))
            actions.extend([Action.FORWARD] * (abs(dy) + 2))
        
        return actions
    
    elif env_name == "RadialArmMaze":
        # RadialArmMaze: Visit all 4 rewarded arms (0=E, 2=N, 4=W, 6=S)
        # Arm ends are 10 cells from center, rewards collected within 1 cell of end
        # Walk 9 steps to reach reward zone, then 9 steps back to center
        actions = []
        
        # First, rotate to face East (arm 0) from whatever angle we're at
        actions.extend(rotate_to_angle(env.agent.angle, 0))
        
        # Visit arm 0 (East)
        actions.extend([Action.FORWARD] * 9)  # To reward zone
        actions.extend([Action.ROTATE_LEFT] * 4)  # Turn 180°
        actions.extend([Action.FORWARD] * 9)  # Back to center
        
        # From West, rotate to North: 4 -> 2 = RIGHT 2
        actions.extend([Action.ROTATE_RIGHT] * 2)
        
        # Visit arm 2 (North)
        actions.extend([Action.FORWARD] * 9)
        actions.extend([Action.ROTATE_LEFT] * 4)
        actions.extend([Action.FORWARD] * 9)
        
        # From South, rotate to West: 6 -> 4 = RIGHT 2
        actions.extend([Action.ROTATE_RIGHT] * 2)
        
        # Visit arm 4 (West)
        actions.extend([Action.FORWARD] * 9)
        actions.extend([Action.ROTATE_LEFT] * 4)
        actions.extend([Action.FORWARD] * 9)
        
        # From East, rotate to South: 0 -> 6 = RIGHT 2
        actions.extend([Action.ROTATE_RIGHT] * 2)
        
        # Visit arm 6 (South)
        actions.extend([Action.FORWARD] * 9)
        actions.extend([Action.ROTATE_LEFT] * 4)
        actions.extend([Action.FORWARD] * 9)
        
        return actions
    
    elif env_name == "StarMaze":
        # StarMaze: Navigate through the star-shaped maze to goal
        # Strategy: walk from start arm to center, turn to goal arm, walk to goal
        actions = []
        
        # Agent starts facing toward center (opposite of start arm direction)
        # Walk toward center (about 9 steps gets us there from arm end)
        actions.extend([Action.FORWARD] * 9)
        
        # Turn to face goal arm direction
        goal_dir = env._arm_to_direction[env.goal_arm]
        # After walking toward center, agent is still facing opposite of start arm direction
        current_angle = (env._arm_to_direction[env.start_arm] + 4) % 8
        actions.extend(rotate_to_angle(current_angle, goal_dir))
        
        # Walk down goal arm to reach goal (arm length + some extra)
        actions.extend([Action.FORWARD] * 12)
        
        return actions
    
    elif env_name == "OperantChamber":
        actions = []
        if env.active_lever == 0:
            actions.append(Action.ROTATE_LEFT)
        else:
            actions.append(Action.ROTATE_RIGHT)
        actions.extend([Action.FORWARD] * 15)
        return actions
    
    elif env_name == "ShuttleBox":
        actions = []
        actions.extend([Action.STAY] * 5)
        actions.extend([Action.FORWARD] * 8)
        actions.extend([Action.STAY] * 5)
        actions.extend([Action.FORWARD] * 8)
        actions.extend([Action.STAY] * 5)
        actions.extend([Action.FORWARD] * 8)
        return actions
    
    elif env_name == "PlacePreference":
        actions = []
        if env.agent.angle < 2:
            actions.extend([Action.ROTATE_LEFT] * 4)
        actions.extend([Action.FORWARD] * 8)
        actions.extend([Action.STAY] * 25)
        return actions
    
    elif env_name == "DNMSTask":
        actions = [Action.STAY]
        actions.extend([Action.STAY] * 12)
        if env.correct_choice == 0:
            actions.append(Action.ROTATE_LEFT)
        else:
            actions.append(Action.ROTATE_RIGHT)
        actions.append(Action.FORWARD)
        actions.extend([Action.STAY] * 15)
        return actions
    
    return [Action.FORWARD] * 20


def run_env_demo_combined(env_class, env_name: str) -> Tuple[List[np.ndarray], bool]:
    """Run a demo for one environment with all 3 view modes side by side.
    
    Uses a SINGLE environment instance and switches view_mode for rendering
    to ensure all views show the exact same state.
    """
    
    view_modes = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
    
    # Create a SINGLE environment instance
    env = env_class(view_mode=ViewMode.ASCII_2D)
    env.reset()
    
    # Get actions based on the environment state
    actions = generate_optimal_actions(env_name, env)
    
    frames = []
    total_reward = 0.0
    
    def get_all_observations():
        """Render the same env state in all view modes."""
        observations = {}
        original_mode = env.view_mode
        for vm in view_modes:
            env.view_mode = vm
            observations[vm] = env.render()
        env.view_mode = original_mode
        return observations
    
    # Initial frame
    observations = get_all_observations()
    frame = create_combined_frame(observations, env_name, 0, 0.0, total_reward, "START")
    frames.append(np.array(frame))
    
    success = False
    initial_trial = env.session.current_trial
    
    for i, action in enumerate(actions):
        # Check if done or moved to next trial (success)
        if env.is_done:
            success = True
            break
        
        # Check if we moved to next trial (means previous trial succeeded)
        if env.session.current_trial > initial_trial:
            success = True
            break
        
        # Step the single environment
        result = env.step(action)
        if isinstance(result, tuple):
            obs, reward = result
        else:
            reward = result
        
        total_reward += reward
        
        # Check if reward indicates success (1.0 = task complete)
        if reward >= 1.0:
            success = True
        
        # Collect observations from all view modes (same env state)
        observations = get_all_observations()
        
        frame = create_combined_frame(observations, env_name, i + 1, reward, total_reward, action.name)
        frames.append(np.array(frame))
        
        # Check again after step
        if env.is_done or env.session.current_trial > initial_trial:
            success = True
            break
    
    # Add pause at end
    if frames:
        for _ in range(5):
            frames.append(frames[-1])
    
    return frames, success


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    environments = {
        "TMaze": TMaze,
        "BarnesMaze": BarnesMaze,
        "MorrisWaterMaze": MorrisWaterMaze,
        "RadialArmMaze": RadialArmMaze,
        "StarMaze": StarMaze,
        "OperantChamber": OperantChamber,
        "ShuttleBox": ShuttleBox,
        "PlacePreference": PlacePreference,
        "DNMSTask": DNMSTask,
    }
    
    results = {}
    
    for env_name, env_class in environments.items():
        print(f"\n{'='*50}")
        print(f"Processing {env_name}...", end=" ")
        
        try:
            frames, success = run_env_demo_combined(env_class, env_name)
            
            if frames:
                output_file = os.path.join(OUTPUT_DIR, f"{env_name}_all_views.gif")
                imageio.mimsave(output_file, frames, fps=FPS, loop=0)
                
                status = "SUCCESS" if success else "INCOMPLETE"
                print(f"{status} ({len(frames)} frames)")
                results[env_name] = {"status": status, "frames": len(frames)}
            else:
                print("No frames generated")
                results[env_name] = {"status": "ERROR", "frames": 0}
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[env_name] = {"status": f"ERROR: {e}", "frames": 0}
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - All 3 views side by side per environment")
    print("="*70)
    for env_name, data in results.items():
        print(f"  {env_name}: {data['status']} ({data['frames']} frames)")
    
    print(f"\nVideos saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
