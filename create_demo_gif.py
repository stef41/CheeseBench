"""
Create an MP4 video showing parallel views of an environment exploration.
Shows ASCII_2D, ASCII_2D_FPV, and ASCII_3D side by side.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import cv2
import os

from environments import StarMaze, MorrisWaterMaze, RadialArmMaze
from environments.base_env import ViewMode, Action

# Configuration
ENV_CLASS = StarMaze  # Most visually interesting with the arm structure
FPS = 10  # frames per second
OUTPUT_FILE = "demo_views.mp4"

# StarMaze arm structure:
# - Arm 0: direction 0 (E)  -> end at (22, 12)
# - Arm 1: direction 2 (N)  -> end at (12, 22)  
# - Arm 2: direction 3 (NW) -> end at (2, 22) - START ARM
# - Arm 3: direction 5 (SW) -> end at (2, 2)
# - Arm 4: direction 6 (S)  -> end at (12, 2)

# We'll fix goal to arm 3 (SW) and visit it LAST
GOAL_ARM = 3  # SW

def generate_exploration_actions():
    """
    Generate actions to explore StarMaze, visiting goal arm last.
    
    Agent starts at Arm 2 (NW), facing SE (angle 7).
    We visit arms in order: Start(2) -> Center -> 0(E) -> 4(S) -> 1(N) -> 3(SW/GOAL)
    
    Direction indices: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    TURN_LEFT: angle = (angle + 1) % 8
    TURN_RIGHT: angle = (angle - 1) % 8
    """
    actions = []
    
    def add(action, count):
        actions.extend([action] * count)
    
    # ========== Phase 1: Start arm (NW) to center ==========
    # Start at angle=7 (SE), go toward center
    add(Action.FORWARD, 10)
    add(Action.STAY, 2)
    # Now at center, angle=7
    
    # ========== Phase 2: Explore Arm 0 (E) ==========
    # From 7, turn left to 0: one turn
    add(Action.TURN_LEFT, 1)  # angle = 0 (E)
    add(Action.FORWARD, 12)   # Go East to arm end
    add(Action.STAY, 2)
    add(Action.TURN_LEFT, 4)  # Turn around: 0 -> 4 (W)
    add(Action.FORWARD, 12)   # Return to center
    add(Action.STAY, 2)
    # angle = 4 (W)
    
    # ========== Phase 3: Explore Arm 4 (S) ==========
    # From 4, turn right to 6: 4 -> 3 -> 2 -> 1 -> 0 -> 7 -> 6 (six turns)
    # Or turn left: 4 -> 5 -> 6 (two turns)
    add(Action.TURN_LEFT, 2)  # angle = 6 (S)
    add(Action.FORWARD, 12)   # Go South to arm end
    add(Action.STAY, 2)
    add(Action.TURN_LEFT, 4)  # Turn around: 6 -> 2 (N)
    add(Action.FORWARD, 12)   # Return to center
    add(Action.STAY, 2)
    # angle = 2 (N)
    
    # ========== Phase 4: Explore Arm 1 (N) ==========
    # Already facing N (2)!
    add(Action.FORWARD, 12)   # Go North to arm end
    add(Action.STAY, 2)
    add(Action.TURN_RIGHT, 4) # Turn around: 2 -> 6 (S)
    add(Action.FORWARD, 12)   # Return to center  
    add(Action.STAY, 2)
    # angle = 6 (S)
    
    # ========== Phase 5: Go to Arm 3 (SW) - THE GOAL ==========
    # From 6, turn right to 5: one turn
    add(Action.TURN_RIGHT, 1)  # angle = 5 (SW)
    add(Action.FORWARD, 15)    # Go SW to goal!
    
    # ========== Celebration! ==========
    add(Action.STAY, 5)
    add(Action.TURN_LEFT, 8)   # Full spin
    add(Action.TURN_RIGHT, 8)  # Spin back
    add(Action.STAY, 5)
    
    return actions

# Generate the full exploration sequence
ACTIONS = generate_exploration_actions()

def render_ascii_to_image(ascii_text: str, title: str, width: int = 400, height: int = 400) -> Image.Image:
    """Convert ASCII art to a PIL Image."""
    # Create image with dark background
    img = Image.new('RGB', (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        title_font = font
    
    # Draw title
    draw.text((width // 2 - len(title) * 4, 10), title, fill=(100, 200, 255), font=title_font, anchor="mm")
    
    # Draw ASCII content
    lines = ascii_text.split('\n')
    y_offset = 35
    
    # Color mapping for different characters
    color_map = {
        '#': (100, 100, 120),   # Walls - gray
        '█': (100, 100, 120),   # Walls - gray
        '▓': (80, 80, 100),     # Dark walls
        '.': (60, 60, 80),      # Floor
        ' ': (40, 40, 50),      # Empty
        'G': (50, 255, 50),     # Goal - green
        '*': (255, 215, 0),     # Reward - gold
        '^': (255, 100, 100),   # Agent facing up - red
        'v': (255, 100, 100),   # Agent facing down
        '<': (255, 100, 100),   # Agent facing left
        '>': (255, 100, 100),   # Agent facing right
        'A': (255, 100, 100),   # Agent
        '@': (255, 100, 100),   # Agent alt
        'O': (200, 150, 100),   # Holes
        'o': (150, 100, 50),    # Small holes
        '~': (50, 100, 200),    # Water
        '|': (80, 80, 100),     # Vertical wall
        '-': (80, 80, 100),     # Horizontal wall
        '+': (80, 80, 100),     # Corner
        '/': (120, 80, 60),     # Diagonal
        '\\': (120, 80, 60),    # Diagonal
        '_': (60, 60, 80),      # Floor
    }
    
    for line in lines:
        x_offset = 20
        for char in line:
            color = color_map.get(char, (180, 180, 180))
            draw.text((x_offset, y_offset), char, fill=color, font=font)
            x_offset += 7  # Approximate char width
        y_offset += 12  # Line height
    
    return img


def create_combined_frame(frames: list, titles: list) -> Image.Image:
    """Combine multiple frames horizontally with titles."""
    # Calculate dimensions
    frame_width = 400
    frame_height = 400
    padding = 10
    total_width = len(frames) * frame_width + (len(frames) + 1) * padding
    total_height = frame_height + 2 * padding
    
    # Create combined image
    combined = Image.new('RGB', (total_width, total_height), color=(10, 10, 15))
    
    # Paste each frame
    for i, (ascii_text, title) in enumerate(zip(frames, titles)):
        frame_img = render_ascii_to_image(ascii_text, title, frame_width, frame_height)
        x_pos = padding + i * (frame_width + padding)
        combined.paste(frame_img, (x_pos, padding))
    
    return combined


def main():
    print("Creating demo GIF with 3 view modes...")
    print(f"Action sequence: {len(ACTIONS)} actions")
    
    # Create environments with different view modes
    view_modes = [ViewMode.ASCII_2D, ViewMode.ASCII_2D_FPV, ViewMode.ASCII_3D]
    titles = ["Top-Down 2D", "First-Person 2D", "Pseudo-3D"]
    
    # Initialize environments
    envs = []
    for vm in view_modes:
        env = ENV_CLASS(view_mode=vm)
        env.reset()
        envs.append(env)
    
    # FIX the goal to arm 3 (SW) for ALL environments
    # This ensures we can explore all other arms without triggering a reset
    base_env = envs[0]
    base_env.goal_arm = GOAL_ARM
    base_env.goal_x, base_env.goal_y = base_env.arm_ends[GOAL_ARM]
    
    # Sync ALL state across environments
    for env in envs[1:]:
        # Sync agent state
        env.agent.x = base_env.agent.x
        env.agent.y = base_env.agent.y
        env.agent.angle = base_env.agent.angle
        # Sync goal (fixed to arm 3)
        env.goal_arm = GOAL_ARM
        env.goal_x = base_env.goal_x
        env.goal_y = base_env.goal_y
        # Sync maze structure
        env.valid_positions = base_env.valid_positions.copy()
        env.arm_ends = base_env.arm_ends.copy()
    
    print(f"Agent starts at ({base_env.agent.x}, {base_env.agent.y})")
    print(f"Goal FIXED at arm {GOAL_ARM} ({base_env.goal_x}, {base_env.goal_y})")
    print(f"Arm ends: {base_env.arm_ends}")
    
    # Collect frames
    frames = []
    
    # Initial frame
    renders = [env.render() for env in envs]
    combined = create_combined_frame(renders, titles)
    frames.append(np.array(combined))
    
    # Execute actions and capture frames
    for i, action in enumerate(ACTIONS):
        # Apply same action to all environments
        for env in envs:
            env.step(action)
        
        # Render all views
        renders = [env.render() for env in envs]
        combined = create_combined_frame(renders, titles)
        frames.append(np.array(combined))
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(ACTIONS)} actions...")
    
    # Save MP4 using OpenCV with H.264 codec
    print(f"Saving {len(frames)} frames to {OUTPUT_FILE}...")
    height, width = frames[0].shape[:2]
    # Use avc1 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (width, height))
    if not writer.isOpened():
        # Fallback: save as temp images and use ffmpeg
        print("  OpenCV writer failed, using ffmpeg...")
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                img.save(f"{tmpdir}/frame_{i:04d}.png")
            subprocess.run([
                'ffmpeg', '-y', '-framerate', str(FPS),
                '-i', f'{tmpdir}/frame_%04d.png',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                OUTPUT_FILE
            ], check=True, capture_output=True)
    else:
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        writer.release()
    print(f"Done! Video saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
