"""
Create a GIF showing valid trajectories for all 9 environments across all visualization modes.
Shows FPV_3D, TOPDOWN_2D, ASCII_2D, ASCII_2D_FPV, ASCII_3D for each environment.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from collections import deque
import math
import os

from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze, StarMaze,
    OperantChamber, ShuttleBox, PlacePreference, DNMSTask
)
from environments.base_env import ViewMode, Action

# Configuration
FPS = 8
OUTPUT_DIR = "demo_gifs"
FRAME_SIZE = 224  # Size for each view


# ============== Path Finding Utilities ==============

def get_valid_positions_from_map(env):
    """Extract valid positions from a pool_map."""
    valid = set()
    for y, row in enumerate(env.pool_map):
        for x, cell in enumerate(row):
            if cell in ('~', 'P'):
                gx = x - env.map_offset
                gy = y - env.map_offset
                valid.add((gx, gy))
    return valid


def get_valid_positions_from_collision(env, bounds=(-20, 20)):
    """Generate valid positions by testing collision function."""
    valid = set()
    for x in range(bounds[0], bounds[1]+1):
        for y in range(bounds[0], bounds[1]+1):
            if not env._check_collision_at(x, y):
                valid.add((x, y))
    return valid


def bfs_path(valid_positions, start, goal):
    """BFS that respects corner-cutting restrictions."""
    queue = deque([(start, [start])])
    visited = {start}
    dirs = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in valid_positions or (nx, ny) in visited:
                continue
            # Check corner-cutting for diagonals
            if dx != 0 and dy != 0:
                if (x+dx, y) not in valid_positions or (x, y+dy) not in valid_positions:
                    continue
            visited.add((nx, ny))
            queue.append(((nx, ny), path + [(nx, ny)]))
    return None


def angle_for_dir(dx, dy):
    """Convert direction to angle index (0=E, 1=NE, ...)"""
    return round(math.atan2(dy, dx) / (math.pi/4)) % 8


def turn_toward(current, target):
    """Return action to turn toward target angle."""
    diff = (target - current) % 8
    if diff == 0:
        return None
    return Action.TURN_LEFT if diff <= 4 else Action.TURN_RIGHT


# ============== Action Generation for Each Environment ==============

def generate_navigation_actions(env, valid_positions, goal_x, goal_y, interact_at_goal=False):
    """Generate actions to navigate from current position to goal."""
    actions = []
    path = bfs_path(valid_positions, (int(env.agent.x), int(env.agent.y)), (goal_x, goal_y))
    
    if not path:
        return [Action.STAY] * 10
    
    # Simulate navigation to generate action sequence
    agent_x, agent_y = env.agent.x, env.agent.y
    agent_angle = env.agent.angle
    
    idx = 1
    while idx < len(path):
        target = path[idx]
        dx = target[0] - agent_x
        dy = target[1] - agent_y
        
        if dx == 0 and dy == 0:
            idx += 1
            continue
        
        target_angle = angle_for_dir(dx, dy)
        turn = turn_toward(agent_angle, target_angle)
        
        if turn:
            actions.append(turn)
            if turn == Action.TURN_LEFT:
                agent_angle = (agent_angle + 1) % 8
            else:
                agent_angle = (agent_angle - 1) % 8
        else:
            actions.append(Action.FORWARD)
            agent_x, agent_y = target
            idx += 1
    
    if interact_at_goal:
        actions.append(Action.INTERACT)
    
    # Add celebration
    actions.extend([Action.STAY] * 3)
    
    return actions


def generate_morris_actions(env):
    """Generate actions for MorrisWaterMaze."""
    valid = get_valid_positions_from_map(env)
    return generate_navigation_actions(env, valid, env.goal_x, env.goal_y)


def generate_tmaze_actions(env):
    """Generate actions for TMaze."""
    valid = get_valid_positions_from_collision(env, (-10, 10))
    return generate_navigation_actions(env, valid, env.goal_x, env.goal_y)


def generate_barnes_actions(env):
    """Generate actions for BarnesMaze (needs INTERACT)."""
    return generate_navigation_actions(env, env.valid_positions, env.goal_x, env.goal_y, interact_at_goal=True)


def generate_radial_actions(env):
    """Generate actions for RadialArmMaze - must collect ALL rewards."""
    actions = []
    
    # Need to visit all rewarded arm ends
    # Agent starts at center, facing random direction
    agent_x, agent_y = env.agent.x, env.agent.y
    agent_angle = env.agent.angle
    
    def navigate_to_target(target_x, target_y):
        """Navigate from current position to target."""
        nonlocal agent_x, agent_y, agent_angle
        nav_actions = []
        
        path = bfs_path(env.valid_positions, (int(agent_x), int(agent_y)), (target_x, target_y))
        if not path:
            return nav_actions
        
        idx = 0
        while idx < len(path):
            target = path[idx]
            dx = target[0] - agent_x
            dy = target[1] - agent_y
            
            if dx == 0 and dy == 0:
                idx += 1
                continue
            
            tgt_angle = angle_for_dir(dx, dy)
            turn = turn_toward(agent_angle, tgt_angle)
            
            if turn:
                nav_actions.append(turn)
                if turn == Action.TURN_LEFT:
                    agent_angle = (agent_angle + 1) % 8
                else:
                    agent_angle = (agent_angle - 1) % 8
            else:
                nav_actions.append(Action.FORWARD)
                agent_x, agent_y = target
                idx += 1
        
        return nav_actions
    
    # Visit each rewarded arm in order
    for arm_idx in env.rewarded_arms:
        ex, ey = env.arm_ends[arm_idx]
        actions.extend(navigate_to_target(ex, ey))
        # Return to center for next arm
        actions.extend(navigate_to_target(env.center_x, env.center_y))
    
    actions.extend([Action.STAY] * 3)
    return actions


def generate_star_actions(env):
    """Generate actions for StarMaze."""
    return generate_navigation_actions(env, env.valid_positions, env.goal_x, env.goal_y)


def generate_operant_actions(env):
    """Generate actions for OperantChamber."""
    actions = []
    # Turn to face active lever
    if env.active_lever == 0:
        actions.append(Action.TURN_LEFT)
    else:
        actions.append(Action.TURN_RIGHT)
    # Press lever multiple times
    for _ in range(5):
        actions.append(Action.INTERACT)
        actions.append(Action.STAY)
    return actions


def generate_shuttle_actions(env):
    """Generate actions for ShuttleBox."""
    actions = []
    # Wait for cue
    actions.extend([Action.STAY] * 5)
    # Move to other chamber
    for _ in range(15):
        actions.append(Action.FORWARD)
    actions.extend([Action.STAY] * 3)
    return actions


def generate_place_actions(env):
    """Generate actions for PlacePreference."""
    actions = []
    # Turn toward conditioning chamber
    if env.conditioning_chamber == 0:  # Left
        actions.extend([Action.TURN_LEFT] * 4)
    # Move forward into chamber
    for _ in range(10):
        actions.append(Action.FORWARD)
    actions.extend([Action.STAY] * 5)
    return actions


def generate_dnms_actions(env):
    """Generate actions for DNMSTask."""
    actions = []
    # Acknowledge sample
    actions.append(Action.INTERACT)
    # Wait through delay
    actions.extend([Action.STAY] * 10)
    # Make correct choice
    if env.correct_choice == 0:
        actions.append(Action.TURN_LEFT)
    else:
        actions.append(Action.TURN_RIGHT)
    # Confirm choice
    actions.append(Action.INTERACT)
    actions.extend([Action.STAY] * 3)
    return actions


# ============== Rendering Utilities ==============

def render_ascii_to_image(ascii_text: str, width: int = FRAME_SIZE, height: int = FRAME_SIZE) -> np.ndarray:
    """Convert ASCII art to a numpy image array."""
    img = Image.new('RGB', (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 8)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 8)
        except:
            font = ImageFont.load_default()
    
    # Color mapping
    color_map = {
        '#': (100, 100, 120), '█': (100, 100, 120), '▓': (80, 80, 100),
        '.': (60, 60, 80), ' ': (40, 40, 50),
        'G': (50, 255, 50), '*': (255, 215, 0), 'P': (200, 200, 50),
        '↑': (255, 100, 100), '↓': (255, 100, 100), '←': (255, 100, 100), '→': (255, 100, 100),
        '↖': (255, 100, 100), '↗': (255, 100, 100), '↙': (255, 100, 100), '↘': (255, 100, 100),
        '^': (255, 100, 100), 'v': (255, 100, 100), '<': (255, 100, 100), '>': (255, 100, 100),
        'A': (255, 100, 100), '@': (255, 100, 100),
        'O': (200, 150, 100), 'o': (150, 100, 50), 'E': (50, 200, 50),
        '~': (50, 100, 200), '|': (80, 80, 100), '-': (80, 80, 100),
        '/': (120, 80, 60), '\\': (120, 80, 60), '_': (60, 60, 80),
        'L': (180, 180, 200), 'M': (255, 200, 100), '!': (255, 50, 50), 'X': (255, 50, 50),
    }
    
    lines = ascii_text.split('\n')
    y_offset = 5
    for line in lines:
        x_offset = 5
        for char in line:
            color = color_map.get(char, (180, 180, 180))
            draw.text((x_offset, y_offset), char, fill=color, font=font)
            x_offset += 6
        y_offset += 9
    
    return np.array(img)


def create_env_frame(env, view_mode, env_name: str, view_name: str) -> np.ndarray:
    """Create a frame for a single environment/view combination."""
    env.view_mode = view_mode
    obs = env.render()
    
    if isinstance(obs, str):
        # ASCII mode
        frame = render_ascii_to_image(obs)
    else:
        # Image mode - resize to FRAME_SIZE
        if obs.shape[0] != FRAME_SIZE or obs.shape[1] != FRAME_SIZE:
            img = Image.fromarray(obs)
            img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.LANCZOS)
            frame = np.array(img)
        else:
            frame = obs
    
    # Add labels
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    # Environment name at top
    draw.rectangle([(0, 0), (FRAME_SIZE, 14)], fill=(0, 0, 0, 180))
    draw.text((FRAME_SIZE // 2, 2), env_name, fill=(255, 255, 100), font=font, anchor="mt")
    
    # View name at bottom
    draw.rectangle([(0, FRAME_SIZE - 14), (FRAME_SIZE, FRAME_SIZE)], fill=(0, 0, 0, 180))
    draw.text((FRAME_SIZE // 2, FRAME_SIZE - 12), view_name, fill=(100, 200, 255), font=font, anchor="mt")
    
    return np.array(img)


def create_grid_frame(env_frames: list, cols: int = 5) -> np.ndarray:
    """Create a grid of environment frames."""
    n = len(env_frames)
    rows = (n + cols - 1) // cols
    
    padding = 2
    grid_width = cols * FRAME_SIZE + (cols + 1) * padding
    grid_height = rows * FRAME_SIZE + (rows + 1) * padding
    
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid[:] = [10, 10, 15]  # Dark background
    
    for i, frame in enumerate(env_frames):
        row = i // cols
        col = i % cols
        x = padding + col * (FRAME_SIZE + padding)
        y = padding + row * (FRAME_SIZE + padding)
        grid[y:y+FRAME_SIZE, x:x+FRAME_SIZE] = frame
    
    return grid


# ============== Main GIF Generation ==============

ENV_CONFIGS = [
    ("MorrisWaterMaze", MorrisWaterMaze, generate_morris_actions),
    ("TMaze", TMaze, generate_tmaze_actions),
    ("BarnesMaze", BarnesMaze, generate_barnes_actions),
    ("RadialArmMaze", RadialArmMaze, generate_radial_actions),
    ("StarMaze", StarMaze, generate_star_actions),
    ("OperantChamber", OperantChamber, generate_operant_actions),
    ("ShuttleBox", ShuttleBox, generate_shuttle_actions),
    ("PlacePreference", PlacePreference, generate_place_actions),
    ("DNMSTask", DNMSTask, generate_dnms_actions),
]

VIEW_MODES = [
    (ViewMode.FPV_3D, "FPV 3D"),
    (ViewMode.TOPDOWN_2D, "Top-Down"),
    (ViewMode.ASCII_2D, "ASCII 2D"),
    (ViewMode.ASCII_2D_FPV, "ASCII FPV"),
    (ViewMode.ASCII_3D, "ASCII 3D"),
]


def create_single_env_gif(env_name: str, env_class, action_generator, output_path: str):
    """Create a GIF for a single environment showing all view modes."""
    print(f"  Creating GIF for {env_name}...")
    
    # Create environments for each view mode
    envs = []
    for view_mode, _ in VIEW_MODES:
        env = env_class(view_mode=view_mode)
        env.reset()
        envs.append(env)
    
    # Generate actions using first env
    actions = action_generator(envs[0])
    
    # Sync state across all envs
    base_env = envs[0]
    for env in envs[1:]:
        env.agent.x = base_env.agent.x
        env.agent.y = base_env.agent.y
        env.agent.angle = base_env.agent.angle
        if hasattr(base_env, 'goal_x'):
            env.goal_x = base_env.goal_x
            env.goal_y = base_env.goal_y
        if hasattr(base_env, 'valid_positions'):
            env.valid_positions = base_env.valid_positions.copy() if hasattr(base_env.valid_positions, 'copy') else base_env.valid_positions
    
    # Collect frames
    frames = []
    
    # Initial frame
    env_frames = [create_env_frame(env, vm, env_name, vn) for env, (vm, vn) in zip(envs, VIEW_MODES)]
    frames.append(create_grid_frame(env_frames, cols=5))
    
    # Execute actions
    for action in actions:
        for env in envs:
            try:
                env.step(action)
            except:
                pass
        
        env_frames = [create_env_frame(env, vm, env_name, vn) for env, (vm, vn) in zip(envs, VIEW_MODES)]
        frames.append(create_grid_frame(env_frames, cols=5))
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=FPS, loop=0)
    print(f"    Saved {len(frames)} frames to {output_path}")


def create_all_envs_single_view_gif(view_mode: ViewMode, view_name: str, output_path: str):
    """Create a GIF showing all environments in a single view mode."""
    print(f"  Creating GIF for {view_name}...")
    
    # Create all environments
    envs = []
    action_lists = []
    for env_name, env_class, action_gen in ENV_CONFIGS:
        env = env_class(view_mode=view_mode)
        env.reset()
        envs.append((env_name, env))
        action_lists.append(action_gen(env))
    
    # Find max action length
    max_actions = max(len(a) for a in action_lists)
    
    # Pad action lists
    for i, actions in enumerate(action_lists):
        while len(action_lists[i]) < max_actions:
            action_lists[i].append(Action.STAY)
    
    # Collect frames
    frames = []
    
    # Initial frame
    env_frames = [create_env_frame(env, view_mode, name, view_name) for name, env in envs]
    frames.append(create_grid_frame(env_frames, cols=3))
    
    # Execute actions
    for step in range(max_actions):
        for i, (name, env) in enumerate(envs):
            try:
                env.step(action_lists[i][step])
            except:
                pass
        
        env_frames = [create_env_frame(env, view_mode, name, view_name) for name, env in envs]
        frames.append(create_grid_frame(env_frames, cols=3))
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=FPS, loop=0)
    print(f"    Saved {len(frames)} frames to {output_path}")


def main():
    """Generate all demo GIFs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Creating Demo GIFs for All Environments")
    print("=" * 60)
    
    # 1. Create individual environment GIFs (each env, all views)
    print("\n[1/3] Creating per-environment GIFs (all view modes)...")
    for env_name, env_class, action_gen in ENV_CONFIGS:
        output_path = os.path.join(OUTPUT_DIR, f"{env_name.lower()}_all_views.gif")
        create_single_env_gif(env_name, env_class, action_gen, output_path)
    
    # 2. Create per-view-mode GIFs (all envs, single view)
    print("\n[2/3] Creating per-view-mode GIFs (all environments)...")
    for view_mode, view_name in VIEW_MODES:
        safe_name = view_name.lower().replace(" ", "_").replace("-", "_")
        output_path = os.path.join(OUTPUT_DIR, f"all_envs_{safe_name}.gif")
        create_all_envs_single_view_gif(view_mode, view_name, output_path)
    
    # 3. Create master GIF with one representative frame per env/view
    print("\n[3/3] Creating master overview GIF...")
    create_master_gif()
    
    print("\n" + "=" * 60)
    print(f"Done! GIFs saved to {OUTPUT_DIR}/")
    print("=" * 60)


def create_master_gif():
    """Create a master GIF showing all environments cycling through views."""
    output_path = os.path.join(OUTPUT_DIR, "master_all_envs_all_views.gif")
    print(f"  Creating master GIF...")
    
    frames = []
    
    for view_mode, view_name in VIEW_MODES:
        # Create all environments for this view
        envs = []
        for env_name, env_class, _ in ENV_CONFIGS:
            env = env_class(view_mode=view_mode)
            env.reset()
            envs.append((env_name, env))
        
        # Create frame showing all envs in this view
        env_frames = [create_env_frame(env, view_mode, name, view_name) for name, env in envs]
        grid = create_grid_frame(env_frames, cols=3)
        
        # Hold this view for a few frames
        for _ in range(FPS * 2):  # 2 seconds per view
            frames.append(grid)
    
    imageio.mimsave(output_path, frames, fps=FPS, loop=0)
    print(f"    Saved {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
