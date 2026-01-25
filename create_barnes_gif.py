"""
Create a GIF showing BarnesMaze 2D top-down view with LLM learnings on the side.
Uses actual observations from benchmark logs.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import textwrap
import re

from environments import BarnesMaze
from environments.base_env import ViewMode, Action

# Configuration
FPS = 5
OUTPUT_FILE = "barnes_maze_demo.mp4"
LOG_FILE = "llm_traces.log"


def parse_log_observations(log_file: str, env_name: str = "BarnesMaze", view_mode: str = "ASCII_2D"):
    """Parse observations, rewards, and learnings from the benchmark log file."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the start of our section
    start_idx = None
    end_idx = None
    target_header = f"Environment: {env_name} | Mode: {view_mode}"
    
    for i, line in enumerate(lines):
        if line.strip() == target_header:
            start_idx = i
        elif start_idx is not None and line.startswith("Environment:") and i > start_idx + 2:
            end_idx = i
            break
    
    if start_idx is None:
        print(f"Could not find {env_name} {view_mode} in log")
        return []
    
    if end_idx is None:
        end_idx = len(lines)
    
    section = ''.join(lines[start_idx:end_idx])
    
    # Parse trials
    trials = []
    trial_splits = re.split(r'--- Trial (\d+)/(\d+) ---', section)
    
    # trial_splits: ['', '1', '3', content1, '2', '3', content2, ...]
    i = 1
    while i < len(trial_splits) - 2:
        trial_num = int(trial_splits[i])
        trial_content = trial_splits[i + 2]
        
        # Parse steps - look for "Step N: ACTION -> reward=X.XX" followed by the ASCII map
        steps = []
        
        # Split by "Step N:" pattern
        step_parts = re.split(r'(  Step \d+: \w+ -> reward=[\d\.\-]+\n)', trial_content)
        
        # Also extract learnings from the LLM calls in this trial
        # Find all "LEARNINGS: ..." lines (not "unchanged")
        learnings_pattern = r'LEARNINGS: ([^\n]+)'
        all_learnings_matches = re.findall(learnings_pattern, trial_content)
        learnings_list = []
        for l in all_learnings_matches:
            l = l.strip()
            if l and l != "unchanged" and l not in learnings_list:
                learnings_list.append(l)
        
        for j in range(1, len(step_parts) - 1, 2):
            header = step_parts[j].strip()
            obs_block = step_parts[j + 1] if j + 1 < len(step_parts) else ""
            
            # Parse header: "Step N: ACTION -> reward=X.XX"
            header_match = re.match(r'Step (\d+): (\w+) -> reward=([\d\.\-]+)', header)
            if header_match:
                step_num = int(header_match.group(1))
                action = header_match.group(2)
                reward = float(header_match.group(3))
                
                # Extract observation - everything until next "Step" or "---" or "Result:"
                # Skip "Final observation after" lines and everything after them
                obs_lines = []
                skip_rest = False
                for line in obs_block.split('\n'):
                    if line.strip().startswith('Step ') or line.strip().startswith('---') or line.strip().startswith('Result:'):
                        break
                    if line.strip().startswith('Final observation after'):
                        skip_rest = True
                        continue
                    if skip_rest:
                        continue
                    obs_lines.append(line)
                
                # Remove leading/trailing empty lines but preserve internal whitespace
                while obs_lines and not obs_lines[0].strip():
                    obs_lines.pop(0)
                while obs_lines and not obs_lines[-1].strip():
                    obs_lines.pop()
                obs_text = '\n'.join(obs_lines)
                
                if obs_text:  # Only add if there's actual observation
                    steps.append({
                        'step': step_num,
                        'action': action,
                        'reward': reward,
                        'observation': obs_text
                    })
        
        # Check for result
        result_match = re.search(r'Result: (SUCCESS|FAILURE|TIMEOUT) in (\d+) steps', trial_content)
        result = result_match.group(1) if result_match else "TIMEOUT"
        result_steps = int(result_match.group(2)) if result_match else len(steps)
        
        trials.append({
            'trial_num': trial_num,
            'steps': steps,
            'result': result,
            'total_steps': result_steps,
            'learnings': learnings_list
        })
        
        i += 3
    
    return trials


def get_font(size=12, bold=False):
    """Get a font, falling back to default if needed."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            pass
    return ImageFont.load_default()


def render_ascii_to_image(ascii_text: str, width: int = 500, height: int = 500) -> Image.Image:
    """Convert ASCII art to a PIL Image."""
    img = Image.new('RGB', (width, height), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    font = get_font(14)
    
    # Color mapping for different characters
    color_map = {
        '#': (100, 100, 120),   # Walls - gray
        '.': (60, 80, 60),      # Floor - dark green
        ' ': (30, 30, 40),      # Empty
        'E': (50, 255, 50),     # Exit - green
        'G': (50, 255, 50),     # Goal - green
        '?': (255, 200, 100),   # Unknown - orange
        '*': (255, 215, 0),     # Reward - gold
        '1': (255, 100, 100),   # Landmarks - red tones
        '2': (100, 255, 100),   # Landmarks - green
        '3': (100, 100, 255),   # Landmarks - blue  
        '4': (255, 255, 100),   # Landmarks - yellow
        '↑': (255, 80, 80),     # Agent directions - bright red
        '↓': (255, 80, 80),
        '←': (255, 80, 80),
        '→': (255, 80, 80),
        '↗': (255, 80, 80),
        '↘': (255, 80, 80),
        '↙': (255, 80, 80),
        '↖': (255, 80, 80),
        '@': (255, 80, 80),     # Agent alt
    }
    
    lines = ascii_text.split('\n')
    y_offset = 20
    
    for line in lines:
        x_offset = 20
        for char in line:
            color = color_map.get(char, (180, 180, 180))
            draw.text((x_offset, y_offset), char, fill=color, font=font)
            x_offset += 10
        y_offset += 16
    
    return img


def render_learnings_panel(learnings: list, current_step: int, total_steps: int,
                           step_reward: float,
                           width: int = 450, height: int = 500) -> Image.Image:
    """Create a panel showing learnings and progress."""
    img = Image.new('RGB', (width, height), color=(25, 25, 35))
    draw = ImageDraw.Draw(img)
    
    title_font = get_font(22)
    text_font = get_font(12)
    small_font = get_font(10)
    
    y = 12
    
    # Title
    draw.text((20, y), "LLM Agent Learnings", fill=(100, 200, 255), font=title_font)
    y += 34
    
    # Divider
    draw.line([(20, y), (width - 20, y)], fill=(60, 60, 80), width=1)
    y += 10
    
    # Calculate max y before progress bar
    max_y = height - 80  # Leave space for progress bar
    
    # Learnings with NEW indicator for most recent (show newest first, limited to fit)
    max_learnings = 4  # Limit to prevent overflow
    display_learnings = list(reversed(learnings[-max_learnings:])) if learnings else []
    
    for i, learning in enumerate(display_learnings):
        is_newest = (i == 0)  # First one is newest since we reversed
        
        # Bullet with highlight for new
        bullet_color = (255, 255, 100) if is_newest else (100, 255, 100)
        draw.text((20, y), "•", fill=bullet_color, font=text_font)
        
        # NEW badge for newest learning
        if is_newest and len(learnings) > 0:
            draw.text((width - 50, y), "NEW", fill=(255, 200, 50), font=small_font)
        
        # Wrap text - show full learning but limit lines if near bottom
        text_color = (255, 255, 200) if is_newest else (180, 180, 180)
        wrapped = textwrap.wrap(learning, width=45)
        
        # Limit lines to avoid overlap with progress bar
        lines_available = (max_y - y) // 16
        if lines_available < len(wrapped):
            wrapped = wrapped[:max(1, lines_available - 1)]
            if len(wrapped) > 0:
                wrapped[-1] = wrapped[-1][:28] + "..."
        
        for j, line in enumerate(wrapped):
            draw.text((35, y + j * 16), line, fill=text_color, font=text_font)
        y += len(wrapped) * 16 + 10
        
        if y > max_y:  # Stop if running out of space
            break
    
    # Progress bar at bottom (always show 200 steps scale)
    y = height - 60
    max_steps = 200
    draw.text((20, y), f"Step: {current_step}/{max_steps}", fill=(150, 150, 150), font=small_font)
    y += 20
    
    # Progress bar
    bar_width = width - 40
    progress = current_step / max_steps
    draw.rectangle([(20, y), (20 + bar_width, y + 15)], outline=(80, 80, 100))
    draw.rectangle([(20, y), (20 + int(bar_width * progress), y + 15)], fill=(100, 200, 100))
    
    return img


def create_combined_frame(ascii_img: Image.Image, learnings_panel: Image.Image,
                          trial: int, total_trials: int, trial_result: str = "") -> Image.Image:
    """Combine the ASCII view and learnings panel."""
    padding = 10
    total_width = ascii_img.width + learnings_panel.width + 3 * padding
    total_height = max(ascii_img.height, learnings_panel.height) + 2 * padding
    
    combined = Image.new('RGB', (total_width, total_height), color=(15, 15, 20))
    
    # Title bar
    draw = ImageDraw.Draw(combined)
    title_font = get_font(20)
    small_font = get_font(14)
    
    # Trial indicator on top left
    trial_text = f"Trial: {trial}/{total_trials}"
    draw.text((padding, 5), trial_text, fill=(100, 200, 255), font=small_font)
    if trial_result:
        result_color = (100, 255, 100) if trial_result == "SUCCESS" else (255, 150, 100)
        draw.text((padding + 100, 5), f"[{trial_result}]", fill=result_color, font=small_font)
    
    # Center title
    draw.text((total_width // 2, 5), "Barnes Maze - LLM Navigation Demo", 
              fill=(150, 200, 255), font=title_font, anchor="mt")
    
    # Paste images
    combined.paste(ascii_img, (padding, padding + 25))
    combined.paste(learnings_panel, (ascii_img.width + 2 * padding, padding + 25))
    
    return combined


def generate_exploration_actions():
    """
    Actions from actual successful benchmark run (Trial 1 - 36 steps to success).
    Extracted from llm_traces.log BarnesMaze ASCII_2D.
    """
    # Trial 1 actions that led to SUCCESS in 36 steps
    actions_str = [
        # LLM Call 1-8
        'ROTATE_LEFT', 'ROTATE_LEFT', 'ROTATE_LEFT', 'FORWARD', 'FORWARD', 'ROTATE_LEFT', 'ROTATE_LEFT', 'FORWARD',
        # LLM Call 9-16
        'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'ROTATE_RIGHT', 'ROTATE_RIGHT', 'FORWARD', 'FORWARD',
        # LLM Call 17-23
        'ROTATE_RIGHT', 'FORWARD', 'ROTATE_RIGHT', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD',
        # LLM Call 24-31
        'FORWARD', 'FORWARD', 'FORWARD', 'ROTATE_RIGHT', 'FORWARD', 'ROTATE_LEFT', 'FORWARD', 'FORWARD',
        # LLM Call 32-36 (SUCCESS!)
        'ROTATE_RIGHT', 'ROTATE_RIGHT', 'FORWARD', 'FORWARD', 'FORWARD', 'ROTATE_LEFT', 'ROTATE_LEFT', 'FORWARD',
    ]
    
    action_map = {
        'FORWARD': Action.FORWARD,
        'ROTATE_LEFT': Action.ROTATE_LEFT,
        'ROTATE_RIGHT': Action.ROTATE_RIGHT,
        'STAY': Action.STAY,
        'INTERACT': Action.INTERACT,
    }
    
    return [action_map[a] for a in actions_str]


def generate_trial2_actions():
    """Trial 2 actions - SUCCESS in 6 steps (agent learned from trial 1)."""
    actions_str = [
        'ROTATE_LEFT', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'ROTATE_RIGHT', 'FORWARD',
    ]
    action_map = {
        'FORWARD': Action.FORWARD,
        'ROTATE_LEFT': Action.ROTATE_LEFT,
        'ROTATE_RIGHT': Action.ROTATE_RIGHT,
        'STAY': Action.STAY,
    }
    return [action_map[a] for a in actions_str]


def generate_trial3_actions():
    """Trial 3 actions - longer exploration."""
    actions_str = [
        'ROTATE_RIGHT', 'FORWARD', 'ROTATE_LEFT', 'ROTATE_LEFT', 'FORWARD', 'FORWARD', 'ROTATE_RIGHT', 'ROTATE_RIGHT',
        'ROTATE_LEFT', 'ROTATE_LEFT', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD',
        'ROTATE_RIGHT', 'ROTATE_RIGHT', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD', 'FORWARD',
        'ROTATE_RIGHT', 'FORWARD', 'FORWARD', 'FORWARD', 'ROTATE_LEFT', 'FORWARD', 'FORWARD', 'ROTATE_RIGHT',
    ]
    action_map = {
        'FORWARD': Action.FORWARD,
        'ROTATE_LEFT': Action.ROTATE_LEFT,
        'ROTATE_RIGHT': Action.ROTATE_RIGHT,
        'STAY': Action.STAY,
    }
    return [action_map[a] for a in actions_str]


def main():
    print("Creating Barnes Maze demo GIF from actual benchmark logs...")
    
    # Parse the actual log
    trials = parse_log_observations(LOG_FILE, "BarnesMaze", "ASCII_2D")
    
    if not trials:
        print("ERROR: No trials found in log. Make sure benchmark has run.")
        return
    
    print(f"Found {len(trials)} trials in log")
    for t in trials:
        print(f"  Trial {t['trial_num']}: {t['result']} in {t['total_steps']} steps ({len(t['steps'])} observations)")
        print(f"    Learnings: {t['learnings']}")
    
    # Collect frames
    frames = []
    NUM_TRIALS = len(trials)
    total_reward = 0.0
    
    # Track which learnings have been shown (across all trials)
    shown_learnings = []
    
    for trial_data in trials:
        trial = trial_data['trial_num']
        steps = trial_data['steps']
        result = trial_data['result']
        trial_learnings = trial_data['learnings']
        
        print(f"  Processing Trial {trial}...")
        
        if not steps:
            continue
        
        # Process each step from the log
        for i, step_data in enumerate(steps):
            obs = step_data['observation']
            reward = step_data['reward']
            action = step_data['action']
            step_num = step_data['step']
            
            total_reward += reward
            
            # Gradually reveal learnings based on step progress within trial
            # Learnings come from LLM calls, which happen every 8 steps roughly
            # First learning appears after step 8 (when first batch completes)
            num_learnings_to_show = min(len(trial_learnings), step_num // 8)
            
            # Add new learnings to shown_learnings
            for l in trial_learnings[:num_learnings_to_show]:
                if l not in shown_learnings:
                    shown_learnings.append(l)
            
            # Check if this is a success step
            trial_result = ""
            if i == len(steps) - 1:
                trial_result = result
                # Show all learnings from this trial at end
                for l in trial_learnings:
                    if l not in shown_learnings:
                        shown_learnings.append(l)
            
            # Render frame
            ascii_img = render_ascii_to_image(obs)
            learnings_panel = render_learnings_panel(
                shown_learnings, step_num, trial_data['total_steps'],
                reward
            )
            combined = create_combined_frame(ascii_img, learnings_panel, trial, NUM_TRIALS, trial_result)
            frames.append(np.array(combined))
        
        # Add pause frames after trial ends
        if frames:
            for _ in range(10):
                frames.append(frames[-1])
    
    # Save MP4 video
    print(f"Saving {len(frames)} frames to {OUTPUT_FILE}...")
    writer = imageio.get_writer(OUTPUT_FILE, fps=FPS, codec='libx264', pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Done! Video saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
