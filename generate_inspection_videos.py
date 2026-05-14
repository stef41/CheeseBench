"""
Generate individual MP4 videos for each environment × each view mode.
Each video shows 40 steps of random exploration at larger resolution for close inspection.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import os
import random

from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze, StarMaze,
    OperantChamber, ShuttleBox, PlacePreference, DNMSTask,
)
from environments.base_env import ViewMode, Action

OUTPUT_DIR = "inspection_videos"
FPS = 4
STEPS = 40
IMG_SIZE = 480  # Larger for close inspection

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

COLOR_MAP = {
    '#': (100, 100, 120), '█': (140, 140, 160), '▓': (110, 110, 130),
    '▒': (80, 80, 100),   '░': (60, 70, 90),    '·': (50, 50, 60),
    '.': (70, 80, 100),    ' ': (30, 30, 40),    ',': (60, 60, 70),
    'G': (50, 255, 50),    '*': (255, 215, 0),   'P': (200, 200, 50),
    'E': (50, 200, 50),    '?': (180, 150, 50),
    '↑': (255, 80, 80),   '↓': (255, 80, 80),   '←': (255, 80, 80),
    '→': (255, 80, 80),   '↖': (255, 80, 80),   '↗': (255, 80, 80),
    '↙': (255, 80, 80),   '↘': (255, 80, 80),
    '~': (50, 100, 200),  '|': (80, 80, 100),   '─': (80, 80, 100),
    '═': (100, 100, 120), '║': (100, 100, 120),
    '╔': (100, 100, 120), '╗': (100, 100, 120), '╚': (100, 100, 120), '╝': (100, 100, 120),
    '!': (255, 50, 50),   'X': (255, 50, 50),
    '●': (200, 200, 50),  '■': (50, 150, 255),  '▲': (255, 100, 50), '◆': (200, 50, 200),
    '[': (150, 150, 170),  ']': (150, 150, 170), '=': (180, 180, 200),
    'm': (200, 160, 50),
    '1': (255, 200, 50),  '2': (255, 200, 50),  '3': (255, 200, 50),  '4': (255, 200, 50),
    '5': (255, 200, 50),  '6': (255, 200, 50),  '7': (255, 200, 50),  '8': (255, 200, 50),
}


def ascii_to_image(text: str, size: int = IMG_SIZE) -> np.ndarray:
    """Render ASCII text to a crisp image with syntax coloring."""
    lines = text.split('\n')
    if not lines:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Calculate font size to fit
    max_cols = max(len(l) for l in lines)
    max_rows = len(lines)
    if max_cols == 0 or max_rows == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Pick font size that fits the content in the image
    font_w = max(4, size // max(max_cols + 2, 1))
    font_h = max(6, size // max(max_rows + 2, 1))
    font_size = min(font_w, font_h)
    font_size = max(6, min(font_size, 20))

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure char dimensions
    bbox = font.getbbox("X")
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]
    if cw == 0:
        cw = font_size
    if ch == 0:
        ch = font_size

    img = Image.new('RGB', (size, size), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    # Center the text block
    text_w = max_cols * cw
    text_h = max_rows * ch
    x_off = max(0, (size - text_w) // 2)
    y_off = max(0, (size - text_h) // 2)

    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            color = COLOR_MAP.get(char, (180, 180, 180))
            draw.text((x_off + col_idx * cw, y_off + row_idx * ch), char, fill=color, font=font)

    return np.array(img)


def front_block_to_image(text: str, size: int = IMG_SIZE) -> np.ndarray:
    """Render FRONT_BLOCK single character as a large centered glyph."""
    char = text.strip()
    if not char:
        char = ' '

    try:
        font = ImageFont.truetype(FONT_PATH, size // 2)
    except Exception:
        font = ImageFont.load_default()

    img = Image.new('RGB', (size, size), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    color = COLOR_MAP.get(char, (200, 200, 200))
    draw.text((size // 2, size // 2), char, fill=color, font=font, anchor="mm")
    # Label
    try:
        label_font = ImageFont.truetype(FONT_PATH, 14)
    except Exception:
        label_font = ImageFont.load_default()
    draw.text((size // 2, size - 20), f"FRONT_BLOCK: '{char}'", fill=(150, 150, 150), font=label_font, anchor="mm")
    return np.array(img)


def obs_to_frame(obs, view_mode: ViewMode, env_name: str, view_name: str, step: int) -> np.ndarray:
    """Convert an observation to a labeled image frame."""
    if view_mode == ViewMode.FRONT_BLOCK:
        frame = front_block_to_image(obs)
    elif isinstance(obs, str):
        frame = ascii_to_image(obs)
    else:
        # numpy image — resize
        pil = Image.fromarray(obs)
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
        frame = np.array(pil)

    # Add label overlay
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    try:
        label_font = ImageFont.truetype(FONT_PATH, 13)
    except Exception:
        label_font = ImageFont.load_default()

    # Top bar
    draw.rectangle([(0, 0), (IMG_SIZE, 18)], fill=(0, 0, 0))
    draw.text((IMG_SIZE // 2, 2), f"{env_name}  |  {view_name}  |  step {step}", fill=(200, 200, 100), font=label_font, anchor="mt")

    return np.array(pil)


ENV_CONFIGS = [
    ("MorrisWaterMaze", MorrisWaterMaze),
    ("TMaze", TMaze),
    ("BarnesMaze", BarnesMaze),
    ("RadialArmMaze", RadialArmMaze),
    ("StarMaze", StarMaze),
    ("OperantChamber", OperantChamber),
    ("ShuttleBox", ShuttleBox),
    ("PlacePreference", PlacePreference),
    ("DNMSTask", DNMSTask),
]

VIEW_MODES = [
    (ViewMode.FPV_3D, "FPV_3D"),
    (ViewMode.TOPDOWN_2D, "TOPDOWN_2D"),
    (ViewMode.ASCII_2D, "ASCII_2D"),
    (ViewMode.ASCII_2D_FPV, "ASCII_2D_FPV"),
    (ViewMode.ASCII_3D, "ASCII_3D"),
    (ViewMode.FRONT_BLOCK, "FRONT_BLOCK"),
]

ACTIONS = [Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT]


def make_video(env_name: str, env_class, view_mode: ViewMode, view_name: str):
    """Generate a single MP4 for one env + one view mode."""
    env = env_class(view_mode=view_mode)
    obs = env.reset()
    
    frames = [obs_to_frame(obs, view_mode, env_name, view_name, 0)]

    random.seed(42)
    for step in range(1, STEPS + 1):
        action = random.choice(ACTIONS)
        obs, _ = env.step(action)
        if env.is_done:
            env.reset()
        frames.append(obs_to_frame(obs, view_mode, env_name, view_name, step))

    filename = f"{env_name.lower()}_{view_name.lower()}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)
    imageio.mimsave(path, frames, fps=FPS)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(ENV_CONFIGS) * len(VIEW_MODES)
    print(f"Generating {total} inspection videos ({len(ENV_CONFIGS)} envs × {len(VIEW_MODES)} views)...")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    count = 0
    for env_name, env_class in ENV_CONFIGS:
        for view_mode, view_name in VIEW_MODES:
            count += 1
            path = make_video(env_name, env_class, view_mode, view_name)
            size_kb = os.path.getsize(path) / 1024
            print(f"  [{count:2d}/{total}] {os.path.basename(path):45s} ({size_kb:.0f} KB)")

    print(f"\nDone! {count} videos in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
