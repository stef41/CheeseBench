"""Generate environment visualization figure for the ALIFE paper.

Creates a 3x3 grid showing all 9 environments in ASCII_2D mode,
plus a companion figure showing the 3 ASCII view modes for a single environment.
"""
import sys
sys.path.insert(0, '/data/users/zacharie/CheeseBench')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze, StarMaze,
    OperantChamber, ShuttleBox, PlacePreference, DNMSTask,
    ViewMode, Action
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ENVS = [
    ('Morris Water Maze', MorrisWaterMaze),
    ('Barnes Maze', BarnesMaze),
    ('T-Maze', TMaze),
    ('Radial Arm Maze', RadialArmMaze),
    ('Star Maze', StarMaze),
    ('Operant Chamber', OperantChamber),
    ('Shuttle Box', ShuttleBox),
    ('Place Preference', PlacePreference),
    ('DNMS Task', DNMSTask),
]


def render_ascii_to_image(ascii_text, target_w=220, target_h=220):
    """Render ASCII text onto a black image with green monospace font."""
    img = Image.new('RGB', (target_w, target_h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    
    lines = ascii_text.split('\n')
    # Strip leading/trailing blank lines but preserve per-line spacing
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    max_line_len = max(len(l) for l in lines)
    
    # Find font size that fits
    try:
        for sz in range(14, 3, -1):
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", sz)
            # Use getlength for accurate advance width (not getbbox which is glyph bounds)
            char_w = font.getlength('M')
            line_h = sz + 2  # font size + small leading
            text_w = char_w * max_line_len
            text_h = line_h * len(lines)
            if text_w <= target_w - 8 and text_h <= target_h - 8:
                break
    except (IOError, OSError):
        font = ImageFont.load_default()
        line_h = 10
        text_h = line_h * len(lines)
        text_w = 6 * max_line_len
    
    # Center text block
    x0 = max(4, (target_w - text_w) // 2)
    y0 = max(4, (target_h - text_h) // 2)
    
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * line_h), line, fill=(0, 220, 0), font=font)
    
    return img


def make_env_grid():
    """3x3 grid of all environments in ASCII_2D mode."""
    cell_size = 220
    padding = 4
    label_h = 22
    cols, rows = 3, 3
    
    total_w = cols * cell_size + (cols - 1) * padding
    total_h = rows * (cell_size + label_h) + (rows - 1) * padding
    
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except (IOError, OSError):
        label_font = ImageFont.load_default()
    
    for idx, (name, EnvClass) in enumerate(ENVS):
        row, col = divmod(idx, cols)
        x0 = col * (cell_size + padding)
        y0 = row * (cell_size + label_h + padding)
        
        env = EnvClass(view_mode=ViewMode.ASCII_2D)
        obs = env.reset()
        
        # Take a few steps for more interesting state (except DNMS)
        if EnvClass is not DNMSTask:
            np.random.seed(42 + idx)
            for _ in range(5):
                action = np.random.choice([Action.FORWARD, Action.ROTATE_LEFT, Action.ROTATE_RIGHT])
                obs, _ = env.step(action)
                obs_ascii = env.render(mode=ViewMode.ASCII_2D)
            obs = obs_ascii
        
        img = render_ascii_to_image(obs, cell_size, cell_size)
        canvas.paste(img, (x0, y0 + label_h))
        
        # Label
        bbox = draw.textbbox((0, 0), name, font=label_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (cell_size - tw) // 2
        draw.text((tx, y0 + 2), name, fill=(0, 0, 0), font=label_font)
    
    canvas.save('fig_environments.png', dpi=(300, 300))
    fig, ax = plt.subplots(figsize=(total_w/100, total_h/100), dpi=300)
    ax.imshow(np.array(canvas))
    ax.axis('off')
    fig.savefig('fig_environments.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f'Created fig_environments.png/pdf ({total_w}x{total_h})')


def make_viewmode_comparison():
    """Show 3 ASCII view modes for StarMaze (good corridor depth)."""
    env = StarMaze()
    env.reset()
    for a in [Action.FORWARD, Action.FORWARD]:
        env.step(a)
    
    modes = [
        ('ASCII_2D\n(Top-down)', ViewMode.ASCII_2D),
        ('ASCII_2D_FPV\n(Egocentric)', ViewMode.ASCII_2D_FPV),
        ('ASCII_3D\n(First-person)', ViewMode.ASCII_3D),
    ]
    
    cell_w, cell_h = 220, 260
    padding = 8
    label_h = 36  # space for two-line title
    
    total_w = 3 * cell_w + 2 * padding
    total_h = cell_h + label_h
    
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (IOError, OSError):
        label_font = ImageFont.load_default()
    
    for idx, (title, mode) in enumerate(modes):
        obs = env.render(mode=mode)
        x0 = idx * (cell_w + padding)
        
        # Render ASCII into cell
        img = render_ascii_to_image(obs, cell_w, cell_h)
        canvas.paste(img, (x0, label_h))
        
        # Draw centered two-line title
        for line_idx, line in enumerate(title.split('\n')):
            bbox = draw.textbbox((0, 0), line, font=label_font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (cell_w - tw) // 2
            ty = 2 + line_idx * 16
            draw.text((tx, ty), line, fill=(0, 0, 0), font=label_font)
    
    canvas.save('fig_viewmodes.png', dpi=(300, 300))
    fig, ax = plt.subplots(figsize=(total_w/100, total_h/100), dpi=300)
    ax.imshow(np.array(canvas))
    ax.axis('off')
    fig.savefig('fig_viewmodes.pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f'Created fig_viewmodes.pdf/png ({total_w}x{total_h})')


if __name__ == '__main__':
    make_env_grid()
    make_viewmode_comparison()
    print('Done.')
