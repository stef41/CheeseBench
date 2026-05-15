#!/usr/bin/env python3
"""
Interactive keyboard-controlled environment player.

Play any environment with keyboard controls in terminal.
Supports ASCII_2D, ASCII_2D_FPV, and ASCII_3D view modes.

Usage:
    python play.py                     # Menu to select environment
    python play.py TMaze               # Play specific environment
    python play.py TMaze --mode 3d     # Play with ASCII_3D view
    python play.py --list              # List all environments
"""

import sys
import argparse
import termios
import tty
import select

sys.path.insert(0, '.')
from .environments import (
    MorrisWaterMaze, TMaze, BarnesMaze, RadialArmMaze,
    OperantChamber, ShuttleBox, PlacePreference, StarMaze, DNMSTask,
    Action, ViewMode
)

# =============================================================================
# TERMINAL UTILITIES
# =============================================================================

class TerminalInput:
    """Handle raw keyboard input without requiring Enter."""
    
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None
    
    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self
    
    def __exit__(self, *args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self, timeout=None):
        """Get a single keypress. Returns None on timeout."""
        if timeout:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return None
        
        ch = sys.stdin.read(1)
        
        # Handle escape sequences (arrow keys)
        if ch == '\x1b':
            # Check if more characters are available
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'UP'
                    elif ch3 == 'B':
                        return 'DOWN'
                    elif ch3 == 'C':
                        return 'RIGHT'
                    elif ch3 == 'D':
                        return 'LEFT'
            return 'ESC'
        
        return ch


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='', flush=True)


def move_cursor(row, col):
    """Move cursor to position."""
    print(f'\033[{row};{col}H', end='', flush=True)


def raw_print(text=''):
    """Print text in raw terminal mode (handles newlines properly)."""
    # In raw mode, \n doesn't return to column 1, so we need \r\n
    if text:
        lines = text.split('\n')
        print('\r\n'.join(lines), end='\r\n', flush=True)
    else:
        print(end='\r\n', flush=True)


# =============================================================================
# KEY MAPPINGS
# =============================================================================

KEY_ACTIONS = {
    # WASD controls
    'w': Action.FORWARD,
    'W': Action.FORWARD,
    's': Action.STAY,
    'S': Action.STAY,
    'a': Action.ROTATE_LEFT,
    'A': Action.ROTATE_LEFT,
    'd': Action.ROTATE_RIGHT,
    'D': Action.ROTATE_RIGHT,
    
    # Arrow keys
    'UP': Action.FORWARD,
    'DOWN': Action.STAY,
    'LEFT': Action.ROTATE_LEFT,
    'RIGHT': Action.ROTATE_RIGHT,
    
    # Other actions
    'x': Action.STAY,
    'X': Action.STAY,
}

VIEW_MODE_KEYS = {
    '1': ViewMode.ASCII_2D,
    '2': ViewMode.ASCII_2D_FPV,
    '3': ViewMode.ASCII_3D,
}

# =============================================================================
# ENVIRONMENT REGISTRY
# =============================================================================

ENVIRONMENTS = {
    'MorrisWaterMaze': MorrisWaterMaze,
    'TMaze': TMaze,
    'BarnesMaze': BarnesMaze,
    'RadialArmMaze': RadialArmMaze,
    'OperantChamber': OperantChamber,
    'ShuttleBox': ShuttleBox,
    'PlacePreference': PlacePreference,
    'StarMaze': StarMaze,
    'DNMSTask': DNMSTask,
}

# Short aliases
ALIASES = {
    'morris': 'MorrisWaterMaze',
    'mwm': 'MorrisWaterMaze',
    'tmaze': 'TMaze',
    't': 'TMaze',
    'barnes': 'BarnesMaze',
    'radial': 'RadialArmMaze',
    'ram': 'RadialArmMaze',
    'operant': 'OperantChamber',
    'skinner': 'OperantChamber',
    'shuttle': 'ShuttleBox',
    'cpp': 'PlacePreference',
    'place': 'PlacePreference',
    'star': 'StarMaze',
    'dnms': 'DNMSTask',
}


def resolve_env_name(name: str) -> str:
    """Resolve environment name from alias or partial match."""
    name_lower = name.lower()
    
    # Check aliases
    if name_lower in ALIASES:
        return ALIASES[name_lower]
    
    # Check exact match
    if name in ENVIRONMENTS:
        return name
    
    # Check case-insensitive match
    for env_name in ENVIRONMENTS:
        if env_name.lower() == name_lower:
            return env_name
    
    # Check partial match
    for env_name in ENVIRONMENTS:
        if name_lower in env_name.lower():
            return env_name
    
    return None


# =============================================================================
# GAME LOOP
# =============================================================================

def render_ui(env, view_mode: ViewMode, step: int, total_reward: float, last_action: str = None):
    """Render the game UI."""
    clear_screen()
    
    # Get observation
    env.view_mode = view_mode
    obs = env.get_observation()
    
    # Header
    env_name = env.__class__.__name__
    mode_name = view_mode.name.replace('ASCII_', '').replace('_', ' ')
    raw_print(f"{'='*70}")
    raw_print(f"Environment: {env_name} | View: {mode_name}")
    raw_print(f"Steps: {step} | Total Reward: {total_reward:.2f}")
    
    # Position info if available
    if hasattr(env, 'agent'):
        # Handle both integer grid (T-maze) and float (circular mazes) positions
        x, y = env.agent.x, env.agent.y
        if isinstance(x, int) and isinstance(y, int):
            pos_str = f"({x}, {y})"
        else:
            pos_str = f"({x:.1f}, {y:.1f})"
        
        # Handle angle - could be integer 0-7 direction or float radians
        if isinstance(env.agent.angle, int) or (isinstance(env.agent.angle, float) and env.agent.angle < 8):
            # Integer direction index (0-7)
            angle_deg = int((env.agent.angle * 45) % 360)
        else:
            # Float radians
            import math
            angle_deg = int(math.degrees(env.agent.angle) % 360)
        raw_print(f"Position: {pos_str} | Angle: {angle_deg}°")
    
    raw_print('='*70)
    
    # Controls
    raw_print("Controls: [w↑]=forward [a←]=left [d→]=right [s]=stay")
    raw_print("          [1]=2D [2]=2D_FPV [3]=3D [r]=reset [q]=quit")
    
    if last_action:
        raw_print(f"Last action: {last_action}")
    
    raw_print()
    
    # Observation (already contains newlines, raw_print handles them)
    raw_print(obs)
    
    # Trial info if available
    if hasattr(env, 'session'):
        trial = env.session.current_trial + 1
        max_trials = env.session.max_trials
        raw_print(f"\nTrial: {trial}/{max_trials}")


def play_environment(env_name: str, initial_mode: ViewMode = ViewMode.ASCII_2D):
    """Main game loop for playing an environment."""
    
    # Create environment
    env_class = ENVIRONMENTS.get(env_name)
    if not env_class:
        print(f"Unknown environment: {env_name}")
        return
    
    env = env_class(view_mode=initial_mode)
    env.reset()
    
    view_mode = initial_mode
    step = 0
    total_reward = 0.0
    last_action = None
    
    # Initial render
    render_ui(env, view_mode, step, total_reward)
    
    with TerminalInput() as term:
        while True:
            key = term.get_key()
            
            if key is None:
                continue
            
            # Quit
            if key in ('q', 'Q', '\x03'):  # q or Ctrl+C
                clear_screen()
                raw_print("Thanks for playing!")
                raw_print(f"Final stats: {step} steps, {total_reward:.2f} total reward")
                break
            
            # Reset
            if key in ('r', 'R'):
                env.reset()
                step = 0
                total_reward = 0.0
                last_action = "RESET"
                render_ui(env, view_mode, step, total_reward, last_action)
                raw_print("\nEnvironment reset!")
                continue
            
            # View mode change
            if key in VIEW_MODE_KEYS:
                view_mode = VIEW_MODE_KEYS[key]
                render_ui(env, view_mode, step, total_reward, last_action)
                continue
            
            # Action
            if key in KEY_ACTIONS:
                action = KEY_ACTIONS[key]
                
                try:
                    obs, reward = env.step(action)
                    step += 1
                    total_reward += reward
                    last_action = action.name
                    
                    if reward > 0:
                        last_action += f" [+{reward:.1f}]"
                    elif reward < 0:
                        last_action += f" [{reward:.1f}]"
                    
                except Exception as e:
                    last_action = f"{action.name} (error: {e})"
                
                render_ui(env, view_mode, step, total_reward, last_action)
                
                # Check if done
                if env.is_done:
                    raw_print("\n" + "="*50)
                    raw_print("SESSION COMPLETE!")
                    raw_print(f"Total steps: {step}")
                    raw_print(f"Total reward: {total_reward:.2f}")
                    if hasattr(env, 'session') and env.session.trial_results:
                        successes = sum(1 for t in env.session.trial_results if t.success)
                        raw_print(f"Successes: {successes}/{len(env.session.trial_results)}")
                    raw_print("Press 'r' to restart or 'q' to quit")
                
                continue


def show_menu():
    """Show interactive menu to select environment."""
    clear_screen()
    print("="*60)
    print("  ENVIRONMENT PLAYER - SELECT AN ENVIRONMENT")
    print("="*60)
    print()
    
    env_list = list(ENVIRONMENTS.keys())
    for i, name in enumerate(env_list, 1):
        print(f"  [{i}] {name}")
    
    print()
    print("  [q] Quit")
    print()
    print("="*60)
    print("Enter number or environment name: ", end='', flush=True)
    
    # Read input (allow normal input for menu)
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        return None
    
    if choice.lower() == 'q':
        return None
    
    # Try as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(env_list):
            return env_list[idx]
    except ValueError:
        pass
    
    # Try as name
    resolved = resolve_env_name(choice)
    if resolved:
        return resolved
    
    print(f"Unknown environment: {choice}")
    return show_menu()


def select_view_mode():
    """Let user select initial view mode."""
    print()
    print("Select view mode:")
    print("  [1] ASCII_2D      - Top-down full map")
    print("  [2] ASCII_2D_FPV  - Top-down cropped around agent")
    print("  [3] ASCII_3D      - First-person 3D view")
    print()
    print("Enter choice [1-3, default=1]: ", end='', flush=True)
    
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        return ViewMode.ASCII_2D
    
    modes = {
        '1': ViewMode.ASCII_2D,
        '2': ViewMode.ASCII_2D_FPV,
        '3': ViewMode.ASCII_3D,
        '': ViewMode.ASCII_2D,
    }
    
    return modes.get(choice, ViewMode.ASCII_2D)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Play behavioral neuroscience environments with keyboard controls.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python play.py                  # Interactive menu
    python play.py TMaze            # Play T-Maze
    python play.py morris --mode 3d # Play Morris Water Maze in 3D view
    python play.py --list           # List environments

Controls:
    WASD / Arrow keys   Move and turn
    1 / 2 / 3           Switch view mode
    R                   Reset environment
    Q                   Quit
        """
    )
    
    parser.add_argument('environment', nargs='?', help='Environment name or alias')
    parser.add_argument('--mode', '-m', choices=['2d', '2d_fpv', '3d'], default='2d',
                        help='Initial view mode (default: 2d)')
    parser.add_argument('--list', '-l', action='store_true', help='List available environments')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available environments:")
        print("-" * 40)
        for name in ENVIRONMENTS:
            print(f"  {name}")
        print()
        print("Aliases:")
        for alias, full in sorted(ALIASES.items()):
            print(f"  {alias} -> {full}")
        return
    
    # Determine view mode
    mode_map = {
        '2d': ViewMode.ASCII_2D,
        '2d_fpv': ViewMode.ASCII_2D_FPV,
        '3d': ViewMode.ASCII_3D,
    }
    view_mode = mode_map.get(args.mode, ViewMode.ASCII_2D)
    
    # Determine environment
    if args.environment:
        env_name = resolve_env_name(args.environment)
        if not env_name:
            print(f"Unknown environment: {args.environment}")
            print("Use --list to see available environments")
            return
    else:
        env_name = show_menu()
        if not env_name:
            print("Goodbye!")
            return
        view_mode = select_view_mode()
    
    # Play!
    play_environment(env_name, view_mode)


if __name__ == "__main__":
    main()
