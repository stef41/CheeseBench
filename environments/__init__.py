"""
Virtual Environments for LLM Evaluation

Based on verified animal learning protocols from neuroscience literature.

Main interface:
    view, reward = environment.step(action)

Environment automatically handles:
    - Trial counting
    - Timeout detection
    - Success criteria checking
    - Agent teleportation between trials

Observation formats:
    - FPV_3D: First-person 3D rendered view (224x224 RGB)
    - TOPDOWN_2D: Top-down 2D view (224x224 RGB)
    - ASCII_2D: ASCII art top-down view (full map)
    - ASCII_3D: ASCII art pseudo-3D view
    - ASCII_2D_FPV: ASCII 2D cropped around agent (partial map, egocentric)
"""

from .base_env import (
    BaseEnvironment,
    NavigationEnvironment,
    EnvironmentConfig,
    SessionState,
    AgentState,
    TrialResult,
    ViewMode,
    Action,
    AsciiCanvas,
    DIRECTION_ARROWS,
)

# Navigation environments
from .morris_water_maze import MorrisWaterMaze, create_morris_water_maze
from .t_maze import TMaze, create_t_maze
from .barnes_maze import BarnesMaze, create_barnes_maze
from .radial_arm_maze import RadialArmMaze, create_radial_arm_maze
from .star_maze import StarMaze

# Operant environments
from .operant_chamber import OperantChamber, create_operant_chamber

# Aversive/avoidance environments
from .shuttle_box import ShuttleBox

# Conditioning environments
from .place_preference import PlacePreference

# Working memory environments
from .dnms_task import DNMSTask

# Registry for loading from verified protocols
from .registry import (
    EnvironmentRegistry,
    ProtocolSpec,
    load_verified_environments,
    get_available_environment_types
)

__all__ = [
    # Base
    'BaseEnvironment', 'NavigationEnvironment', 'EnvironmentConfig',
    'SessionState', 'AgentState', 'TrialResult', 'ViewMode', 'Action',
    'AsciiCanvas', 'DIRECTION_ARROWS',
    # Environments
    'MorrisWaterMaze', 'TMaze', 'BarnesMaze', 'RadialArmMaze', 'StarMaze',
    'OperantChamber', 'ShuttleBox', 'PlacePreference', 'DNMSTask',
    # Factory functions
    'create_morris_water_maze', 'create_t_maze', 'create_barnes_maze',
    'create_radial_arm_maze', 'create_operant_chamber',
    # Registry
    'EnvironmentRegistry', 'ProtocolSpec',
    'load_verified_environments', 'get_available_environment_types',
]
