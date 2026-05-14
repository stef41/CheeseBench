"""
Environment Registry - Load and create environments from verified protocols.

Parses verified_strict.json and creates appropriate environment instances.
"""

import json
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass

from .base_env import BaseEnvironment, ViewMode


@dataclass
class ProtocolSpec:
    """Specification for a verified protocol."""
    pmc_id: str
    task_type: str
    expected_trials: Optional[int]
    expected_sessions: Optional[int]
    success_rate: Optional[float]
    quotes: Dict[str, str]
    environment_class: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pmc_id': self.pmc_id,
            'task_type': self.task_type,
            'expected_trials': self.expected_trials,
            'expected_sessions': self.expected_sessions,
            'success_rate': self.success_rate,
            'quotes': self.quotes,
            'environment_class': self.environment_class
        }


class EnvironmentRegistry:
    """
    Registry for environment types and protocol-to-environment mapping.
    """
    
    # Keywords to environment class mapping
    TASK_KEYWORDS = {
        'morris': 'MorrisWaterMaze',
        'water maze': 'MorrisWaterMaze',
        'swim': 'MorrisWaterMaze',
        'mwm': 'MorrisWaterMaze',
        'hidden platform': 'MorrisWaterMaze',
        
        't-maze': 'TMaze',
        't maze': 'TMaze',
        'tmaze': 'TMaze',
        'alternation': 'TMaze',
        'reversal learning': 'TMaze',
        
        'barnes': 'BarnesMaze',
        'barnes maze': 'BarnesMaze',
        'escape hole': 'BarnesMaze',
        
        'radial': 'RadialArmMaze',
        'radial arm': 'RadialArmMaze',
        '8-arm': 'RadialArmMaze',
        'eight-arm': 'RadialArmMaze',
        'ram': 'RadialArmMaze',
        
        'operant': 'OperantChamber',
        'skinner': 'OperantChamber',
        'lever press': 'OperantChamber',
        'lever-press': 'OperantChamber',
        'nose poke': 'OperantChamber',
        'nose-poke': 'OperantChamber',
        'fr1': 'OperantChamber',
        'fixed ratio': 'OperantChamber',
        'variable ratio': 'OperantChamber',
        'touch screen': 'OperantChamber',
        'touchscreen': 'OperantChamber',
        'autoshaping': 'OperantChamber',
        
        # New environments
        'shuttle': 'ShuttleBox',
        'shuttle box': 'ShuttleBox',
        'shuttlebox': 'ShuttleBox',
        'active avoidance': 'ShuttleBox',
        'fear conditioning': 'ShuttleBox',
        'footshock': 'ShuttleBox',
        'escapable': 'ShuttleBox',
        'inescapable': 'ShuttleBox',
        'two-way': 'ShuttleBox',
        '2cap': 'ShuttleBox',
        'cued access': 'ShuttleBox',
        
        'place preference': 'PlacePreference',
        'cpp': 'PlacePreference',
        'conditioned place': 'PlacePreference',
        
        'star maze': 'StarMaze',
        'starmaze': 'StarMaze',
        'sunburst': 'StarMaze',
        'transformer maze': 'StarMaze',
        'complex maze': 'StarMaze',
        
        'dnms': 'DNMSTask',
        'dnmtp': 'DNMSTask',
        'delayed non-match': 'DNMSTask',
        'non-match-to-sample': 'DNMSTask',
        'working memory task': 'DNMSTask',
        'olfactory delayed': 'DNMSTask',
        
        # Map these to existing envs or new ones
        'y-maze': 'TMaze',  # Y-maze similar to T-maze
        'y maze': 'TMaze',
        'ymaze': 'TMaze',
        
        'probabilistic reversal': 'TMaze',  # Reversal learning variant
        
        'open field': 'MorrisWaterMaze',  # Generic navigation
        'open-field': 'MorrisWaterMaze',
        
        'social conditioning': 'PlacePreference',  # Social preference similar to CPP
    }
    
    # Environment classes available
    AVAILABLE_ENVIRONMENTS = {
        'MorrisWaterMaze': 'morris_water_maze',
        'TMaze': 't_maze',
        'BarnesMaze': 'barnes_maze',
        'RadialArmMaze': 'radial_arm_maze',
        'OperantChamber': 'operant_chamber',
        'ShuttleBox': 'shuttle_box',
        'PlacePreference': 'place_preference',
        'StarMaze': 'star_maze',
        'DNMSTask': 'dnms_task',
    }
    
    def __init__(self, verified_data_path: Optional[str] = None):
        """Initialize registry with optional verified data."""
        self.protocols: List[ProtocolSpec] = []
        self._env_cache: Dict[str, Type[BaseEnvironment]] = {}
        
        if verified_data_path:
            self.load_verified_data(verified_data_path)
    
    def load_verified_data(self, path: str):
        """Load verified protocols from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle the grouped format from verified_strict.json
        if isinstance(data, dict) and 'protocols' in data:
            protocols_list = data['protocols']
        elif isinstance(data, list):
            protocols_list = data
        else:
            protocols_list = []
        
        for item in protocols_list:
            protocol = self._parse_protocol(item)
            if protocol:
                self.protocols.append(protocol)
        
        print(f"Loaded {len(self.protocols)} verified protocols")
    
    def _parse_protocol(self, item: Dict[str, Any]) -> Optional[ProtocolSpec]:
        """Parse a protocol item into ProtocolSpec."""
        
        # Handle the verified_strict.json format
        task = item.get('task', '')
        
        # Get PMC ID from trials_to_criterion or other verified fields
        pmc_id = 'unknown'
        for field in ['trials_to_criterion', 'sessions_to_criterion', 'trials_per_session']:
            if field in item and isinstance(item[field], dict):
                pmc_id = item[field].get('source_pmc', pmc_id)
                break
        
        # Collect quotes from verified fields
        quotes = {}
        for field in ['trials_to_criterion', 'sessions_to_criterion', 'trials_per_session']:
            if field in item and isinstance(item[field], dict):
                quote = item[field].get('quote', '')
                if quote:
                    quotes[field] = quote
        
        # Determine environment type from task name
        all_text = (task + ' ' + ' '.join(quotes.values())).lower()
        env_class = self._classify_task(all_text)
        
        # Extract values from verified fields
        def get_value(field):
            if field in item:
                if isinstance(item[field], dict):
                    return item[field].get('value')
                return item[field]
            return None
        
        # Get trials info
        expected_trials = get_value('trials_to_criterion')
        if expected_trials is None:
            expected_trials = item.get('calculated_total_trials')
        
        return ProtocolSpec(
            pmc_id=pmc_id,
            task_type=item.get('environment_type', 'unknown'),
            expected_trials=expected_trials,
            expected_sessions=get_value('sessions_to_criterion'),
            success_rate=None,  # Not in this format
            quotes=quotes,
            environment_class=env_class
        )
    
    def _classify_task(self, text: str) -> str:
        """Classify task type from text."""
        text_lower = text.lower()
        
        for keyword, env_class in self.TASK_KEYWORDS.items():
            if keyword in text_lower:
                return env_class
        
        # Default to open field if can't classify
        return 'OpenField'
    
    def get_environment_class(self, class_name: str) -> Optional[Type[BaseEnvironment]]:
        """Get environment class by name."""
        if class_name in self._env_cache:
            return self._env_cache[class_name]
        
        if class_name not in self.AVAILABLE_ENVIRONMENTS:
            return None
        
        module_name = self.AVAILABLE_ENVIRONMENTS[class_name]
        
        try:
            if class_name == 'MorrisWaterMaze':
                from .morris_water_maze import MorrisWaterMaze
                self._env_cache[class_name] = MorrisWaterMaze
                return MorrisWaterMaze
            elif class_name == 'TMaze':
                from .t_maze import TMaze
                self._env_cache[class_name] = TMaze
                return TMaze
            elif class_name == 'BarnesMaze':
                from .barnes_maze import BarnesMaze
                self._env_cache[class_name] = BarnesMaze
                return BarnesMaze
            elif class_name == 'RadialArmMaze':
                from .radial_arm_maze import RadialArmMaze
                self._env_cache[class_name] = RadialArmMaze
                return RadialArmMaze
            elif class_name == 'OperantChamber':
                from .operant_chamber import OperantChamber
                self._env_cache[class_name] = OperantChamber
                return OperantChamber
        except ImportError as e:
            print(f"Could not import {class_name}: {e}")
            return None
        
        return None
    
    def create_environment(
        self, 
        protocol: ProtocolSpec,
        view_mode: ViewMode = ViewMode.FPV_3D
    ) -> Optional[BaseEnvironment]:
        """Create environment instance from protocol specification."""
        
        env_class = self.get_environment_class(protocol.environment_class)
        if env_class is None:
            print(f"Environment class not available: {protocol.environment_class}")
            return None
        
        # Get main quote for source
        main_quote = ""
        for field in ['expected_trials', 'expected_sessions', 'success_rate']:
            if field in protocol.quotes:
                main_quote = protocol.quotes[field]
                break
        
        # Create config based on protocol
        if protocol.environment_class == 'MorrisWaterMaze':
            from .morris_water_maze import create_morris_water_maze
            return create_morris_water_maze(
                trials_to_criterion=protocol.expected_trials or 15,
                trials_per_session=3,
                view_mode=view_mode,
                source_pmc=protocol.pmc_id,
                source_quote=main_quote
            )
        elif protocol.environment_class == 'TMaze':
            from .t_maze import create_t_maze
            return create_t_maze(
                trials_to_criterion=protocol.expected_trials or 20,
                trials_per_session=5,
                view_mode=view_mode,
                source_pmc=protocol.pmc_id,
                source_quote=main_quote
            )
        elif protocol.environment_class == 'BarnesMaze':
            from .barnes_maze import create_barnes_maze
            return create_barnes_maze(
                trials_to_criterion=protocol.expected_trials or 16,
                trials_per_session=4,
                view_mode=view_mode,
                source_pmc=protocol.pmc_id,
                source_quote=main_quote
            )
        elif protocol.environment_class == 'RadialArmMaze':
            from .radial_arm_maze import create_radial_arm_maze
            return create_radial_arm_maze(
                trials_to_criterion=protocol.expected_trials or 20,
                trials_per_session=4,
                view_mode=view_mode,
                source_pmc=protocol.pmc_id,
                source_quote=main_quote
            )
        elif protocol.environment_class == 'OperantChamber':
            from .operant_chamber import create_operant_chamber
            return create_operant_chamber(
                schedule='FR1',
                trials_to_criterion=protocol.expected_trials or 50,
                view_mode=view_mode,
                source_pmc=protocol.pmc_id,
                source_quote=main_quote
            )
        
        return None
    
    def get_protocols_by_type(self, env_class: str) -> List[ProtocolSpec]:
        """Get all protocols of a specific environment type."""
        return [p for p in self.protocols if p.environment_class == env_class]
    
    def get_protocol_summary(self) -> Dict[str, int]:
        """Get count of protocols by environment type."""
        summary = {}
        for protocol in self.protocols:
            env_class = protocol.environment_class
            summary[env_class] = summary.get(env_class, 0) + 1
        return summary
    
    def create_all_environments(
        self, 
        view_mode: ViewMode = ViewMode.FPV_3D
    ) -> List[BaseEnvironment]:
        """Create all available environments from protocols."""
        environments = []
        
        for protocol in self.protocols:
            env = self.create_environment(protocol, view_mode)
            if env is not None:
                environments.append(env)
        
        return environments


def load_verified_environments(
    data_path: str = "output/verified_strict.json",
    view_mode: ViewMode = ViewMode.FPV_3D
) -> List[BaseEnvironment]:
    """
    Convenience function to load all verified environments.
    
    Returns list of environment instances ready for VLM evaluation.
    """
    registry = EnvironmentRegistry(data_path)
    
    print("\nProtocol Summary:")
    for env_type, count in registry.get_protocol_summary().items():
        print(f"  {env_type}: {count}")
    
    print("\nCreating environments...")
    environments = registry.create_all_environments(view_mode)
    print(f"Created {len(environments)} environments")
    
    return environments


def get_available_environment_types() -> List[str]:
    """Get list of available environment types."""
    return list(EnvironmentRegistry.AVAILABLE_ENVIRONMENTS.keys())
