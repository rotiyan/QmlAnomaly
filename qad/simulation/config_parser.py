"""
Configuration parser for Pythia-8 simulation parameters.
"""

import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysicsProcess:
    """Configuration for a physics process."""
    process: str  # e.g., "pp -> jj", "pp -> ttbar"
    energy: float  # Center-of-mass energy in GeV
    parameters: Dict[str, Any]  # Additional Pythia parameters


@dataclass
class SimulationSettings:
    """General simulation settings."""
    n_events: int
    output_file: str
    random_seed: Optional[int] = None
    debug_level: int = 0


@dataclass
class JetSettings:
    """Jet reconstruction settings."""
    algorithm: str = "antikt"  # jet algorithm
    r_parameter: float = 0.4  # jet radius
    pt_min: float = 20.0  # minimum jet pT in GeV
    eta_max: float = 2.5  # maximum jet |eta|


class SimulationConfig:
    """Parser for YAML configuration files."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration sections."""
        required_sections = ['physics_process', 'simulation', 'jets']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
    
    @property
    def physics_process(self) -> PhysicsProcess:
        """Get physics process configuration."""
        proc_config = self.config['physics_process']
        return PhysicsProcess(
            process=proc_config['process'],
            energy=proc_config['energy'],
            parameters=proc_config.get('parameters', {})
        )
    
    @property
    def simulation(self) -> SimulationSettings:
        """Get simulation settings."""
        sim_config = self.config['simulation']
        return SimulationSettings(
            n_events=sim_config['n_events'],
            output_file=sim_config['output_file'],
            random_seed=sim_config.get('random_seed'),
            debug_level=sim_config.get('debug_level', 0)
        )
    
    @property
    def jets(self) -> JetSettings:
        """Get jet settings."""
        jet_config = self.config['jets']
        return JetSettings(
            algorithm=jet_config.get('algorithm', 'antikt'),
            r_parameter=jet_config.get('r_parameter', 0.4),
            pt_min=jet_config.get('pt_min', 20.0),
            eta_max=jet_config.get('eta_max', 2.5)
        )
    
    def get_pythia_commands(self) -> List[str]:
        """Generate Pythia-8 command strings from configuration."""
        commands = []
        
        # Basic settings
        commands.append(f"Beams:eCM = {self.physics_process.energy}")
        commands.append(f"Random:seed = {self.simulation.random_seed or 12345}")
        
        # Process-specific settings
        if "pp" in self.physics_process.process.lower():
            commands.append("Beams:idA = 2212")  # proton
            commands.append("Beams:idB = 2212")  # proton
        
        # Process selection
        if "jj" in self.physics_process.process.lower():
            commands.append("HardQCD:all = on")
        elif "ttbar" in self.physics_process.process.lower():
            commands.append("Top:gg2ttbar = on")
            commands.append("Top:qqbar2ttbar = on")
        
        # Additional parameters
        for key, value in self.physics_process.parameters.items():
            commands.append(f"{key} = {value}")
        
        return commands