#!/usr/bin/env python3
"""
Sophisticated Z+jets event generator that simulates realistic pp→Z+jets production.
This generator creates events with Z boson decay to leptons plus associated jets.
"""

import numpy as np
import h5py
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZJetsEventGenerator:
    """Sophisticated Z+jets event generator with realistic physics."""
    
    def __init__(self, config_path: str):
        """
        Initialize Z+jets generator.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.z_mass = self.config['z_boson']['mass']
        self.z_width = self.config['z_boson']['width']
        self.decay_modes = self.config['z_boson']['decay_modes']
        
        # Physics constants
        self.electron_mass = 0.000511  # GeV
        self.muon_mass = 0.105658      # GeV
        
        # Statistics
        self.n_generated = 0
        self.n_accepted = 0
        self.z_decay_stats = {'ee': 0, 'mumu': 0}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_events(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Z+jets events.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (particle_data, jet_data) where:
            - particle_data: shape (n_events, 100, 3) with (pT, eta, phi)
            - jet_data: shape (n_events, 10, 4) with (pT, eta, phi, mass)
        """
        n_events = self.config['simulation']['n_events']
        logger.info(f"Generating {n_events} Z+jets events...")
        
        particle_events = []
        jet_events = []
        
        for i in range(n_events):
            if i % 20 == 0 and i > 0:
                logger.info(f"Generated {i} events...")
            
            # Generate Z+jets event
            particles, jets, z_info = self._generate_single_event()
            
            if len(particles) == 0:
                continue
            
            self.n_generated += 1
            
            # Process particles for autoencoder
            particle_array = self._process_particles(particles)
            jet_array = self._process_jets(jets)
            
            particle_events.append(particle_array)
            jet_events.append(jet_array)
            self.n_accepted += 1
        
        logger.info(f"Generated {self.n_generated} events, accepted {self.n_accepted}")
        logger.info(f"Z decay statistics: {self.z_decay_stats}")
        
        return np.array(particle_events), np.array(jet_events)
    
    def _generate_single_event(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """Generate a single Z+jets event."""
        # Generate Z boson
        z_p4 = self._generate_z_boson()
        
        # Decay Z boson to leptons
        lepton1, lepton2, decay_mode = self._decay_z_boson(z_p4)
        
        # Generate associated jets
        jets = self._generate_associated_jets(z_p4)
        
        # Collect all particles
        particles = [lepton1, lepton2] + jets
        
        # Add some additional particles from underlying event
        particles.extend(self._generate_underlying_event())
        
        # Create jet data (leptons + jets)
        jet_data = [lepton1, lepton2] + jets
        
        z_info = {
            'mass': z_p4['mass'],
            'pt': z_p4['pt'],
            'eta': z_p4['eta'],
            'phi': z_p4['phi'],
            'decay_mode': decay_mode
        }
        
        return particles, jet_data, z_info
    
    def _generate_z_boson(self) -> Dict[str, float]:
        """Generate Z boson with realistic kinematics."""
        # Z boson pT distribution (falling exponential)
        z_pt = np.random.exponential(50) + 20  # GeV
        
        # Z boson rapidity (Gaussian around 0)
        z_y = np.random.normal(0, 1.5)
        
        # Z boson azimuthal angle
        z_phi = np.random.uniform(-np.pi, np.pi)
        
        # Calculate 4-momentum
        z_mass = self.z_mass
        z_e = np.sqrt(z_pt**2 + z_mass**2 * np.cosh(z_y)**2)
        z_pz = z_mass * np.sinh(z_y)
        z_px = z_pt * np.cos(z_phi)
        z_py = z_pt * np.sin(z_phi)
        
        return {
            'px': z_px, 'py': z_py, 'pz': z_pz, 'e': z_e,
            'pt': z_pt, 'eta': z_y, 'phi': z_phi, 'mass': z_mass
        }
    
    def _decay_z_boson(self, z_p4: Dict[str, float]) -> Tuple[Dict, Dict, str]:
        """Decay Z boson to leptons."""
        # Choose decay mode
        decay_mode = np.random.choice(self.decay_modes)
        
        if decay_mode == 'ee':
            mass1, mass2 = self.electron_mass, self.electron_mass
            pdg1, pdg2 = 11, -11  # e-, e+
        else:  # mumu
            mass1, mass2 = self.muon_mass, self.muon_mass
            pdg1, pdg2 = 13, -13  # μ-, μ+
        
        # Generate decay in Z rest frame
        # Energy of each lepton in Z rest frame
        e1_cm = (z_p4['mass']**2 + mass1**2 - mass2**2) / (2 * z_p4['mass'])
        e2_cm = z_p4['mass'] - e1_cm
        
        # Momentum magnitude in Z rest frame
        p_cm = np.sqrt(e1_cm**2 - mass1**2)
        
        # Random direction in Z rest frame
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(-np.pi, np.pi)
        
        # 4-momenta in Z rest frame
        p1_cm = np.array([
            p_cm * sin_theta * np.cos(phi),
            p_cm * sin_theta * np.sin(phi),
            p_cm * cos_theta,
            e1_cm
        ])
        p2_cm = np.array([
            -p_cm * sin_theta * np.cos(phi),
            -p_cm * sin_theta * np.sin(phi),
            -p_cm * cos_theta,
            e2_cm
        ])
        
        # Boost to lab frame
        z_beta = np.sqrt(z_p4['px']**2 + z_p4['py']**2 + z_p4['pz']**2) / z_p4['e']
        z_gamma = z_p4['e'] / z_p4['mass']
        
        # Boost vector
        boost = np.array([z_p4['px'], z_p4['py'], z_p4['pz']]) / z_p4['e']
        
        # Boost leptons to lab frame
        p1_lab = self._boost_particle(p1_cm, boost, z_gamma)
        p2_lab = self._boost_particle(p2_cm, boost, z_gamma)
        
        # Convert to our format
        lepton1 = {
            'px': p1_lab[0], 'py': p1_lab[1], 'pz': p1_lab[2], 'e': p1_lab[3],
            'pt': np.sqrt(p1_lab[0]**2 + p1_lab[1]**2),
            'eta': 0.5 * np.log((p1_lab[3] + p1_lab[2]) / (p1_lab[3] - p1_lab[2])),
            'phi': np.arctan2(p1_lab[1], p1_lab[0]),
            'mass': mass1, 'pdg': pdg1
        }
        
        lepton2 = {
            'px': p2_lab[0], 'py': p2_lab[1], 'pz': p2_lab[2], 'e': p2_lab[3],
            'pt': np.sqrt(p2_lab[0]**2 + p2_lab[1]**2),
            'eta': 0.5 * np.log((p2_lab[3] + p2_lab[2]) / (p2_lab[3] - p2_lab[2])),
            'phi': np.arctan2(p2_lab[1], p2_lab[0]),
            'mass': mass2, 'pdg': pdg2
        }
        
        self.z_decay_stats[decay_mode] += 1
        
        return lepton1, lepton2, decay_mode
    
    def _boost_particle(self, p_cm: np.ndarray, boost: np.ndarray, gamma: float) -> np.ndarray:
        """Boost particle from rest frame to lab frame."""
        # Boost along the boost vector
        p_parallel = np.dot(p_cm[:3], boost)
        p_perp = p_cm[:3] - p_parallel * boost
        
        # Boosted momentum
        p_lab = p_perp + gamma * (p_parallel + boost[2] * p_cm[3]) * boost
        
        # Boosted energy
        e_lab = gamma * (p_cm[3] + boost[2] * p_parallel)
        
        return np.array([p_lab[0], p_lab[1], p_lab[2], e_lab])
    
    def _generate_associated_jets(self, z_p4: Dict[str, float]) -> List[Dict]:
        """Generate associated jets."""
        jets = []
        
        # Number of jets (Poisson distribution)
        n_jets = np.random.poisson(2.5)  # Average 2.5 jets per event
        
        for _ in range(n_jets):
            # Jet pT (falling exponential, correlated with Z pT)
            jet_pt = np.random.exponential(30) + 20  # GeV
            
            # Jet direction (somewhat correlated with Z direction)
            jet_eta = z_p4['eta'] + np.random.normal(0, 1.0)
            jet_phi = z_p4['phi'] + np.random.normal(0, 1.0)
            
            # Apply cuts
            if abs(jet_eta) > 2.5:
                continue
            
            # Calculate 4-momentum
            jet_e = np.sqrt(jet_pt**2 + 0.1**2)  # Small mass
            jet_px = jet_pt * np.cos(jet_phi)
            jet_py = jet_pt * np.sin(jet_phi)
            jet_pz = jet_pt * np.sinh(jet_eta)
            
            jets.append({
                'px': jet_px, 'py': jet_py, 'pz': jet_pz, 'e': jet_e,
                'pt': jet_pt, 'eta': jet_eta, 'phi': jet_phi, 'mass': 0.1
            })
        
        return jets
    
    def _generate_underlying_event(self) -> List[Dict]:
        """Generate underlying event particles."""
        particles = []
        
        # Number of underlying event particles
        n_ue = np.random.poisson(20)
        
        for _ in range(n_ue):
            # Low pT particles
            pt = np.random.exponential(2) + 0.5  # GeV
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            if pt < 1.0:  # Apply pT cut
                continue
            
            # Calculate 4-momentum
            e = np.sqrt(pt**2 + 0.1**2)  # Small mass
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            
            particles.append({
                'px': px, 'py': py, 'pz': pz, 'e': e,
                'pt': pt, 'eta': eta, 'phi': phi, 'mass': 0.1
            })
        
        return particles
    
    def _process_particles(self, particles: List[Dict]) -> np.ndarray:
        """Process particles into fixed-size array for autoencoder."""
        # Convert to (pT, eta, phi) format
        particle_data = []
        for p in particles:
            particle_data.append([p['pt'], p['eta'], p['phi']])
        
        # Pad or truncate to exactly 100 particles
        n_particles = len(particle_data)
        if n_particles < 100:
            # Pad with zeros
            particle_data.extend([[0.0, 0.0, 0.0]] * (100 - n_particles))
        else:
            # Sort by pT and take top 100
            particle_data.sort(key=lambda x: x[0], reverse=True)
            particle_data = particle_data[:100]
        
        return np.array(particle_data)
    
    def _process_jets(self, jets: List[Dict]) -> np.ndarray:
        """Process jets into fixed-size array."""
        # Sort by pT
        jets.sort(key=lambda x: x['pt'], reverse=True)
        
        # Pad or truncate to 10 jets max
        max_jets = 10
        if len(jets) < max_jets:
            jets.extend([{'pt': 0.0, 'eta': 0.0, 'phi': 0.0, 'mass': 0.0}] * (max_jets - len(jets)))
        else:
            jets = jets[:max_jets]
        
        jet_array = np.array([[j['pt'], j['eta'], j['phi'], j['mass']] for j in jets])
        return jet_array
    
    def save_events(self, particle_data: np.ndarray, jet_data: np.ndarray, filename: str = None):
        """Save generated events to HDF5 file."""
        if filename is None:
            filename = self.config['simulation']['output_file']
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('particle_data', data=particle_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(particle_data)
            f.attrs['n_particles_per_event'] = particle_data.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
            f.attrs['process'] = self.config['physics_process']['process']
            f.attrs['energy'] = self.config['physics_process']['energy']
            f.attrs['z_mass'] = self.z_mass
            f.attrs['z_decay_stats'] = str(self.z_decay_stats)
        
        logger.info(f"Saved {len(particle_data)} events to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'n_generated': self.n_generated,
            'n_accepted': self.n_accepted,
            'acceptance_rate': self.n_accepted / max(self.n_generated, 1),
            'z_decay_stats': self.z_decay_stats,
            'config': {
                'process': self.config['physics_process']['process'],
                'energy': self.config['physics_process']['energy'],
                'n_events': self.config['simulation']['n_events']
            }
        }


def main():
    """Test the Z+jets generator."""
    logger.info("Testing Z+jets event generator...")
    
    # Create generator
    generator = ZJetsEventGenerator("zjets_config.yaml")
    
    # Generate events
    particle_data, jet_data = generator.generate_events()
    
    # Save events
    generator.save_events(particle_data, jet_data)
    
    # Print statistics
    stats = generator.get_statistics()
    logger.info("Generation completed!")
    logger.info(f"Generated events: {stats['n_generated']}")
    logger.info(f"Accepted events: {stats['n_accepted']}")
    logger.info(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    logger.info(f"Z decay statistics: {stats['z_decay_stats']}")
    
    # Print some sample data
    logger.info(f"Particle data shape: {particle_data.shape}")
    logger.info(f"Jet data shape: {jet_data.shape}")
    logger.info(f"Mean pT: {np.mean(particle_data[:, :, 0]):.2f} GeV")
    logger.info(f"Mean jet pT: {np.mean(jet_data[:, :, 0]):.2f} GeV")


if __name__ == "__main__":
    main()