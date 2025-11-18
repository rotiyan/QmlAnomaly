"""
Pythia-8 event generator for particle physics simulations.
"""

import numpy as np
import h5py
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    import pythia8
    PYTHIA_AVAILABLE = True
except ImportError:
    PYTHIA_AVAILABLE = False
    pythia8 = None

from .config_parser import SimulationConfig


class PythiaEventGenerator:
    """Pythia-8 event generator with jet reconstruction."""
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize Pythia-8 generator.
        
        Parameters
        ----------
        config : SimulationConfig
            Configuration object containing simulation parameters
        """
        if not PYTHIA_AVAILABLE:
            raise ImportError(
                "Pythia-8 Python bindings not available. "
                "Please install pythia8 with Python support."
            )
        
        self.config = config
        self.pythia = pythia8.Pythia()
        self._setup_pythia()
        
        # Jet reconstruction setup
        self._setup_jet_reconstruction()
        
        # Event storage
        self.events = []
        self.jet_data = []
        
        # Statistics
        self.n_generated = 0
        self.n_accepted = 0
    
    def _setup_pythia(self):
        """Configure Pythia-8 based on configuration."""
        commands = self.config.get_pythia_commands()
        
        for command in commands:
            if not self.pythia.readString(command):
                raise ValueError(f"Failed to set Pythia parameter: {command}")
        
        # Initialize Pythia
        if not self.pythia.init():
            raise RuntimeError("Failed to initialize Pythia")
        
        logging.info(f"Pythia-8 initialized with {len(commands)} commands")
    
    def _setup_jet_reconstruction(self):
        """Setup jet reconstruction using FastJet."""
        try:
            import fastjet as fj
            self.fastjet_available = True
            self.jet_algorithm = fj.JetAlgorithm.antikt_algorithm
            self.jet_def = fj.JetDefinition(self.jet_algorithm, self.config.jets.r_parameter)
        except ImportError:
            self.fastjet_available = False
            logging.warning("FastJet not available. Using simple jet clustering.")
    
    def generate_events(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate events and return particle and jet data.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (particle_data, jet_data) where:
            - particle_data: shape (n_events, 100, 3) with (pT, eta, phi)
            - jet_data: shape (n_events, n_jets, 4) with (pT, eta, phi, mass)
        """
        logging.info(f"Generating {self.config.simulation.n_events} events...")
        
        particle_events = []
        jet_events = []
        
        for i in range(self.config.simulation.n_events):
            if i % 1000 == 0 and i > 0:
                logging.info(f"Generated {i} events...")
            
            # Generate event
            if not self.pythia.next():
                continue
            
            self.n_generated += 1
            
            # Extract particles
            particles = self._extract_particles()
            if len(particles) == 0:
                continue
            
            # Reconstruct jets
            jets = self._reconstruct_jets(particles)
            if len(jets) == 0:
                continue
            
            # Process and store data
            particle_array = self._process_particles(particles)
            jet_array = self._process_jets(jets)
            
            particle_events.append(particle_array)
            jet_events.append(jet_array)
            self.n_accepted += 1
        
        logging.info(f"Generated {self.n_generated} events, accepted {self.n_accepted}")
        
        return np.array(particle_events), np.array(jet_events)
    
    def _extract_particles(self) -> List[Dict[str, float]]:
        """Extract final state particles from Pythia event."""
        particles = []
        
        for i in range(self.pythia.event.size()):
            particle = self.pythia.event[i]
            
            # Only final state particles
            if not particle.isFinal():
                continue
            
            # Only stable particles (exclude neutrinos)
            if abs(particle.id()) in [12, 14, 16]:  # neutrinos
                continue
            
            particles.append({
                'px': particle.px(),
                'py': particle.py(),
                'pz': particle.pz(),
                'e': particle.e(),
                'id': particle.id(),
                'status': particle.status()
            })
        
        return particles
    
    def _reconstruct_jets(self, particles: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Reconstruct jets from particles."""
        if not particles:
            return []
        
        if self.fastjet_available:
            return self._reconstruct_jets_fastjet(particles)
        else:
            return self._reconstruct_jets_simple(particles)
    
    def _reconstruct_jets_fastjet(self, particles: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Reconstruct jets using FastJet."""
        import fastjet as fj
        
        # Convert particles to FastJet format
        fj_particles = []
        for p in particles:
            fj_particles.append(fj.PseudoJet(p['px'], p['py'], p['pz'], p['e']))
        
        # Cluster jets
        cluster = fj.ClusterSequence(fj_particles, self.jet_def)
        jets = cluster.inclusive_jets(self.config.jets.pt_min)
        
        # Convert back to our format
        jet_list = []
        for jet in jets:
            if abs(jet.eta()) < self.config.jets.eta_max:
                jet_list.append({
                    'pt': jet.pt(),
                    'eta': jet.eta(),
                    'phi': jet.phi(),
                    'mass': jet.m()
                })
        
        return jet_list
    
    def _reconstruct_jets_simple(self, particles: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Simple jet reconstruction (fallback when FastJet unavailable)."""
        # Simple approach: treat each particle as a "jet"
        jets = []
        for p in particles:
            pt = np.sqrt(p['px']**2 + p['py']**2)
            if pt > self.config.jets.pt_min:
                eta = 0.5 * np.log((p['e'] + p['pz']) / (p['e'] - p['pz']))
                phi = np.arctan2(p['py'], p['px'])
                
                if abs(eta) < self.config.jets.eta_max:
                    jets.append({
                        'pt': pt,
                        'eta': eta,
                        'phi': phi,
                        'mass': 0.0  # Simplified
                    })
        
        return jets
    
    def _process_particles(self, particles: List[Dict[str, float]]) -> np.ndarray:
        """Process particles into fixed-size array for autoencoder."""
        # Convert to (pT, eta, phi) format
        particle_data = []
        for p in particles:
            pt = np.sqrt(p['px']**2 + p['py']**2)
            eta = 0.5 * np.log((p['e'] + p['pz']) / (p['e'] - p['pz']))
            phi = np.arctan2(p['py'], p['px'])
            particle_data.append([pt, eta, phi])
        
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
    
    def _process_jets(self, jets: List[Dict[str, float]]) -> np.ndarray:
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
            filename = self.config.simulation.output_file
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('particle_data', data=particle_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(particle_data)
            f.attrs['n_particles_per_event'] = particle_data.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
            f.attrs['process'] = self.config.physics_process.process
            f.attrs['energy'] = self.config.physics_process.energy
        
        logging.info(f"Saved {len(particle_data)} events to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            'n_generated': self.n_generated,
            'n_accepted': self.n_accepted,
            'acceptance_rate': self.n_accepted / max(self.n_generated, 1),
            'config': {
                'process': self.config.physics_process.process,
                'energy': self.config.physics_process.energy,
                'n_events': self.config.simulation.n_events
            }
        }