#!/usr/bin/env python3
"""
Main entry point for the particle physics simulation and autoencoder pipeline.

This script provides a unified interface for running the complete pipeline:
1. Generate particle physics events (dijet, Z+jets, etc.)
2. Process data for autoencoder input
3. Train autoencoder for compression
4. Generate latent representations for quantum ML

Usage:
    python -m qad.simulation.run_pipeline --help
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from qad.simulation import SimulationConfig, EventDataProcessor
from qad.autoencoder.autoencoder import ParticleAutoencoder
from qad.autoencoder.util import get_mean, get_std

# Import our custom generators
try:
    from examples.simulation.zjets_generator import ZJetsEventGenerator
    ZJETS_AVAILABLE = True
except ImportError:
    ZJETS_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for particle data compression."""
    
    def __init__(self, input_dim=300, latent_dim=6):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


def preprocess_particles(particle_data):
    """Preprocess particle data for autoencoder."""
    processed_data = particle_data.copy()
    
    # Log transform pT for better numerical stability
    processed_data[:, :, 0] = np.log1p(processed_data[:, :, 0])
    
    # Normalize eta and phi to [-1, 1] range
    processed_data[:, :, 1] = np.tanh(processed_data[:, :, 1])  # eta
    processed_data[:, :, 2] = processed_data[:, :, 2] / np.pi  # phi
    
    return processed_data


def train_autoencoder(particle_data, latent_dim=6, epochs=100):
    """Train autoencoder on particle data."""
    logger = logging.getLogger(__name__)
    logger.info("Training autoencoder...")
    
    # Convert to PyTorch tensors
    data_tensor = torch.FloatTensor(particle_data.reshape(particle_data.shape[0], -1))
    
    # Create training/validation split
    n_events = len(data_tensor)
    n_train = int(n_events * 0.8)
    
    train_data = data_tensor[:n_train]
    val_data = data_tensor[n_train:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)
    
    # Create model
    model = SimpleAutoencoder(input_dim=300, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    logger.info("Autoencoder training completed!")
    return model


def generate_mock_dijet_events(n_events=100):
    """Generate mock dijet events for demonstration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {n_events} mock dijet events...")
    
    particle_events = []
    jet_events = []
    
    for i in range(n_events):
        # Generate mock particle data for dijet events
        n_particles = np.random.randint(20, 80)
        
        # Create two main jets
        jet1_pt = np.random.exponential(100) + 50
        jet1_eta = np.random.uniform(-2.5, 2.5)
        jet1_phi = np.random.uniform(-np.pi, np.pi)
        
        jet2_pt = np.random.exponential(80) + 40
        jet2_eta = np.random.uniform(-2.5, 2.5)
        jet2_phi = jet1_phi + np.pi + np.random.normal(0, 0.5)
        
        # Generate particles around jets
        particles = []
        for j in range(n_particles):
            if j < n_particles // 2:
                pt = np.random.exponential(jet1_pt / 10) + 5
                eta = jet1_eta + np.random.normal(0, 0.3)
                phi = jet1_phi + np.random.normal(0, 0.3)
            else:
                pt = np.random.exponential(jet2_pt / 10) + 5
                eta = jet2_eta + np.random.normal(0, 0.3)
                phi = jet2_phi + np.random.normal(0, 0.3)
            
            if pt > 1.0 and abs(eta) < 2.5:
                particles.append([pt, eta, phi])
        
        # Pad or truncate to exactly 100 particles
        if len(particles) < 100:
            particles.extend([[0.0, 0.0, 0.0]] * (100 - len(particles)))
        else:
            particles.sort(key=lambda x: x[0], reverse=True)
            particles = particles[:100]
        
        particle_events.append(particles)
        
        # Generate jet data
        jets = [
            [jet1_pt, jet1_eta, jet1_phi, jet1_pt * 0.1],
            [jet2_pt, jet2_eta, jet2_phi, jet2_pt * 0.1],
        ]
        
        # Add additional jets
        for _ in range(8):
            pt = np.random.exponential(20) + 10
            if pt > 20:
                eta = np.random.uniform(-2.5, 2.5)
                phi = np.random.uniform(-np.pi, np.pi)
                mass = pt * 0.1
                jets.append([pt, eta, phi, mass])
            else:
                jets.append([0.0, 0.0, 0.0, 0.0])
        
        while len(jets) < 10:
            jets.append([0.0, 0.0, 0.0, 0.0])
        
        jet_events.append(jets[:10])
    
    return np.array(particle_events), np.array(jet_events)


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Run particle physics simulation and autoencoder pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dijet events with mock data
  python -m qad.simulation.run_pipeline --process dijet --n-events 100

  # Generate Z+jets events (requires zjets_generator.py)
  python -m qad.simulation.run_pipeline --process zjets --n-events 100

  # Use configuration file
  python -m qad.simulation.run_pipeline --config examples/simulation/dijet_10_events.yaml

  # With custom output directory
  python -m qad.simulation.run_pipeline --process dijet --n-events 50 --output-dir my_output
        """
    )
    
    parser.add_argument('--process', choices=['dijet', 'zjets'], 
                       help='Physics process to simulate')
    parser.add_argument('--n-events', type=int, default=100,
                       help='Number of events to generate')
    parser.add_argument('--config', type=str,
                       help='Path to YAML configuration file')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results')
    parser.add_argument('--latent-dim', type=int, default=6,
                       help='Latent dimension for autoencoder')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate events
        if args.config:
            # Use configuration file
            logger.info(f"Loading configuration from {args.config}")
            config = SimulationConfig(args.config)
            
            if config.physics_process.process == "pp -> jj":
                particle_data, jet_data = generate_mock_dijet_events(config.simulation.n_events)
            elif "zjets" in config.physics_process.process.lower() and ZJETS_AVAILABLE:
                generator = ZJetsEventGenerator(args.config)
                particle_data, jet_data = generator.generate_events()
            else:
                raise ValueError(f"Unsupported process: {config.physics_process.process}")
        else:
            # Use command line arguments
            if args.process == 'dijet':
                particle_data, jet_data = generate_mock_dijet_events(args.n_events)
            elif args.process == 'zjets':
                if not ZJETS_AVAILABLE:
                    raise ImportError("Z+jets generator not available. Please ensure zjets_generator.py is in the examples/simulation/ directory.")
                # Create temporary config
                config_data = {
                    'physics_process': {'process': 'pp -> Z+jets', 'energy': 13000.0, 'parameters': {}},
                    'simulation': {'n_events': args.n_events, 'output_file': 'temp.h5', 'random_seed': 12345},
                    'jets': {'algorithm': 'antikt', 'r_parameter': 0.4, 'pt_min': 20.0, 'eta_max': 2.5},
                    'z_boson': {'mass': 91.1876, 'width': 2.4952, 'decay_modes': ['ee', 'mumu']}
                }
                import yaml
                config_path = output_dir / 'temp_config.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f)
                
                generator = ZJetsEventGenerator(str(config_path))
                particle_data, jet_data = generator.generate_events()
            else:
                raise ValueError(f"Unsupported process: {args.process}")
        
        logger.info(f"Generated {len(particle_data)} events")
        logger.info(f"Particle data shape: {particle_data.shape}")
        logger.info(f"Jet data shape: {jet_data.shape}")
        
        # Process data
        logger.info("Preprocessing particle data...")
        processed_particles = preprocess_particles(particle_data)
        
        # Train autoencoder
        logger.info("Training autoencoder...")
        autoencoder = train_autoencoder(processed_particles, args.latent_dim, args.epochs)
        
        # Encode to latent space
        logger.info("Encoding events to latent space...")
        with torch.no_grad():
            data_tensor = torch.FloatTensor(processed_particles.reshape(processed_particles.shape[0], -1))
            latent_data = autoencoder.encode(data_tensor).numpy()
        
        # Verify reconstruction
        with torch.no_grad():
            reconstructed_data = autoencoder(data_tensor).numpy()
            reconstructed_data = reconstructed_data.reshape(processed_particles.shape)
        
        reconstruction_error = np.mean(np.square(processed_particles - reconstructed_data))
        logger.info(f"Reconstruction MSE: {reconstruction_error:.6f}")
        
        # Save results
        logger.info("Saving results...")
        
        # Save raw events
        raw_output_path = output_dir / "raw_events.h5"
        with h5py.File(raw_output_path, 'w') as f:
            f.create_dataset('particle_data', data=particle_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(particle_data)
            f.attrs['process'] = args.process or 'unknown'
        
        # Save processed data
        processed_output_path = output_dir / "processed_events.h5"
        with h5py.File(processed_output_path, 'w') as f:
            f.create_dataset('particle_data', data=processed_particles)
            f.create_dataset('latent_data', data=latent_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(processed_particles)
            f.attrs['latent_dim'] = latent_data.shape[1]
            f.attrs['n_particles_per_event'] = processed_particles.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
        
        # Save autoencoder
        autoencoder_path = output_dir / "autoencoder.pth"
        torch.save(autoencoder.state_dict(), autoencoder_path)
        
        # Print summary
        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated events: {len(particle_data)}")
        logger.info(f"Latent data shape: {latent_data.shape}")
        logger.info(f"Compression ratio: {processed_particles.size / latent_data.size:.1f}x")
        logger.info(f"Reconstruction MSE: {reconstruction_error:.6f}")
        logger.info(f"Output files saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()