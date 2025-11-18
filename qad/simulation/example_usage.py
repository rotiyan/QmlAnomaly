#!/usr/bin/env python3
"""
Example usage of the Pythia-8 simulation module.
This example demonstrates how to use the module without requiring Pythia-8 installation.
"""

import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import directly to avoid dependency issues
from qad.simulation.config_parser import SimulationConfig
from qad.simulation.data_processor import EventDataProcessor


def create_example_config():
    """Create an example configuration file."""
    config = {
        'physics_process': {
            'process': 'pp -> jj',
            'energy': 13000.0,
            'parameters': {
                'PhaseSpace:pTHatMin': 50.0,
                'PhaseSpace:pTHatMax': 2000.0
            }
        },
        'simulation': {
            'n_events': 1000,
            'output_file': 'example_events.h5',
            'random_seed': 12345,
            'debug_level': 0
        },
        'jets': {
            'algorithm': 'antikt',
            'r_parameter': 0.4,
            'pt_min': 20.0,
            'eta_max': 2.5
        }
    }
    
    config_path = Path("example_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(config_path)


def generate_mock_events(n_events=1000):
    """Generate mock particle physics events for demonstration."""
    print(f"Generating {n_events} mock events...")
    
    # Generate mock particle data (pT, eta, phi)
    particle_data = np.random.rand(n_events, 100, 3)
    
    # Scale pT values (log-normal distribution)
    particle_data[:, :, 0] = np.exp(np.random.normal(2, 1, (n_events, 100)))
    
    # Scale eta values (uniform in [-2.5, 2.5])
    particle_data[:, :, 1] = (particle_data[:, :, 1] - 0.5) * 5
    
    # Scale phi values (uniform in [-π, π])
    particle_data[:, :, 2] = (particle_data[:, :, 2] - 0.5) * 2 * np.pi
    
    # Generate mock jet data (pT, eta, phi, mass)
    jet_data = np.random.rand(n_events, 10, 4)
    jet_data[:, :, 0] = np.exp(np.random.normal(3, 0.5, (n_events, 10)))  # pT
    jet_data[:, :, 1] = (jet_data[:, :, 1] - 0.5) * 4  # eta
    jet_data[:, :, 2] = (jet_data[:, :, 2] - 0.5) * 2 * np.pi  # phi
    jet_data[:, :, 3] = np.random.exponential(10, (n_events, 10))  # mass
    
    return particle_data, jet_data


def main():
    """Main example function."""
    print("Pythia-8 Simulation Module Example")
    print("=" * 40)
    
    try:
        # 1. Create and load configuration
        print("\n1. Loading configuration...")
        config_path = create_example_config()
        config = SimulationConfig(config_path)
        
        print(f"   Process: {config.physics_process.process}")
        print(f"   Energy: {config.physics_process.energy} GeV")
        print(f"   Events: {config.simulation.n_events}")
        
        # 2. Generate mock events (simulating Pythia-8 output)
        print("\n2. Generating mock events...")
        particle_data, jet_data = generate_mock_events(config.simulation.n_events)
        
        print(f"   Particle data shape: {particle_data.shape}")
        print(f"   Jet data shape: {jet_data.shape}")
        print(f"   Mean pT: {np.mean(particle_data[:, :, 0]):.2f} GeV")
        
        # 3. Process data for autoencoder
        print("\n3. Processing data for autoencoder...")
        processor = EventDataProcessor()
        
        # Preprocess particle data
        processed_particles = processor.preprocess_particles(particle_data)
        print(f"   Processed shape: {processed_particles.shape}")
        print(f"   Preprocessing stats: {processor.data_stats}")
        
        # 4. Create mock latent representations (simulating autoencoder output)
        print("\n4. Creating mock latent representations...")
        latent_dim = 6
        latent_data = np.random.randn(len(processed_particles), latent_dim)
        print(f"   Latent data shape: {latent_data.shape}")
        
        # 5. Save processed data
        print("\n5. Saving processed data...")
        output_path = "example_processed_events.h5"
        processor.save_processed_data(
            processed_particles, 
            latent_data, 
            jet_data, 
            output_path
        )
        print(f"   Saved to: {output_path}")
        
        # 6. Demonstrate data loading
        print("\n6. Loading and verifying data...")
        loaded_particles, loaded_jets = processor.load_events(output_path)
        print(f"   Loaded particle data shape: {loaded_particles.shape}")
        print(f"   Loaded jet data shape: {loaded_jets.shape}")
        
        # 7. Show Pythia-8 commands that would be generated
        print("\n7. Pythia-8 commands that would be generated:")
        commands = config.get_pythia_commands()
        for i, cmd in enumerate(commands[:5], 1):  # Show first 5 commands
            print(f"   {i}. {cmd}")
        if len(commands) > 5:
            print(f"   ... and {len(commands) - 5} more commands")
        
        print("\n✅ Example completed successfully!")
        print("\nTo use with real Pythia-8:")
        print("1. Install Pythia-8 with Python bindings")
        print("2. Install FastJet (optional)")
        print("3. Run: python run_simulation.py --config example_config.yaml")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        for file in ["example_config.yaml", "example_processed_events.h5"]:
            if Path(file).exists():
                Path(file).unlink()
                print(f"   Cleaned up: {file}")


if __name__ == "__main__":
    main()