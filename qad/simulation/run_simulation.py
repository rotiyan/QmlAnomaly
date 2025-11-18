#!/usr/bin/env python3
"""
Main script for running Pythia-8 simulations and processing with autoencoder.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from qad.simulation import PythiaEventGenerator, SimulationConfig, EventDataProcessor
from qad.autoencoder.autoencoder import ParticleAutoencoder
from qad.autoencoder.util import get_mean, get_std


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


def main():
    """Main simulation function."""
    parser = argparse.ArgumentParser(description='Run Pythia-8 simulation with autoencoder processing')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--autoencoder-path', help='Path to pre-trained autoencoder')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    try:
        # Load configuration
        logging.info(f"Loading configuration from {args.config}")
        config = SimulationConfig(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Pythia generator
        logging.info("Initializing Pythia-8 generator...")
        generator = PythiaEventGenerator(config)
        
        # Generate events
        logging.info("Generating events...")
        particle_data, jet_data = generator.generate_events()
        
        # Save raw events
        raw_output_path = output_dir / "raw_events.h5"
        generator.save_events(particle_data, jet_data, str(raw_output_path))
        
        # Initialize data processor
        processor = EventDataProcessor()
        
        # Preprocess particle data
        logging.info("Preprocessing particle data...")
        processed_particles = processor.preprocess_particles(particle_data)
        
        # Load autoencoder if provided
        autoencoder = None
        if args.autoencoder_path:
            logging.info(f"Loading autoencoder from {args.autoencoder_path}")
            try:
                autoencoder = ParticleAutoencoder.load(args.autoencoder_path)
                processor.autoencoder = autoencoder
            except Exception as e:
                logging.warning(f"Failed to load autoencoder: {e}")
                logging.warning("Continuing without autoencoder...")
        
        # Encode events if autoencoder is available
        latent_data = None
        if autoencoder is not None:
            logging.info("Encoding events to latent space...")
            latent_data = processor.encode_events(processed_particles)
        else:
            # Create dummy latent data for compatibility
            latent_data = np.zeros((len(processed_particles), 6))
            logging.info("No autoencoder available, using dummy latent data")
        
        # Save processed data
        processed_output_path = output_dir / "processed_events.h5"
        processor.save_processed_data(
            processed_particles, 
            latent_data, 
            jet_data, 
            str(processed_output_path)
        )
        
        # Print statistics
        stats = generator.get_statistics()
        logging.info("Simulation completed successfully!")
        logging.info(f"Generated events: {stats['n_generated']}")
        logging.info(f"Accepted events: {stats['n_accepted']}")
        logging.info(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
        logging.info(f"Process: {stats['config']['process']}")
        logging.info(f"Energy: {stats['config']['energy']} GeV")
        
        print(f"\nSimulation completed!")
        print(f"Raw events saved to: {raw_output_path}")
        print(f"Processed events saved to: {processed_output_path}")
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()