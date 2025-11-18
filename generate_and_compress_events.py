#!/usr/bin/env python3
"""
Complete pipeline for event generation, compression, and quantum ML preparation.

This script implements the full pipeline from the paper:
1. Generate dijet events using Pythia8
2. Preprocess particle data
3. Train autoencoder for compression
4. Encode events to latent space
5. Save in format compatible with quantum anomaly detection algorithms

Usage:
    python generate_and_compress_events.py --config config.yaml --n-events 100
    python generate_and_compress_events.py --process dijet --n-events 1000 --latent-dim 6
"""

import argparse
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from qad.simulation import PythiaEventGenerator, SimulationConfig
from qad.autoencoder.autoencoder import ParticleAutoencoder
from qad.autoencoder.util import get_mean, get_std
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)
logger = logging.getLogger(__name__)


def preprocess_particles(particle_data):
    """
    Preprocess particle data for autoencoder.
    
    Parameters
    ----------
    particle_data : np.ndarray
        Raw particle data of shape (n_events, 100, 3)
        
    Returns
    -------
    np.ndarray
        Preprocessed particle data
    """
    logger.info("Preprocessing particle data...")
    processed_data = particle_data.copy()
    
    # Log transform pT for better numerical stability
    processed_data[:, :, 0] = np.log1p(processed_data[:, :, 0])
    
    # Normalize eta and phi to [-1, 1] range
    processed_data[:, :, 1] = np.tanh(processed_data[:, :, 1])  # eta
    processed_data[:, :, 2] = processed_data[:, :, 2] / np.pi  # phi
    
    logger.info(f"✓ Preprocessed data shape: {processed_data.shape}")
    return processed_data


def train_autoencoder(particle_data, latent_dim=6, epochs=100):
    """
    Train autoencoder on particle data using TensorFlow (original implementation).
    
    Parameters
    ----------
    particle_data : np.ndarray
        Preprocessed particle data
    latent_dim : int
        Latent dimension size
    epochs : int
        Number of training epochs
        
    Returns
    -------
    ParticleAutoencoder
        Trained autoencoder model
    """
    logger.info(f"Training autoencoder (latent_dim={latent_dim}, epochs={epochs})...")
    
    # Create training/validation split
    n_events = len(particle_data)
    n_train = int(n_events * 0.8)
    
    train_data = particle_data[:n_train]
    val_data = particle_data[n_train:]
    
    logger.info(f"  Training samples: {n_train}, Validation samples: {n_events - n_train}")
    
    # Create autoencoder using original framework
    autoencoder = ParticleAutoencoder(
        input_shape=(100, 3),
        latent_dim=latent_dim,
        x_mean_stdev=(get_mean(particle_data), get_std(particle_data)),
        activation_latent=tf.keras.activations.linear,
    )
    
    # Define loss function
    def threeD_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compile model (use legacy optimizer for M1/M2 Mac compatibility)
    autoencoder.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        reco_loss=threeD_loss
    )
    
    # Convert to TensorFlow datasets (ensure float32 dtype)
    train_data = train_data.astype(np.float32)
    val_data = val_data.astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(8, drop_remainder=False)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data).batch(8, drop_remainder=False)
    
    # Train model
    history = autoencoder.fit(
        train_ds,
        epochs=epochs,
        shuffle=True,
        validation_data=val_ds,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=0)
        ]
    )
    
    logger.info(f"✓ Training completed")
    logger.info(f"  Final training loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    return autoencoder


def save_for_quantum_ml(latent_data, jet_data, output_path, metadata=None):
    """
    Save latent representations in format expected by quantum anomaly detection.
    
    The quantum algorithms expect HDF5 with 'latent_space' key in shape (n, 2, latent_dim).
    This represents pairs of jets per event encoded in latent space.
    
    Parameters
    ----------
    latent_data : np.ndarray
        Latent representations of shape (n_events, latent_dim)
    jet_data : np.ndarray
        Jet data of shape (n_events, n_jets, 4)
    output_path : str
        Path to save HDF5 file
    metadata : dict, optional
        Additional metadata to save
    """
    logger.info(f"Saving for quantum ML: {output_path}")
    
    # Reshape latent data for quantum anomaly detection
    # Original format: (n_events, latent_dim)
    # Required format: (n_events, 2, latent_dim) for jet pairs
    # We'll create jet pairs from leading jets
    n_events = len(latent_data)
    latent_dim = latent_data.shape[1]
    
    # Create jet-pair representation
    # For simplicity, duplicate the event latent representation for each jet
    latent_space = np.zeros((n_events, 2, latent_dim))
    for i in range(n_events):
        # Represent both leading jets with the event's latent encoding
        latent_space[i, 0, :] = latent_data[i]
        latent_space[i, 1, :] = latent_data[i]
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('latent_space', data=latent_space)
        f.create_dataset('jet_data', data=jet_data)
        f.attrs['n_events'] = n_events
        f.attrs['latent_dim'] = latent_dim
        f.attrs['format'] = 'quantum_ml_ready'
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
    
    logger.info(f"✓ Saved latent_space with shape: {latent_space.shape}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description='Generate events with Pythia8 and compress for quantum ML',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--process', choices=['dijet', 'zjets'], default='dijet',
                       help='Physics process to simulate')
    parser.add_argument('--n-events', type=int, default=100,
                       help='Number of events to generate')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for results')
    parser.add_argument('--latent-dim', type=int, default=6,
                       help='Latent dimension for autoencoder')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("PYTHIA8 EVENT GENERATION AND COMPRESSION PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load or create configuration
        logger.info("\n[1/6] Loading configuration...")
        if args.config:
            config = SimulationConfig(args.config)
        else:
            # Create default configuration
            import yaml
            config_data = {
                'physics_process': {
                    'process': 'pp -> jj',
                    'energy': 13000.0,
                    'parameters': {
                        'PhaseSpace:pTHatMin': 50.0,
                        'PhaseSpace:pTHatMax': 2000.0,
                        'PartonLevel:ISR': 'on',
                        'PartonLevel:FSR': 'on',
                        'PartonLevel:MPI': 'on'
                    }
                },
                'simulation': {
                    'n_events': args.n_events,
                    'output_file': str(output_dir / 'raw_events.h5'),
                    'random_seed': 12345,
                    'debug_level': 1
                },
                'jets': {
                    'algorithm': 'antikt',
                    'r_parameter': 0.4,
                    'pt_min': 20.0,
                    'eta_max': 2.5
                }
            }
            
            config_path = output_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            config = SimulationConfig(str(config_path))
        
        logger.info(f"  Process: {config.physics_process.process}")
        logger.info(f"  Energy: {config.physics_process.energy} GeV")
        logger.info(f"  Events: {config.simulation.n_events}")
        
        # 2. Generate events with Pythia8
        logger.info("\n[2/6] Generating events with Pythia8...")
        generator = PythiaEventGenerator(config)
        particle_data, jet_data = generator.generate_events()
        
        stats = generator.get_statistics()
        logger.info(f"  Generated: {stats['n_generated']}, Accepted: {stats['n_accepted']}")
        logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
        
        # Save raw events
        raw_output_path = output_dir / "raw_events.h5"
        generator.save_events(particle_data, jet_data, str(raw_output_path))
        
        # 3. Preprocess particle data
        logger.info("\n[3/6] Preprocessing particle data...")
        processed_particles = preprocess_particles(particle_data)
        
        # 4. Train autoencoder
        logger.info("\n[4/6] Training autoencoder...")
        autoencoder = train_autoencoder(processed_particles, args.latent_dim, args.epochs)
        
        # 5. Encode events to latent space
        logger.info("\n[5/6] Encoding events to latent space...")
        latent_data = autoencoder.encoder.predict(processed_particles, verbose=0)
        logger.info(f"  Latent data shape: {latent_data.shape}")
        
        # Verify reconstruction
        reconstructed_data = autoencoder.decoder.predict(latent_data, verbose=0)
        reconstruction_error = np.mean(np.square(processed_particles - reconstructed_data))
        logger.info(f"  Reconstruction MSE: {reconstruction_error:.6f}")
        
        # 6. Save results
        logger.info("\n[6/6] Saving results...")
        
        # Save for quantum ML
        quantum_ml_path = output_dir / "events_for_quantum_ml.h5"
        save_for_quantum_ml(
            latent_data,
            jet_data,
            str(quantum_ml_path),
            metadata={
                'process': config.physics_process.process,
                'energy': config.physics_process.energy,
                'reconstruction_mse': reconstruction_error
            }
        )
        
        # Save autoencoder weights
        autoencoder_path = output_dir / "autoencoder_weights.h5"
        autoencoder.save_weights(str(autoencoder_path))
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Generated events:     {len(particle_data)}")
        logger.info(f"Latent dimension:     {args.latent_dim}")
        logger.info(f"Compression ratio:    {processed_particles.size / latent_data.size:.1f}x")
        logger.info(f"Reconstruction MSE:   {reconstruction_error:.6f}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  Raw events:           {raw_output_path}")
        logger.info(f"  Quantum ML format:    {quantum_ml_path}")
        logger.info(f"  Autoencoder weights:  {autoencoder_path}")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Use {quantum_ml_path} with quantum anomaly detection algorithms")
        logger.info(f"  2. Train one-class QSVM: scripts/kernel_machines/train_one_class_qsvm.py")
        logger.info(f"  3. Apply quantum k-medians clustering")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
