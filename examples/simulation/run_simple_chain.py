#!/usr/bin/env python3
"""
Simplified full chain script for proton-proton collision to dijet event production
with variational autoencoder compressed representation.

This script runs the complete pipeline:
1. Generate 10 dijet events (using mock data)
2. Process particle data for autoencoder
3. Train autoencoder on the data
4. Encode events to latent space
5. Verify the complete pipeline
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

import numpy as np
import tensorflow as tf
import h5py
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mock_dijet_events(n_events=10):
    """
    Generate mock dijet events that simulate Pythia8 output.
    
    Parameters
    ----------
    n_events : int
        Number of events to generate
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (particle_data, jet_data) where:
        - particle_data: shape (n_events, 100, 3) with (pT, eta, phi)
        - jet_data: shape (n_events, 10, 4) with (pT, eta, phi, mass)
    """
    logger.info(f"Generating {n_events} mock dijet events...")
    
    particle_events = []
    jet_events = []
    
    for i in range(n_events):
        # Generate mock particle data for dijet events
        n_particles = np.random.randint(20, 80)  # Variable number of particles per event
        
        # Generate particles with realistic dijet characteristics
        particles = []
        
        # Create two main jets (dijet structure)
        jet1_pt = np.random.exponential(100) + 50  # GeV
        jet1_eta = np.random.uniform(-2.5, 2.5)
        jet1_phi = np.random.uniform(-np.pi, np.pi)
        
        jet2_pt = np.random.exponential(80) + 40   # GeV
        jet2_eta = np.random.uniform(-2.5, 2.5)
        jet2_phi = jet1_phi + np.pi + np.random.normal(0, 0.5)  # Back-to-back with some spread
        
        # Generate particles around the two jets
        for j in range(n_particles):
            if j < n_particles // 2:
                # Particles from jet 1
                pt = np.random.exponential(jet1_pt / 10) + 5
                eta = jet1_eta + np.random.normal(0, 0.3)
                phi = jet1_phi + np.random.normal(0, 0.3)
            else:
                # Particles from jet 2
                pt = np.random.exponential(jet2_pt / 10) + 5
                eta = jet2_eta + np.random.normal(0, 0.3)
                phi = jet2_phi + np.random.normal(0, 0.3)
            
            # Apply cuts
            if pt > 1.0 and abs(eta) < 2.5:
                particles.append([pt, eta, phi])
        
        # Pad or truncate to exactly 100 particles
        if len(particles) < 100:
            particles.extend([[0.0, 0.0, 0.0]] * (100 - len(particles)))
        else:
            # Sort by pT and take top 100
            particles.sort(key=lambda x: x[0], reverse=True)
            particles = particles[:100]
        
        particle_events.append(particles)
        
        # Generate jet data (2 main jets + some additional jets)
        jets = [
            [jet1_pt, jet1_eta, jet1_phi, jet1_pt * 0.1],  # jet mass ~ 10% of pT
            [jet2_pt, jet2_eta, jet2_phi, jet2_pt * 0.1],
        ]
        
        # Add some additional smaller jets
        for _ in range(8):  # Total 10 jets max
            pt = np.random.exponential(20) + 10
            if pt > 20:  # Apply pT cut
                eta = np.random.uniform(-2.5, 2.5)
                phi = np.random.uniform(-np.pi, np.pi)
                mass = pt * 0.1
                jets.append([pt, eta, phi, mass])
            else:
                jets.append([0.0, 0.0, 0.0, 0.0])
        
        # Pad to exactly 10 jets
        while len(jets) < 10:
            jets.append([0.0, 0.0, 0.0, 0.0])
        
        jet_events.append(jets[:10])
    
    particle_data = np.array(particle_events)
    jet_data = np.array(jet_events)
    
    logger.info(f"Generated particle data shape: {particle_data.shape}")
    logger.info(f"Generated jet data shape: {jet_data.shape}")
    logger.info(f"Mean pT: {np.mean(particle_data[:, :, 0]):.2f} GeV")
    logger.info(f"Mean jet pT: {np.mean(jet_data[:, :, 0]):.2f} GeV")
    
    return particle_data, jet_data


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
    
    # Apply basic preprocessing
    processed_data = particle_data.copy()
    
    # Remove events with no particles (all zeros)
    non_zero_events = np.any(processed_data.sum(axis=(1, 2)) > 0, axis=0)
    processed_data = processed_data[non_zero_events]
    
    # Log transform pT for better numerical stability
    processed_data[:, :, 0] = np.log1p(processed_data[:, :, 0])
    
    # Normalize eta and phi to [-1, 1] range
    processed_data[:, :, 1] = np.tanh(processed_data[:, :, 1])  # eta
    processed_data[:, :, 2] = processed_data[:, :, 2] / np.pi  # phi
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    return processed_data


def create_simple_autoencoder(input_shape=(100, 3), latent_dim=6):
    """Create a simple autoencoder model."""
    
    # Encoder
    inputs = tf.keras.Input(shape=input_shape)
    
    # Flatten input
    x = tf.keras.layers.Flatten()(inputs)
    
    # Dense layers
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Latent space
    latent = tf.keras.layers.Dense(latent_dim, activation='linear', name='latent')(x)
    
    # Decoder
    x = tf.keras.layers.Dense(64, activation='relu')(latent)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='linear')(x)
    
    # Reshape to original shape
    outputs = tf.keras.layers.Reshape(input_shape)(x)
    
    # Create model
    autoencoder = tf.keras.Model(inputs, outputs)
    
    # Create encoder and decoder separately
    encoder = tf.keras.Model(inputs, latent)
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    decoder_output = autoencoder.layers[-2](decoder_input)
    decoder_output = autoencoder.layers[-1](decoder_output)
    decoder = tf.keras.Model(decoder_input, decoder_output)
    
    return autoencoder, encoder, decoder


def train_autoencoder(particle_data, latent_dim=6, epochs=50):
    """
    Train autoencoder on particle data.
    
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
    Tuple[autoencoder, encoder, decoder]
        Trained models
    """
    logger.info("Training autoencoder...")
    
    # Create training/validation split
    n_events = len(particle_data)
    n_train = int(n_events * 0.8)
    
    train_data = particle_data[:n_train]
    val_data = particle_data[n_train:]
    
    # Convert to TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(4, drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices(val_data).batch(4, drop_remainder=True)
    
    # Create autoencoder
    autoencoder, encoder, decoder = create_simple_autoencoder(
        input_shape=(100, 3),
        latent_dim=latent_dim
    )
    
    # Compile model
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train model
    history = autoencoder.fit(
        train_ds,
        epochs=epochs,
        shuffle=True,
        validation_data=val_ds,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)
        ]
    )
    
    logger.info("Autoencoder training completed!")
    logger.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    return autoencoder, encoder, decoder


def main():
    """Main function to run the complete pipeline."""
    logger.info("Starting full chain: Proton-proton collision to dijet events with VAE compression")
    logger.info("=" * 80)
    
    try:
        # 1. Generate mock dijet events
        logger.info("\n1. Generating mock dijet events...")
        particle_data, jet_data = generate_mock_dijet_events(10)
        
        # 2. Process data for autoencoder
        logger.info("\n2. Processing data for autoencoder...")
        processed_particles = preprocess_particles(particle_data)
        
        # 3. Train autoencoder
        logger.info("\n3. Training autoencoder...")
        autoencoder, encoder, decoder = train_autoencoder(processed_particles, latent_dim=6, epochs=50)
        
        # 4. Encode events to latent space
        logger.info("\n4. Encoding events to latent space...")
        latent_data = encoder.predict(processed_particles, verbose=0)
        logger.info(f"   Latent data shape: {latent_data.shape}")
        logger.info(f"   Latent dimension: {latent_data.shape[1]}")
        
        # 5. Verify reconstruction quality
        logger.info("\n5. Verifying reconstruction quality...")
        reconstructed_data = decoder.predict(latent_data, verbose=0)
        reconstruction_error = np.mean(np.square(processed_particles - reconstructed_data))
        logger.info(f"   Reconstruction MSE: {reconstruction_error:.6f}")
        
        # 6. Save results
        logger.info("\n6. Saving results...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw events
        raw_output_path = output_dir / "raw_dijet_events.h5"
        with h5py.File(raw_output_path, 'w') as f:
            f.create_dataset('particle_data', data=particle_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(particle_data)
            f.attrs['process'] = 'pp -> jj'
            f.attrs['energy'] = 13000.0
        
        # Save processed data
        processed_output_path = output_dir / "processed_dijet_events.h5"
        with h5py.File(processed_output_path, 'w') as f:
            f.create_dataset('particle_data', data=processed_particles)
            f.create_dataset('latent_data', data=latent_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(processed_particles)
            f.attrs['latent_dim'] = latent_data.shape[1]
            f.attrs['n_particles_per_event'] = processed_particles.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
        
        # Save autoencoder
        autoencoder_path = output_dir / "dijet_autoencoder"
        autoencoder.save(str(autoencoder_path))
        
        # 7. Display final statistics
        logger.info("\n7. Final Statistics:")
        logger.info("=" * 40)
        logger.info(f"Generated events: {len(particle_data)}")
        logger.info(f"Particle data shape: {particle_data.shape}")
        logger.info(f"Jet data shape: {jet_data.shape}")
        logger.info(f"Processed data shape: {processed_particles.shape}")
        logger.info(f"Latent data shape: {latent_data.shape}")
        logger.info(f"Compression ratio: {processed_particles.size / latent_data.size:.2f}x")
        logger.info(f"Reconstruction MSE: {reconstruction_error:.6f}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  Raw events: {raw_output_path}")
        logger.info(f"  Processed events: {processed_output_path}")
        logger.info(f"  Autoencoder: {autoencoder_path}")
        
        # 8. Show sample latent representations
        logger.info(f"\n8. Sample latent representations (first 3 events):")
        for i in range(min(3, len(latent_data))):
            logger.info(f"   Event {i+1}: {latent_data[i]}")
        
        logger.info("\n✅ Full chain completed successfully!")
        logger.info("Pipeline: Mock Pythia8 → Data Processing → Autoencoder Training → Latent Encoding")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())