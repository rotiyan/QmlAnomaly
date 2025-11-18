"""
Data processor for converting Pythia-8 output to autoencoder-compatible format.
"""

import numpy as np
import h5py
from typing import Tuple, Optional
from pathlib import Path
import logging

from qad.autoencoder.autoencoder import ParticleAutoencoder
from qad.autoencoder.util import get_mean, get_std


class EventDataProcessor:
    """Process Pythia-8 events for autoencoder training/inference."""
    
    def __init__(self, autoencoder: Optional[ParticleAutoencoder] = None):
        """
        Initialize data processor.
        
        Parameters
        ----------
        autoencoder : ParticleAutoencoder, optional
            Pre-trained autoencoder for encoding events
        """
        self.autoencoder = autoencoder
        self.data_stats = None
    
    def load_events(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load events from HDF5 file.
        
        Parameters
        ----------
        file_path : str
            Path to HDF5 file containing events
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (particle_data, jet_data)
        """
        with h5py.File(file_path, 'r') as f:
            particle_data = f['particle_data'][:]
            jet_data = f['jet_data'][:]
            
            # Load metadata
            metadata = dict(f.attrs)
            logging.info(f"Loaded {metadata['n_events']} events from {file_path}")
            logging.info(f"Process: {metadata['process']}, Energy: {metadata['energy']} GeV")
        
        return particle_data, jet_data
    
    def preprocess_particles(self, particle_data: np.ndarray) -> np.ndarray:
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
        
        # Store statistics for denormalization
        self.data_stats = {
            'original_shape': particle_data.shape,
            'n_non_zero_events': len(processed_data),
            'pt_log_transform': True,
            'eta_tanh_transform': True,
            'phi_normalized': True
        }
        
        return processed_data
    
    def encode_events(self, particle_data: np.ndarray) -> np.ndarray:
        """
        Encode events using the autoencoder.
        
        Parameters
        ----------
        particle_data : np.ndarray
            Preprocessed particle data
            
        Returns
        -------
        np.ndarray
            Latent representations of shape (n_events, latent_dim)
        """
        if self.autoencoder is None:
            raise ValueError("No autoencoder provided. Cannot encode events.")
        
        # Ensure data is in correct format
        if particle_data.shape[1:] != (100, 3):
            raise ValueError(f"Expected particle data shape (n_events, 100, 3), got {particle_data.shape}")
        
        # Encode using autoencoder
        latent_representations = self.autoencoder.encoder(particle_data).numpy()
        
        logging.info(f"Encoded {len(particle_data)} events to {latent_representations.shape[1]}D latent space")
        
        return latent_representations
    
    def decode_events(self, latent_data: np.ndarray) -> np.ndarray:
        """
        Decode latent representations back to particle data.
        
        Parameters
        ----------
        latent_data : np.ndarray
            Latent representations
            
        Returns
        -------
        np.ndarray
            Reconstructed particle data
        """
        if self.autoencoder is None:
            raise ValueError("No autoencoder provided. Cannot decode events.")
        
        # Decode using autoencoder
        reconstructed_data = self.autoencoder.decoder(latent_data).numpy()
        
        logging.info(f"Decoded {len(latent_data)} latent representations to particle data")
        
        return reconstructed_data
    
    def denormalize_events(self, processed_data: np.ndarray) -> np.ndarray:
        """
        Reverse preprocessing to get original scale data.
        
        Parameters
        ----------
        processed_data : np.ndarray
            Preprocessed particle data
            
        Returns
        -------
        np.ndarray
            Denormalized particle data
        """
        if self.data_stats is None:
            logging.warning("No preprocessing statistics available. Returning data as-is.")
            return processed_data
        
        denormalized_data = processed_data.copy()
        
        # Reverse phi normalization
        if self.data_stats.get('phi_normalized', False):
            denormalized_data[:, :, 2] = denormalized_data[:, :, 2] * np.pi
        
        # Reverse eta tanh transform
        if self.data_stats.get('eta_tanh_transform', False):
            denormalized_data[:, :, 1] = np.arctanh(denormalized_data[:, :, 1])
        
        # Reverse pT log transform
        if self.data_stats.get('pt_log_transform', False):
            denormalized_data[:, :, 0] = np.expm1(denormalized_data[:, :, 0])
        
        return denormalized_data
    
    def save_processed_data(self, 
                          particle_data: np.ndarray, 
                          latent_data: np.ndarray,
                          jet_data: np.ndarray,
                          output_path: str):
        """
        Save processed data to HDF5 file.
        
        Parameters
        ----------
        particle_data : np.ndarray
            Processed particle data
        latent_data : np.ndarray
            Latent representations
        jet_data : np.ndarray
            Jet data
        output_path : str
            Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('particle_data', data=particle_data)
            f.create_dataset('latent_data', data=latent_data)
            f.create_dataset('jet_data', data=jet_data)
            
            # Save metadata
            f.attrs['n_events'] = len(particle_data)
            f.attrs['latent_dim'] = latent_data.shape[1]
            f.attrs['n_particles_per_event'] = particle_data.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
            
            if self.data_stats:
                for key, value in self.data_stats.items():
                    f.attrs[f'preprocessing_{key}'] = value
        
        logging.info(f"Saved processed data to {output_file}")
    
    def create_autoencoder_training_data(self, 
                                       particle_data: np.ndarray,
                                       train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training/validation split for autoencoder training.
        
        Parameters
        ----------
        particle_data : np.ndarray
            Preprocessed particle data
        train_split : float
            Fraction of data to use for training
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (train_data, val_data)
        """
        n_events = len(particle_data)
        n_train = int(n_events * train_split)
        
        # Shuffle data
        indices = np.random.permutation(n_events)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_data = particle_data[train_indices]
        val_data = particle_data[val_indices]
        
        logging.info(f"Created training split: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data