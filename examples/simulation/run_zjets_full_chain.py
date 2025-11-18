#!/usr/bin/env python3
"""
Full chain script for Z+jets event production with variational autoencoder compressed representation.

This script runs the complete pipeline:
1. Generate 100 Z+jets events using sophisticated physics simulation
2. Process particle data for autoencoder
3. Train autoencoder on the data
4. Encode events to latent space
5. Verify the complete pipeline
6. Compare with dijet results
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our Z+jets generator
from zjets_generator import ZJetsEventGenerator


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
    
    # Log transform pT for better numerical stability
    processed_data[:, :, 0] = np.log1p(processed_data[:, :, 0])
    
    # Normalize eta and phi to [-1, 1] range
    processed_data[:, :, 1] = np.tanh(processed_data[:, :, 1])  # eta
    processed_data[:, :, 2] = processed_data[:, :, 2] / np.pi  # phi
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    return processed_data


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


def train_autoencoder(particle_data, latent_dim=6, epochs=100):
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
    SimpleAutoencoder
        Trained autoencoder model
    """
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
    train_losses = []
    val_losses = []
    
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    logger.info("Autoencoder training completed!")
    logger.info(f"Final training loss: {train_losses[-1]:.6f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.6f}")
    
    return model


def load_dijet_results():
    """Load previous dijet results for comparison."""
    try:
        with h5py.File('output/processed_dijet_events.h5', 'r') as f:
            dijet_particles = f['particle_data'][:]
            dijet_latent = f['latent_data'][:]
            dijet_jets = f['jet_data'][:]
        return dijet_particles, dijet_latent, dijet_jets
    except FileNotFoundError:
        logger.warning("Dijet results not found for comparison")
        return None, None, None


def create_comparison_plots(zjets_data, dijet_data, output_dir):
    """Create comparison plots between Z+jets and dijet events."""
    logger.info("Creating comparison plots...")
    
    zjets_particles, zjets_latent, zjets_jets = zjets_data
    dijet_particles, dijet_latent, dijet_jets = dijet_data
    
    if dijet_particles is None:
        logger.warning("No dijet data available for comparison")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Z+jets vs Dijet Events Comparison', fontsize=16)
    
    # 1. Particle pT distribution comparison
    ax1 = axes[0, 0]
    zjets_pt = zjets_particles[:, :, 0].flatten()
    zjets_pt = zjets_pt[zjets_pt > 0]
    dijet_pt = dijet_particles[:, :, 0].flatten()
    dijet_pt = dijet_pt[dijet_pt > 0]
    
    ax1.hist(zjets_pt, bins=50, alpha=0.7, label='Z+jets', color='blue', density=True)
    ax1.hist(dijet_pt, bins=50, alpha=0.7, label='Dijet', color='red', density=True)
    ax1.set_xlabel('pT (GeV)')
    ax1.set_ylabel('Density')
    ax1.set_title('Particle pT Distribution')
    ax1.set_yscale('log')
    ax1.legend()
    
    # 2. Jet pT distribution comparison
    ax2 = axes[0, 1]
    zjets_jet_pt = zjets_jets[:, :, 0].flatten()
    zjets_jet_pt = zjets_jet_pt[zjets_jet_pt > 0]
    dijet_jet_pt = dijet_jets[:, :, 0].flatten()
    dijet_jet_pt = dijet_jet_pt[dijet_jet_pt > 0]
    
    ax2.hist(zjets_jet_pt, bins=30, alpha=0.7, label='Z+jets', color='blue', density=True)
    ax2.hist(dijet_jet_pt, bins=30, alpha=0.7, label='Dijet', color='red', density=True)
    ax2.set_xlabel('Jet pT (GeV)')
    ax2.set_ylabel('Density')
    ax2.set_title('Jet pT Distribution')
    ax2.legend()
    
    # 3. Latent space comparison (first 2 dimensions)
    ax3 = axes[0, 2]
    ax3.scatter(zjets_latent[:, 0], zjets_latent[:, 1], alpha=0.6, label='Z+jets', s=20)
    ax3.scatter(dijet_latent[:, 0], dijet_latent[:, 1], alpha=0.6, label='Dijet', s=20)
    ax3.set_xlabel('Latent Dimension 1')
    ax3.set_ylabel('Latent Dimension 2')
    ax3.set_title('Latent Space (Dim 1 vs Dim 2)')
    ax3.legend()
    
    # 4. Latent space statistics comparison
    ax4 = axes[0, 3]
    zjets_means = np.mean(zjets_latent, axis=0)
    dijet_means = np.mean(dijet_latent, axis=0)
    dims = range(len(zjets_means))
    
    x = np.arange(len(dims))
    width = 0.35
    
    ax4.bar(x - width/2, zjets_means, width, label='Z+jets', alpha=0.7)
    ax4.bar(x + width/2, dijet_means, width, label='Dijet', alpha=0.7)
    ax4.set_xlabel('Latent Dimension')
    ax4.set_ylabel('Mean Value')
    ax4.set_title('Latent Space Statistics')
    ax4.set_xticks(x)
    ax4.legend()
    
    # 5. Event multiplicity comparison
    ax5 = axes[1, 0]
    zjets_multiplicity = np.sum(zjets_particles[:, :, 0] > 0, axis=1)
    dijet_multiplicity = np.sum(dijet_particles[:, :, 0] > 0, axis=1)
    
    ax5.hist(zjets_multiplicity, bins=20, alpha=0.7, label='Z+jets', color='blue', density=True)
    ax5.hist(dijet_multiplicity, bins=20, alpha=0.7, label='Dijet', color='red', density=True)
    ax5.set_xlabel('Particles per Event')
    ax5.set_ylabel('Density')
    ax5.set_title('Event Multiplicity')
    ax5.legend()
    
    # 6. Jet multiplicity comparison
    ax6 = axes[1, 1]
    zjets_jet_multiplicity = np.sum(zjets_jets[:, :, 0] > 0, axis=1)
    dijet_jet_multiplicity = np.sum(dijet_jets[:, :, 0] > 0, axis=1)
    
    ax6.hist(zjets_jet_multiplicity, bins=10, alpha=0.7, label='Z+jets', color='blue', density=True)
    ax6.hist(dijet_jet_multiplicity, bins=10, alpha=0.7, label='Dijet', color='red', density=True)
    ax6.set_xlabel('Jets per Event')
    ax6.set_ylabel('Density')
    ax6.set_title('Jet Multiplicity')
    ax6.legend()
    
    # 7. Latent space distance analysis
    ax7 = axes[1, 2]
    # Calculate pairwise distances in latent space
    from scipy.spatial.distance import pdist
    zjets_distances = pdist(zjets_latent)
    dijet_distances = pdist(dijet_latent)
    
    ax7.hist(zjets_distances, bins=30, alpha=0.7, label='Z+jets', color='blue', density=True)
    ax7.hist(dijet_distances, bins=30, alpha=0.7, label='Dijet', color='red', density=True)
    ax7.set_xlabel('Pairwise Distance in Latent Space')
    ax7.set_ylabel('Density')
    ax7.set_title('Latent Space Clustering')
    ax7.legend()
    
    # 8. Summary statistics
    ax8 = axes[1, 3]
    ax8.axis('off')
    
    # Calculate summary statistics
    stats_text = f"""
    Z+jets Events: {len(zjets_particles)}
    Dijet Events: {len(dijet_particles)}
    
    Z+jets Mean pT: {np.mean(zjets_pt):.2f} GeV
    Dijet Mean pT: {np.mean(dijet_pt):.2f} GeV
    
    Z+jets Mean Jet pT: {np.mean(zjets_jet_pt):.2f} GeV
    Dijet Mean Jet pT: {np.mean(dijet_jet_pt):.2f} GeV
    
    Z+jets Mean Multiplicity: {np.mean(zjets_multiplicity):.1f}
    Dijet Mean Multiplicity: {np.mean(dijet_multiplicity):.1f}
    
    Z+jets Mean Jet Multiplicity: {np.mean(zjets_jet_multiplicity):.1f}
    Dijet Mean Jet Multiplicity: {np.mean(dijet_jet_multiplicity):.1f}
    """
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'zjets_vs_dijet_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the complete Z+jets pipeline."""
    logger.info("Starting full chain: Z+jets event production with VAE compression")
    logger.info("=" * 80)
    
    try:
        # 1. Generate Z+jets events
        logger.info("\n1. Generating Z+jets events...")
        generator = ZJetsEventGenerator("zjets_config.yaml")
        particle_data, jet_data = generator.generate_events()
        
        # Print Z+jets statistics
        stats = generator.get_statistics()
        logger.info(f"Z+jets generation completed!")
        logger.info(f"Generated events: {stats['n_generated']}")
        logger.info(f"Accepted events: {stats['n_accepted']}")
        logger.info(f"Z decay statistics: {stats['z_decay_stats']}")
        
        # 2. Process data for autoencoder
        logger.info("\n2. Processing data for autoencoder...")
        processed_particles = preprocess_particles(particle_data)
        
        # 3. Train autoencoder
        logger.info("\n3. Training autoencoder...")
        autoencoder = train_autoencoder(processed_particles, latent_dim=6, epochs=100)
        
        # 4. Encode events to latent space
        logger.info("\n4. Encoding events to latent space...")
        with torch.no_grad():
            data_tensor = torch.FloatTensor(processed_particles.reshape(processed_particles.shape[0], -1))
            latent_data = autoencoder.encode(data_tensor).numpy()
        
        logger.info(f"   Latent data shape: {latent_data.shape}")
        logger.info(f"   Latent dimension: {latent_data.shape[1]}")
        
        # 5. Verify reconstruction quality
        logger.info("\n5. Verifying reconstruction quality...")
        with torch.no_grad():
            reconstructed_data = autoencoder(data_tensor).numpy()
            reconstructed_data = reconstructed_data.reshape(processed_particles.shape)
        
        reconstruction_error = np.mean(np.square(processed_particles - reconstructed_data))
        logger.info(f"   Reconstruction MSE: {reconstruction_error:.6f}")
        
        # 6. Save results
        logger.info("\n6. Saving results...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw events
        raw_output_path = output_dir / "raw_zjets_events.h5"
        generator.save_events(particle_data, jet_data, str(raw_output_path))
        
        # Save processed data
        processed_output_path = output_dir / "processed_zjets_events.h5"
        with h5py.File(processed_output_path, 'w') as f:
            f.create_dataset('particle_data', data=processed_particles)
            f.create_dataset('latent_data', data=latent_data)
            f.create_dataset('jet_data', data=jet_data)
            f.attrs['n_events'] = len(processed_particles)
            f.attrs['latent_dim'] = latent_data.shape[1]
            f.attrs['n_particles_per_event'] = processed_particles.shape[1]
            f.attrs['n_jets_per_event'] = jet_data.shape[1]
            f.attrs['z_decay_stats'] = str(stats['z_decay_stats'])
        
        # Save autoencoder
        autoencoder_path = output_dir / "zjets_autoencoder.pth"
        torch.save(autoencoder.state_dict(), autoencoder_path)
        
        # 7. Load dijet results for comparison
        logger.info("\n7. Loading dijet results for comparison...")
        dijet_particles, dijet_latent, dijet_jets = load_dijet_results()
        
        # 8. Create comparison plots
        logger.info("\n8. Creating comparison plots...")
        create_comparison_plots(
            (processed_particles, latent_data, jet_data),
            (dijet_particles, dijet_latent, dijet_jets),
            output_dir
        )
        
        # 9. Display final statistics
        logger.info("\n9. Final Statistics:")
        logger.info("=" * 40)
        logger.info(f"Z+jets Events: {len(particle_data)}")
        logger.info(f"Particle data shape: {particle_data.shape}")
        logger.info(f"Jet data shape: {jet_data.shape}")
        logger.info(f"Processed data shape: {processed_particles.shape}")
        logger.info(f"Latent data shape: {latent_data.shape}")
        logger.info(f"Compression ratio: {processed_particles.size / latent_data.size:.1f}x")
        logger.info(f"Reconstruction MSE: {reconstruction_error:.6f}")
        logger.info(f"Z decay statistics: {stats['z_decay_stats']}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  Raw Z+jets events: {raw_output_path}")
        logger.info(f"  Processed Z+jets events: {processed_output_path}")
        logger.info(f"  Z+jets autoencoder: {autoencoder_path}")
        logger.info(f"  Comparison plots: {output_dir / 'zjets_vs_dijet_comparison.png'}")
        
        # 10. Show sample latent representations
        logger.info(f"\n10. Sample Z+jets latent representations (first 3 events):")
        for i in range(min(3, len(latent_data))):
            logger.info(f"   Event {i+1}: {latent_data[i]}")
        
        logger.info("\n✅ Z+jets full chain completed successfully!")
        logger.info("Pipeline: Z+jets Generation → Data Processing → Autoencoder Training → Latent Encoding → Comparison")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())