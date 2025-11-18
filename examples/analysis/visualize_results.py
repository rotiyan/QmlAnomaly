#!/usr/bin/env python3
"""
Visualization script for the dijet event production and VAE compression results.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load the generated results."""
    # Load raw events
    with h5py.File('output/raw_dijet_events.h5', 'r') as f:
        particle_data = f['particle_data'][:]
        jet_data = f['jet_data'][:]
        raw_attrs = dict(f.attrs)
    
    # Load processed events
    with h5py.File('output/processed_dijet_events.h5', 'r') as f:
        processed_particles = f['particle_data'][:]
        latent_data = f['latent_data'][:]
        processed_attrs = dict(f.attrs)
    
    return particle_data, jet_data, processed_particles, latent_data, raw_attrs, processed_attrs

def plot_results():
    """Create visualization plots."""
    # Load data
    particle_data, jet_data, processed_particles, latent_data, raw_attrs, processed_attrs = load_results()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Proton-Proton Collision to Dijet Events with VAE Compression', fontsize=16)
    
    # 1. Raw particle pT distribution
    ax1 = axes[0, 0]
    pT_values = particle_data[:, :, 0].flatten()
    pT_values = pT_values[pT_values > 0]  # Remove zero padding
    ax1.hist(pT_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('pT (GeV)')
    ax1.set_ylabel('Count')
    ax1.set_title('Raw Particle pT Distribution')
    ax1.set_yscale('log')
    
    # 2. Jet pT distribution
    ax2 = axes[0, 1]
    jet_pT_values = jet_data[:, :, 0].flatten()
    jet_pT_values = jet_pT_values[jet_pT_values > 0]  # Remove zero padding
    ax2.hist(jet_pT_values, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Jet pT (GeV)')
    ax2.set_ylabel('Count')
    ax2.set_title('Jet pT Distribution')
    
    # 3. Processed particle data (first event)
    ax3 = axes[0, 2]
    event_0 = processed_particles[0]
    pT_proc = event_0[:, 0]
    eta_proc = event_0[:, 1]
    phi_proc = event_0[:, 2]
    
    # Only plot non-zero particles
    mask = pT_proc > 0
    scatter = ax3.scatter(eta_proc[mask], phi_proc[mask], c=pT_proc[mask], 
                         cmap='viridis', alpha=0.7, s=20)
    ax3.set_xlabel('η (processed)')
    ax3.set_ylabel('φ (processed)')
    ax3.set_title('Event 0: Processed Particles')
    plt.colorbar(scatter, ax=ax3, label='log(1+pT)')
    
    # 4. Latent space visualization (first 2 dimensions)
    ax4 = axes[1, 0]
    ax4.scatter(latent_data[:, 0], latent_data[:, 1], c=range(len(latent_data)), 
               cmap='tab10', s=100, alpha=0.8)
    ax4.set_xlabel('Latent Dimension 1')
    ax4.set_ylabel('Latent Dimension 2')
    ax4.set_title('Latent Space (Dim 1 vs Dim 2)')
    
    # 5. Latent space statistics
    ax5 = axes[1, 1]
    latent_means = np.mean(latent_data, axis=0)
    latent_stds = np.std(latent_data, axis=0)
    dims = range(len(latent_means))
    
    ax5.bar(dims, latent_means, yerr=latent_stds, capsize=5, alpha=0.7, color='green')
    ax5.set_xlabel('Latent Dimension')
    ax5.set_ylabel('Mean Value')
    ax5.set_title('Latent Space Statistics')
    ax5.set_xticks(dims)
    
    # 6. Compression ratio visualization
    ax6 = axes[1, 2]
    original_size = processed_particles.size
    compressed_size = latent_data.size
    compression_ratio = original_size / compressed_size
    
    sizes = [original_size, compressed_size]
    labels = ['Original\n(3000 values)', f'Compressed\n({compressed_size} values)']
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax6.bar(labels, sizes, color=colors, alpha=0.7)
    ax6.set_ylabel('Number of Values')
    ax6.set_title(f'Compression: {compression_ratio:.1f}x')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('output/dijet_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DIJET EVENT PRODUCTION AND VAE COMPRESSION SUMMARY")
    print("="*60)
    print(f"Generated Events: {raw_attrs['n_events']}")
    print(f"Process: {raw_attrs['process']}")
    print(f"Energy: {raw_attrs['energy']} GeV")
    print(f"Particle Data Shape: {particle_data.shape}")
    print(f"Jet Data Shape: {jet_data.shape}")
    print(f"Processed Data Shape: {processed_particles.shape}")
    print(f"Latent Data Shape: {latent_data.shape}")
    print(f"Compression Ratio: {compression_ratio:.1f}x")
    print(f"Latent Dimensions: {processed_attrs['latent_dim']}")
    print("\nLatent Space Statistics:")
    print(f"  Mean: {np.mean(latent_data, axis=0)}")
    print(f"  Std:  {np.std(latent_data, axis=0)}")
    print(f"  Min:  {np.min(latent_data, axis=0)}")
    print(f"  Max:  {np.max(latent_data, axis=0)}")
    print("\nParticle Statistics:")
    print(f"  Mean pT: {np.mean(pT_values):.2f} GeV")
    print(f"  Mean Jet pT: {np.mean(jet_pT_values):.2f} GeV")
    print(f"  Total Particles: {len(pT_values)}")
    print(f"  Total Jets: {len(jet_pT_values)}")
    print("="*60)

if __name__ == "__main__":
    plot_results()