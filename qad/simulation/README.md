# Particle Physics Simulation Module

This module provides a complete pipeline for generating particle physics events using **Pythia8** and compressing them using variational autoencoders for quantum machine learning applications.

## Features

- **Pythia8 Integration**: Full integration with Pythia8 event generator
- **Multiple Physics Processes**: Dijet production, Z+jets, and custom processes
- **Realistic Event Generation**: Production-grade physics simulation with proper kinematics
- **Autoencoder Integration**: Seamless compression to latent space
- **Quantum ML Ready**: Compressed representations compatible with quantum anomaly detection algorithms
- **Easy Configuration**: YAML-based configuration system
- **Command Line Interface**: Simple CLI for running simulations

## Requirements

- Pythia8 with Python bindings (installed via conda)
- TensorFlow or PyTorch for autoencoder
- FastJet (optional, for jet reconstruction)

## Quick Start

### 1. Basic Usage

Generate dijet events and compress them to latent space:

```bash
python -m qad.simulation.run_pipeline --process dijet --n-events 100
```

Generate Z+jets events:

```bash
python -m qad.simulation.run_pipeline --process zjets --n-events 100
```

### 2. Using Configuration Files

Create a YAML configuration file:

```yaml
# dijet_config.yaml
physics_process:
  process: "pp -> jj"
  energy: 13000.0
  parameters:
    "PhaseSpace:pTHatMin": 50.0

simulation:
  n_events: 1000
  output_file: "data/dijet_events.h5"
  random_seed: 12345

jets:
  algorithm: "antikt"
  r_parameter: 0.4
  pt_min: 20.0
  eta_max: 2.5
```

Run with configuration:

```bash
python -m qad.simulation.run_pipeline --config dijet_config.yaml
```

### 3. Python API

```python
from qad.simulation import PythiaEventGenerator, SimulationConfig, EventDataProcessor
from qad.autoencoder.autoencoder import ParticleAutoencoder

# Load configuration
config = SimulationConfig("config.yaml")

# Generate events with Pythia8
generator = PythiaEventGenerator(config)
particle_data, jet_data = generator.generate_events()

# Process data
processor = EventDataProcessor()
processed_particles = processor.preprocess_particles(particle_data)

# Train autoencoder
autoencoder = ParticleAutoencoder()
# ... training code ...

# Encode to latent space
latent_data = processor.encode_events(processed_particles)
```

## Supported Physics Processes

### Dijet Production (pp → jj)
- **Description**: Proton-proton collisions producing two jets
- **Features**: Realistic pT distributions, jet clustering
- **Usage**: `--process dijet`

### Z+jets Production (pp → Z+jets)
- **Description**: Z boson production with associated jets
- **Features**: Z → e⁺e⁻ and Z → μ⁺μ⁻ decays, proper kinematics
- **Usage**: `--process zjets`

### Custom Processes
- **Description**: Define custom physics processes via YAML
- **Features**: Flexible parameter configuration
- **Usage**: `--config custom_config.yaml`

## Data Format

### Input Data
- **Particle Data**: Shape `(n_events, 100, 3)` with `[pT, η, φ]` coordinates
- **Jet Data**: Shape `(n_events, 10, 4)` with `[pT, η, φ, mass]` coordinates

### Preprocessing
- **pT**: Log transformation for numerical stability
- **η**: Tanh normalization to `[-1, 1]`
- **φ**: Division by π to normalize to `[-1, 1]`

### Latent Space
- **Shape**: `(n_events, latent_dim)` where `latent_dim=6` by default
- **Compression**: 50x reduction (30000 → 600 values)
- **Usage**: Ready for quantum ML algorithms

## Output Files

The pipeline generates several output files:

- **`raw_events.h5`**: Raw particle and jet data
- **`processed_events.h5`**: Preprocessed data with latent representations
- **`autoencoder.pth`**: Trained PyTorch model weights
- **`simulation.log`**: Detailed execution log

## Configuration Options

### Physics Process
```yaml
physics_process:
  process: "pp -> jj"  # or "pp -> Z+jets"
  energy: 13000.0      # Center-of-mass energy in GeV
  parameters:          # Additional Pythia8 parameters
    "PhaseSpace:pTHatMin": 50.0
```

### Simulation Settings
```yaml
simulation:
  n_events: 1000       # Number of events to generate
  output_file: "data/events.h5"
  random_seed: 12345   # For reproducibility
  debug_level: 1       # Pythia8 debug level
```

### Jet Reconstruction
```yaml
jets:
  algorithm: "antikt"  # Jet algorithm
  r_parameter: 0.4     # Jet radius
  pt_min: 20.0         # Minimum jet pT in GeV
  eta_max: 2.5         # Maximum |η| for jets
```

### Z Boson Settings (for Z+jets)
```yaml
z_boson:
  mass: 91.1876        # Z boson mass in GeV
  width: 2.4952        # Z boson width in GeV
  decay_modes: ["ee", "mumu"]  # Allowed decay modes
```

## Command Line Options

```bash
python -m qad.simulation.run_pipeline [OPTIONS]

Options:
  --process {dijet,zjets}    Physics process to simulate
  --n-events INTEGER         Number of events to generate (default: 100)
  --config PATH              Path to YAML configuration file
  --output-dir PATH          Output directory (default: output)
  --latent-dim INTEGER       Latent dimension (default: 6)
  --epochs INTEGER           Training epochs (default: 100)
  --verbose, -v              Verbose logging
  --help                     Show help message
```

## Examples

### Example 1: Quick Dijet Simulation
```bash
# Generate 50 dijet events
python -m qad.simulation.run_pipeline --process dijet --n-events 50 --output-dir dijet_results
```

### Example 2: Z+jets with Custom Settings
```bash
# Generate 200 Z+jets events with 8D latent space
python -m qad.simulation.run_pipeline --process zjets --n-events 200 --latent-dim 8 --epochs 150
```

### Example 3: Using Configuration File
```bash
# Use custom configuration
python -m qad.simulation.run_pipeline --config examples/simulation/dijet_10_events.yaml --verbose
```

## Integration with Quantum ML

The generated latent representations are ready for quantum machine learning:

```python
import h5py
import numpy as np

# Load latent data
with h5py.File('output/processed_events.h5', 'r') as f:
    latent_data = f['latent_data'][:]

# Use with quantum algorithms
# latent_data shape: (n_events, latent_dim)
# Each row is a 6D latent representation ready for quantum circuits
```

## Installation

### Install Pythia8

```bash
# Activate your conda environment
conda activate qad_env

# Install Pythia8 from conda-forge
conda install -c conda-forge pythia8

# Optional: Install FastJet for better jet reconstruction
conda install -c conda-forge fastjet
```

### Verify Installation

```bash
python -c "import pythia8; print('Pythia8 installed successfully')"
```

## Troubleshooting

### Common Issues

1. **Pythia8 Not Found**: Install with `conda install -c conda-forge pythia8`
2. **Python Version Conflicts**: Pythia8 requires Python 3.8-3.11
3. **Memory Issues**: Reduce `n_events` for large datasets
4. **CUDA Errors**: The code will fall back to CPU if GPU is not available
5. **File Not Found**: Check that output directory exists and is writable

### Debug Mode

Enable verbose logging for debugging:

```bash
python -m qad.simulation.run_pipeline --process dijet --verbose
```

## Advanced Usage

### Custom Event Generators

You can create custom event generators by extending the base classes:

```python
from qad.simulation import EventDataProcessor

class CustomEventGenerator:
    def generate_events(self, n_events):
        # Your custom event generation logic
        return particle_data, jet_data

# Use with the pipeline
generator = CustomEventGenerator()
particle_data, jet_data = generator.generate_events(100)
```

### Batch Processing

For large datasets, process events in batches:

```python
def process_large_dataset(n_events, batch_size=1000):
    for i in range(0, n_events, batch_size):
        batch_events = min(batch_size, n_events - i)
        # Process batch
        particle_data, jet_data = generate_events(batch_events)
        # Save batch results
```

## Performance Tips

1. **Use GPU**: Install PyTorch with CUDA support for faster training
2. **Batch Size**: Adjust batch size based on available memory
3. **Epochs**: More epochs may improve reconstruction quality
4. **Latent Dimension**: Higher dimensions may capture more details

## Citation

If you use this module in your research, please cite:

```bibtex
@article{Belis_2024,
    title={Quantum anomaly detection in the latent space of proton collision events at the LHC},
    volume={7},
    journal={Communications Physics},
    author={Belis, Vasilis and Woźniak, Kinga Anna and Puljak, Ema and others},
    year={2024}
}
```

## License

This module is part of the `qad` package and is licensed under the MIT License.