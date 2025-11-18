# Latent Adaptive Quantum Machine Learning (QAD)

> **⚠️ Attribution Notice**: This is a fork with enhancements of the original work by **Vasilis Belis, Kinga Anna Woźniak, Ema Puljak, and collaborators**. The original research and codebase were published in *Communications Physics* (2024). This repository contains improvements to code organization, documentation, and usability while maintaining full attribution to the original authors.
>
> 📄 **Original Paper**: [Quantum anomaly detection in the latent space of proton collision events at the LHC](https://www.nature.com/articles/s42005-024-01811-6)  
> 🔗 **Original Repository**: [vbelis/latent-ad-qml](https://github.com/vbelis/latent-ad-qml)

## Overview

This package implements a complete pipeline for:
1. **Particle Physics Simulation**: Generate realistic proton-proton collision events
2. **Data Compression**: Use variational autoencoders to compress events to latent space
3. **Quantum ML**: Apply quantum machine learning algorithms to detect anomalies
4. **Analysis Tools**: Comprehensive analysis and visualization tools

The figure below shows the quantum-classical pipeline for detecting anomalous new-physics events in proton collisions at the LHC.

![Pipeline](docs/Pipeline_QML.png)

## Key Features

- **Multiple Physics Processes**: Dijet production, Z+jets, top quark pairs, and more
- **Realistic Event Generation**: Sophisticated physics simulation with proper kinematics
- **Autoencoder Compression**: 50x compression while preserving physics characteristics
- **Quantum ML Integration**: Compressed representations ready for quantum algorithms
- **Easy-to-Use Interface**: Simple command-line tools and Python API
- **Comprehensive Documentation**: Detailed guides and examples

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Standard Installation

```bash
# Clone this repository
git clone https://github.com/rotiyan/QmlAnomaly.git
cd QmlAnomaly

# Install the package
pip install .
```

**Note**: To access the original repository, use:
```bash
git clone https://github.com/vbelis/latent-ad-qml.git
```

### Development Install

```bash
# Clone and install in development mode
git clone https://github.com/rotiyan/QmlAnomaly.git
cd QmlAnomaly
pip install -e .
```

### Dependencies

The package requires several dependencies that will be installed automatically:

- **Core**: numpy, scipy, pandas, h5py, pyyaml
- **ML**: scikit-learn, tensorflow, torch
- **Quantum**: qiskit, qibo
- **Visualization**: matplotlib, mplhep
- **Physics**: Custom simulation modules

## Quick Start

### 1. Generate and Compress Events

Generate dijet events and compress them to latent space:

```bash
# Generate 100 dijet events
python -m qad.simulation.run_pipeline --process dijet --n-events 100

# Generate 100 Z+jets events
python -m qad.simulation.run_pipeline --process zjets --n-events 100
```

### 2. Use Configuration Files

Create a configuration file for custom settings:

```yaml
# my_config.yaml
physics_process:
  process: "pp -> jj"
  energy: 13000.0
  parameters:
    "PhaseSpace:pTHatMin": 50.0

simulation:
  n_events: 1000
  output_file: "data/my_events.h5"
  random_seed: 12345

jets:
  algorithm: "antikt"
  r_parameter: 0.4
  pt_min: 20.0
  eta_max: 2.5
```

Run with configuration:

```bash
python -m qad.simulation.run_pipeline --config my_config.yaml
```

### 3. Python API

```python
from qad.simulation import SimulationConfig, EventDataProcessor
import numpy as np

# Generate events
particle_data, jet_data = generate_mock_dijet_events(100)

# Process data
processor = EventDataProcessor()
processed_particles = processor.preprocess_particles(particle_data)

# Train autoencoder and encode to latent space
# ... (see examples for complete code)

# Load latent data for quantum ML
with h5py.File('output/processed_events.h5', 'r') as f:
    latent_data = f['latent_data'][:]
    # latent_data shape: (n_events, 6)
    # Ready for quantum algorithms!
```

## Supported Physics Processes

| Process | Description | Command |
|---------|-------------|---------|
| **Dijet** | pp → jj | `--process dijet` |
| **Z+jets** | pp → Z+jets | `--process zjets` |
| **Custom** | User-defined | `--config custom.yaml` |

## Data Format

### Input Data
- **Particle Data**: `(n_events, 100, 3)` with `[pT, η, φ]` coordinates
- **Jet Data**: `(n_events, 10, 4)` with `[pT, η, φ, mass]` coordinates

### Latent Space
- **Shape**: `(n_events, latent_dim)` where `latent_dim=6` by default
- **Compression**: 50x reduction (30000 → 600 values)
- **Usage**: Ready for quantum ML algorithms

## Examples

### Example 1: Basic Dijet Simulation

```bash
# Generate 100 dijet events
python -m qad.simulation.run_pipeline --process dijet --n-events 100 --output-dir dijet_results
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

## Output Files

The pipeline generates several output files:

- **`raw_events.h5`**: Raw particle and jet data
- **`processed_events.h5`**: Preprocessed data with latent representations
- **`autoencoder.pth`**: Trained PyTorch model weights
- **`simulation.log`**: Detailed execution log

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

## Documentation

- **API Reference**: [Read the Docs](https://latent-ad-qml.readthedocs.io/en/latest/)
- **Examples**: See `examples/` directory
- **Simulation Module**: See `qad/simulation/README.md`

## Examples Directory

The `examples/` directory contains:

- **`simulation/`**: Complete simulation examples
  - `run_pytorch_chain.py`: PyTorch-based pipeline
  - `zjets_generator.py`: Z+jets event generator
  - `dijet_10_events.yaml`: Dijet configuration
  - `zjets_config.yaml`: Z+jets configuration

- **`analysis/`**: Analysis and visualization tools
  - `visualize_results.py`: Result visualization
  - `verify_results.py`: Result verification
  - `PIPELINE_SUMMARY.md`: Detailed analysis summary

## Performance Tips

1. **Use GPU**: Install PyTorch with CUDA support for faster training
2. **Batch Size**: Adjust batch size based on available memory
3. **Epochs**: More epochs may improve reconstruction quality
4. **Latent Dimension**: Higher dimensions may capture more details

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `n_events` for large datasets
3. **CUDA Errors**: The code will fall back to CPU if GPU is not available
4. **File Not Found**: Check that output directory exists and is writable

### Debug Mode

Enable verbose logging for debugging:

```bash
python -m qad.simulation.run_pipeline --process dijet --verbose
```

## Citation

**Important**: If you use this package in your research, please cite the **original authors' work**:

```bibtex
@article{Belis_2024,
    title={Quantum anomaly detection in the latent space of proton collision events at the LHC},
    volume={7},
    journal={Communications Physics},
    author={Belis, Vasilis and Woźniak, Kinga Anna and Puljak, Ema and others},
    year={2024},
    doi={10.1038/s42005-024-01811-6},
    url={https://www.nature.com/articles/s42005-024-01811-6}
}
```

**All scientific credit belongs to the original authors.** This repository contains organizational and documentation improvements to make the codebase more accessible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

**Original Research**: The core research and algorithms in this package were developed by Vasilis Belis, Kinga Anna Woźniak, Ema Puljak, and collaborators. Their work was supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme.

**This Fork**: This repository contains organizational improvements, enhanced documentation, and usability enhancements to make the original codebase more accessible to users and researchers.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
