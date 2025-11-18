# Setup Guide for QAD Package

> **⚠️ Attribution Notice**: This is a setup guide for a fork of the original work by **Vasilis Belis, Kinga Anna Woźniak, Ema Puljak, and collaborators**, published in *Communications Physics* (2024). All core algorithms and scientific contributions are the work of the original authors. This guide documents improved installation and usage procedures.
> 
> 📄 **Original Paper**: [Nature Communications Physics](https://www.nature.com/articles/s42005-024-01811-6)  
> 🔗 **Original Repository**: [vbelis/latent-ad-qml](https://github.com/vbelis/latent-ad-qml)

This guide will help you set up the QAD package for particle physics simulation and quantum machine learning.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Methods

### Method 1: Direct Installation from GitHub

```bash
# Clone this repository (fork with improvements)
git clone https://github.com/rotiyan/QmlAnomaly.git
cd QmlAnomaly

# Install the package
pip install .
```

**Note**: To install from the original repository:
```bash
git clone https://github.com/vbelis/latent-ad-qml.git
cd latent-ad-qml
pip install .
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/rotiyan/QmlAnomaly.git
cd QmlAnomaly

# Install in development mode
pip install -e .
```

### Method 3: Using pip directly

```bash
pip install https://github.com/rotiyan/QmlAnomaly/archive/main.zip
```

## Verification

After installation, verify that everything is working:

```bash
python test_installation.py
```

This will run a comprehensive test suite to check:
- All required dependencies are installed
- Basic functionality works
- Configuration parsing works
- Example files are accessible

## Quick Start

### 1. Generate Dijet Events

```bash
# Generate 100 dijet events
python -m qad.simulation.run_pipeline --process dijet --n-events 100
```

### 2. Generate Z+jets Events

```bash
# Generate 100 Z+jets events
python -m qad.simulation.run_pipeline --process zjets --n-events 100
```

### 3. Use Configuration Files

```bash
# Use a configuration file
python -m qad.simulation.run_pipeline --config examples/simulation/dijet_10_events.yaml
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues

The package will automatically fall back to CPU if GPU is not available. To force CPU usage:

```bash
export CUDA_VISIBLE_DEVICES=""
python -m qad.simulation.run_pipeline --process dijet --n-events 100
```

#### 3. Memory Issues

For large datasets, reduce the number of events or use smaller batch sizes:

```bash
# Generate fewer events
python -m qad.simulation.run_pipeline --process dijet --n-events 50

# Or use configuration file with smaller batch size
```

#### 4. File Permission Issues

Ensure you have write permissions in the output directory:

```bash
# Create output directory with proper permissions
mkdir -p output
chmod 755 output
```

### Debug Mode

Enable verbose logging to see detailed information:

```bash
python -m qad.simulation.run_pipeline --process dijet --n-events 100 --verbose
```

## Dependencies

The package requires several dependencies that will be installed automatically:

### Core Dependencies
- numpy >= 1.21
- scipy >= 1.9
- pandas == 1.4.0
- h5py >= 3.6.0
- pyyaml >= 6.0

### Machine Learning
- scikit-learn == 1.1.1
- tensorflow >= 2.6
- torch >= 1.9.0

### Quantum Computing
- qiskit == 0.36.2
- qibo == 0.1.10

### Visualization
- matplotlib >= 3.5
- mplhep >= 0.3.26

### Physics
- Custom simulation modules (included)

## Environment Setup

### Using Conda (Recommended)

```bash
# Create a new environment
conda create -n qad_env python=3.8
conda activate qad_env

# Install the package
pip install https://github.com/rotiyan/QmlAnomaly/archive/main.zip
```

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv qad_env
source qad_env/bin/activate  # On Windows: qad_env\Scripts\activate

# Install the package
pip install https://github.com/rotiyan/QmlAnomaly/archive/main.zip
```

## Testing the Installation

Run the test suite to verify everything is working:

```bash
python test_installation.py
```

Expected output:
```
============================================================
QAD Package Installation Test
============================================================

Import Test:
----------------------------------------
✓ numpy
✓ h5py
✓ torch
✓ tensorflow
✓ pyyaml
✓ qad
✓ qad.simulation
✓ qad.autoencoder

Basic Functionality:
----------------------------------------
✓ Generated 10 dijet events
  Particle data shape: (10, 100, 3)
  Jet data shape: (10, 10, 4)
✓ Preprocessed data shape: (10, 100, 3)
✓ Created autoencoder model
✓ Encoded to latent space: (10, 6)

Configuration:
----------------------------------------
✓ Configuration parsing
  Process: pp -> jj
  Energy: 13000.0 GeV
  Events: 100

Examples:
----------------------------------------
✓ examples/simulation/run_pytorch_chain.py
✓ examples/simulation/zjets_generator.py
✓ examples/simulation/dijet_10_events.yaml
✓ examples/simulation/zjets_config.yaml
✓ examples/analysis/visualize_results.py
✓ examples/analysis/verify_results.py

============================================================
Test Results Summary:
============================================================
Import Test           : PASS
Basic Functionality   : PASS
Configuration         : PASS
Examples              : PASS

============================================================
🎉 All tests passed! The installation is working correctly.

You can now run the simulation pipeline:
  python -m qad.simulation.run_pipeline --process dijet --n-events 100
============================================================
```

## Next Steps

Once the installation is verified, you can:

1. **Run Examples**: Check the `examples/` directory for complete examples
2. **Read Documentation**: See the main README.md for detailed usage
3. **Explore the API**: Use the Python API for custom applications
4. **Generate Data**: Create your own particle physics datasets
5. **Apply Quantum ML**: Use the compressed representations with quantum algorithms

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Run the test suite to identify specific problems
3. Check the GitHub issues page
4. Contact the maintainers

## Contributing

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
