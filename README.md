# Quantum Anomaly Detection in the Latent Space of Proton Collision Events

> **⚠️ Attribution Notice**: This is a fork with enhancements of the original work by **Vasilis Belis, Kinga Anna Woźniak, Ema Puljak, and collaborators**. The original research and codebase were published in *Communications Physics* (2024).
>
> 📄 **Original Paper**: [Quantum anomaly detection in the latent space of proton collision events at the LHC](https://www.nature.com/articles/s42005-024-01811-6)  
> 🔗 **Original Repository**: [vbelis/latent-ad-qml](https://github.com/vbelis/latent-ad-qml)

## Overview

This repository implements a complete pipeline for quantum machine learning-based anomaly detection in high-energy physics:

1. **Event Generation**: Use Pythia8 to generate realistic proton-proton collision events
2. **Compression**: Train autoencoders to compress events to 6D latent space (50x compression)
3. **Quantum ML**: Apply quantum kernel methods for anomaly detection
4. **Analysis**: Comprehensive visualization and statistical tools

![Pipeline](docs/Pipeline_QML.png)

## Key Features

- ✅ **Real Pythia8 Integration**: Generate physics events at 13 TeV
- ✅ **Autoencoder Compression**: 50x compression while preserving physics
- ✅ **Quantum Algorithms**: One-class QSVM, quantum k-medians, quantum k-means
- ✅ **Easy Installation**: Single command conda environment setup
- ✅ **Production Ready**: Tested and documented pipeline

## Installation

### Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/rotiyan/QmlAnomaly.git
cd QmlAnomaly

# Create environment from file (includes Pythia8)
conda env create -f environment.yaml

# Activate environment
conda activate qad_env

# Verify installation
python -c "import pythia8; print('✓ Pythia8 installed')"
python -c "import qiskit; print('✓ Qiskit installed')"
```

The `environment.yaml` includes all dependencies:
- **Pythia8 8.311** (event generation)
- **TensorFlow & PyTorch** (autoencoders)
- **Qiskit 0.36.2** (quantum computing)
- **Scientific stack** (numpy, scipy, pandas, h5py)

### Alternative: pip install

If you don't need Pythia8:

```bash
pip install -e .
```

## Complete Pipeline: Event Generation → Quantum Anomaly Detection

### Step 1: Generate Signal Events

```bash
conda activate qad_env

# Generate 1000 QCD dijet events (signal)
python generate_and_compress_events.py \
    --process dijet \
    --n-events 1000 \
    --latent-dim 6 \
    --epochs 100 \
    --output-dir output/signal
```

**Output:**
- `output/signal/raw_events.h5` - Raw Pythia8 data
- `output/signal/events_for_quantum_ml.h5` - **Compressed events for quantum algorithms**
- `output/signal/autoencoder_weights.h5` - Trained model

### Step 2: Generate Background Events

Create `background_config.yaml`:
```yaml
physics_process:
  process: "pp -> jj"
  energy: 13000.0
  parameters:
    "PhaseSpace:pTHatMin": 100.0  # Different pT range
    "PhaseSpace:pTHatMax": 1000.0

simulation:
  n_events: 1000
  random_seed: 54321

jets:
  algorithm: "antikt"
  r_parameter: 0.4
  pt_min: 20.0
  eta_max: 2.5
```

```bash
python generate_and_compress_events.py \
    --config background_config.yaml \
    --output-dir output/background
```

### Step 3: Train Quantum Anomaly Detector

```bash
# Train one-class quantum SVM
python scripts/kernel_machines/train_one_class_qsvm.py \
    --train-file output/signal/events_for_quantum_ml.h5 \
    --test-file output/background/events_for_quantum_ml.h5 \
    --nqubits 6 \
    --feature-map ZZFeatureMap \
    --reps 2 \
    --backend qasm_simulator \
    --nu-param 0.05
```

### Step 4: Visualize Results

```bash
python examples/analysis/visualize_results.py \
    --signal-file output/signal/events_for_quantum_ml.h5 \
    --background-file output/background/events_for_quantum_ml.h5
```

## Configuration Files

Example configuration for dijet production:

```yaml
# dijet_config.yaml
physics_process:
  process: "pp -> jj"
  energy: 13000.0
  parameters:
    "PhaseSpace:pTHatMin": 50.0
    "PhaseSpace:pTHatMax": 2000.0
    "PartonLevel:ISR": "on"
    "PartonLevel:FSR": "on"
    "PartonLevel:MPI": "on"

simulation:
  n_events: 1000
  output_file: "output/raw_events.h5"
  random_seed: 12345
  debug_level: 1

jets:
  algorithm: "antikt"
  r_parameter: 0.4
  pt_min: 20.0
  eta_max: 2.5
```

Run with:
```bash
python generate_and_compress_events.py --config dijet_config.yaml
```

## Python API

```python
from qad.simulation import PythiaEventGenerator, SimulationConfig
from qad.autoencoder.autoencoder import ParticleAutoencoder

# Generate events
config = SimulationConfig("dijet_config.yaml")
generator = PythiaEventGenerator(config)
particle_data, jet_data = generator.generate_events()

# Train autoencoder
autoencoder = ParticleAutoencoder(input_shape=(100, 3), latent_dim=6)
# ... training code ...

# Encode to latent space
latent_data = autoencoder.encoder.predict(processed_particles)
```

## Repository Structure

```
latent-ad-qml/
├── environment.yaml              # Conda environment with all dependencies
├── generate_and_compress_events.py  # Main pipeline script
├── qad/
│   ├── simulation/
│   │   ├── pythia_generator.py   # Pythia8 integration
│   │   ├── config_parser.py      # Configuration handling
│   │   └── data_processor.py     # Data preprocessing
│   ├── autoencoder/               # Compression models
│   └── algorithms/
│       ├── kernel_machines/       # Quantum SVM
│       ├── kmeans/                # Quantum k-means
│       └── kmedians/              # Quantum k-medians
├── scripts/                       # Training scripts
│   ├── kernel_machines/
│   ├── kmeans/
│   └── kmedians/
└── examples/
    ├── simulation/                # Example configs
    └── analysis/                  # Visualization tools
```

## Data Format

### Quantum ML Format

The pipeline outputs data in the format expected by quantum algorithms:

```python
import h5py

with h5py.File('output/events_for_quantum_ml.h5', 'r') as f:
    latent_space = f['latent_space'][:]  # Shape: (n_events, 2, latent_dim)
    jet_data = f['jet_data'][:]          # Shape: (n_events, 10, 4)
    
    # Metadata
    print(f"Events: {f.attrs['n_events']}")
    print(f"Latent dim: {f.attrs['latent_dim']}")
    print(f"Process: {f.attrs['process']}")
```

### Pipeline Flow

```
Pythia8 (pp → jj)
    ↓
Raw Events (100 particles × 3 features)
    ↓
Preprocessing (log pT, normalize η, φ)
    ↓
Autoencoder (300D → 6D latent space)
    ↓
Quantum Format (n_events, 2, 6)
    ↓
Quantum Algorithms (QSVM, k-medians, etc.)
```

## Available Quantum Algorithms

### 1. One-Class Quantum SVM

Detects anomalies using quantum kernels:

```bash
python scripts/kernel_machines/train_one_class_qsvm.py \
    --train-file data/signal.h5 \
    --nqubits 6 \
    --feature-map ZZFeatureMap \
    --backend qasm_simulator
```

### 2. Quantum K-Medians

Clustering in latent space:

```bash
python scripts/kmedians/run_qkmedians.py \
    --data-file data/events.h5 \
    --n-clusters 2 \
    --backend aer_simulator
```

### 3. Quantum K-Means

Alternative clustering approach:

```bash
python scripts/kmeans/run_qkmeans.py \
    --data-file data/events.h5 \
    --n-clusters 2
```

## Supported Physics Processes

- **QCD Dijet** (pp → jj): Dominant LHC background
- **Z+jets** (pp → Z+jets): Electroweak process
- **Custom processes**: Define via Pythia8 configuration

## Performance

| Stage | Events | Time | Output |
|-------|--------|------|--------|
| Pythia8 Generation | 1000 | ~45s | raw_events.h5 |
| Autoencoder Training | 1000 | ~15s | 6D latent |
| Quantum QSVM | 1000 | varies | predictions |

## Troubleshooting

### Low Event Acceptance

If Pythia8 generates few accepted events:

```yaml
# Relax cuts in configuration
jets:
  pt_min: 15.0    # Lower from 20.0
  eta_max: 3.0    # Increase from 2.5
```

### Autoencoder Not Converging

```bash
# Increase training
python generate_and_compress_events.py --epochs 200

# Or increase latent dimension
python generate_and_compress_events.py --latent-dim 8
```

### Memory Issues

```bash
# Generate in smaller batches
python generate_and_compress_events.py --n-events 500
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Belis_2024,
    title={Quantum anomaly detection in the latent space of proton collision events at the LHC},
    volume={7},
    journal={Communications Physics},
    author={Belis, Vasilis and Woźniak, Kinga Anna and Puljak, Ema and others},
    year={2024}
}
```

## Documentation

- **`qad/simulation/README.md`**: Simulation module API
- **`examples/simulation/`**: Example configuration files
- **`examples/analysis/`**: Analysis and visualization examples

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/rotiyan/QmlAnomaly/issues)
- **Original Repository**: [vbelis/latent-ad-qml](https://github.com/vbelis/latent-ad-qml)
- **Paper**: [Communications Physics 7, 1 (2024)](https://www.nature.com/articles/s42005-024-01811-6)
