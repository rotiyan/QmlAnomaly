#!/usr/bin/env python3
"""
Test script to verify the installation and basic functionality of the qad package.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import h5py
        print("✓ h5py")
    except ImportError as e:
        print(f"✗ h5py: {e}")
        return False
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ tensorflow")
    except ImportError as e:
        print(f"✗ tensorflow: {e}")
        return False
    
    try:
        import yaml
        print("✓ pyyaml")
    except ImportError as e:
        print(f"✗ pyyaml: {e}")
        return False
    
    try:
        import qad
        print("✓ qad")
    except ImportError as e:
        print(f"✗ qad: {e}")
        return False
    
    try:
        from qad.simulation import SimulationConfig, EventDataProcessor
        print("✓ qad.simulation")
    except ImportError as e:
        print(f"✗ qad.simulation: {e}")
        return False
    
    try:
        from qad.autoencoder.autoencoder import ParticleAutoencoder
        print("✓ qad.autoencoder")
    except ImportError as e:
        print(f"✗ qad.autoencoder: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the simulation module."""
    print("\nTesting basic functionality...")
    
    try:
        # Test mock dijet generation
        import sys
        sys.path.append('.')
        from qad.simulation.run_pipeline import generate_mock_dijet_events
        
        particle_data, jet_data = generate_mock_dijet_events(10)
        print(f"✓ Generated {len(particle_data)} dijet events")
        print(f"  Particle data shape: {particle_data.shape}")
        print(f"  Jet data shape: {jet_data.shape}")
        
        # Test data preprocessing
        from qad.simulation.run_pipeline import preprocess_particles
        
        processed_data = preprocess_particles(particle_data)
        print(f"✓ Preprocessed data shape: {processed_data.shape}")
        
        # Test autoencoder creation
        from qad.simulation.run_pipeline import SimpleAutoencoder
        
        autoencoder = SimpleAutoencoder(input_dim=300, latent_dim=6)
        print("✓ Created autoencoder model")
        
        # Test encoding
        import torch
        data_tensor = torch.FloatTensor(processed_data.reshape(processed_data.shape[0], -1))
        
        with torch.no_grad():
            latent_data = autoencoder.encode(data_tensor).numpy()
        
        print(f"✓ Encoded to latent space: {latent_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration file parsing."""
    print("\nTesting configuration...")
    
    try:
        from qad.simulation import SimulationConfig
        
        # Create a test configuration
        config_data = {
            'physics_process': {
                'process': 'pp -> jj',
                'energy': 13000.0,
                'parameters': {'PhaseSpace:pTHatMin': 50.0}
            },
            'simulation': {
                'n_events': 100,
                'output_file': 'test.h5',
                'random_seed': 12345
            },
            'jets': {
                'algorithm': 'antikt',
                'r_parameter': 0.4,
                'pt_min': 20.0,
                'eta_max': 2.5
            }
        }
        
        import yaml
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        config = SimulationConfig(config_path)
        print("✓ Configuration parsing")
        print(f"  Process: {config.physics_process.process}")
        print(f"  Energy: {config.physics_process.energy} GeV")
        print(f"  Events: {config.simulation.n_events}")
        
        # Clean up
        Path(config_path).unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_examples():
    """Test that example files exist and are accessible."""
    print("\nTesting examples...")
    
    example_files = [
        'examples/simulation/run_pytorch_chain.py',
        'examples/simulation/zjets_generator.py',
        'examples/simulation/dijet_10_events.yaml',
        'examples/simulation/zjets_config.yaml',
        'examples/analysis/visualize_results.py',
        'examples/analysis/verify_results.py'
    ]
    
    all_exist = True
    for file_path in example_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("QAD Package Installation Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration", test_configuration),
        ("Examples", test_examples)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The installation is working correctly.")
        print("\nYou can now run the simulation pipeline:")
        print("  python -m qad.simulation.run_pipeline --process dijet --n-events 100")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("You may need to install missing dependencies or check your installation.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
