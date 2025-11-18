#!/usr/bin/env python3
"""
Standalone test for the Pythia-8 simulation module.
This test runs without importing the main qad package to avoid dependency issues.
"""

import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add current directory to path for direct imports
sys.path.insert(0, str(Path(__file__).parent))

# Import modules directly
from config_parser import SimulationConfig, PhysicsProcess, SimulationSettings, JetSettings
from data_processor import EventDataProcessor


def test_config_parser():
    """Test configuration parser."""
    print("Testing configuration parser...")
    
    # Create test configuration
    test_config = {
        'physics_process': {
            'process': 'pp -> jj',
            'energy': 13000.0,
            'parameters': {
                'PhaseSpace:pTHatMin': 50.0
            }
        },
        'simulation': {
            'n_events': 1000,
            'output_file': 'test_events.h5',
            'random_seed': 12345,
            'debug_level': 0
        },
        'jets': {
            'algorithm': 'antikt',
            'r_parameter': 0.4,
            'pt_min': 20.0,
            'eta_max': 2.5
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # Test configuration loading
        config = SimulationConfig(config_path)
        
        # Test property access
        assert config.physics_process.process == 'pp -> jj'
        assert config.physics_process.energy == 13000.0
        assert config.simulation.n_events == 1000
        assert config.jets.algorithm == 'antikt'
        
        # Test Pythia command generation
        commands = config.get_pythia_commands()
        assert 'Beams:eCM = 13000.0' in commands
        assert 'Random:seed = 12345' in commands
        assert 'Beams:idA = 2212' in commands
        assert 'HardQCD:all = on' in commands
        
        print("✓ Configuration parser test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration parser test failed: {e}")
        return False
    finally:
        os.unlink(config_path)


def test_data_processor():
    """Test data processor with mock data."""
    print("Testing data processor...")
    
    try:
        # Create mock particle data
        n_events = 100
        particle_data = np.random.rand(n_events, 100, 3)
        particle_data[:, :, 0] *= 100  # pT values
        particle_data[:, :, 1] = (particle_data[:, :, 1] - 0.5) * 4  # eta values
        particle_data[:, :, 2] = (particle_data[:, :, 2] - 0.5) * 2 * np.pi  # phi values
        
        # Create mock jet data
        jet_data = np.random.rand(n_events, 10, 4)
        jet_data[:, :, 0] *= 200  # pT values
        
        # Test data processor
        processor = EventDataProcessor()
        
        # Test preprocessing
        processed_data = processor.preprocess_particles(particle_data)
        assert processed_data.shape == particle_data.shape
        assert processor.data_stats is not None
        
        # Test denormalization
        denormalized_data = processor.denormalize_events(processed_data)
        assert denormalized_data.shape == processed_data.shape
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving/loading
            processor.save_processed_data(
                processed_data, 
                np.random.rand(n_events, 6),  # dummy latent data
                jet_data, 
                temp_path
            )
            
            loaded_particles, loaded_jets = processor.load_events(temp_path)
            assert loaded_particles.shape == processed_data.shape
            assert loaded_jets.shape == jet_data.shape
            
            print("✓ Data processor test passed")
            return True
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ Data processor test failed: {e}")
        return False


def test_mock_pythia_generator():
    """Test mock Pythia generator (without actual Pythia-8)."""
    print("Testing mock Pythia generator...")
    
    try:
        # Create test configuration
        test_config = {
            'physics_process': {
                'process': 'pp -> jj',
                'energy': 13000.0,
                'parameters': {}
            },
            'simulation': {
                'n_events': 10,
                'output_file': 'test_events.h5',
                'random_seed': 12345
            },
            'jets': {
                'algorithm': 'antikt',
                'r_parameter': 0.4,
                'pt_min': 20.0,
                'eta_max': 2.5
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            config = SimulationConfig(config_path)
            
            # Mock the PythiaEventGenerator class to avoid Pythia-8 dependency
            class MockPythiaEventGenerator:
                def __init__(self, config):
                    self.config = config
                    self.n_generated = 0
                    self.n_accepted = 0
                
                def generate_events(self):
                    # Generate mock events
                    n_events = self.config.simulation.n_events
                    particle_data = np.random.rand(n_events, 100, 3)
                    particle_data[:, :, 0] *= 100  # pT
                    particle_data[:, :, 1] = (particle_data[:, :, 1] - 0.5) * 4  # eta
                    particle_data[:, :, 2] = (particle_data[:, :, 2] - 0.5) * 2 * np.pi  # phi
                    
                    jet_data = np.random.rand(n_events, 10, 4)
                    jet_data[:, :, 0] *= 200  # pT
                    
                    self.n_generated = n_events
                    self.n_accepted = n_events
                    
                    return particle_data, jet_data
                
                def get_statistics(self):
                    return {
                        'n_generated': self.n_generated,
                        'n_accepted': self.n_accepted,
                        'acceptance_rate': self.n_accepted / max(self.n_generated, 1)
                    }
            
            # Test mock generator
            generator = MockPythiaEventGenerator(config)
            particle_data, jet_data = generator.generate_events()
            
            assert particle_data.shape[0] == config.simulation.n_events
            assert particle_data.shape[1:] == (100, 3)
            assert jet_data.shape[0] == config.simulation.n_events
            assert jet_data.shape[1:] == (10, 4)
            
            stats = generator.get_statistics()
            assert stats['n_generated'] == config.simulation.n_events
            assert stats['n_accepted'] == config.simulation.n_events
            
            print("✓ Mock Pythia generator test passed")
            return True
            
        finally:
            os.unlink(config_path)
            
    except Exception as e:
        print(f"✗ Mock Pythia generator test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running Pythia-8 simulation module tests...\n")
    
    tests = [
        test_config_parser,
        test_data_processor,
        test_mock_pythia_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)