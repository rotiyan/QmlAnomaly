"""
Pythia-8 particle physics simulation module for quantum anomaly detection.

This module provides interfaces for generating particle physics events using Pythia-8
and preprocessing them for use with the quantum machine learning pipeline.
"""

from .pythia_generator import PythiaEventGenerator
from .config_parser import SimulationConfig
from .data_processor import EventDataProcessor

__all__ = [
    "PythiaEventGenerator",
    "SimulationConfig", 
    "EventDataProcessor"
]