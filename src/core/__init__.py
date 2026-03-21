"""
Core Module

This module contains the core functionality for managing HELICS federations
and federates in the COSIM Gym framework.

Author: COSIM Gym Team
Date: 2025
"""

from .ScenarioManager import ScenarioManager, main
from .BaseFederate import BaseFederate

__all__ = [
    'ScenarioManager',
    'BaseFederate',
    'main'
]