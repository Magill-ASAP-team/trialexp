"""
Task processors for PyControl data analysis.

This package provides a flexible processor-based architecture for analyzing
behavioral task data, allowing task-specific customization through inheritance.
"""

from .base import BaseTaskProcessor
from .pycontrol_processor import PyControlProcessor
from .photometry_processor import PhotometryProcessor
from .behavioral_processor import BehavioralProcessor
from .factory import get_processor, register_processor, list_processors

__all__ = [
    'BaseTaskProcessor',
    'PyControlProcessor',
    'PhotometryProcessor', 
    'BehavioralProcessor',
    'get_processor', 
    'register_processor',
    'list_processors'
]