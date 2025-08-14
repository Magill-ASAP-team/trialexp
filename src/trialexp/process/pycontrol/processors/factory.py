"""
Factory system for PyControl task processors.

Provides a registry and factory for instantiating appropriate processor classes
based on configuration. Supports both built-in and custom processor classes.
"""

from typing import Dict, Type, Any, Union
from .base import BaseTaskProcessor
from .pycontrol_processor import PyControlProcessor
from .photometry_processor import PhotometryProcessor
from .behavioral_processor import BehavioralProcessor


# Global registry of processor classes
_PROCESSOR_REGISTRY: Dict[str, Type[Union[BaseTaskProcessor, PyControlProcessor, PhotometryProcessor, BehavioralProcessor]]] = {}


def register_processor(name: str, processor_class: Type[Any]) -> None:
    """
    Register a processor class in the global registry.
    
    Args:
        name: Name to register the processor under
        processor_class: Processor class to register
        
    Raises:
        ValueError: If name is already registered
        TypeError: If processor_class is not a valid processor class
    """
    if name in _PROCESSOR_REGISTRY:
        raise ValueError(f"Processor '{name}' is already registered")
    
    # Check if it's a valid processor class (has callable methods)
    if not callable(processor_class):
        raise TypeError(f"Processor class must be callable")
    
    _PROCESSOR_REGISTRY[name] = processor_class


def get_processor(processor_name: str) -> Union[BaseTaskProcessor, PyControlProcessor, PhotometryProcessor, BehavioralProcessor]:
    """
    Get a processor instance by name from the registry.
    
    Args:
        processor_name: Name of the processor to instantiate
        
    Returns:
        BaseTaskProcessor: Instance of the requested processor
        
    Raises:
        ValueError: If processor_name is not found in registry
    """
    if processor_name not in _PROCESSOR_REGISTRY:
        raise ValueError(
            f"Processor '{processor_name}' not found in registry. "
            f"Available processors: {list(_PROCESSOR_REGISTRY.keys())}"
        )
    
    processor_class = _PROCESSOR_REGISTRY[processor_name]
    return processor_class()


def list_processors() -> list:
    """
    List all registered processor names.
    
    Returns:
        list: List of registered processor names
    """
    return list(_PROCESSOR_REGISTRY.keys())


def clear_registry() -> None:
    """
    Clear all registered processors. Mainly for testing.
    """
    global _PROCESSOR_REGISTRY
    _PROCESSOR_REGISTRY.clear()


# Register all processor classes by default
register_processor("BaseTaskProcessor", BaseTaskProcessor)
register_processor("PyControlProcessor", PyControlProcessor)
register_processor("PhotometryProcessor", PhotometryProcessor)
register_processor("BehavioralProcessor", BehavioralProcessor)