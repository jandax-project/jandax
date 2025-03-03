"""
Jandax - Traceable and Portable DataFrames for C++ Integration
"""

# Import core components first
from jandax.core import DataFrame, GroupBy, GroupByRolling, Rolling

# Version information
__version__ = "0.1.0"

# Make all important classes and functions available at package level
__all__ = ["DataFrame", "GroupBy", "GroupByRolling", "Rolling"]
