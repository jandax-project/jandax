"""
Jandax - Traceable and Portable DataFrames for C++ Integration
"""

# Import core components first
# Import pytree registrations (this executes the registrations)
# This must happen after the classes are imported but before any instances are created
from jandax.core import DataFrame, GroupBy, GroupByRolling, Rolling

import jandax.pytree  # isort:skip

# Version information
__version__ = "0.1.0"

# Make all important classes and functions available at package level
__all__ = ["DataFrame", "GroupBy", "GroupByRolling", "Rolling"]
