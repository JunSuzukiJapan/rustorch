"""
RusTorch: PyTorch-compatible deep learning library in Rust - Phase 1
===================================================================

Minimal implementation of core Tensor functionality.

Basic usage:
    >>> import rustorch
    >>> t1 = rustorch.zeros([2, 3])
    >>> t2 = rustorch.ones([2, 3])
    >>> result = t1 + t2
    >>> print(result.shape)
    [2, 3]
"""

from typing import List, Optional, Union, Any

# Import the compiled Rust extension
try:
    from rustorch._rustorch_py import *  # type: ignore
except ImportError as e:
    raise ImportError(
        f"Failed to import RusTorch native extension: {e}\n"
        "Make sure RusTorch is properly built with: maturin develop"
    ) from e

# Version information
__version__ = "0.3.3"
__author__ = "Jun Suzuki"
__email__ = "jun.suzuki.japan@gmail.com"

# Re-export core classes and functions for Phase 3
from rustorch._rustorch_py import (
    # Core tensor class
    Tensor,

    # Variable for autograd
    Variable,

    # Neural network layers
    Linear,

    # Optimizers
    SGD,

    # Loss functions
    MSELoss,

    # Activation functions
    ReLU,
    Sigmoid,
    Tanh,

    # Tensor creation functions
    tensor,
    zeros,
    ones,
)

# Utility functions
def is_tensor(obj: Any) -> bool:
    """
    Check if an object is a RusTorch tensor.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Tensor, False otherwise
    """
    return isinstance(obj, Tensor)

# Module documentation
__all__ = [
    # Core classes
    "Tensor",
    "Variable",
    "Linear",

    # Optimizers
    "SGD",

    # Loss functions
    "MSELoss",

    # Activation functions
    "ReLU",
    "Sigmoid",
    "Tanh",

    # Creation functions
    "tensor",
    "zeros",
    "ones",

    # Utilities
    "is_tensor",

    # Version info
    "__version__",
]