"""
RusTorch: PyTorch-compatible deep learning library in Rust
========================================================

RusTorch provides a high-performance, memory-safe alternative to PyTorch
while maintaining API compatibility and familiar workflows.

Basic usage:
    >>> import rustorch as rt
    >>> tensor = rt.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> print(tensor.shape)
    [2, 2]

Key features:
- Zero-cost abstractions with Rust performance
- Memory safety without garbage collection  
- WebAssembly support for browser deployment
- PyTorch-compatible API design
- Automatic differentiation (autograd)
- Neural network modules and optimizers
"""

from typing import List, Optional, Union, Any
import sys

# Import the compiled Rust extension
try:
    from rustorch._rustorch_py import *  # type: ignore
except ImportError as e:
    raise ImportError(
        f"Failed to import RusTorch native extension: {e}\n"
        "Make sure RusTorch is properly installed with: pip install rustorch"
    ) from e

# Version information
__version__ = "0.3.3"
__author__ = "Jun Suzuki"
__email__ = "jun.suzuki.japan@gmail.com"

# Re-export core classes and functions
from rustorch._rustorch_py import (
    Tensor,
    Variable, 
    Linear,
    SGD,
    Adam,
    tensor,
    zeros,
    ones,
    randn,
    from_numpy,
    matmul,
    add,
    sum as tensor_sum,
    mean as tensor_mean,
)

# Module imports
from . import nn
from . import optim  
from . import autograd
from . import utils

# NumPy interoperability
def from_numpy(array: "numpy.ndarray") -> Tensor:
    """
    Create a RusTorch tensor from a NumPy array.
    
    Args:
        array: NumPy array to convert
        
    Returns:
        RusTorch tensor with the same data and shape
        
    Example:
        >>> import numpy as np
        >>> import rustorch as rt
        >>> np_array = np.array([1, 2, 3], dtype=np.float32)
        >>> rt_tensor = rt.from_numpy(np_array)
    """
    import numpy as np
    
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    return from_numpy(array.flatten())

def manual_seed(seed: int) -> None:
    """
    Set the random seed for reproducible results.
    
    Args:
        seed: Random seed value
        
    Example:
        >>> import rustorch as rt
        >>> rt.manual_seed(42)
        >>> tensor = rt.randn([2, 2])  # Reproducible random values
    """
    # This would need to be implemented in the Rust side
    # For now, this is a placeholder
    pass

def set_num_threads(num_threads: int) -> None:
    """
    Set the number of threads for parallel operations.
    
    Args:
        num_threads: Number of threads to use
        
    Example:
        >>> import rustorch as rt
        >>> rt.set_num_threads(4)
    """
    # This would need to be implemented in the Rust side
    # For now, this is a placeholder
    pass

def get_num_threads() -> int:
    """
    Get the current number of threads for parallel operations.
    
    Returns:
        Current number of threads
    """
    # This would need to be implemented in the Rust side
    return 1  # Placeholder

# Utility functions for PyTorch compatibility
def is_tensor(obj: Any) -> bool:
    """
    Check if an object is a RusTorch tensor.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is a Tensor, False otherwise
    """
    return isinstance(obj, Tensor)

def numel(tensor: Tensor) -> int:
    """
    Get the total number of elements in a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Total number of elements
    """
    return tensor.size

# Device management (placeholder for future CUDA support)
class device:
    """
    Device specification for tensor placement.
    
    Currently supports CPU only, with plans for GPU support.
    """
    
    def __init__(self, device_type: str = "cpu"):
        if device_type != "cpu":
            raise NotImplementedError("Only CPU device is currently supported")
        self.type = device_type
    
    def __str__(self) -> str:
        return self.type
    
    def __repr__(self) -> str:
        return f"device(type='{self.type}')"

# Common devices
cpu = device("cpu")

# Global settings
class _TorchSettings:
    """Global settings for RusTorch behavior."""
    
    def __init__(self):
        self.default_dtype = "float32"
        self.grad_enabled = True
    
    def set_default_dtype(self, dtype: str) -> None:
        """Set the default data type for new tensors."""
        self.default_dtype = dtype
    
    def get_default_dtype(self) -> str:
        """Get the current default data type."""
        return self.default_dtype

_settings = _TorchSettings()

def set_default_dtype(dtype: str) -> None:
    """
    Set the default floating point data type.
    
    Args:
        dtype: Default data type (currently only 'float32' supported)
    """
    _settings.set_default_dtype(dtype)

def get_default_dtype() -> str:
    """
    Get the current default floating point data type.
    
    Returns:
        Current default data type
    """
    return _settings.get_default_dtype()

# Context managers
class no_grad:
    """
    Context manager to disable gradient computation.
    
    Example:
        >>> import rustorch as rt
        >>> x = rt.Variable(rt.tensor([1.0]), requires_grad=True)
        >>> with rt.no_grad():
        ...     y = x * 2  # No gradient tracking
    """
    
    def __init__(self):
        self.prev_grad_enabled = True
    
    def __enter__(self):
        self.prev_grad_enabled = _settings.grad_enabled
        _settings.grad_enabled = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _settings.grad_enabled = self.prev_grad_enabled

class enable_grad:
    """
    Context manager to enable gradient computation.
    
    Example:
        >>> import rustorch as rt
        >>> with rt.enable_grad():
        ...     x = rt.Variable(rt.tensor([1.0]), requires_grad=True)
        ...     y = x * 2  # Gradient tracking enabled
    """
    
    def __init__(self):
        self.prev_grad_enabled = True
    
    def __enter__(self):
        self.prev_grad_enabled = _settings.grad_enabled
        _settings.grad_enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _settings.grad_enabled = self.prev_grad_enabled

# Error classes for better error handling
class RusTorchError(Exception):
    """Base exception class for RusTorch errors."""
    pass

class ShapeMismatchError(RusTorchError):
    """Raised when tensor shapes are incompatible for an operation."""
    
    def __init__(self, shape1: List[int], shape2: List[int], operation: str = "operation"):
        self.shape1 = shape1
        self.shape2 = shape2
        self.operation = operation
        super().__init__(
            f"Shape mismatch in {operation}: {shape1} vs {shape2}"
        )

class DeviceError(RusTorchError):
    """Raised when there are device-related errors."""
    pass

# Backward compatibility aliases
Tensor = Tensor
FloatTensor = Tensor  # For PyTorch compatibility

# Module documentation
__all__ = [
    # Core classes
    "Tensor",
    "Variable", 
    "device",
    
    # Creation functions
    "tensor",
    "zeros", 
    "ones",
    "randn",
    "from_numpy",
    
    # Operations
    "matmul",
    "add", 
    "tensor_sum",
    "tensor_mean",
    
    # Modules
    "nn",
    "optim",
    "autograd", 
    "utils",
    
    # Utilities
    "manual_seed",
    "set_num_threads", 
    "get_num_threads",
    "is_tensor",
    "numel",
    "set_default_dtype",
    "get_default_dtype",
    
    # Context managers
    "no_grad",
    "enable_grad",
    
    # Devices
    "cpu",
    
    # Exceptions
    "RusTorchError",
    "ShapeMismatchError", 
    "DeviceError",
    
    # Version info
    "__version__",
]