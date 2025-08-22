"""
Neural Network modules for RusTorch
===================================

This module provides PyTorch-compatible neural network building blocks
including layers, activation functions, and loss functions.

Example:
    >>> import rustorch as rt
    >>> import rustorch.nn as nn
    
    >>> model = nn.Sequential(
    ...     nn.Linear(784, 128),
    ...     nn.ReLU(),
    ...     nn.Linear(128, 10)
    ... )
    >>> output = model(input_tensor)
"""

from typing import List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod

# Import from the Rust extension
from rustorch._rustorch_py import Linear as _Linear, ReLU as _ReLU, Sequential as _Sequential, MSELoss as _MSELoss
from rustorch import Variable, Tensor

__all__ = [
    "Module",
    "Linear", 
    "ReLU",
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "Conv2d",
    "BatchNorm2d",
    "Dropout",
    "Parameter",
]

class Parameter(Variable):
    """
    A special Variable that is considered a module parameter.
    
    Parameters are Variables that are typically learnable weights
    of a neural network module.
    """
    
    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__(data, requires_grad)
    
    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"

class Module(ABC):
    """
    Base class for all neural network modules.
    
    This is the base class that all layers and networks should inherit from.
    It provides basic functionality for parameter management and training/eval modes.
    """
    
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, 'Module'] = {}
        self._training = True
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make modules callable by forwarding to forward method."""
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Define the forward pass of the module.
        
        This method should be overridden by all subclasses.
        """
        raise NotImplementedError
    
    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, recursively yield parameters of submodules
            
        Returns:
            List of parameters
        """
        params = list(self._parameters.values())
        
        if recurse:
            for module in self._modules.values():
                params.extend(module.parameters(recurse=True))
        
        return params
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> List[tuple[str, Parameter]]:
        """
        Return an iterator over module parameters with names.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, recursively yield parameters of submodules
            
        Returns:
            List of (name, parameter) tuples
        """
        named_params = []
        
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            named_params.append((full_name, param))
        
        if recurse:
            for name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                named_params.extend(module.named_parameters(submodule_prefix, recurse=True))
        
        return named_params
    
    def train(self, mode: bool = True) -> 'Module':
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)
    
    def training(self) -> bool:
        """
        Check if the module is in training mode.
        
        Returns:
            True if in training mode, False if in evaluation mode
        """
        return self._training
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.parameters():
            param.zero_grad()
    
    def _register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Register a parameter with the module."""
        if param is not None:
            self._parameters[name] = param
    
    def _register_module(self, name: str, module: Optional['Module']) -> None:
        """Register a submodule with the module."""
        if module is not None:
            self._modules[name] = module
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Linear(Module):
    """
    Linear transformation layer: y = xA^T + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, add learnable bias. Default: True
        
    Example:
        >>> import rustorch.nn as nn
        >>> layer = nn.Linear(784, 128)
        >>> output = layer(input)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use the Rust implementation
        self._linear = _Linear(in_features, out_features)
        
        # Get parameters from Rust implementation
        rust_params = self._linear.parameters()
        if len(rust_params) >= 1:
            self._register_parameter('weight', Parameter(rust_params[0].detach()))
        if len(rust_params) >= 2 and bias:
            self._register_parameter('bias', Parameter(rust_params[1].detach()))
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through the linear layer."""
        return self._linear.forward(input)
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

class ReLU(Module):
    """
    Rectified Linear Unit activation function.
    
    Applies the function element-wise: ReLU(x) = max(0, x)
    
    Example:
        >>> import rustorch.nn as nn
        >>> activation = nn.ReLU()
        >>> output = activation(input)
    """
    
    def __init__(self):
        super().__init__()
        self._relu = _ReLU()
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through ReLU activation."""
        return self._relu.forward(input)
    
    def __repr__(self) -> str:
        return "ReLU()"

class Sequential(Module):
    """
    Sequential container for layers.
    
    Layers will be added to it in the order they are passed in the constructor.
    
    Args:
        *args: Modules to be added in order
        
    Example:
        >>> import rustorch.nn as nn
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 128),
        ...     nn.ReLU(),
        ...     nn.Linear(128, 10)
        ... )
    """
    
    def __init__(self, *args):
        super().__init__()
        self._layers = list(args)
        
        # Register submodules
        for i, layer in enumerate(self._layers):
            self._register_module(str(i), layer)
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through all layers in sequence."""
        x = input
        for layer in self._layers:
            x = layer(x)
        return x
    
    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return '\n'.join(lines)
    
    def __len__(self) -> int:
        return len(self._layers)
    
    def __getitem__(self, idx: int) -> Module:
        return self._layers[idx]

class MSELoss(Module):
    """
    Mean Squared Error loss function.
    
    Creates a criterion that measures the mean squared error between
    input and target.
    
    Example:
        >>> import rustorch.nn as nn
        >>> criterion = nn.MSELoss()
        >>> loss = criterion(output, target)
    """
    
    def __init__(self):
        super().__init__()
        self._mse = _MSELoss()
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute MSE loss between input and target."""
        return self._mse.forward(input, target)
    
    def __call__(self, input: Variable, target: Variable) -> Variable:
        return self.forward(input, target)
    
    def __repr__(self) -> str:
        return "MSELoss()"

class CrossEntropyLoss(Module):
    """
    Cross Entropy loss function.
    
    This criterion computes cross entropy loss between input and target.
    
    Example:
        >>> import rustorch.nn as nn
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss = criterion(output, target)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input: Variable, target: Variable) -> Variable:
        """Compute cross entropy loss."""
        # This is a simplified implementation
        # In practice, you'd want proper softmax + log-likelihood
        # For now, using MSE as placeholder
        return MSELoss().forward(input, target)
    
    def __call__(self, input: Variable, target: Variable) -> Variable:
        return self.forward(input, target)
    
    def __repr__(self) -> str:
        return "CrossEntropyLoss()"

class Conv2d(Module):
    """
    2D convolution layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Zero-padding added to both sides of the input. Default: 0
        bias: If True, adds a learnable bias. Default: True
        
    Example:
        >>> import rustorch.nn as nn
        >>> conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> output = conv(input)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Union[int, tuple[int, int]], 
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # TODO: Implement with Rust Conv2d
        # For now, this is a placeholder
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through convolution layer."""
        # Placeholder implementation
        return input
    
    def __repr__(self) -> str:
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")

class BatchNorm2d(Module):
    """
    2D batch normalization layer.
    
    Args:
        num_features: Number of features (channels)
        eps: A small value added to denominator for numerical stability. Default: 1e-5
        momentum: Value used for running mean and variance computation. Default: 0.1
        
    Example:
        >>> import rustorch.nn as nn
        >>> bn = nn.BatchNorm2d(64)
        >>> output = bn(input)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # TODO: Implement with Rust BatchNorm2d
        # For now, this is a placeholder
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through batch normalization."""
        # Placeholder implementation
        return input
    
    def __repr__(self) -> str:
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum})"

class Dropout(Module):
    """
    Dropout layer for regularization.
    
    Args:
        p: Probability of an element to be zeroed. Default: 0.5
        
    Example:
        >>> import rustorch.nn as nn
        >>> dropout = nn.Dropout(0.5)
        >>> output = dropout(input)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, input: Variable) -> Variable:
        """Forward pass through dropout."""
        if self.training:
            # Apply dropout during training
            # TODO: Implement proper dropout with Rust
            return input
        else:
            # No dropout during evaluation
            return input
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

# Functional API (similar to torch.nn.functional)
class functional:
    """Functional interface for neural network operations."""
    
    @staticmethod
    def relu(input: Variable) -> Variable:
        """Apply ReLU activation function."""
        return ReLU()(input)
    
    @staticmethod
    def mse_loss(input: Variable, target: Variable) -> Variable:
        """Compute mean squared error loss."""
        return MSELoss()(input, target)
    
    @staticmethod
    def cross_entropy(input: Variable, target: Variable) -> Variable:
        """Compute cross entropy loss."""
        return CrossEntropyLoss()(input, target)

# Add functional to module exports
F = functional
__all__.append("F")
__all__.append("functional")