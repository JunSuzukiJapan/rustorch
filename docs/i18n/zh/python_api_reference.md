# RusTorch Python API 参考

## 概述

本文档为 RusTorch Python 绑定提供完整的 API 参考。详细介绍所有模块、类和函数。

## 模块结构

```python
rustorch/
├── tensor          # 核心张量操作和类
├── autograd        # 自动微分系统
├── nn              # 神经网络层和函数
├── optim           # 优化器和学习率调度器
├── data            # 数据加载和数据集
├── training        # 高级训练 API
├── distributed     # 分布式训练支持
├── visualization   # 可视化和绘图
└── utils           # 工具函数
```

---

## rustorch.tensor

### 类

#### PyTensor
```python
class PyTensor:
    """核心多维数组（张量）类"""
    
    def __init__(self, data: Union[List, np.ndarray], dtype: Optional[str] = None, requires_grad: bool = False)
    def shape(self) -> List[int]
    def reshape(self, new_shape: List[int]) -> PyTensor
    def transpose(self, dim0: int, dim1: int) -> PyTensor
    def permute(self, dims: List[int]) -> PyTensor
    def squeeze(self, dim: Optional[int] = None) -> PyTensor
    def unsqueeze(self, dim: int) -> PyTensor
    def view(self, shape: List[int]) -> PyTensor
    def size(self) -> List[int]
    def numel(self) -> int
    def dim(self) -> int
    def dtype(self) -> str
    def device(self) -> str
    def to(self, device: str) -> PyTensor
    def cpu(self) -> PyTensor
    def cuda(self) -> PyTensor
    def clone(self) -> PyTensor
    def detach(self) -> PyTensor
    def requires_grad_(self, requires_grad: bool = True) -> PyTensor
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False)
    def grad(self) -> Optional[PyTensor]
    def zero_grad(self)
    
    # 算术运算
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # 原地操作
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # 线性代数
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # 统计函数
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # 比较操作
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # 索引操作
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # 转换函数
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # 特殊方法
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### 函数

#### 工厂函数
```python
def tensor(data: Union[List, np.ndarray], dtype: Optional[str] = None, requires_grad: bool = False) -> PyTensor
def zeros(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def ones(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def zeros_like(input: PyTensor, dtype: Optional[str] = None) -> PyTensor
def ones_like(input: PyTensor, dtype: Optional[str] = None) -> PyTensor
def empty(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def full(shape: List[int], fill_value: float, dtype: str = "float32", requires_grad: bool = False) -> PyTensor

def eye(n: int, m: Optional[int] = None, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def arange(start: float, end: Optional[float] = None, step: float = 1.0, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def linspace(start: float, end: float, steps: int, dtype: str = "float32", requires_grad: bool = False) -> PyTensor

def rand(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def randn(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def randint(low: int, high: int, shape: List[int], dtype: str = "int64", requires_grad: bool = False) -> PyTensor

def from_numpy(array: np.ndarray) -> PyTensor
def as_tensor(data: Union[List, np.ndarray, PyTensor], dtype: Optional[str] = None) -> PyTensor
```

#### 数学函数
```python
# 元素级运算
def add(input: PyTensor, other: Union[PyTensor, float], alpha: float = 1.0) -> PyTensor
def sub(input: PyTensor, other: Union[PyTensor, float], alpha: float = 1.0) -> PyTensor
def mul(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def div(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def pow(input: PyTensor, exponent: Union[PyTensor, float]) -> PyTensor
def sqrt(input: PyTensor) -> PyTensor
def abs(input: PyTensor) -> PyTensor
def neg(input: PyTensor) -> PyTensor

def exp(input: PyTensor) -> PyTensor
def log(input: PyTensor) -> PyTensor
def sin(input: PyTensor) -> PyTensor
def cos(input: PyTensor) -> PyTensor
def tan(input: PyTensor) -> PyTensor
def tanh(input: PyTensor) -> PyTensor
def sigmoid(input: PyTensor) -> PyTensor

def clamp(input: PyTensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> PyTensor

# 线性代数
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor

# 归约操作
def sum(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def mean(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def std(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def var(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def max(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
def min(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
def argmax(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
def argmin(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
```

---

## rustorch.autograd

### 类

#### PyVariable
```python
class PyVariable:
    """支持自动微分的张量包装器"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    
    # Variable 操作（带自动微分支持）
    def add(self, other: Union[PyVariable, float]) -> PyVariable
    def sub(self, other: Union[PyVariable, float]) -> PyVariable
    def mul(self, other: Union[PyVariable, float]) -> PyVariable
    def div(self, other: Union[PyVariable, float]) -> PyVariable
    def pow(self, exponent: Union[PyVariable, float]) -> PyVariable
    def sqrt(self) -> PyVariable
    def exp(self) -> PyVariable
    def log(self) -> PyVariable
    def sin(self) -> PyVariable
    def cos(self) -> PyVariable
    def tanh(self) -> PyVariable
    def sigmoid(self) -> PyVariable
    def relu(self) -> PyVariable
    
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    
    def matmul(self, other: PyVariable) -> PyVariable
    def mm(self, other: PyVariable) -> PyVariable
    
    def reshape(self, shape: List[int]) -> PyVariable
    def transpose(self, dim0: int, dim1: int) -> PyVariable
    def view(self, shape: List[int]) -> PyVariable
```

### 函数

#### 梯度计算
```python
def grad(outputs: List[PyVariable], 
         inputs: List[PyVariable], 
         grad_outputs: Optional[List[PyTensor]] = None,
         retain_graph: bool = False,
         create_graph: bool = False,
         only_inputs: bool = True,
         allow_unused: bool = False) -> List[Optional[PyTensor]]

def backward(tensors: List[PyVariable],
             grad_tensors: Optional[List[PyTensor]] = None,
             retain_graph: bool = False,
             create_graph: bool = False) -> None
```

#### 上下文管理器
```python
class no_grad:
    """禁用梯度计算的上下文管理器"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class enable_grad:
    """启用梯度计算的上下文管理器"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class set_grad_enabled:
    """设置梯度计算启用/禁用的上下文管理器"""
    def __init__(self, mode: bool)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
```

---

## rustorch.nn

### 基类

#### Module
```python
class Module:
    """所有神经网络模块的基类"""
    
    def __init__(self)
    def forward(self, *input) -> PyTensor
    def __call__(self, *input) -> PyTensor
    def parameters(self, recurse: bool = True) -> Iterator[PyTensor]
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, PyTensor]]
    def modules(self) -> Iterator[Module]
    def children(self) -> Iterator[Module]
    def train(self, mode: bool = True) -> Module
    def eval(self) -> Module
    def cuda(self, device: Optional[str] = None) -> Module
    def cpu(self) -> Module
    def to(self, device: str) -> Module
    def zero_grad(self)
    def state_dict(self) -> Dict[str, PyTensor]
    def load_state_dict(self, state_dict: Dict[str, PyTensor], strict: bool = True)
```

### 线性层

#### PyLinear
```python
class PyLinear(Module):
    """线性变换（全连接）层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
```

### 卷积层

#### PyConv2d
```python
class PyConv2d(Module):
    """2D 卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_parameters(self)
```

### 归一化层

#### PyBatchNorm2d
```python
class PyBatchNorm2d(Module):
    """2D 批量归一化层"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_running_stats(self)
    def reset_parameters(self)
```

### 激活函数

#### PyReLU
```python
class PyReLU(Module):
    """ReLU 激活函数"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PySigmoid
```python
class PySigmoid(Module):
    """Sigmoid 激活函数"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PyTanh
```python
class PyTanh(Module):
    """双曲正切激活函数"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

### 损失函数

#### PyMSELoss
```python
class PyMSELoss(Module):
    """均方误差损失"""
    
    def __init__(self, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### PyCrossEntropyLoss
```python
class PyCrossEntropyLoss(Module):
    """交叉熵损失"""
    
    def __init__(self, weight: Optional[PyTensor] = None, ignore_index: int = -100, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

### 容器模块

#### Sequential
```python
class Sequential(Module):
    """模块的顺序容器"""
    
    def __init__(self, *args)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Module
    def append(self, module: Module) -> Sequential
    def forward(self, input: PyTensor) -> PyTensor
```

---

## rustorch.optim

### 优化器

#### PySGD
```python
class PySGD:
    """随机梯度下降优化器"""
    
    def __init__(self, params: List[PyTensor], lr: float, momentum: float = 0, weight_decay: float = 0)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

#### PyAdam
```python
class PyAdam:
    """Adam 优化器"""
    
    def __init__(self, params: List[PyTensor], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

### 学习率调度器

#### PyStepLR
```python
class PyStepLR:
    """步长学习率调度器"""
    
    def __init__(self, optimizer: Union[PySGD, PyAdam], step_size: int, gamma: float = 0.1)
    def step(self, epoch: Optional[int] = None)
    def get_last_lr(self) -> List[float]
    def get_lr(self) -> List[float]
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

---

## rustorch.data

### 数据集

#### PyTensorDataset
```python
class PyTensorDataset:
    """包装张量的数据集"""
    
    def __init__(self, *tensors: PyTensor)
    def __getitem__(self, index: int) -> List[PyTensor]
    def __len__(self) -> int
```

### 数据加载

#### PyDataLoader
```python
class PyDataLoader:
    """数据集的数据加载器"""
    
    def __init__(self, dataset: PyTensorDataset, batch_size: int = 1, 
                 shuffle: bool = False, num_workers: int = 0, drop_last: bool = False)
    def __iter__(self) -> Iterator[List[PyTensor]]
    def __len__(self) -> int
```

### 变换

#### PyTransform
```python
class PyTransform:
    """数据变换"""
    
    def __init__(self, name: str, params: Optional[Dict[str, float]] = None)
    def __call__(self, input: PyTensor) -> PyTensor
```

---

## rustorch.training

### 高级训练 API

#### PyModel
```python
class PyModel:
    """高级 Keras 风格模型"""
    
    def __init__(self, name: Optional[str] = None)
    def add(self, layer: Union[str, Module])
    def compile(self, optimizer: Union[str, Dict], loss: Union[str, Callable], metrics: Optional[List[str]] = None)
    def fit(self, train_data: PyDataLoader, 
            validation_data: Optional[PyDataLoader] = None,
            epochs: int = 10, verbose: bool = True) -> PyTrainingHistory
    def evaluate(self, data: PyDataLoader) -> Dict[str, float]
    def predict(self, data: PyDataLoader) -> List[PyTensor]
    def summary(self) -> str
```

#### PyTrainingHistory
```python
class PyTrainingHistory:
    """训练历史容器"""
    
    def __init__(self)
    def add_epoch(self, train_loss: float, val_loss: Optional[float] = None, metrics: Optional[Dict[str, float]] = None)
    def train_loss(self) -> List[float]
    def val_loss(self) -> List[float]
    def metrics(self) -> Dict[str, List[float]]
    def summary(self) -> str
    def plot_data(self) -> Tuple[List[float], List[float], List[float]]
```

---

## rustorch.distributed

### 配置

#### PyDistributedConfig
```python
class PyDistributedConfig:
    """分布式训练配置"""
    
    def __init__(self, backend: str = "nccl", world_size: int = 1, rank: int = 0,
                 master_addr: str = "localhost", master_port: int = 29500)
    def backend(self) -> str
    def world_size(self) -> int
    def rank(self) -> int
```

### 分布式训练

#### PyDistributedDataParallel
```python
class PyDistributedDataParallel:
    """分布式数据并行包装器"""
    
    def __init__(self, model: PyModel, device_ids: Optional[List[int]] = None)
    def forward(self, *inputs) -> PyTensor
    def sync_gradients(self)
    def module(self) -> PyModel
```

---

## rustorch.visualization

### 绘图

#### PyPlotter
```python
class PyPlotter:
    """绘图工具"""
    
    def __init__(self, backend: str = "matplotlib", style: str = "default")
    def plot_training_history(self, history: PyTrainingHistory, save_path: Optional[str] = None)
    def plot_tensor_as_image(self, tensor: PyTensor, title: Optional[str] = None, save_path: Optional[str] = None)
    def line_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def scatter_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def histogram(self, data: List[float], bins: int = 30, title: Optional[str] = None, save_path: Optional[str] = None)
```

---

## rustorch.utils

### 模型管理

#### 函数
```python
def save_model(model: PyModel, path: str)
def load_model(path: str) -> PyModel
def save_checkpoint(model: PyModel, optimizer: Union[PySGD, PyAdam], epoch: int, loss: float, path: str)
def load_checkpoint(path: str) -> Dict[str, Any]
```

### 性能

#### PyProfiler
```python
class PyProfiler:
    """性能分析器"""
    
    def __init__(self)
    def start(self)
    def stop(self)
    def reset(self)
    def summary(self) -> str
    def profile(self) -> ContextManager
```

### 配置

#### PyConfig
```python
class PyConfig:
    """全局配置管理"""
    
    def __init__(self)
    def set(self, key: str, value: Any)
    def get(self, key: str) -> Any
    def reset(self)
```

---

## 错误处理

### 异常

#### RusTorchError
所有 RusTorch 错误的基异常类。

```python
class RusTorchError(Exception):
    """RusTorch 错误的基异常"""
    pass

class TensorError(RusTorchError):
    """张量操作错误"""
    pass

class ShapeError(RusTorchError):
    """形状不匹配错误"""
    pass

class DeviceError(RusTorchError):
    """设备相关错误"""
    pass

class SerializationError(RusTorchError):
    """模型保存/加载错误"""
    pass
```

---

## 类型提示

RusTorch 提供全面的类型提示支持：

```python
import rustorch
from typing import Optional, List, Tuple, Union, Dict, Any

def train_model(model: rustorch.Model, 
               data: rustorch.data.DataLoader,
               optimizer: Union[rustorch.optim.SGD, rustorch.optim.Adam],
               epochs: int = 10) -> rustorch.training.TrainingHistory:
    return model.fit(data, epochs=epochs)
```

---

本 API 参考涵盖了 RusTorch Python 绑定的核心功能。详细的使用示例请参见 [examples/](../examples/) 目录。