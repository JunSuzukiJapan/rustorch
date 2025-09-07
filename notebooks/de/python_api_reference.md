# RusTorch Python API-Referenz

## Übersicht

Dieses Dokument bietet eine vollständige API-Referenz für RusTorch Python-Bindings. Es behandelt alle Module, Klassen und Funktionen im Detail.

## Modulstruktur

```python
rustorch/
├── tensor          # Zentrale Tensor-Operationen und Klassen
├── autograd        # Automatisches Differenzierungssystem
├── nn              # Neuronale Netzwerk-Schichten und Funktionen
├── optim           # Optimierer und Lernraten-Scheduler
├── data            # Datenladen und Datensätze
├── training        # High-Level Training-API
├── distributed     # Verteiltes Training-Support
├── visualization   # Visualisierung und Plotting
└── utils           # Hilfsfunktionen
```

---

## rustorch.tensor

### Klassen

#### PyTensor
```python
class PyTensor:
    """Zentrale mehrdimensionale Array (Tensor) Klasse"""
    
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
    
    # Arithmetische Operationen
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # In-Place-Operationen
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # Lineare Algebra
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # Statistische Funktionen
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # Vergleichsoperationen
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # Indizierungsoperationen
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # Konvertierungsfunktionen
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # Spezielle Methoden
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### Funktionen

#### Factory-Funktionen
```python
def tensor(data: Union[List, np.ndarray], dtype: Optional[str] = None, requires_grad: bool = False) -> PyTensor
def zeros(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def ones(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def zeros_like(input: PyTensor, dtype: Optional[str] = None) -> PyTensor
def ones_like(input: PyTensor, dtype: Optional[str] = None) -> PyTensor
def empty(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def empty_like(input: PyTensor, dtype: Optional[str] = None) -> PyTensor
def full(shape: List[int], fill_value: float, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def full_like(input: PyTensor, fill_value: float, dtype: Optional[str] = None) -> PyTensor

def eye(n: int, m: Optional[int] = None, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def arange(start: float, end: Optional[float] = None, step: float = 1.0, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def linspace(start: float, end: float, steps: int, dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def logspace(start: float, end: float, steps: int, base: float = 10.0, dtype: str = "float32", requires_grad: bool = False) -> PyTensor

def rand(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def randn(shape: List[int], dtype: str = "float32", requires_grad: bool = False) -> PyTensor
def randint(low: int, high: int, shape: List[int], dtype: str = "int64", requires_grad: bool = False) -> PyTensor
def randperm(n: int, dtype: str = "int64", requires_grad: bool = False) -> PyTensor

def from_numpy(array: np.ndarray) -> PyTensor
def as_tensor(data: Union[List, np.ndarray, PyTensor], dtype: Optional[str] = None) -> PyTensor
```

#### Mathematische Funktionen
```python
# Elementweise Operationen
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

# Lineare Algebra
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor

# Reduktionsoperationen
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

### Klassen

#### PyVariable
```python
class PyVariable:
    """Tensor-Wrapper mit Unterstützung für automatische Differenzierung"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    
    # Variable-Operationen (mit Autograd-Unterstützung)
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

### Funktionen

#### Gradientenberechnung
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

#### Kontext-Manager
```python
class no_grad:
    """Kontext-Manager zum Deaktivieren der Gradientenberechnung"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class enable_grad:
    """Kontext-Manager zum Aktivieren der Gradientenberechnung"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class set_grad_enabled:
    """Kontext-Manager zum Setzen der Gradientenberechnung aktiviert/deaktiviert"""
    def __init__(self, mode: bool)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
```

---

## rustorch.nn

### Basisklassen

#### Module
```python
class Module:
    """Basisklasse für alle neuronalen Netzwerk-Module"""
    
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

### Lineare Schichten

#### PyLinear
```python
class PyLinear(Module):
    """Lineare Transformation (vollständig verbundene) Schicht"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
```

### Faltungsschichten

#### PyConv2d
```python
class PyConv2d(Module):
    """2D-Faltungsschicht"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_parameters(self)
```

### Normalisierungsschichten

#### PyBatchNorm2d
```python
class PyBatchNorm2d(Module):
    """2D-Batch-Normalisierungsschicht"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_running_stats(self)
    def reset_parameters(self)
```

### Aktivierungsfunktionen

#### PyReLU
```python
class PyReLU(Module):
    """ReLU-Aktivierungsfunktion"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PySigmoid
```python
class PySigmoid(Module):
    """Sigmoid-Aktivierungsfunktion"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PyTanh
```python
class PyTanh(Module):
    """Hyperbolische Tangens-Aktivierungsfunktion"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

### Verlustfunktionen

#### PyMSELoss
```python
class PyMSELoss(Module):
    """Mittlerer quadratischer Fehler-Verlust"""
    
    def __init__(self, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### PyCrossEntropyLoss
```python
class PyCrossEntropyLoss(Module):
    """Kreuzentropie-Verlust"""
    
    def __init__(self, weight: Optional[PyTensor] = None, ignore_index: int = -100, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

### Container-Module

#### Sequential
```python
class Sequential(Module):
    """Sequenzieller Container für Module"""
    
    def __init__(self, *args)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Module
    def append(self, module: Module) -> Sequential
    def forward(self, input: PyTensor) -> PyTensor
```

---

## rustorch.optim

### Optimierer

#### PySGD
```python
class PySGD:
    """Stochastischer Gradientenabstieg-Optimierer"""
    
    def __init__(self, params: List[PyTensor], lr: float, momentum: float = 0, weight_decay: float = 0)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

#### PyAdam
```python
class PyAdam:
    """Adam-Optimierer"""
    
    def __init__(self, params: List[PyTensor], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

### Lernraten-Scheduler

#### PyStepLR
```python
class PyStepLR:
    """Schritt-Lernraten-Scheduler"""
    
    def __init__(self, optimizer: Union[PySGD, PyAdam], step_size: int, gamma: float = 0.1)
    def step(self, epoch: Optional[int] = None)
    def get_last_lr(self) -> List[float]
    def get_lr(self) -> List[float]
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

---

## rustorch.data

### Datensätze

#### PyTensorDataset
```python
class PyTensorDataset:
    """Datensatz, der Tensoren umschließt"""
    
    def __init__(self, *tensors: PyTensor)
    def __getitem__(self, index: int) -> List[PyTensor]
    def __len__(self) -> int
```

### Datenladen

#### PyDataLoader
```python
class PyDataLoader:
    """Daten-Loader für Datensätze"""
    
    def __init__(self, dataset: PyTensorDataset, batch_size: int = 1, 
                 shuffle: bool = False, num_workers: int = 0, drop_last: bool = False)
    def __iter__(self) -> Iterator[List[PyTensor]]
    def __len__(self) -> int
```

### Transformationen

#### PyTransform
```python
class PyTransform:
    """Datentransformation"""
    
    def __init__(self, name: str, params: Optional[Dict[str, float]] = None)
    def __call__(self, input: PyTensor) -> PyTensor
```

---

## rustorch.training

### High-Level Training-API

#### PyModel
```python
class PyModel:
    """High-Level Keras-artiges Modell"""
    
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
    """Training-Verlauf-Container"""
    
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

### Konfiguration

#### PyDistributedConfig
```python
class PyDistributedConfig:
    """Verteilte Training-Konfiguration"""
    
    def __init__(self, backend: str = "nccl", world_size: int = 1, rank: int = 0,
                 master_addr: str = "localhost", master_port: int = 29500)
    def backend(self) -> str
    def world_size(self) -> int
    def rank(self) -> int
```

### Verteiltes Training

#### PyDistributedDataParallel
```python
class PyDistributedDataParallel:
    """Verteilter Datenparallel-Wrapper"""
    
    def __init__(self, model: PyModel, device_ids: Optional[List[int]] = None)
    def forward(self, *inputs) -> PyTensor
    def sync_gradients(self)
    def module(self) -> PyModel
```

---

## rustorch.visualization

### Plotting

#### PyPlotter
```python
class PyPlotter:
    """Plotting-Hilfsprogramme"""
    
    def __init__(self, backend: str = "matplotlib", style: str = "default")
    def plot_training_history(self, history: PyTrainingHistory, save_path: Optional[str] = None)
    def plot_tensor_as_image(self, tensor: PyTensor, title: Optional[str] = None, save_path: Optional[str] = None)
    def line_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def scatter_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def histogram(self, data: List[float], bins: int = 30, title: Optional[str] = None, save_path: Optional[str] = None)
```

---

## rustorch.utils

### Modellverwaltung

#### Funktionen
```python
def save_model(model: PyModel, path: str)
def load_model(path: str) -> PyModel
def save_checkpoint(model: PyModel, optimizer: Union[PySGD, PyAdam], epoch: int, loss: float, path: str)
def load_checkpoint(path: str) -> Dict[str, Any]
```

### Performance

#### PyProfiler
```python
class PyProfiler:
    """Performance-Profiler"""
    
    def __init__(self)
    def start(self)
    def stop(self)
    def reset(self)
    def summary(self) -> str
    def profile(self) -> ContextManager
```

### Konfiguration

#### PyConfig
```python
class PyConfig:
    """Globale Konfigurationsverwaltung"""
    
    def __init__(self)
    def set(self, key: str, value: Any)
    def get(self, key: str) -> Any
    def reset(self)
```

---

## Fehlerbehandlung

### Exceptions

#### RusTorchError
Basis-Exception-Klasse für alle RusTorch-Fehler.

```python
class RusTorchError(Exception):
    """Basis-Exception für RusTorch-Fehler"""
    pass

class TensorError(RusTorchError):
    """Tensor-Operationsfehler"""
    pass

class ShapeError(RusTorchError):
    """Shape-Mismatch-Fehler"""
    pass

class DeviceError(RusTorchError):
    """Gerätebezogene Fehler"""
    pass

class SerializationError(RusTorchError):
    """Modell-Speichern/Laden-Fehler"""
    pass
```

---

## Typisierungshinweise

RusTorch bietet umfassende Typisierungshinweise-Unterstützung:

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

Diese API-Referenz deckt die Kernfunktionalität der RusTorch Python-Bindings ab. Für detaillierte Verwendungsbeispiele siehe das [examples/](../examples/) Verzeichnis.