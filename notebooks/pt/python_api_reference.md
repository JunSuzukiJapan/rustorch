# Referência da API Python RusTorch

## Visão Geral

Este documento fornece uma referência completa da API para os bindings Python do RusTorch. Abrange todos os módulos, classes e funções em detalhes.

## Estrutura dos Módulos

```python
rustorch/
├── tensor          # Operações e classes de tensor principais
├── autograd        # Sistema de diferenciação automática
├── nn              # Camadas e funções de redes neurais
├── optim           # Otimizadores e programadores de taxa de aprendizado
├── data            # Carregamento de dados e conjuntos de dados
├── training        # API de treinamento de alto nível
├── distributed     # Suporte para treinamento distribuído
├── visualization   # Visualização e gráficos
└── utils           # Funções utilitárias
```

---

## rustorch.tensor

### Classes

#### PyTensor
```python
class PyTensor:
    """Classe principal de array multidimensional (tensor)"""
    
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
    
    # Operações aritméticas
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # Operações in-place
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # Álgebra linear
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # Funções estatísticas
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # Operações de comparação
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # Operações de indexação
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # Funções de conversão
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # Métodos especiais
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### Funções

#### Funções de Fábrica
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

#### Funções Matemáticas
```python
# Operações elemento por elemento
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

# Álgebra linear
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor

# Operações de redução
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

### Classes

#### PyVariable
```python
class PyVariable:
    """Wrapper de tensor com suporte para diferenciação automática"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    
    # Operações de variáveis (com suporte autograd)
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

---

## rustorch.nn

### Classes Base

#### Module
```python
class Module:
    """Classe base para todos os módulos de rede neural"""
    
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

### Camadas Lineares

#### PyLinear
```python
class PyLinear(Module):
    """Camada de transformação linear (totalmente conectada)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
```

---

## rustorch.optim

### Otimizadores

#### PySGD
```python
class PySGD:
    """Otimizador de descida do gradiente estocástico"""
    
    def __init__(self, params: List[PyTensor], lr: float, momentum: float = 0, weight_decay: float = 0)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

#### PyAdam
```python
class PyAdam:
    """Otimizador Adam"""
    
    def __init__(self, params: List[PyTensor], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

---

## rustorch.data

### Conjuntos de Dados

#### PyTensorDataset
```python
class PyTensorDataset:
    """Conjunto de dados que envolve tensores"""
    
    def __init__(self, *tensors: PyTensor)
    def __getitem__(self, index: int) -> List[PyTensor]
    def __len__(self) -> int
```

### Carregamento de Dados

#### PyDataLoader
```python
class PyDataLoader:
    """Carregador de dados para conjuntos de dados"""
    
    def __init__(self, dataset: PyTensorDataset, batch_size: int = 1, 
                 shuffle: bool = False, num_workers: int = 0, drop_last: bool = False)
    def __iter__(self) -> Iterator[List[PyTensor]]
    def __len__(self) -> int
```

---

## rustorch.training

### API de Treinamento de Alto Nível

#### PyModel
```python
class PyModel:
    """Modelo de alto nível estilo Keras"""
    
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

---

## rustorch.utils

### Gerência de Modelos

#### Funções
```python
def save_model(model: PyModel, path: str)
def load_model(path: str) -> PyModel
def save_checkpoint(model: PyModel, optimizer: Union[PySGD, PyAdam], epoch: int, loss: float, path: str)
def load_checkpoint(path: str) -> Dict[str, Any]
```

---

## Tratamento de Erros

### Exceções

#### RusTorchError
Classe base de exceção para todos os erros do RusTorch.

```python
class RusTorchError(Exception):
    """Exceção base para erros do RusTorch"""
    pass

class TensorError(RusTorchError):
    """Erros de operações de tensor"""
    pass

class ShapeError(RusTorchError):
    """Erros de incompatibilidade de forma"""
    pass

class DeviceError(RusTorchError):
    """Erros relacionados ao dispositivo"""
    pass

class SerializationError(RusTorchError):
    """Erros de salvamento/carregamento de modelo"""
    pass
```

---

## Dicas de Tipo

RusTorch fornece suporte abrangente para dicas de tipo:

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

Esta referência da API abrange a funcionalidade principal dos bindings Python do RusTorch. Para exemplos de uso detalhados, consulte o diretório [examples/](../examples/).