# Référence API Python RusTorch

## Aperçu

Ce document fournit une référence API complète pour les bindings Python de RusTorch. Il couvre tous les modules, classes et fonctions en détail.

## Structure des Modules

```python
rustorch/
├── tensor          # Opérations et classes de tenseurs principales
├── autograd        # Système de différenciation automatique
├── nn              # Couches et fonctions de réseaux de neurones
├── optim           # Optimiseurs et planificateurs de taux d'apprentissage
├── data            # Chargement de données et jeux de données
├── training        # API d'entraînement de haut niveau
├── distributed     # Support d'entraînement distribué
├── visualization   # Visualisation et graphiques
└── utils           # Fonctions utilitaires
```

---

## rustorch.tensor

### Classes

#### PyTensor
```python
class PyTensor:
    """Classe de tableau multidimensionnel (tenseur) principal"""
    
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
    
    # Opérations arithmétiques
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # Opérations en place
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # Algèbre linéaire
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # Fonctions statistiques
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # Opérations de comparaison
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # Opérations d'indexation
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # Fonctions de conversion
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # Méthodes spéciales
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### Fonctions

#### Fonctions Fabrique
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

#### Fonctions Mathématiques
```python
# Opérations élément par élément
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

# Algèbre linéaire
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor

# Opérations de réduction
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
    """Enveloppe de tenseur supportant la différenciation automatique"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    
    # Opérations de variables (avec support autograd)
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

### Classes de Base

#### Module
```python
class Module:
    """Classe de base pour tous les modules de réseau de neurones"""
    
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

### Couches Linéaires

#### PyLinear
```python
class PyLinear(Module):
    """Couche de transformation linéaire (entièrement connectée)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
```

---

## rustorch.optim

### Optimiseurs

#### PySGD
```python
class PySGD:
    """Optimiseur de descente de gradient stochastique"""
    
    def __init__(self, params: List[PyTensor], lr: float, momentum: float = 0, weight_decay: float = 0)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

#### PyAdam
```python
class PyAdam:
    """Optimiseur Adam"""
    
    def __init__(self, params: List[PyTensor], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

---

## rustorch.data

### Jeux de Données

#### PyTensorDataset
```python
class PyTensorDataset:
    """Jeu de données enveloppant des tenseurs"""
    
    def __init__(self, *tensors: PyTensor)
    def __getitem__(self, index: int) -> List[PyTensor]
    def __len__(self) -> int
```

### Chargement de Données

#### PyDataLoader
```python
class PyDataLoader:
    """Chargeur de données pour jeux de données"""
    
    def __init__(self, dataset: PyTensorDataset, batch_size: int = 1, 
                 shuffle: bool = False, num_workers: int = 0, drop_last: bool = False)
    def __iter__(self) -> Iterator[List[PyTensor]]
    def __len__(self) -> int
```

---

## rustorch.training

### API d'Entraînement de Haut Niveau

#### PyModel
```python
class PyModel:
    """Modèle de haut niveau style Keras"""
    
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

### Gestion de Modèles

#### Fonctions
```python
def save_model(model: PyModel, path: str)
def load_model(path: str) -> PyModel
def save_checkpoint(model: PyModel, optimizer: Union[PySGD, PyAdam], epoch: int, loss: float, path: str)
def load_checkpoint(path: str) -> Dict[str, Any]
```

---

## Gestion d'Erreurs

### Exceptions

#### RusTorchError
Classe d'exception de base pour toutes les erreurs RusTorch.

```python
class RusTorchError(Exception):
    """Exception de base pour les erreurs RusTorch"""
    pass

class TensorError(RusTorchError):
    """Erreurs d'opérations de tenseur"""
    pass

class ShapeError(RusTorchError):
    """Erreurs de non-correspondance de forme"""
    pass

class DeviceError(RusTorchError):
    """Erreurs liées aux dispositifs"""
    pass

class SerializationError(RusTorchError):
    """Erreurs de sauvegarde/chargement de modèle"""
    pass
```

---

## Annotations de Type

RusTorch fournit un support complet d'annotations de type :

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

Cette référence API couvre la fonctionnalité principale des bindings Python de RusTorch. Pour des exemples d'utilisation détaillés, voir le répertoire [examples/](../examples/).
