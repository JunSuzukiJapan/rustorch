# Справочник API Python RusTorch

## Обзор

Этот документ предоставляет полный справочник API для Python-привязок RusTorch. Он охватывает все модули, классы и функции в деталях.

## Структура модулей

```python
rustorch/
├── tensor          # Основные операции с тензорами и классы
├── autograd        # Система автоматического дифференцирования
├── nn              # Слои нейронных сетей и функции
├── optim           # Оптимизаторы и планировщики скорости обучения
├── data            # Загрузка данных и наборы данных
├── training        # API высокого уровня для обучения
├── distributed     # Поддержка распределенного обучения
├── visualization   # Визуализация и графики
└── utils           # Вспомогательные функции
```

---

## rustorch.tensor

### Классы

#### PyTensor
```python
class PyTensor:
    """Основной класс многомерного массива (тензора)"""
    
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
    
    # Арифметические операции
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # Операции на месте
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # Линейная алгебра
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # Статистические функции
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # Операции сравнения
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # Операции индексирования
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # Функции преобразования
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # Специальные методы
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### Функции

#### Фабричные функции
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

#### Математические функции
```python
# Поэлементные операции
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

# Линейная алгебра
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor

# Операции редукции
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

### Классы

#### PyVariable
```python
class PyVariable:
    """Обертка тензора с поддержкой автоматического дифференцирования"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    
    # Операции с переменными (с поддержкой autograd)
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

### Функции

#### Вычисление градиентов
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

#### Менеджеры контекста
```python
class no_grad:
    """Менеджер контекста для отключения вычисления градиентов"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class enable_grad:
    """Менеджер контекста для включения вычисления градиентов"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class set_grad_enabled:
    """Менеджер контекста для установки вычисления градиентов включено/выключено"""
    def __init__(self, mode: bool)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
```

---

## rustorch.nn

### Базовые классы

#### Module
```python
class Module:
    """Базовый класс для всех модулей нейронной сети"""
    
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

### Линейные слои

#### PyLinear
```python
class PyLinear(Module):
    """Слой линейного преобразования (полносвязный)"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
```

### Сверточные слои

#### PyConv2d
```python
class PyConv2d(Module):
    """2D сверточный слой"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_parameters(self)
```

### Слои нормализации

#### PyBatchNorm2d
```python
class PyBatchNorm2d(Module):
    """2D слой пакетной нормализации"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_running_stats(self)
    def reset_parameters(self)
```

### Функции активации

#### PyReLU
```python
class PyReLU(Module):
    """Функция активации ReLU"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PySigmoid
```python
class PySigmoid(Module):
    """Сигмоидная функция активации"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PyTanh
```python
class PyTanh(Module):
    """Гиперболический тангенс функция активации"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

### Функции потерь

#### PyMSELoss
```python
class PyMSELoss(Module):
    """Среднеквадратичная ошибка"""
    
    def __init__(self, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### PyCrossEntropyLoss
```python
class PyCrossEntropyLoss(Module):
    """Кросс-энтропийная потеря"""
    
    def __init__(self, weight: Optional[PyTensor] = None, ignore_index: int = -100, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

### Контейнерные модули

#### Sequential
```python
class Sequential(Module):
    """Последовательный контейнер для модулей"""
    
    def __init__(self, *args)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Module
    def append(self, module: Module) -> Sequential
    def forward(self, input: PyTensor) -> PyTensor
```

---

## rustorch.optim

### Оптимизаторы

#### PySGD
```python
class PySGD:
    """Стохастический градиентный спуск оптимизатор"""
    
    def __init__(self, params: List[PyTensor], lr: float, momentum: float = 0, weight_decay: float = 0)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

#### PyAdam
```python
class PyAdam:
    """Оптимизатор Adam"""
    
    def __init__(self, params: List[PyTensor], lr: float = 0.001, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8)
    def step(self)
    def zero_grad(self)
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

### Планировщики скорости обучения

#### PyStepLR
```python
class PyStepLR:
    """Ступенчатый планировщик скорости обучения"""
    
    def __init__(self, optimizer: Union[PySGD, PyAdam], step_size: int, gamma: float = 0.1)
    def step(self, epoch: Optional[int] = None)
    def get_last_lr(self) -> List[float]
    def get_lr(self) -> List[float]
    def state_dict(self) -> Dict
    def load_state_dict(self, state_dict: Dict)
```

---

## rustorch.data

### Наборы данных

#### PyTensorDataset
```python
class PyTensorDataset:
    """Набор данных, оборачивающий тензоры"""
    
    def __init__(self, *tensors: PyTensor)
    def __getitem__(self, index: int) -> List[PyTensor]
    def __len__(self) -> int
```

### Загрузка данных

#### PyDataLoader
```python
class PyDataLoader:
    """Загрузчик данных для наборов данных"""
    
    def __init__(self, dataset: PyTensorDataset, batch_size: int = 1, 
                 shuffle: bool = False, num_workers: int = 0, drop_last: bool = False)
    def __iter__(self) -> Iterator[List[PyTensor]]
    def __len__(self) -> int
```

### Преобразования

#### PyTransform
```python
class PyTransform:
    """Преобразование данных"""
    
    def __init__(self, name: str, params: Optional[Dict[str, float]] = None)
    def __call__(self, input: PyTensor) -> PyTensor
```

---

## rustorch.training

### API высокого уровня для обучения

#### PyModel
```python
class PyModel:
    """Модель высокого уровня в стиле Keras"""
    
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
    """Контейнер истории обучения"""
    
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

### Конфигурация

#### PyDistributedConfig
```python
class PyDistributedConfig:
    """Конфигурация распределенного обучения"""
    
    def __init__(self, backend: str = "nccl", world_size: int = 1, rank: int = 0,
                 master_addr: str = "localhost", master_port: int = 29500)
    def backend(self) -> str
    def world_size(self) -> int
    def rank(self) -> int
```

### Распределенное обучение

#### PyDistributedDataParallel
```python
class PyDistributedDataParallel:
    """Обертка для распределенного параллелизма данных"""
    
    def __init__(self, model: PyModel, device_ids: Optional[List[int]] = None)
    def forward(self, *inputs) -> PyTensor
    def sync_gradients(self)
    def module(self) -> PyModel
```

---

## rustorch.visualization

### Построение графиков

#### PyPlotter
```python
class PyPlotter:
    """Утилиты для построения графиков"""
    
    def __init__(self, backend: str = "matplotlib", style: str = "default")
    def plot_training_history(self, history: PyTrainingHistory, save_path: Optional[str] = None)
    def plot_tensor_as_image(self, tensor: PyTensor, title: Optional[str] = None, save_path: Optional[str] = None)
    def line_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def scatter_plot(self, x_data: List[float], y_data: List[float], title: Optional[str] = None, save_path: Optional[str] = None)
    def histogram(self, data: List[float], bins: int = 30, title: Optional[str] = None, save_path: Optional[str] = None)
```

---

## rustorch.utils

### Управление моделями

#### Функции
```python
def save_model(model: PyModel, path: str)
def load_model(path: str) -> PyModel
def save_checkpoint(model: PyModel, optimizer: Union[PySGD, PyAdam], epoch: int, loss: float, path: str)
def load_checkpoint(path: str) -> Dict[str, Any]
```

### Производительность

#### PyProfiler
```python
class PyProfiler:
    """Профилировщик производительности"""
    
    def __init__(self)
    def start(self)
    def stop(self)
    def reset(self)
    def summary(self) -> str
    def profile(self) -> ContextManager
```

### Конфигурация

#### PyConfig
```python
class PyConfig:
    """Управление глобальной конфигурацией"""
    
    def __init__(self)
    def set(self, key: str, value: Any)
    def get(self, key: str) -> Any
    def reset(self)
```

---

## Обработка ошибок

### Исключения

#### RusTorchError
Базовый класс исключений для всех ошибок RusTorch.

```python
class RusTorchError(Exception):
    """Базовое исключение для ошибок RusTorch"""
    pass

class TensorError(RusTorchError):
    """Ошибки операций с тензорами"""
    pass

class ShapeError(RusTorchError):
    """Ошибки несоответствия формы"""
    pass

class DeviceError(RusTorchError):
    """Ошибки, связанные с устройствами"""
    pass

class SerializationError(RusTorchError):
    """Ошибки сохранения/загрузки модели"""
    pass
```

---

## Подсказки типов

RusTorch предоставляет полную поддержку подсказок типов:

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

Этот справочник API покрывает основную функциональность Python-привязок RusTorch. Для подробных примеров использования см. директорию [examples/](../examples/).