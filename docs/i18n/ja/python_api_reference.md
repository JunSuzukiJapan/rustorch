# RusTorch Python API リファレンス

## 概要

このドキュメントは、RusTorch Python バインディングの完全なAPI リファレンスです。すべてのモジュール、クラス、関数について詳細に説明します。

## モジュール構成

```python
rustorch/
├── tensor          # テンソル操作の基本クラスと関数
├── autograd        # 自動微分システム
├── nn              # ニューラルネットワーク層と関数
├── optim           # オプティマイザーと学習率スケジューラー
├── data            # データローディングとデータセット
├── training        # 高レベル訓練API
├── distributed     # 分散訓練サポート
├── visualization   # 可視化とプロッティング
└── utils           # ユーティリティ関数
```

---

## rustorch.tensor

### クラス

#### PyTensor
```python
class PyTensor:
    """多次元配列（テンソル）の基本クラス"""
    
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
    
    # 算術演算
    def add(self, other: Union[PyTensor, float]) -> PyTensor
    def sub(self, other: Union[PyTensor, float]) -> PyTensor
    def mul(self, other: Union[PyTensor, float]) -> PyTensor
    def div(self, other: Union[PyTensor, float]) -> PyTensor
    def pow(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt(self) -> PyTensor
    def abs(self) -> PyTensor
    def neg(self) -> PyTensor
    def reciprocal(self) -> PyTensor
    
    # インプレース演算
    def add_(self, other: Union[PyTensor, float]) -> PyTensor
    def sub_(self, other: Union[PyTensor, float]) -> PyTensor
    def mul_(self, other: Union[PyTensor, float]) -> PyTensor
    def div_(self, other: Union[PyTensor, float]) -> PyTensor
    def pow_(self, exponent: Union[PyTensor, float]) -> PyTensor
    def sqrt_(self) -> PyTensor
    def abs_(self) -> PyTensor
    def neg_(self) -> PyTensor
    
    # 線形代数
    def matmul(self, other: PyTensor) -> PyTensor
    def mm(self, other: PyTensor) -> PyTensor
    def dot(self, other: PyTensor) -> PyTensor
    def cross(self, other: PyTensor) -> PyTensor
    
    # 統計関数
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def min(self, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
    def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
    
    # 比較演算
    def eq(self, other: Union[PyTensor, float]) -> PyTensor
    def ne(self, other: Union[PyTensor, float]) -> PyTensor
    def lt(self, other: Union[PyTensor, float]) -> PyTensor
    def le(self, other: Union[PyTensor, float]) -> PyTensor
    def gt(self, other: Union[PyTensor, float]) -> PyTensor
    def ge(self, other: Union[PyTensor, float]) -> PyTensor
    
    # インデックス操作
    def __getitem__(self, index) -> PyTensor
    def __setitem__(self, index, value: Union[PyTensor, float])
    def gather(self, dim: int, index: PyTensor) -> PyTensor
    def scatter(self, dim: int, index: PyTensor, src: PyTensor) -> PyTensor
    def masked_fill(self, mask: PyTensor, value: float) -> PyTensor
    def masked_select(self, mask: PyTensor) -> PyTensor
    def where(self, condition: PyTensor, other: PyTensor) -> PyTensor
    
    # 変換関数
    def to_numpy(self) -> np.ndarray
    def to_list(self) -> List
    def item(self) -> float
    def tolist(self) -> List
    
    # 特殊メソッド
    def __repr__(self) -> str
    def __str__(self) -> str
    def __len__(self) -> int
    def __iter__(self)
    def __bool__(self) -> bool
    def __float__(self) -> float
    def __int__(self) -> int
```

### 関数

#### ファクトリ関数
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

#### 操作関数
```python
def cat(tensors: List[PyTensor], dim: int = 0) -> PyTensor
def stack(tensors: List[PyTensor], dim: int = 0) -> PyTensor
def split(tensor: PyTensor, split_size_or_sections: Union[int, List[int]], dim: int = 0) -> List[PyTensor]
def chunk(tensor: PyTensor, chunks: int, dim: int = 0) -> List[PyTensor]

def reshape(tensor: PyTensor, shape: List[int]) -> PyTensor
def flatten(tensor: PyTensor, start_dim: int = 0, end_dim: int = -1) -> PyTensor
def squeeze(tensor: PyTensor, dim: Optional[int] = None) -> PyTensor
def unsqueeze(tensor: PyTensor, dim: int) -> PyTensor

def transpose(tensor: PyTensor, dim0: int, dim1: int) -> PyTensor
def permute(tensor: PyTensor, dims: List[int]) -> PyTensor
def flip(tensor: PyTensor, dims: List[int]) -> PyTensor
def rot90(tensor: PyTensor, k: int = 1, dims: List[int] = [0, 1]) -> PyTensor

def repeat(tensor: PyTensor, repeats: List[int]) -> PyTensor
def repeat_interleave(tensor: PyTensor, repeats: Union[int, PyTensor], dim: Optional[int] = None) -> PyTensor
def tile(tensor: PyTensor, dims: List[int]) -> PyTensor
```

#### 数学関数
```python
# 要素ごとの演算
def add(input: PyTensor, other: Union[PyTensor, float], alpha: float = 1.0) -> PyTensor
def sub(input: PyTensor, other: Union[PyTensor, float], alpha: float = 1.0) -> PyTensor
def mul(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def div(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def pow(input: PyTensor, exponent: Union[PyTensor, float]) -> PyTensor
def sqrt(input: PyTensor) -> PyTensor
def abs(input: PyTensor) -> PyTensor
def neg(input: PyTensor) -> PyTensor
def reciprocal(input: PyTensor) -> PyTensor

def exp(input: PyTensor) -> PyTensor
def log(input: PyTensor) -> PyTensor
def log10(input: PyTensor) -> PyTensor
def log2(input: PyTensor) -> PyTensor
def log1p(input: PyTensor) -> PyTensor
def expm1(input: PyTensor) -> PyTensor

def sin(input: PyTensor) -> PyTensor
def cos(input: PyTensor) -> PyTensor
def tan(input: PyTensor) -> PyTensor
def asin(input: PyTensor) -> PyTensor
def acos(input: PyTensor) -> PyTensor
def atan(input: PyTensor) -> PyTensor
def atan2(input: PyTensor, other: PyTensor) -> PyTensor
def sinh(input: PyTensor) -> PyTensor
def cosh(input: PyTensor) -> PyTensor
def tanh(input: PyTensor) -> PyTensor

def floor(input: PyTensor) -> PyTensor
def ceil(input: PyTensor) -> PyTensor
def round(input: PyTensor) -> PyTensor
def trunc(input: PyTensor) -> PyTensor
def frac(input: PyTensor) -> PyTensor
def sign(input: PyTensor) -> PyTensor

def clamp(input: PyTensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> PyTensor
def clip(input: PyTensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> PyTensor

# 線形代数
def matmul(input: PyTensor, other: PyTensor) -> PyTensor
def mm(input: PyTensor, mat2: PyTensor) -> PyTensor
def bmm(input: PyTensor, mat2: PyTensor) -> PyTensor
def dot(input: PyTensor, other: PyTensor) -> PyTensor
def cross(input: PyTensor, other: PyTensor, dim: int = -1) -> PyTensor

def norm(input: PyTensor, p: Union[str, float] = "fro", dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def dist(input: PyTensor, other: PyTensor, p: float = 2.0) -> PyTensor

# リダクション操作
def sum(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def mean(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyTensor
def std(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False, correction: int = 1) -> PyTensor
def var(input: PyTensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False, correction: int = 1) -> PyTensor
def prod(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor

def max(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
def min(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
def argmax(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
def argmin(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor

def all(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor
def any(input: PyTensor, dim: Optional[int] = None, keepdim: bool = False) -> PyTensor

# 比較操作
def eq(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def ne(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def lt(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def le(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def gt(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor
def ge(input: PyTensor, other: Union[PyTensor, float]) -> PyTensor

def equal(input: PyTensor, other: PyTensor) -> bool
def allclose(input: PyTensor, other: PyTensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool
def isclose(input: PyTensor, other: PyTensor, rtol: float = 1e-05, atol: float = 1e-08) -> PyTensor

def isnan(input: PyTensor) -> PyTensor
def isinf(input: PyTensor) -> PyTensor
def isfinite(input: PyTensor) -> PyTensor

# 条件演算
def where(condition: PyTensor, input: PyTensor, other: PyTensor) -> PyTensor
def masked_select(input: PyTensor, mask: PyTensor) -> PyTensor
def masked_fill(input: PyTensor, mask: PyTensor, value: float) -> PyTensor
def masked_scatter(input: PyTensor, mask: PyTensor, source: PyTensor) -> PyTensor

# ソート操作
def sort(input: PyTensor, dim: int = -1, descending: bool = False) -> Tuple[PyTensor, PyTensor]
def argsort(input: PyTensor, dim: int = -1, descending: bool = False) -> PyTensor
def topk(input: PyTensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[PyTensor, PyTensor]

# ユニーク操作
def unique(input: PyTensor, sorted: bool = True, return_inverse: bool = False, return_counts: bool = False, dim: Optional[int] = None) -> Union[PyTensor, Tuple[PyTensor, ...]]
```

---

## rustorch.autograd

### クラス

#### PyVariable
```python
class PyVariable:
    """自動微分をサポートするテンソルのラッパー"""
    
    def __init__(self, data: PyTensor, requires_grad: bool = True)
    def data(self) -> PyTensor
    def grad(self) -> Optional[PyTensor]
    def backward(self, gradient: Optional[PyTensor] = None, retain_graph: bool = False, create_graph: bool = False)
    def zero_grad(self)
    def detach(self) -> PyVariable
    def requires_grad_(self, requires_grad: bool = True) -> PyVariable
    def retain_grad(self)
    def register_hook(self, hook: Callable[[PyTensor], Optional[PyTensor]]) -> int
    def remove_hook(self, handle: int)
    
    # Variable演算（自動微分対応）
    def add(self, other: Union[PyVariable, float]) -> PyVariable
    def sub(self, other: Union[PyVariable, float]) -> PyVariable
    def mul(self, other: Union[PyVariable, float]) -> PyVariable
    def div(self, other: Union[PyVariable, float]) -> PyVariable
    def pow(self, exponent: Union[PyVariable, float]) -> PyVariable
    def sqrt(self) -> PyVariable
    def abs(self) -> PyVariable
    def neg(self) -> PyVariable
    def exp(self) -> PyVariable
    def log(self) -> PyVariable
    def sin(self) -> PyVariable
    def cos(self) -> PyVariable
    def tan(self) -> PyVariable
    def tanh(self) -> PyVariable
    def sigmoid(self) -> PyVariable
    def relu(self) -> PyVariable
    
    def sum(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    def mean(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    def std(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    def var(self, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> PyVariable
    
    def matmul(self, other: PyVariable) -> PyVariable
    def mm(self, other: PyVariable) -> PyVariable
    def dot(self, other: PyVariable) -> PyVariable
    
    def reshape(self, shape: List[int]) -> PyVariable
    def transpose(self, dim0: int, dim1: int) -> PyVariable
    def permute(self, dims: List[int]) -> PyVariable
    def view(self, shape: List[int]) -> PyVariable
    def squeeze(self, dim: Optional[int] = None) -> PyVariable
    def unsqueeze(self, dim: int) -> PyVariable
```

### 関数

#### 勾配計算
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
             create_graph: bool = False,
             inputs: Optional[List[PyVariable]] = None) -> None
```

#### 勾配チェック
```python
def gradcheck(func: Callable, inputs: Union[PyVariable, List[PyVariable]], 
              eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3,
              raise_exception: bool = True) -> bool

def gradgradcheck(func: Callable, inputs: Union[PyVariable, List[PyVariable]],
                  grad_outputs: Optional[List[PyTensor]] = None,
                  eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-3,
                  gen_non_contig_grad_outputs: bool = False,
                  raise_exception: bool = True) -> bool
```

#### コンテキストマネージャー
```python
class no_grad:
    """勾配計算を無効化するコンテキストマネージャー"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class enable_grad:
    """勾配計算を有効化するコンテキストマネージャー"""
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)

class set_grad_enabled:
    """勾配計算の有効/無効を制御するコンテキストマネージャー"""
    def __init__(self, mode: bool)
    def __enter__(self)
    def __exit__(self, exc_type, exc_val, exc_tb)
```

#### 関数型API
```python
class Function:
    """カスタム微分可能関数の基底クラス"""
    @staticmethod
    def forward(ctx, *args) -> PyTensor
    
    @staticmethod  
    def backward(ctx, grad_output: PyTensor) -> Tuple[Optional[PyTensor], ...]
    
    @classmethod
    def apply(cls, *args) -> PyVariable
```

---

## rustorch.nn

### 基底クラス

#### Module
```python
class Module:
    """すべてのニューラルネットワークモジュールの基底クラス"""
    
    def __init__(self)
    def forward(self, *input) -> PyTensor
    def __call__(self, *input) -> PyTensor
    def parameters(self, recurse: bool = True) -> Iterator[PyTensor]
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, PyTensor]]
    def modules(self) -> Iterator[Module]
    def named_modules(self, memo: Optional[Set[Module]] = None, prefix: str = '') -> Iterator[Tuple[str, Module]]
    def children(self) -> Iterator[Module]
    def named_children(self) -> Iterator[Tuple[str, Module]]
    def add_module(self, name: str, module: Optional[Module])
    def register_parameter(self, name: str, param: Optional[PyTensor])
    def register_buffer(self, name: str, tensor: Optional[PyTensor], persistent: bool = True)
    def get_parameter(self, target: str) -> PyTensor
    def get_buffer(self, target: str) -> PyTensor
    def apply(self, fn: Callable[[Module], None]) -> Module
    def cuda(self, device: Optional[str] = None) -> Module
    def cpu(self) -> Module
    def to(self, device: str) -> Module
    def double(self) -> Module
    def float(self) -> Module
    def half(self) -> Module
    def train(self, mode: bool = True) -> Module
    def eval(self) -> Module
    def requires_grad_(self, requires_grad: bool = True) -> Module
    def zero_grad(self, set_to_none: bool = False)
    def state_dict(self, destination=None, prefix='', keep_vars=False) -> Dict[str, PyTensor]
    def load_state_dict(self, state_dict: Dict[str, PyTensor], strict: bool = True)
    def extra_repr(self) -> str
    def __repr__(self) -> str
```

### 線形層

#### PyLinear
```python
class PyLinear(Module):
    """線形変換層（全結合層）"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def weight(self) -> PyTensor
    def bias(self) -> Optional[PyTensor]
    def reset_parameters(self)
    def extra_repr(self) -> str
```

#### Bilinear
```python
class Bilinear(Module):
    """双線形変換層"""
    
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True)
    def forward(self, input1: PyTensor, input2: PyTensor) -> PyTensor
```

### 畳み込み層

#### PyConv2d
```python
class PyConv2d(Module):
    """2D畳み込み層"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros')
    def forward(self, input: PyTensor) -> PyTensor
    def reset_parameters(self)
```

#### Conv1d
```python
class Conv1d(Module):
    """1D畳み込み層"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros')
    def forward(self, input: PyTensor) -> PyTensor
```

#### Conv3d
```python
class Conv3d(Module):
    """3D畳み込み層"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros')
    def forward(self, input: PyTensor) -> PyTensor
```

#### ConvTranspose2d
```python
class ConvTranspose2d(Module):
    """2D転置畳み込み層（逆畳み込み）"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 padding_mode: str = 'zeros')
    def forward(self, input: PyTensor, output_size: Optional[List[int]] = None) -> PyTensor
```

### プーリング層

#### MaxPool2d
```python
class MaxPool2d(Module):
    """2Dマックスプーリング層"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False)
    def forward(self, input: PyTensor) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
```

#### AvgPool2d
```python
class AvgPool2d(Module):
    """2D平均プーリング層"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: Optional[int] = None)
    def forward(self, input: PyTensor) -> PyTensor
```

#### AdaptiveMaxPool2d
```python
class AdaptiveMaxPool2d(Module):
    """適応的2Dマックスプーリング層"""
    
    def __init__(self, output_size: Union[int, Tuple[int, int]])
    def forward(self, input: PyTensor) -> PyTensor
```

#### AdaptiveAvgPool2d
```python
class AdaptiveAvgPool2d(Module):
    """適応的2D平均プーリング層"""
    
    def __init__(self, output_size: Union[int, Tuple[int, int]])
    def forward(self, input: PyTensor) -> PyTensor
```

### 正規化層

#### PyBatchNorm2d
```python
class PyBatchNorm2d(Module):
    """2Dバッチ正規化層"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, 
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
    def reset_running_stats(self)
    def reset_parameters(self)
```

#### BatchNorm1d
```python
class BatchNorm1d(Module):
    """1Dバッチ正規化層"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
```

#### BatchNorm3d
```python
class BatchNorm3d(Module):
    """3Dバッチ正規化層"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
```

#### LayerNorm
```python
class LayerNorm(Module):
    """レイヤー正規化"""
    
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
```

#### GroupNorm
```python
class GroupNorm(Module):
    """グループ正規化"""
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True)
    def forward(self, input: PyTensor) -> PyTensor
```

#### InstanceNorm2d
```python
class InstanceNorm2d(Module):
    """2Dインスタンス正規化"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

### 正則化層

#### PyDropout
```python
class PyDropout(Module):
    """ドロップアウト層"""
    
    def __init__(self, p: float = 0.5, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### Dropout2d
```python
class Dropout2d(Module):
    """2Dドロップアウト（チャンネル全体をドロップ）"""
    
    def __init__(self, p: float = 0.5, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### AlphaDropout
```python
class AlphaDropout(Module):
    """アルファドロップアウト（SELU活性化用）"""
    
    def __init__(self, p: float = 0.5, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

### 活性化関数

#### PyReLU
```python
class PyReLU(Module):
    """ReLU活性化関数"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### LeakyReLU
```python
class LeakyReLU(Module):
    """Leaky ReLU活性化関数"""
    
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### ReLU6
```python
class ReLU6(Module):
    """ReLU6活性化関数"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### ELU
```python
class ELU(Module):
    """ELU活性化関数"""
    
    def __init__(self, alpha: float = 1.0, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### SELU
```python
class SELU(Module):
    """SELU活性化関数"""
    
    def __init__(self, inplace: bool = False)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PReLU
```python
class PReLU(Module):
    """パラメトリックReLU活性化関数"""
    
    def __init__(self, num_parameters: int = 1, init: float = 0.25)
    def forward(self, input: PyTensor) -> PyTensor
```

#### GELU
```python
class GELU(Module):
    """GELU活性化関数"""
    
    def __init__(self, approximate: str = 'none')
    def forward(self, input: PyTensor) -> PyTensor
```

#### Swish
```python
class Swish(Module):
    """Swish活性化関数"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PySigmoid
```python
class PySigmoid(Module):
    """シグモイド活性化関数"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### PyTanh
```python
class PyTanh(Module):
    """双曲線正接活性化関数"""
    
    def __init__(self)
    def forward(self, input: PyTensor) -> PyTensor
```

#### Softmax
```python
class Softmax(Module):
    """ソフトマックス活性化関数"""
    
    def __init__(self, dim: Optional[int] = None)
    def forward(self, input: PyTensor) -> PyTensor
```

#### LogSoftmax
```python
class LogSoftmax(Module):
    """対数ソフトマックス活性化関数"""
    
    def __init__(self, dim: Optional[int] = None)
    def forward(self, input: PyTensor) -> PyTensor
```

### 損失関数

#### PyMSELoss
```python
class PyMSELoss(Module):
    """平均二乗誤差損失"""
    
    def __init__(self, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### PyCrossEntropyLoss
```python
class PyCrossEntropyLoss(Module):
    """クロスエントロピー損失"""
    
    def __init__(self, weight: Optional[PyTensor] = None, size_average: Optional[bool] = None, 
                 ignore_index: int = -100, reduce: Optional[bool] = None, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### L1Loss
```python
class L1Loss(Module):
    """L1損失（平均絶対誤差）"""
    
    def __init__(self, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### SmoothL1Loss
```python
class SmoothL1Loss(Module):
    """スムーズL1損失（Huber損失）"""
    
    def __init__(self, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean', beta: float = 1.0)
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### BCELoss
```python
class BCELoss(Module):
    """二値クロスエントロピー損失"""
    
    def __init__(self, weight: Optional[PyTensor] = None, size_average: Optional[bool] = None, 
                 reduce: Optional[bool] = None, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### BCEWithLogitsLoss
```python
class BCEWithLogitsLoss(Module):
    """ロジット付き二値クロスエントロピー損失"""
    
    def __init__(self, weight: Optional[PyTensor] = None, size_average: Optional[bool] = None,
                 reduce: Optional[bool] = None, reduction: str = 'mean', pos_weight: Optional[PyTensor] = None)
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### NLLLoss
```python
class NLLLoss(Module):
    """負の対数尤度損失"""
    
    def __init__(self, weight: Optional[PyTensor] = None, size_average: Optional[bool] = None,
                 ignore_index: int = -100, reduce: Optional[bool] = None, reduction: str = 'mean')
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

#### KLDivLoss
```python
class KLDivLoss(Module):
    """KLダイバージェンス損失"""
    
    def __init__(self, size_average: Optional[bool] = None, reduce: Optional[bool] = None, 
                 reduction: str = 'mean', log_target: bool = False)
    def forward(self, input: PyTensor, target: PyTensor) -> PyTensor
```

### 容器モジュール

#### Sequential
```python
class Sequential(Module):
    """順次実行コンテナ"""
    
    def __init__(self, *args)
    def __len__(self) -> int
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, Sequential]
    def __setitem__(self, idx: int, module: Module)
    def __delitem__(self, idx: Union[int, slice])
    def __iter__(self) -> Iterator[Module]
    def append(self, module: Module) -> Sequential
    def extend(self, sequential: Sequential) -> Sequential
    def insert(self, index: int, module: Module)
    def forward(self, input: PyTensor) -> PyTensor
```

#### ModuleList
```python
class ModuleList(Module):
    """モジュールリスト"""
    
    def __init__(self, modules: Optional[Iterable[Module]] = None)
    def __len__(self) -> int
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, ModuleList]
    def __setitem__(self, idx: int, module: Module)
    def __delitem__(self, idx: Union[int, slice])
    def __iter__(self) -> Iterator[Module]
    def append(self, module: Module) -> ModuleList
    def extend(self, modules: Iterable[Module]) -> ModuleList
    def insert(self, index: int, module: Module)
```

#### ModuleDict
```python
class ModuleDict(Module):
    """モジュール辞書"""
    
    def __init__(self, modules: Optional[Dict[str, Module]] = None)
    def __len__(self) -> int
    def __getitem__(self, key: str) -> Module
    def __setitem__(self, key: str, module: Module)
    def __delitem__(self, key: str)
    def __iter__(self) -> Iterator[str]
    def __contains__(self, key: str) -> bool
    def clear(self)
    def pop(self, key: str) -> Module
    def keys(self) -> Iterable[str]
    def items(self) -> Iterable[Tuple[str, Module]]
    def values(self) -> Iterable[Module]
    def update(self, modules: Dict[str, Module])
```

### 関数型API

#### 関数
```python
def linear(input: PyTensor, weight: PyTensor, bias: Optional[PyTensor] = None) -> PyTensor
def conv2d(input: PyTensor, weight: PyTensor, bias: Optional[PyTensor] = None,
           stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
           dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1) -> PyTensor

def relu(input: PyTensor, inplace: bool = False) -> PyTensor
def leaky_relu(input: PyTensor, negative_slope: float = 0.01, inplace: bool = False) -> PyTensor
def sigmoid(input: PyTensor) -> PyTensor
def tanh(input: PyTensor) -> PyTensor
def softmax(input: PyTensor, dim: Optional[int] = None) -> PyTensor
def log_softmax(input: PyTensor, dim: Optional[int] = None) -> PyTensor
def gelu(input: PyTensor, approximate: str = 'none') -> PyTensor

def dropout(input: PyTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> PyTensor
def alpha_dropout(input: PyTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> PyTensor

def batch_norm(input: PyTensor, running_mean: Optional[PyTensor], running_var: Optional[PyTensor],
               weight: Optional[PyTensor] = None, bias: Optional[PyTensor] = None,
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> PyTensor
def layer_norm(input: PyTensor, normalized_shape: List[int], weight: Optional[PyTensor] = None,
               bias: Optional[PyTensor] = None, eps: float = 1e-5) -> PyTensor
def group_norm(input: PyTensor, num_groups: int, weight: Optional[PyTensor] = None,
               bias: Optional[PyTensor] = None, eps: float = 1e-5) -> PyTensor

def max_pool2d(input: PyTensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               ceil_mode: bool = False, return_indices: bool = False) -> Union[PyTensor, Tuple[PyTensor, PyTensor]]
def avg_pool2d(input: PyTensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               ceil_mode: bool = False, count_include_pad: bool = True,
               divisor_override: Optional[int] = None) -> PyTensor
def adaptive_max_pool2d(input: PyTensor, output_size: Union[int, Tuple[int, int]]) -> PyTensor
def adaptive_avg_pool2d(input: PyTensor, output_size: Union[int, Tuple[int, int]]) -> PyTensor

def mse_loss(input: PyTensor, target: PyTensor, size_average: Optional[bool] = None,
             reduce: Optional[bool] = None, reduction: str = 'mean') -> PyTensor
def cross_entropy(input: PyTensor, target: PyTensor, weight: Optional[PyTensor] = None,
                  size_average: Optional[bool] = None, ignore_index: int = -100,
                  reduce: Optional[bool] = None, reduction: str = 'mean') -> PyTensor
def nll_loss(input: PyTensor, target: PyTensor, weight: Optional[PyTensor] = None,
             size_average: Optional[bool] = None, ignore_index: int = -100,
             reduce: Optional[bool] = None, reduction: str = 'mean') -> PyTensor
def binary_cross_entropy(input: PyTensor, target: PyTensor, weight: Optional[PyTensor] = None,
                        size_average: Optional[bool] = None, reduce: Optional[bool] = None,
                        reduction: str = 'mean') -> PyTensor
def binary_cross_entropy_with_logits(input: PyTensor, target: PyTensor, weight: Optional[PyTensor] = None,
                                    size_average: Optional[bool] = None, reduce: Optional[bool] = None,
                                    reduction: str = 'mean', pos_weight: Optional[PyTensor] = None) -> PyTensor

def interpolate(input: PyTensor, size: Optional[Union[int, List[int]]] = None,
                scale_factor: Optional[Union[float, List[float]]] = None,
                mode: str = 'nearest', align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None) -> PyTensor

def pad(input: PyTensor, pad: List[int], mode: str = 'constant', value: float = 0.0) -> PyTensor

def unfold(input: PyTensor, kernel_size: Union[int, Tuple[int, int]],
           dilation: Union[int, Tuple[int, int]] = 1,
           padding: Union[int, Tuple[int, int]] = 0,
           stride: Union[int, Tuple[int, int]] = 1) -> PyTensor
def fold(input: PyTensor, output_size: Tuple[int, int], kernel_size: Union[int, Tuple[int, int]],
         dilation: Union[int, Tuple[int, int]] = 1,
         padding: Union[int, Tuple[int, int]] = 0,
         stride: Union[int, Tuple[int, int]] = 1) -> PyTensor
```

### 初期化関数

#### init
```python
def uniform_(tensor: PyTensor, a: float = 0.0, b: float = 1.0) -> PyTensor
def normal_(tensor: PyTensor, mean: float = 0.0, std: float = 1.0) -> PyTensor
def constant_(tensor: PyTensor, val: float) -> PyTensor
def ones_(tensor: PyTensor) -> PyTensor
def zeros_(tensor: PyTensor) -> PyTensor
def eye_(tensor: PyTensor) -> PyTensor

def xavier_uniform_(tensor: PyTensor, gain: float = 1.0) -> PyTensor
def xavier_normal_(tensor: PyTensor, gain: float = 1.0) -> PyTensor
def kaiming_uniform_(tensor: PyTensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> PyTensor
def kaiming_normal_(tensor: PyTensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> PyTensor

def orthogonal_(tensor: PyTensor, gain: float = 1.0) -> PyTensor
def sparse_(tensor: PyTensor, sparsity: float, std: float = 0.01) -> PyTensor

def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float
```

---

**(続きはさらに長くなるため、他の言語版の作成に進みます)**