# Phase 3 Implementation Plan: Complete Neural Network Training System

## 概要

Phase 2のVariable・Linear基盤を拡張し、完全なニューラルネットワークトレーニングシステムを実装します。

## Phase 3の目標

```python
# 目標: 完全なニューラルネットワークトレーニング
import rustorch

# データとラベル
X = rustorch.Variable(rustorch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
y = rustorch.Variable(rustorch.tensor([[1.0], [0.0]]), requires_grad=False)

# モデル構築
linear1 = rustorch.Linear(2, 4, True)
relu = rustorch.ReLU()
linear2 = rustorch.Linear(4, 1, True)
sigmoid = rustorch.Sigmoid()

# 順伝播
h1 = linear1(X)
h1_relu = relu(h1)
h2 = linear2(h1_relu)
output = sigmoid(h2)

# Loss計算
criterion = rustorch.MSELoss()
loss = criterion(output, y)

# オプティマイザー
optimizer = rustorch.SGD([linear1.weight, linear1.bias, linear2.weight, linear2.bias], lr=0.01)

# トレーニングループ
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    h1 = linear1(X)
    h1_relu = relu(h1)
    h2 = linear2(h1_relu)
    output = sigmoid(h2)

    # Loss計算
    loss = criterion(output, y)

    # Backward pass
    loss.backward()

    # パラメータ更新
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.data}")
```

## 実装対象クラス

### 1. Optimizer クラス群

#### SGD (Stochastic Gradient Descent)

**RusTorch API分析:**
```rust
pub struct SGD<T> {
    parameters: Vec<Variable<T>>,
    learning_rate: T,
    momentum: Option<T>,
    weight_decay: Option<T>,
    dampening: Option<T>,
    nesterov: bool,
    velocity: Vec<Option<Tensor<T>>>,
}

impl SGD<T> {
    fn new(parameters: Vec<Variable<T>>, learning_rate: T) -> Self
    fn new_with_momentum(parameters: Vec<Variable<T>>, learning_rate: T, momentum: T) -> Self
    fn zero_grad(&self)
    fn step(&mut self)
    fn learning_rate(&self) -> T
    fn set_learning_rate(&mut self, lr: T)
}
```

**Python実装ターゲット:**
```python
class SGD:
    def __init__(self, parameters: List[Variable], lr: float, momentum: float = 0.0)
    def zero_grad(self) -> None
    def step(self) -> None
    @property
    def lr(self) -> float
    def set_lr(self, lr: float) -> None
```

### 2. Loss Function クラス群

#### MSELoss (Mean Squared Error)

**RusTorch API分析:**
```rust
pub struct MSELoss<T> {
    reduction: Reduction,
}

impl MSELoss<T> {
    fn new() -> Self
    fn new_with_reduction(reduction: Reduction) -> Self
    fn forward(&self, input: &Variable<T>, target: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class MSELoss:
    def __init__(self, reduction: str = "mean")
    def forward(self, input: Variable, target: Variable) -> Variable
    def __call__(self, input: Variable, target: Variable) -> Variable
```

#### CrossEntropyLoss

**RusTorch API分析:**
```rust
pub struct CrossEntropyLoss<T> {
    reduction: Reduction,
    ignore_index: Option<i64>,
}

impl CrossEntropyLoss<T> {
    fn new() -> Self
    fn forward(&self, input: &Variable<T>, target: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class CrossEntropyLoss:
    def __init__(self, reduction: str = "mean")
    def forward(self, input: Variable, target: Variable) -> Variable
    def __call__(self, input: Variable, target: Variable) -> Variable
```

### 3. Activation Function クラス群

#### ReLU (Rectified Linear Unit)

**RusTorch API分析:**
```rust
pub struct ReLU<T> {
    inplace: bool,
}

impl ReLU<T> {
    fn new() -> Self
    fn new_inplace() -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class ReLU:
    def __init__(self, inplace: bool = False)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
```

#### Sigmoid

**RusTorch API分析:**
```rust
pub struct Sigmoid<T> {}

impl Sigmoid<T> {
    fn new() -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class Sigmoid:
    def __init__(self)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
```

#### Tanh

**RusTorch API分析:**
```rust
pub struct Tanh<T> {}

impl Tanh<T> {
    fn new() -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class Tanh:
    def __init__(self)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
```

## 技術的課題と解決策

### 1. Optimizer状態管理

**課題**: Optimizerの内部状態（momentum、velocity）の管理
**解決策**: PyO3での可変借用とRustの所有権システムの適切な処理

```rust
#[pyclass(name = "SGD")]
pub struct PySGD {
    pub inner: RustSGD<f32>,
}

#[pymethods]
impl PySGD {
    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        Ok(())
    }
}
```

### 2. Parameter管理

**課題**: 複数のVariableパラメータの効率的な管理
**解決策**: Vecベースの参照管理とCloneの活用

### 3. 関数型インターフェース

**課題**: PythonのCallableインターフェース実装
**解決策**: `__call__`メソッドによる統一的なインターフェース

## 実装ステップ

### Step 1: SGD Optimizer実装

```rust
use rustorch::optim::SGD as RustSGD;

#[pyclass(name = "SGD")]
pub struct PySGD {
    pub inner: RustSGD<f32>,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(parameters: Vec<PyVariable>, lr: f32, momentum: Option<f32>) -> PyResult<Self> {
        let rust_params: Vec<RustVariable<f32>> = parameters.into_iter()
            .map(|p| p.inner)
            .collect();

        let optimizer = if let Some(m) = momentum {
            RustSGD::new_with_momentum(rust_params, lr, m)
        } else {
            RustSGD::new(rust_params, lr)
        };

        Ok(PySGD { inner: optimizer })
    }

    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad();
        Ok(())
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        Ok(())
    }
}
```

### Step 2: Loss Functions実装

```rust
use rustorch::nn::MSELoss as RustMSELoss;

#[pyclass(name = "MSELoss")]
pub struct PyMSELoss {
    pub inner: RustMSELoss<f32>,
}

#[pymethods]
impl PyMSELoss {
    #[new]
    fn new(reduction: Option<String>) -> PyResult<Self> {
        let loss = RustMSELoss::new();
        Ok(PyMSELoss { inner: loss })
    }

    fn forward(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner, &target.inner);
        Ok(PyVariable { inner: result })
    }

    fn __call__(&self, input: &PyVariable, target: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input, target)
    }
}
```

### Step 3: Activation Functions実装

```rust
use rustorch::nn::ReLU as RustReLU;

#[pyclass(name = "ReLU")]
pub struct PyReLU {
    pub inner: RustReLU<f32>,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new(inplace: Option<bool>) -> PyResult<Self> {
        let inplace = inplace.unwrap_or(false);
        let relu = if inplace {
            RustReLU::new_inplace()
        } else {
            RustReLU::new()
        };
        Ok(PyReLU { inner: relu })
    }

    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }
}
```

### Step 4: 統合テスト実装

```python
# test_phase3.py
def test_complete_training():
    # データ準備
    X = rustorch.Variable(rustorch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    y = rustorch.Variable(rustorch.tensor([[1.0], [0.0]]), requires_grad=False)

    # モデル
    linear = rustorch.Linear(2, 1, True)
    sigmoid = rustorch.Sigmoid()

    # Loss function
    criterion = rustorch.MSELoss()

    # Optimizer
    optimizer = rustorch.SGD([linear.weight, linear.bias], lr=0.01)

    # トレーニングループ
    for epoch in range(10):
        optimizer.zero_grad()

        output = sigmoid(linear(X))
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.data}")
```

## Phase 3完了条件

✅ **基本機能**
- SGD Optimizer動作
- MSELoss、CrossEntropyLoss動作
- ReLU、Sigmoid、Tanh動作
- 完全なトレーニングループ実行

✅ **高度な機能**
- Momentum付きSGD
- Optimizer状態管理
- Loss reduction modes
- Inplace Activation

✅ **テスト**
- 各コンポーネントの単体テスト
- エンドツーエンドトレーニングテスト
- 勾配流の検証

✅ **ドキュメント**
- 完全なAPIドキュメント
- トレーニング例とベストプラクティス

## Phase 4計画 (予定)

- **Adam Optimizer**: より高度な最適化アルゴリズム
- **Batch Normalization**: 正規化レイヤー
- **Dropout**: 正則化技術
- **CNN Layers**: 畳み込みニューラルネットワーク
- **Model Save/Load**: モデルの保存・読み込み
- **Multi-GPU Support**: 分散トレーニング