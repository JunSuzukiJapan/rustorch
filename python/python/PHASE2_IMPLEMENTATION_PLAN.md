# Phase 2 Implementation Plan: Neural Network Foundation

## 概要

Phase 1の基本Tensorクラスを基盤に、Neural Networkの基礎機能を実装します。

## Phase 2の目標

```python
# 目標: 基本的なニューラルネットワークトレーニング
import rustorch

# Variable (自動微分対応)
x = rustorch.Variable(rustorch.tensor([[1.0, 2.0]]), requires_grad=True)
print(x.data.shape)  # [1, 2]

# Linear Layer
linear = rustorch.Linear(2, 1)  # input_size=2, output_size=1
y = linear(x)  # forward pass
print(y.data.shape)  # [1, 1]

# 基本的な自動微分
loss = y.sum()
loss.backward()
print(x.grad)  # 勾配情報
```

## 実装対象クラス

### 1. Variable クラス

**RusTorch API分析:**
```rust
pub struct Variable<T> {
    data: Arc<RwLock<Tensor<T>>>,
    grad: Arc<RwLock<Option<Tensor<T>>>>,
    requires_grad: bool,
    unique_id: usize,
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    _marker: PhantomData<T>,
}

// 主要メソッド
impl Variable<T> {
    fn new(data: Tensor<T>, requires_grad: bool) -> Self
    fn data(&self) -> Arc<RwLock<Tensor<T>>>
    fn grad(&self) -> Arc<RwLock<Option<Tensor<T>>>>
    fn requires_grad(&self) -> bool
    fn zero_grad(&self)
    fn backward(&self)
    fn backward_with_grad(&self, grad_output: Option<Tensor<T>>)
    fn sum(&self) -> Variable<T>
    fn matmul(&self, other: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class Variable:
    def __init__(self, data: Tensor, requires_grad: bool = False)
    @property
    def data(self) -> Tensor
    @property
    def grad(self) -> Optional[Tensor]
    @property
    def requires_grad(self) -> bool
    def zero_grad(self) -> None
    def backward(self) -> None
    def sum(self) -> Variable
```

### 2. Linear クラス

**RusTorch API分析:**
```rust
pub struct Linear<T> {
    weight: Variable<T>,          // (output_features, input_features)
    bias: Option<Variable<T>>,    // (output_features,)
    input_size: usize,
    output_size: usize,
}

impl Linear<T> {
    fn new(input_size: usize, output_size: usize) -> Self
    fn new_no_bias(input_size: usize, output_size: usize) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
    fn input_size(&self) -> usize
    fn output_size(&self) -> usize
}
```

**Python実装ターゲット:**
```python
class Linear:
    def __init__(self, input_size: int, output_size: int, bias: bool = True)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable  # Pythonの慣例
    @property
    def weight(self) -> Variable
    @property
    def bias(self) -> Optional[Variable]
```

## 技術的課題と解決策

### 1. スレッドセーフティ問題

**課題**: RustのArc<RwLock<T>>をPythonに安全に公開
**解決策**: PyO3でのclone()とread()/write()の適切な処理

```rust
// Rust側での実装例
#[pymethods]
impl PyVariable {
    #[getter]
    fn data(&self) -> PyResult<PyTensor> {
        let data = self.inner.data().read().unwrap().clone();
        Ok(PyTensor { inner: data })
    }
}
```

### 2. 自動微分のグラディエント関数

**課題**: 複雑なgrad_fnの適切な抽象化
**解決策**: Phase 2では基本的な演算のみサポート、複雑な機能は段階的実装

### 3. メモリ管理

**課題**: PythonとRustの所有権の違い
**解決策**: Clone()を多用し、パフォーマンスより安全性を優先

## 実装ステップ

### Step 1: Variable基盤実装

```rust
#[pyclass(name = "Variable")]
#[derive(Clone)]
pub struct PyVariable {
    pub inner: RustVariable<f32>,
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new(data: &PyTensor, requires_grad: Option<bool>) -> PyResult<Self> {
        let requires_grad = requires_grad.unwrap_or(false);
        let variable = RustVariable::new(data.inner.clone(), requires_grad);
        Ok(PyVariable { inner: variable })
    }

    #[getter]
    fn data(&self) -> PyResult<PyTensor> {
        let data = self.inner.data().read().unwrap().clone();
        Ok(PyTensor { inner: data })
    }

    fn backward(&self) -> PyResult<()> {
        self.inner.backward();
        Ok(())
    }
}
```

### Step 2: Variable演算実装

```rust
#[pymethods]
impl PyVariable {
    fn sum(&self) -> PyResult<PyVariable> {
        let result = self.inner.sum();
        Ok(PyVariable { inner: result })
    }

    fn __add__(&self, other: &PyVariable) -> PyResult<PyVariable> {
        // Variable同士の加算実装
    }
}
```

### Step 3: Linear Layer実装

```rust
#[pyclass(name = "Linear")]
pub struct PyLinear {
    pub inner: RustLinear<f32>,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(input_size: usize, output_size: usize, bias: Option<bool>) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);
        let linear = if bias {
            RustLinear::new(input_size, output_size)
        } else {
            RustLinear::new_no_bias(input_size, output_size)
        };
        Ok(PyLinear { inner: linear })
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

### Step 4: テスト環境整備

```python
# test_phase2.py
def test_variable():
    # Variable作成
    t = rustorch.tensor([[1.0, 2.0]])
    v = rustorch.Variable(t, requires_grad=True)
    assert v.requires_grad == True
    assert v.data.shape == [1, 2]

def test_linear_layer():
    # Linear層
    linear = rustorch.Linear(2, 1)
    x = rustorch.Variable(rustorch.tensor([[1.0, 2.0]]), requires_grad=True)
    y = linear(x)
    assert y.data.shape == [1, 1]

def test_autograd():
    # 自動微分
    x = rustorch.Variable(rustorch.tensor([[1.0]]), requires_grad=True)
    y = x.sum()
    y.backward()
    # gradの確認
```

## 依存関係の追加

```toml
# Cargo.toml
[dependencies]
rand = "0.8"  # 重みの初期化用
rand_distr = "0.4"  # ガウス分布用
```

## Phase 2完了条件

✅ **基本機能**
- Variable作成とプロパティアクセス
- Linear層の作成と順伝播
- 基本的な自動微分（sum(), backward()）

✅ **テスト**
- 全ての基本機能のテスト成功
- メモリリークなし
- スレッドセーフティ確認

✅ **ドキュメント**
- 使用例とAPIドキュメント
- Phase 3への移行計画

## Phase 3計画 (予定)

- **Optimizer**: SGD, Adam実装
- **Loss Functions**: MSELoss, CrossEntropyLoss
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **より複雑な自動微分**: Chain rule完全サポート