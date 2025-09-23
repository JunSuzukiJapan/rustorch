# Phase 4 Implementation Plan: Advanced Deep Learning Features
高度なディープラーニング機能の実装

## 概要

Phase 3の基本的なニューラルネットワークトレーニングシステムを拡張し、実用的なディープラーニング開発に必要な高度な機能を実装します。

## Phase 4の目標

```python
# 目標: 実用的なディープラーニングシステム
import rustorch

# データとラベル（MNIST風）
X = rustorch.Variable(rustorch.tensor(batch_data), requires_grad=False)  # [batch, 1, 28, 28]
y = rustorch.Variable(rustorch.tensor(batch_labels), requires_grad=False)  # [batch, 10]

# CNN モデル構築
conv1 = rustorch.Conv2d(1, 32, kernel_size=3, padding=1)
bn1 = rustorch.BatchNorm2d(32)
relu1 = rustorch.ReLU()
pool1 = rustorch.MaxPool2d(kernel_size=2)
dropout1 = rustorch.Dropout(0.25)

conv2 = rustorch.Conv2d(32, 64, kernel_size=3, padding=1)
bn2 = rustorch.BatchNorm2d(64)
relu2 = rustorch.ReLU()
pool2 = rustorch.MaxPool2d(kernel_size=2)
dropout2 = rustorch.Dropout(0.25)

# 全結合層
flatten = rustorch.Flatten()
linear1 = rustorch.Linear(64 * 7 * 7, 128)
bn3 = rustorch.BatchNorm1d(128)
relu3 = rustorch.ReLU()
dropout3 = rustorch.Dropout(0.5)
linear2 = rustorch.Linear(128, 10)

# 順伝播パイプライン
def forward(x):
    x = pool1(relu1(bn1(conv1(x))))
    x = dropout1(x)
    x = pool2(relu2(bn2(conv2(x))))
    x = dropout2(x)
    x = flatten(x)
    x = dropout3(relu3(bn3(linear1(x))))
    x = linear2(x)
    return x

# 損失関数とオプティマイザー
criterion = rustorch.CrossEntropyLoss()
optimizer = rustorch.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# 高度なトレーニングループ
for epoch in range(100):
    model.train()  # Training mode

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()  # Evaluation mode
    # バリデーション...

    # モデル保存
    if epoch % 10 == 0:
        rustorch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
```

## 実装対象クラス

### 1. Advanced Optimizer クラス群

#### Adam (Adaptive Moment Estimation)

**RusTorch API分析:**
```rust
pub struct Adam<T> {
    parameters: Vec<Variable<T>>,
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: Option<T>,
    amsgrad: bool,
    // 状態管理
    m: Vec<Option<Tensor<T>>>,  // 1次モーメント
    v: Vec<Option<Tensor<T>>>,  // 2次モーメント
    step_count: usize,
}

impl Adam<T> {
    fn new(parameters: Vec<Variable<T>>, learning_rate: T) -> Self
    fn new_with_params(parameters: Vec<Variable<T>>, learning_rate: T,
                       beta1: T, beta2: T, epsilon: T) -> Self
    fn zero_grad(&self)
    fn step(&mut self)
    fn learning_rate(&self) -> T
    fn set_learning_rate(&mut self, lr: T)
}
```

**Python実装ターゲット:**
```python
class Adam:
    def __init__(self, parameters: List[Variable], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False)
    def zero_grad(self) -> None
    def step(self) -> None
    @property
    def lr(self) -> float
    def set_lr(self, lr: float) -> None
```

### 2. Normalization Layer クラス群

#### BatchNorm2d (2D Batch Normalization)

**RusTorch API分析:**
```rust
pub struct BatchNorm2d<T> {
    num_features: usize,
    eps: T,
    momentum: T,
    affine: bool,
    track_running_stats: bool,
    // パラメータ
    weight: Option<Variable<T>>,
    bias: Option<Variable<T>>,
    // 統計情報
    running_mean: Option<Tensor<T>>,
    running_var: Option<Tensor<T>>,
    num_batches_tracked: usize,
}

impl BatchNorm2d<T> {
    fn new(num_features: usize) -> Self
    fn new_with_params(num_features: usize, eps: T, momentum: T, affine: bool) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
    fn train(&mut self)
    fn eval(&mut self)
}
```

**Python実装ターゲット:**
```python
class BatchNorm2d:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
    def train(self) -> None
    def eval(self) -> None
    @property
    def weight(self) -> Variable
    @property
    def bias(self) -> Variable
```

#### BatchNorm1d (1D Batch Normalization)

同様の実装パターンで1次元版を実装

### 3. Regularization Layer クラス群

#### Dropout

**RusTorch API分析:**
```rust
pub struct Dropout<T> {
    p: T,
    inplace: bool,
    training: bool,
}

impl Dropout<T> {
    fn new(p: T) -> Self
    fn new_inplace(p: T) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
    fn train(&mut self)
    fn eval(&mut self)
}
```

**Python実装ターゲット:**
```python
class Dropout:
    def __init__(self, p: float = 0.5, inplace: bool = False)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
    def train(self) -> None
    def eval(self) -> None
```

### 4. Convolution Layer クラス群

#### Conv2d (2D Convolution)

**RusTorch API分析:**
```rust
pub struct Conv2d<T> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    bias: bool,
    // パラメータ
    weight: Variable<T>,
    bias: Option<Variable<T>>,
}

impl Conv2d<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self
    fn new_with_params(in_channels: usize, out_channels: usize,
                       kernel_size: (usize, usize), stride: (usize, usize),
                       padding: (usize, usize)) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

**Python実装ターゲット:**
```python
class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1, bias: bool = True)
    def forward(self, input: Variable) -> Variable
    def __call__(self, input: Variable) -> Variable
    @property
    def weight(self) -> Variable
    @property
    def bias(self) -> Optional[Variable]
```

#### MaxPool2d (2D Max Pooling)

**RusTorch API分析:**
```rust
pub struct MaxPool2d<T> {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
}

impl MaxPool2d<T> {
    fn new(kernel_size: usize) -> Self
    fn new_with_params(kernel_size: (usize, usize), stride: Option<(usize, usize)>,
                       padding: (usize, usize)) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

### 5. Utility Layer クラス群

#### Flatten

**RusTorch API分析:**
```rust
pub struct Flatten<T> {
    start_dim: usize,
    end_dim: isize,
}

impl Flatten<T> {
    fn new() -> Self
    fn new_with_dims(start_dim: usize, end_dim: isize) -> Self
    fn forward(&self, input: &Variable<T>) -> Variable<T>
}
```

### 6. Enhanced Loss Functions

#### CrossEntropyLoss

**RusTorch API分析:**
```rust
pub struct CrossEntropyLoss<T> {
    weight: Option<Tensor<T>>,
    reduction: Reduction,
    ignore_index: Option<i64>,
    label_smoothing: T,
}

impl CrossEntropyLoss<T> {
    fn new() -> Self
    fn new_with_params(weight: Option<Tensor<T>>, reduction: Reduction) -> Self
    fn forward(&self, input: &Variable<T>, target: &Variable<T>) -> Variable<T>
}
```

## 技術的課題と解決策

### 1. Adam Optimizer状態管理

**課題**: 1次・2次モーメントの効率的な管理とステップカウント
**解決策**: VecベースのOptional状態管理とmut参照の適切な処理

```rust
#[pyclass(name = "Adam")]
pub struct PyAdam {
    pub parameters: Vec<PyVariable>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    pub m: Vec<Option<PyTensor>>,  // 1次モーメント
    pub v: Vec<Option<PyTensor>>,  // 2次モーメント
}
```

### 2. BatchNormalization状態管理

**課題**: 訓練時と推論時の動作切り替え、running統計の管理
**解決策**: training flagとrunning_mean/varの適切な更新

### 3. Convolution パラメータ管理

**課題**: 4次元テンソル（weight）の初期化と管理
**解決策**: Kaiming初期化の適用とテンソル形状の適切な処理

### 4. Memory効率性

**課題**: CNNの大きなfeature mapとメモリ使用量
**解決策**: inplace操作の活用とメモリ効率的な実装

## 実装ステップ

### Step 1: Adam Optimizer実装

```rust
use rustorch::optim::Adam as RustAdam;

#[pyclass(name = "Adam")]
pub struct PyAdam {
    pub inner: RustAdam<f32>,
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(
        parameters: Vec<PyVariable>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>
    ) -> PyResult<Self> {
        let lr = lr.unwrap_or(0.001);
        let (beta1, beta2) = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let rust_params: Vec<RustVariable<f32>> = parameters.into_iter()
            .map(|p| p.inner)
            .collect();

        let optimizer = RustAdam::new_with_params(rust_params, lr, beta1, beta2, eps);
        Ok(PyAdam { inner: optimizer })
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

### Step 2: BatchNorm2d実装

```rust
use rustorch::nn::BatchNorm2d as RustBatchNorm2d;

#[pyclass(name = "BatchNorm2d")]
pub struct PyBatchNorm2d {
    pub inner: RustBatchNorm2d<f32>,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);

        let bn = RustBatchNorm2d::new_with_params(num_features, eps, momentum, affine);
        Ok(PyBatchNorm2d { inner: bn })
    }

    fn forward(&self, input: &PyVariable) -> PyResult<PyVariable> {
        let result = self.inner.forward(&input.inner);
        Ok(PyVariable { inner: result })
    }

    fn __call__(&self, input: &PyVariable) -> PyResult<PyVariable> {
        self.forward(input)
    }

    fn train(&mut self) -> PyResult<()> {
        self.inner.train();
        Ok(())
    }

    fn eval(&mut self) -> PyResult<()> {
        self.inner.eval();
        Ok(())
    }
}
```

## Phase 4完了条件

✅ **基本機能**
- Adam Optimizer動作
- BatchNorm2d、BatchNorm1d動作
- Dropout正則化動作
- Conv2d、MaxPool2d動作
- CrossEntropyLoss動作

✅ **高度な機能**
- CNNアーキテクチャ構築
- 訓練・推論モード切り替え
- メモリ効率的な実装
- 実用的なディープラーニングパイプライン

✅ **テスト**
- MNIST風CNNトレーニングテスト
- 各コンポーネントの単体テスト
- メモリ使用量とパフォーマンステスト

✅ **ドキュメント**
- 完全なAPIドキュメント
- CNNアーキテクチャ例とベストプラクティス

## Phase 5計画 (予定)

- **Model Save/Load**: モデルの永続化とシリアライゼーション
- **Advanced CNN**: ResNet、DenseNet等のアーキテクチャパターン
- **RNN/LSTM**: 時系列データ処理
- **Attention Mechanism**: Transformer関連機能
- **Multi-GPU Support**: 分散トレーニング対応
- **Performance Optimization**: CUDA最適化とメモリ管理向上