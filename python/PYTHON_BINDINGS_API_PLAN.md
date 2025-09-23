# RusTorch Python Bindings API Plan

## 調査結果: 使用可能な RusTorch API

### 1. Tensor Core API (`src/tensor/core.rs`)

#### コンストラクタ
- `Tensor::new(data: ArrayD<T>)` - 基本コンストラクタ
- `Tensor::from_vec(data: Vec<T>, shape: Vec<usize>)` - ベクトルから作成
- `Tensor::zeros(shape: &[usize])` - ゼロテンサー
- `Tensor::ones(shape: &[usize])` - ワンテンサー
- `Tensor::full(shape: &[usize], value: T)` - 指定値テンサー
- `Tensor::from_scalar(value: T)` - スカラーテンサー

#### 基本メソッド
- `shape()` - テンサー形状
- `ndim()` - 次元数
- `size()` - サイズ情報
- `numel()` - 要素数
- `is_empty()` - 空チェック
- `device()` - デバイス情報
- `requires_grad()` - 勾配要求
- `set_requires_grad(bool)` - 勾配設定

#### データアクセス
- `get(index: &[usize])` - 要素取得
- `set(index: &[usize], value: T)` - 要素設定
- `as_slice()` - スライス参照
- `as_array()` - ndarray参照

#### 形状操作
- `reshape(new_shape: &[usize])` - リシェイプ
- `try_view(shape: &[usize])` - ビュー作成

### 2. Tensor Operations API (`src/tensor/ops/`)

#### 算術演算 (`arithmetic.rs`)
- `Add`, `Sub`, `Mul`, `Div` - 四則演算
- `Neg` - 負数
- スカラー演算サポート (`Tensor + T`, `Tensor * T` など)

#### 数学関数 (`mathematical.rs`)
- `sqrt()` - 平方根
- その他の数学関数

### 3. Neural Network API (`src/nn/`)

#### レイヤー
- `Linear<T>` - 線形レイヤー
  - `weight: Variable<T>`
  - `bias: Option<Variable<T>>`
  - `input_size: usize`
  - `output_size: usize`
- `Sequential<T>` - シーケンシャルモデル

#### トレイト
- `Module<T>` - モジュールトレイト

### 4. Optimizer API (`src/optim/`)

#### オプティマイザー
- `SGD` - 確率勾配降下法
- `Adam` - Adamオプティマイザー
- `RMSprop` - RMSpropオプティマイザー
- `AdaGrad` - AdaGradオプティマイザー

#### トレイト
- `Optimizer` - オプティマイザートレイト

### 5. Autograd API

#### Variable
- 自動微分をサポートする変数型
- `Linear`レイヤーで使用

### 6. Device API

#### デバイス管理
- `Device::default()` - デフォルトデバイス
- `is_cpu()`, `is_on_gpu()` - デバイス判定
- `to_cpu()`, `with_device()` - デバイス変換

## 実装計画

### Phase 1: 最小限のTensor (優先度: 高)
```python
# 目標: 基本的なTensor操作
import rustorch

# テンサー作成
t1 = rustorch.zeros([2, 3])
t2 = rustorch.ones([2, 3])
t3 = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 基本情報
print(t1.shape)  # [2, 3]
print(t1.numel()) # 6

# 基本演算
result = t1 + t2
result = t1 * 2.0
```

**実装対象:**
- `Tensor<f32>`, `Tensor<f64>` (float32, float64のみ)
- `zeros()`, `ones()`, `tensor()` 関数
- `shape`, `numel`, `ndim` プロパティ
- `+`, `-`, `*`, `/` 演算子
- `__repr__`, `__str__` メソッド

### Phase 2: Linear Layer (優先度: 中)
```python
# 目標: 基本的なニューラルネットワーク
linear = rustorch.Linear(10, 5)
x = rustorch.randn([32, 10])
y = linear(x)  # [32, 5]
```

**実装対象:**
- `Linear` クラス
- `Variable` 関連機能
- `forward()` メソッド

### Phase 3: Optimizer (優先度: 中)
```python
# 目標: 最適化
optimizer = rustorch.SGD(model.parameters(), lr=0.01)
loss = criterion(output, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**実装対象:**
- `SGD`, `Adam` クラス
- `parameters()` 収集
- `step()`, `zero_grad()` メソッド

### Phase 4: Advanced Features (優先度: 低)
- GPU サポート
- より多くの数学関数
- 複雑なニューラルネットワークレイヤー
- モデルの保存/読み込み

## 技術的制約と注意点

### 1. 型パラメータ
- RustTorchは `Tensor<T: Float>` のジェネリック型
- Pythonでは `f32`, `f64` に限定して実装

### 2. 所有権とライフタイム
- RustとPythonの所有権モデルの違い
- PyO3での適切なラッパー設計が必要

### 3. エラーハンドリング
- RustTorchの `RusTorchResult<T>` 型
- Pythonの例外への変換

### 4. パフォーマンス
- PyO3のオーバーヘッド最小化
- ndarrayとnumpyの相互変換効率

## 実装戦略

### 1. 段階的実装
- Phase 1から順次実装
- 各フェーズで動作確認

### 2. テスト駆動開発
- 各機能に対応するPythonテスト
- RustTorchとの結果一致確認

### 3. PyO3ベストプラクティス
- 最新PyO3 API使用
- 効率的なメモリ管理
- 適切なエラーハンドリング

### 4. ドキュメント
- Python API ドキュメント
- 使用例とチュートリアル