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

#### Variable（自動微分変数）
- 自動微分をサポートする変数型
- テンソルをラップし勾配計算を自動化
- `requires_grad` フラグで勾配計算の有効/無効を制御
- 基本機能:
  - `Variable.new(tensor, requires_grad)` - 変数作成
  - `Variable.data` - 内部テンソルへのアクセス
  - `Variable.grad` - 勾配テンソル（計算後）
  - `Variable.requires_grad` - 勾配計算フラグ
  - `Variable.backward()` - 逆伝播実行
  - `Variable.zero_grad()` - 勾配クリア

#### 高度なAutograd操作
- `Variable.detach()` - 計算グラフから切り離し
- `Variable.retain_grad()` - 中間変数の勾配保持
- `no_grad()` コンテキスト - 勾配計算無効化
- `enable_grad()` コンテキスト - 勾配計算強制有効化

#### 計算グラフ操作
- 自動微分による計算グラフ構築
- 複雑な数学関数の勾配自動計算
- チェーンルールによる合成関数の微分
- 複数出力に対するヤコビアン計算

#### 高次微分
- `grad()` 関数による任意階微分
- Hessian行列計算サポート
- 勾配の勾配計算

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

### Phase 4: Advanced Features (優先度: 低) - ✅ 完了
```python
# 目標: CNN と高度な深層学習機能
conv = rustorch.Conv2d(3, 16, kernel_size=3)
pool = rustorch.MaxPool2d(2)
bn = rustorch.BatchNorm2d(16)
dropout = rustorch.Dropout(0.5)
flatten = rustorch.Flatten()
criterion = rustorch.CrossEntropyLoss()
```

**実装対象:** ✅ 完了
- CNN layers: `Conv2d`, `MaxPool2d`
- 正規化: `BatchNorm1d`, `BatchNorm2d`
- 正則化: `Dropout`
- ユーティリティ: `Flatten`
- 損失関数: `CrossEntropyLoss`
- Adam optimizer の完全実装

### Phase 5: Advanced Autograd API (優先度: 高)
```python
# 目標: 高度な自動微分とグラフ操作
import rustorch

# 高度なAutograd操作
with rustorch.no_grad():
    # 勾配計算無効化
    y = model(x)

# 計算グラフ操作
x = rustorch.Variable(tensor, requires_grad=True)
x.retain_grad()  # 中間変数の勾配保持
y = x.detach()   # 計算グラフから切り離し

# 高次微分
grad_outputs = rustorch.grad(outputs, inputs, create_graph=True)
hessian = rustorch.grad(grad_outputs, inputs)

# 関数型API
def custom_loss(pred, target):
    return rustorch.functional.mse_loss(pred, target)

# フック機能
x.register_hook(lambda grad: grad * 2)
```

**実装対象:**
- Context managers: `no_grad()`, `enable_grad()`
- 高度なVariable操作: `detach()`, `retain_grad()`
- 関数型勾配計算: `grad()` 関数
- フック機能: `register_hook()`, `register_backward_hook()`
- 関数型API: `rustorch.functional` モジュール
- 高次微分サポート
- カスタム autograd Function

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