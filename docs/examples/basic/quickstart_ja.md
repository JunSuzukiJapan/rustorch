# RusTorch クイックスタートガイド

## インストール

### 1. 必要な環境
```bash
# Rust 1.70以降
rustc --version

# Python 3.8以降
python --version

# 必要な依存関係をインストール
pip install maturin numpy matplotlib
```

### 2. RusTorchのビルドとインストール
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 開発モードでビルド・インストール
maturin develop --release
```

## 基本的な使用例

### 1. テンソル作成と基本演算

```python
import rustorch
import numpy as np

# テンソル作成
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tensor x:\n{x}")
print(f"Shape: {x.shape()}")  # [2, 2]

# ゼロ行列と単位行列
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Identity:\n{identity}")

# ランダムテンソル
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Random normal:\n{random_normal}")
print(f"Random uniform:\n{random_uniform}")

# NumPy連携
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"From NumPy:\n{tensor_from_numpy}")

# NumPyに戻す
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"Back to NumPy:\n{back_to_numpy}")
```

### 2. 算術演算

```python
# 基本算術演算
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 要素ごとの演算
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (要素ごと)
div_result = a.div(b)  # a / b (要素ごと)

print(f"Addition:\n{add_result}")
print(f"Subtraction:\n{sub_result}")
print(f"Multiplication:\n{mul_result}")
print(f"Division:\n{div_result}")

# スカラー演算
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Scalar addition (+2):\n{scalar_add}")
print(f"Scalar multiplication (*3):\n{scalar_mul}")

# 行列積
matmul_result = a.matmul(b)
print(f"Matrix multiplication:\n{matmul_result}")

# 数学関数
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Square root:\n{sqrt_result}")
print(f"Exponential:\n{exp_result}")
print(f"Natural log:\n{log_result}")
```

### 3. テンソル形状操作

```python
# 形状操作の例
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Original shape: {original.shape()}")  # [2, 4]

# リシェイプ
reshaped = original.reshape([4, 2])
print(f"Reshaped [4, 2]:\n{reshaped}")

# 転置
transposed = original.transpose(0, 1)
print(f"Transposed:\n{transposed}")

# 次元追加・削除
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Before squeeze: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"After squeeze: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"After unsqueeze: {expanded.shape()}")  # [1, 3]
```

### 4. 統計操作

```python
# 統計関数
data = rustorch.randn([3, 4])
print(f"Data:\n{data}")

# 基本統計
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Mean: {mean_val.item():.4f}")
print(f"Sum: {sum_val.item():.4f}")
print(f"Std: {std_val.item():.4f}")
print(f"Var: {var_val.item():.4f}")
print(f"Max: {max_val.item():.4f}")
print(f"Min: {min_val.item():.4f}")

# 次元を指定した統計
row_mean = data.mean(dim=1)  # 各行の平均
col_sum = data.sum(dim=0)    # 各列の合計

print(f"Row means: {row_mean}")
print(f"Column sums: {col_sum}")
```

## 自動微分の基本

### 1. 勾配計算

```python
# 自動微分の例
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Input tensor: {x}")

# Variable作成
var_x = rustorch.autograd.Variable(x)

# 計算グラフ構築
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Output: {y.data().item()}")

# 逆伝播
y.backward()

# 勾配取得
grad = var_x.grad()
print(f"Gradient: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. 複雑な計算グラフ

```python
# より複雑な例
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# 複雑な関数: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), then sum

print(f"Result: {z.data().item():.4f}")

# 逆伝播
z.backward()
grad = var_x.grad()
print(f"Gradient: {grad}")
```

## ニューラルネットワークの基本

### 1. 単純な線形層

```python
# 線形層の作成
linear_layer = rustorch.nn.Linear(3, 1)  # 3入力 -> 1出力

# ランダム入力
input_data = rustorch.randn([2, 3])  # バッチサイズ2, 特徴数3
print(f"Input: {input_data}")

# フォワードパス
output = linear_layer.forward(input_data)
print(f"Output: {output}")

# パラメータ確認
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Weight shape: {weight.shape()}")
print(f"Weight: {weight}")
if bias is not None:
    print(f"Bias: {bias}")
```

### 2. 活性化関数

```python
# 各種活性化関数
x = rustorch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# ReLU
relu = rustorch.nn.ReLU()
relu_output = relu.forward(x)
print(f"ReLU: {relu_output}")

# Sigmoid
sigmoid = rustorch.nn.Sigmoid()
sigmoid_output = sigmoid.forward(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh
tanh = rustorch.nn.Tanh()
tanh_output = tanh.forward(x)
print(f"Tanh: {tanh_output}")
```

### 3. 損失関数

```python
# 損失関数の使用例
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# 平均二乗誤差
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE Loss: {loss_value.item():.6f}")

# クロスエントロピー（分類用）
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # クラスインデックス

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Cross Entropy Loss: {ce_loss_value.item():.6f}")
```

## データ処理

### 1. データセットとデータローダー

```python
# データセット作成
import numpy as np

# サンプルデータ生成
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100サンプル, 4特徴
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3クラス分類

# テンソルに変換
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# データセット作成
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Dataset size: {len(dataset)}")

# データローダー作成
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# データローダーからバッチ取得
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # 最初の3バッチのみ表示
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Batch {batch_idx}: Input shape {inputs.shape()}, Target shape {targets.shape()}")
```

### 2. データ変換

```python
# データ変換の例
data = rustorch.randn([10, 10])
print(f"Original data mean: {data.mean().item():.4f}")
print(f"Original data std: {data.std().item():.4f}")

# 正規化変換
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Normalized data mean: {normalized_data.mean().item():.4f}")
print(f"Normalized data std: {normalized_data.std().item():.4f}")
```

## 完全なトレーニング例

### 線形回帰

```python
# 線形回帰の完全な例
import numpy as np

# データ生成
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# テンソル変換
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# データセット・ローダー作成
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# モデル定義
model = rustorch.nn.Linear(1, 1)  # 1入力 -> 1出力

# 損失関数とオプティマイザー
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# 訓練ループ
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    dataloader.reset()
    while True:
        batch = dataloader.next_batch()
        if batch is None:
            break
        
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            
            # 勾配をゼロに
            optimizer.zero_grad()
            
            # フォワードパス
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # バックプロパゲーション（簡略化）
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

print("Training completed!")

# 最終的なパラメータ
final_weight = model.weight()
final_bias = model.bias()
print(f"Learned weight: {final_weight.item():.4f} (true: 2.0)")
if final_bias is not None:
    print(f"Learned bias: {final_bias.item():.4f} (true: 1.0)")
```

## トラブルシューティング

### よくある問題と解決法

1. **インストール問題**
```bash
# maturinが見つからない場合
pip install --upgrade maturin

# Rustが古い場合
rustup update

# Python環境の問題
python -m pip install --upgrade pip
```

2. **実行時エラー**
```python
# テンソルの形状確認
print(f"Tensor shape: {tensor.shape()}")
print(f"Tensor dtype: {tensor.dtype()}")

# NumPy変換でのデータ型注意
np_array = np.array(data, dtype=np.float32)  # float32を明示
```

3. **パフォーマンス最適化**
```python
# リリースモードでビルド
# maturin develop --release

# バッチサイズの調整
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # より大きなバッチ
```

## 次のステップ

1. **高度な例を試す**: `docs/examples/neural_networks/` の例を参照
2. **Keras風APIを使用**: `rustorch.training.Model` で簡単なモデル構築
3. **可視化機能**: `rustorch.visualization` でトレーニング進捗を可視化
4. **分散訓練**: `rustorch.distributed` で並列処理

詳細なドキュメント:
- [Python API リファレンス](../ja/python_api_reference.md)
- [概要ドキュメント](../ja/python_bindings_overview.md)
- [サンプル集](../examples/)