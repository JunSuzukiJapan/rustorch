# RusTorch Python バインディング 概要

## 概要

RusTorchは、Rustで実装された高性能な深層学習フレームワークで、PyTorchに似たAPIを提供しながら、Rustの安全性とパフォーマンスを活用します。Python バインディングを通じて、Pythonから直接RusTorchの機能を利用できます。

## 主な特徴

### 🚀 **高性能**
- **Rust製コア**: メモリ安全性を保証しながら、C++並みの性能を実現
- **SIMDサポート**: 自動ベクトル化による数値計算の最適化
- **並列処理**: rayonによる効率的な並列計算
- **ゼロコピー**: NumPyとの間でデータコピーを最小化

### 🛡️ **安全性**
- **メモリ安全**: Rustの所有権システムによるメモリリークとデータ競合の防止
- **型安全**: コンパイル時の型チェックによるランタイムエラーの削減
- **エラーハンドリング**: 包括的なエラー処理とPython例外への自動変換

### 🎯 **使いやすさ**
- **PyTorch互換API**: 既存のPyTorchコードからの移行が容易
- **Keras風高レベルAPI**: model.fit()のような直感的なインターフェース
- **NumPy統合**: NumPy配列との双方向変換をサポート

## アーキテクチャ

RusTorchのPythonバインディングは、以下の10個のモジュールで構成されています：

### 1. **tensor** - テンソル操作
```python
import rustorch

# テンソル作成
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# NumPy連携
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - 自動微分
```python
# 勾配計算
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # 勾配を取得
```

### 3. **nn** - ニューラルネットワーク
```python
# 層の作成
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# 損失関数
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - オプティマイザー
```python
# オプティマイザー
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学習率スケジューラー
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - データローディング
```python
# データセット作成
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# データ変換
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - 高レベル訓練API
```python
# Keras風API
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 訓練実行
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - 分散訓練
```python
# 分散訓練設定
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# データ並列
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - 可視化
```python
# 訓練履歴のプロット
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# テンソル可視化
plotter.plot_tensor_as_image(tensor, title="Feature Map")
```

### 9. **utils** - ユーティリティ
```python
# モデル保存・読み込み
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# プロファイリング
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## インストール

### 前提条件
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (GPU使用時)

### ビルドとインストール
```bash
# リポジトリをクローン
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Python仮想環境を作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install maturin numpy

# ビルドとインストール
maturin develop --release

# またはPyPIからインストール（将来予定）
# pip install rustorch
```

## クイックスタート

### 1. 基本的なテンソル操作
```python
import rustorch
import numpy as np

# テンソル作成
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Shape: {x.shape()}")  # Shape: [2, 2]

# 数学演算
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Result: {z.to_numpy()}")
```

### 2. 線形回帰の例
```python
import rustorch
import numpy as np

# データ生成
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# テンソルに変換
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# モデル定義
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# データセット作成
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# 訓練実行
history = model.fit(dataloader, epochs=100, verbose=True)

# 結果表示
print(f"Final loss: {history.train_loss()[-1]:.4f}")
```

### 3. ニューラルネットワーク分類
```python
import rustorch

# データ準備
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# モデル構築
model = rustorch.Model("ClassificationNet")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# モデルコンパイル
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 訓練設定
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# 訓練実行
history = trainer.train(model, train_loader, val_loader)

# 評価
metrics = model.evaluate(test_loader)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

## パフォーマンス最適化

### SIMD活用
```python
# SIMD最適化を有効にしてビルド
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # SIMD最適化された計算
```

### GPU利用
```python
# CUDA使用（将来実装予定）
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # GPU計算
```

### 並列データローディング
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # 並列ワーカー数
)
```

## ベストプラクティス

### 1. メモリ効率
```python
# ゼロコピー変換を活用
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # コピーなし

# in-place演算を使用
tensor.add_(1.0)  # メモリ効率的
```

### 2. エラーハンドリング
```python
try:
    result = model(invalid_input)
except rustorch.RusTorchError as e:
    print(f"RusTorch error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. デバッグとプロファイリング
```python
# プロファイラーを使用
profiler = rustorch.utils.Profiler()
profiler.start()

# 計算実行
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## 制限事項

### 現在の制限
- **GPU サポート**: CUDA/ROCm サポートは開発中
- **動的グラフ**: 現在は静的グラフのみサポート
- **分散訓練**: 基本機能のみ実装済み

### 将来の拡張予定
- GPU アクセラレーション (CUDA, Metal, ROCm)
- 動的計算グラフのサポート
- より多くのニューラルネットワーク層
- モデル量子化とプルーニング
- ONNX エクスポート機能

## 貢献

### 開発参加
```bash
# 開発環境セットアップ
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# テスト実行
cargo test
python -m pytest tests/

# コード品質チェック
cargo clippy
cargo fmt
```

### コミュニティ
- GitHub Issues: バグレポートや機能リクエスト
- Discussions: 質問や議論
- Discord: リアルタイムサポート

## ライセンス

RusTorchはMITライセンスの下で公開されています。商用・非商用問わず自由に使用できます。

## 関連リンク

- [GitHub リポジトリ](https://github.com/JunSuzukiJapan/RusTorch)
- [API ドキュメント](./api_documentation.md)
- [例とチュートリアル](../examples/)
- [パフォーマンス ベンチマーク](./benchmarks.md)