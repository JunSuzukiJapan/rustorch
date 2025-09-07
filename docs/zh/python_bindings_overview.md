# RusTorch Python 绑定概述

## 概述

RusTorch 是一个用 Rust 实现的高性能深度学习框架，提供类似 PyTorch 的 API，同时利用 Rust 的安全性和性能优势。通过 Python 绑定，您可以直接从 Python 访问 RusTorch 的功能。

## 主要特性

### 🚀 **高性能**
- **Rust 内核**：在保证内存安全的同时实现 C++ 级别的性能
- **SIMD 支持**：通过自动向量化优化数值计算
- **并行处理**：使用 rayon 进行高效的并行计算
- **零拷贝**：NumPy 和 RusTorch 之间的数据拷贝最小化

### 🛡️ **安全性**
- **内存安全**：通过 Rust 的所有权系统防止内存泄漏和数据竞争
- **类型安全**：编译时类型检查减少运行时错误
- **错误处理**：全面的错误处理，自动转换为 Python 异常

### 🎯 **易用性**
- **PyTorch 兼容 API**：从现有 PyTorch 代码轻松迁移
- **Keras 风格高级 API**：如 model.fit() 等直观接口
- **NumPy 集成**：支持与 NumPy 数组的双向转换

## 架构

RusTorch 的 Python 绑定由 10 个模块组成：

### 1. **tensor** - 张量操作
```python
import rustorch

# 张量创建
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# NumPy 集成
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - 自动微分
```python
# 梯度计算
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # 获取梯度
```

### 3. **nn** - 神经网络
```python
# 层创建
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# 损失函数
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - 优化器
```python
# 优化器
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率调度器
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - 数据加载
```python
# 数据集创建
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 数据变换
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - 高级训练 API
```python
# Keras 风格 API
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 训练执行
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - 分布式训练
```python
# 分布式训练配置
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# 数据并行
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - 可视化
```python
# 训练历史绘图
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# 张量可视化
plotter.plot_tensor_as_image(tensor, title="Feature Map")
```

### 9. **utils** - 工具
```python
# 模型保存/加载
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# 性能分析
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## 安装

### 先决条件
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+（GPU 使用时）

### 构建和安装
```bash
# 克隆仓库
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# 创建 Python 虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install maturin numpy

# 构建和安装
maturin develop --release

# 或从 PyPI 安装（计划中）
# pip install rustorch
```

## 快速入门

### 1. 基本张量操作
```python
import rustorch
import numpy as np

# 张量创建
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Shape: {x.shape()}")  # Shape: [2, 2]

# 数学运算
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Result: {z.to_numpy()}")
```

### 2. 线性回归示例
```python
import rustorch
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# 转换为张量
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# 定义模型
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# 创建数据集
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# 训练
history = model.fit(dataloader, epochs=100, verbose=True)

# 显示结果
print(f"Final loss: {history.train_loss()[-1]:.4f}")
```

### 3. 神经网络分类
```python
import rustorch

# 准备数据
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# 构建模型
model = rustorch.Model("ClassificationNet")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# 编译模型
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 训练配置
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# 训练
history = trainer.train(model, train_loader, val_loader)

# 评估
metrics = model.evaluate(test_loader)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

## 性能优化

### SIMD 利用
```python
# 构建时启用 SIMD 优化
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # SIMD 优化计算
```

### GPU 使用
```python
# CUDA 使用（计划中）
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # GPU 计算
```

### 并行数据加载
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # 并行工作进程数
)
```

## 最佳实践

### 1. 内存效率
```python
# 利用零拷贝转换
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # 无拷贝

# 使用原地操作
tensor.add_(1.0)  # 内存高效
```

### 2. 错误处理
```python
try:
    result = model(invalid_input)
except rustorch.RusTorchError as e:
    print(f"RusTorch error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. 调试和性能分析
```python
# 使用分析器
profiler = rustorch.utils.Profiler()
profiler.start()

# 执行计算
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## 限制

### 当前限制
- **GPU 支持**：CUDA/ROCm 支持开发中
- **动态图**：目前仅支持静态图
- **分布式训练**：仅实现基本功能

### 未来扩展
- GPU 加速（CUDA、Metal、ROCm）
- 动态计算图支持
- 更多神经网络层
- 模型量化和剪枝
- ONNX 导出功能

## 贡献

### 开发参与
```bash
# 设置开发环境
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# 运行测试
cargo test
python -m pytest tests/

# 代码质量检查
cargo clippy
cargo fmt
```

### 社区
- GitHub Issues：错误报告和功能请求
- Discussions：问题和讨论
- Discord：实时支持

## 许可证

RusTorch 在 MIT 许可证下发布，可自由用于商业和非商业目的。

## 相关链接

- [GitHub 仓库](https://github.com/JunSuzukiJapan/RusTorch)
- [API 文档](./api_documentation.md)
- [示例和教程](../examples/)
- [性能基准](./benchmarks.md)