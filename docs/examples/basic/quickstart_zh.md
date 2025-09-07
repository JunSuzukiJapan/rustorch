# RusTorch 快速入门指南

## 安装

### 1. 环境要求
```bash
# Rust 1.70 或更高版本
rustc --version

# Python 3.8 或更高版本
python --version

# 安装必需的依赖
pip install maturin numpy matplotlib
```

### 2. 构建和安装 RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# 创建 Python 虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 开发模式构建和安装
maturin develop --release
```

## 基本使用示例

### 1. 张量创建和基本运算

```python
import rustorch
import numpy as np

# 张量创建
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"张量 x:\n{x}")
print(f"形状: {x.shape()}")  # [2, 2]

# 零矩阵和单位矩阵
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"零矩阵:\n{zeros}")
print(f"单位矩阵:\n{ones}")
print(f"对角矩阵:\n{identity}")

# 随机张量
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"正态分布随机:\n{random_normal}")
print(f"均匀分布随机:\n{random_uniform}")

# NumPy 集成
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"从 NumPy 创建:\n{tensor_from_numpy}")

# 转换回 NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"转换回 NumPy:\n{back_to_numpy}")
```

### 2. 算术运算

```python
# 基本算术运算
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 逐元素运算
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b （逐元素）
div_result = a.div(b)  # a / b （逐元素）

print(f"加法:\n{add_result}")
print(f"减法:\n{sub_result}")
print(f"乘法:\n{mul_result}")
print(f"除法:\n{div_result}")

# 标量运算
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"标量加法 (+2):\n{scalar_add}")
print(f"标量乘法 (*3):\n{scalar_mul}")

# 矩阵乘法
matmul_result = a.matmul(b)
print(f"矩阵乘法:\n{matmul_result}")

# 数学函数
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"平方根:\n{sqrt_result}")
print(f"指数函数:\n{exp_result}")
print(f"自然对数:\n{log_result}")
```

### 3. 张量形状操作

```python
# 形状操作示例
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"原始形状: {original.shape()}")  # [2, 4]

# 重塑
reshaped = original.reshape([4, 2])
print(f"重塑为 [4, 2]:\n{reshaped}")

# 转置
transposed = original.transpose(0, 1)
print(f"转置:\n{transposed}")

# 维度添加/移除
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"压缩前: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"压缩后: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"扩展后: {expanded.shape()}")  # [1, 3]
```

### 4. 统计运算

```python
# 统计函数
data = rustorch.randn([3, 4])
print(f"数据:\n{data}")

# 基本统计量
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"均值: {mean_val.item():.4f}")
print(f"求和: {sum_val.item():.4f}")
print(f"标准差: {std_val.item():.4f}")
print(f"方差: {var_val.item():.4f}")
print(f"最大值: {max_val.item():.4f}")
print(f"最小值: {min_val.item():.4f}")

# 按维度统计
row_mean = data.mean(dim=1)  # 每行的均值
col_sum = data.sum(dim=0)    # 每列的和

print(f"行均值: {row_mean}")
print(f"列求和: {col_sum}")
```

## 自动微分基础

### 1. 梯度计算

```python
# 自动微分示例
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"输入张量: {x}")

# 创建变量
var_x = rustorch.autograd.Variable(x)

# 构建计算图
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"输出: {y.data().item()}")

# 反向传播
y.backward()

# 获取梯度
grad = var_x.grad()
print(f"梯度: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. 复杂计算图

```python
# 更复杂的示例
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# 复杂函数: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x)，然后求和

print(f"结果: {z.data().item():.4f}")

# 反向传播
z.backward()
grad = var_x.grad()
print(f"梯度: {grad}")
```

## 神经网络基础

### 1. 简单线性层

```python
# 创建线性层
linear_layer = rustorch.nn.Linear(3, 1)  # 3输入 -> 1输出

# 随机输入
input_data = rustorch.randn([2, 3])  # 批次大小2，特征数3
print(f"输入: {input_data}")

# 前向传播
output = linear_layer.forward(input_data)
print(f"输出: {output}")

# 检查参数
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"权重形状: {weight.shape()}")
print(f"权重: {weight}")
if bias is not None:
    print(f"偏置: {bias}")
```

### 2. 激活函数

```python
# 各种激活函数
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

### 3. 损失函数

```python
# 损失函数使用示例
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# 均方误差
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE 损失: {loss_value.item():.6f}")

# 交叉熵（用于分类）
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # 类别索引

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"交叉熵损失: {ce_loss_value.item():.6f}")
```

## 数据处理

### 1. 数据集和数据加载器

```python
# 创建数据集
import numpy as np

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100个样本，4个特征
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3类分类

# 转换为张量
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# 创建数据集
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"数据集大小: {len(dataset)}")

# 创建数据加载器
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# 从数据加载器获取批次
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # 只显示前3个批次
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"批次 {batch_idx}: 输入形状 {inputs.shape()}, 目标形状 {targets.shape()}")
```

### 2. 数据变换

```python
# 数据变换示例
data = rustorch.randn([10, 10])
print(f"原始数据均值: {data.mean().item():.4f}")
print(f"原始数据标准差: {data.std().item():.4f}")

# 归一化变换
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"归一化数据均值: {normalized_data.mean().item():.4f}")
print(f"归一化数据标准差: {normalized_data.std().item():.4f}")
```

## 完整训练示例

### 线性回归

```python
# 完整的线性回归示例
import numpy as np

# 生成数据
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# 转换为张量
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# 创建数据集和加载器
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# 定义模型
model = rustorch.nn.Linear(1, 1)  # 1输入 -> 1输出

# 损失函数和优化器
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# 训练循环
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
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # 反向传播（简化版）
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"训练轮次 {epoch}: 损失 = {avg_loss:.6f}")

print("训练完成！")

# 最终参数
final_weight = model.weight()
final_bias = model.bias()
print(f"学习到的权重: {final_weight.item():.4f} （真实值: 2.0）")
if final_bias is not None:
    print(f"学习到的偏置: {final_bias.item():.4f} （真实值: 1.0）")
```

## 故障排除

### 常见问题和解决方案

1. **安装问题**
```bash
# 如果找不到 maturin
pip install --upgrade maturin

# 如果 Rust 版本过旧
rustup update

# Python 环境问题
python -m pip install --upgrade pip
```

2. **运行时错误**
```python
# 检查张量形状
print(f"张量形状: {tensor.shape()}")
print(f"张量数据类型: {tensor.dtype()}")

# 在 NumPy 转换中注意数据类型
np_array = np.array(data, dtype=np.float32)  # 明确指定 float32
```

3. **性能优化**
```python
# 使用发布模式构建
# maturin develop --release

# 调整批次大小
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # 更大的批次
```

## 下一步

1. **尝试高级示例**: 查看 `docs/examples/neural_networks/` 中的示例
2. **使用 Keras 风格 API**: `rustorch.training.Model` 用于简化模型构建
3. **可视化功能**: `rustorch.visualization` 用于训练进度可视化
4. **分布式训练**: `rustorch.distributed` 用于并行处理

详细文档：
- [Python API 参考](../zh/python_api_reference.md)
- [概览文档](../zh/python_bindings_overview.md)
- [示例集合](../examples/)