# RusTorch API 文档

## 📚 完整API参考

本文档为RusTorch v0.5.15提供全面的API文档，按模块和功能组织。包含统一错误处理，使用`RusTorchError`和`RusTorchResult<T>`在所有1060+测试中提供一致的错误管理。**第8阶段已完成**添加高级张量工具，包括条件操作、索引和统计函数。**第9阶段已完成**引入全面序列化系统，包含模型保存/加载、JIT编译和多格式支持（包括PyTorch兼容性）。

## 🏗️ 核心架构

### 模块结构

```
rustorch/
├── tensor/              # 核心张量操作和数据结构
├── nn/                  # 神经网络层和函数
├── autograd/            # 自动微分引擎
├── optim/               # 优化器和学习率调度器
├── special/             # 特殊数学函数
├── distributions/       # 统计分布
├── vision/              # 计算机视觉变换
├── linalg/              # 线性代数操作 (BLAS/LAPACK)
├── gpu/                 # GPU加速 (CUDA/Metal/OpenCL/WebGPU)
├── sparse/              # 稀疏张量操作和剪枝 (第12阶段)
├── serialization/       # 模型序列化和JIT编译 (第9阶段)
└── wasm/                # WebAssembly绑定 (见 [WASM API文档](WASM_API_DOCUMENTATION.md))
```

## 📊 张量模块

### 核心张量创建

```rust
use rustorch::tensor::Tensor;

// 基础创建
let tensor = Tensor::new(vec![2, 3]);               // 基于形状的创建
let tensor = Tensor::from_vec(data, vec![2, 3]);    // 从数据向量创建
let tensor = Tensor::zeros(vec![10, 10]);           // 零填充张量
let tensor = Tensor::ones(vec![5, 5]);              // 一填充张量
let tensor = Tensor::randn(vec![3, 3]);             // 随机正态分布
let tensor = Tensor::rand(vec![3, 3]);              // 随机均匀分布 [0,1)
let tensor = Tensor::eye(5);                        // 单位矩阵
let tensor = Tensor::full(vec![2, 2], 3.14);       // 用特定值填充
let tensor = Tensor::arange(0.0, 10.0, 1.0);       // 范围张量
let tensor = Tensor::linspace(0.0, 1.0, 100);      // 线性间距
```

### 张量操作

```rust
// 算术运算
let result = a.add(&b);                             // 逐元素加法
let result = a.sub(&b);                             // 逐元素减法
let result = a.mul(&b);                             // 逐元素乘法
let result = a.div(&b);                             // 逐元素除法
let result = a.pow(&b);                             // 逐元素幂运算
let result = a.rem(&b);                             // 逐元素取余

// 矩阵运算
let result = a.matmul(&b);                          // 矩阵乘法
let result = a.transpose();                         // 矩阵转置
let result = a.dot(&b);                             // 点积

// 数学函数
let result = tensor.exp();                          // 指数
let result = tensor.ln();                           // 自然对数
let result = tensor.log10();                        // 以10为底对数
let result = tensor.sqrt();                         // 平方根
let result = tensor.abs();                          // 绝对值
let result = tensor.sin();                          // 正弦函数
let result = tensor.cos();                          // 余弦函数
let result = tensor.tan();                          // 正切函数
let result = tensor.asin();                         // 反正弦
let result = tensor.acos();                         // 反余弦
let result = tensor.atan();                         // 反正切
let result = tensor.sinh();                         // 双曲正弦
let result = tensor.cosh();                         // 双曲余弦
let result = tensor.tanh();                         // 双曲正切
let result = tensor.floor();                        // 向下取整
let result = tensor.ceil();                         // 向上取整
let result = tensor.round();                        // 四舍五入
let result = tensor.sign();                         // 符号函数
let result = tensor.max();                          // 最大值
let result = tensor.min();                          // 最小值
let result = tensor.sum();                          // 所有元素求和
let result = tensor.mean();                         // 平均值
let result = tensor.std();                          // 标准差
let result = tensor.var();                          // 方差

// 形状操作
let result = tensor.reshape(vec![6, 4]);            // 重塑张量
let result = tensor.squeeze();                      // 去除大小为1的维度
let result = tensor.unsqueeze(1);                   // 在索引处添加维度
let result = tensor.permute(vec![1, 0, 2]);         // 排列维度
let result = tensor.expand(vec![10, 10, 5]);        // 扩展张量维度
```

## 🧠 神经网络(nn)模块

### 基础层

```rust
use rustorch::nn::{Linear, Conv2d, BatchNorm1d, Dropout};

// 线性层
let linear = Linear::new(784, 256)?;                // 输入784，输出256
let output = linear.forward(&input)?;

// 卷积层
let conv = Conv2d::new(3, 64, 3, None, Some(1))?; // in_channels=3, out_channels=64, kernel_size=3
let output = conv.forward(&input)?;

// 批量归一化
let bn = BatchNorm1d::new(256)?;
let normalized = bn.forward(&input)?;

// Dropout
let dropout = Dropout::new(0.5)?;
let output = dropout.forward(&input, true)?;       // training=true
```

### 激活函数

```rust
use rustorch::nn::{ReLU, Sigmoid, Tanh, LeakyReLU, ELU, GELU};

// 基础激活函数
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();

// 参数化激活函数
let leaky_relu = LeakyReLU::new(0.01)?;
let elu = ELU::new(1.0)?;
let gelu = GELU::new();

// 使用示例
let activated = relu.forward(&input)?;
```

## 🚀 GPU加速模块

### 设备管理

```rust
use rustorch::gpu::{Device, get_device_count, set_device};

// 检查可用设备
let device_count = get_device_count()?;
let device = Device::best_available()?;            // 最佳设备选择

// 设备配置
set_device(&device)?;

// 将张量移至GPU
let gpu_tensor = tensor.to_device(&device)?;
```

### CUDA操作

```rust
#[cfg(feature = "cuda")]
use rustorch::gpu::cuda::{CudaDevice, memory_stats};

// CUDA设备操作
let cuda_device = CudaDevice::new(0)?;              // 使用GPU 0
let stats = memory_stats(0)?;                      // 内存统计
println!("已用内存: {} MB", stats.used_memory / (1024 * 1024));
```

## 🎯 优化器(Optim)模块

### 基础优化器

```rust
use rustorch::optim::{Adam, SGD, RMSprop, AdamW};

// Adam优化器
let mut optimizer = Adam::new(vec![x.clone(), y.clone()], 0.001, 0.9, 0.999, 1e-8)?;

// SGD优化器
let mut sgd = SGD::new(vec![x.clone()], 0.01, 0.9, 1e-4)?;

// 优化步骤
optimizer.zero_grad()?;
// ... 前向计算和反向传播 ...
optimizer.step()?;
```

## 📖 使用示例

### 线性回归

```rust
use rustorch::{tensor::Tensor, nn::Linear, optim::Adam, autograd::Variable};

// 数据准备
let x = Variable::new(Tensor::randn(vec![100, 1]), false)?;
let y = Variable::new(Tensor::randn(vec![100, 1]), false)?;

// 模型定义
let mut model = Linear::new(1, 1)?;
let mut optimizer = Adam::new(model.parameters(), 0.001, 0.9, 0.999, 1e-8)?;

// 训练循环
for epoch in 0..1000 {
    optimizer.zero_grad()?;
    let pred = model.forward(&x)?;
    let loss = (pred - &y).pow(&Tensor::from(2.0))?.mean()?;
    backward(&loss, true)?;
    optimizer.step()?;
    
    if epoch % 100 == 0 {
        println!("轮次 {}: 损失 = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

## ⚠️ 已知限制

1. **GPU内存限制**: 大型张量(>8GB)需要显式内存管理
2. **WebAssembly限制**: 某些BLAS操作在WASM环境中不可用
3. **分布式学习**: NCCL后端仅在Linux上支持
4. **Metal限制**: 某些高级操作仅在CUDA后端可用

## 🔗 相关链接

- [主README](../README.md)
- [WASM API文档](WASM_API_DOCUMENTATION.md)
- [Jupyter指南](jupyter-guide.md)
- [GitHub仓库](https://github.com/JunSuzukiJapan/RusTorch)
- [Crates.io包](https://crates.io/crates/rustorch)

---

**最后更新**: v0.5.15 | **许可证**: MIT | **作者**: Jun Suzuki