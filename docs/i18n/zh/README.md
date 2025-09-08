# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-968%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**一个采用类PyTorch API、GPU加速和企业级性能的生产就绪Rust深度学习库**

RusTorch是一个功能完整的深度学习库，利用Rust的安全性和性能，提供全面的张量运算、自动微分、神经网络层、Transformer架构、多后端GPU加速（CUDA/Metal/OpenCL）、高级SIMD优化、企业级内存管理、数据验证和质量保证，以及全面的调试和日志系统。

## 📚 文档

- **[完整API参考](API_DOCUMENTATION.md)** - 所有模块的全面API文档
- **[WASM API参考](WASM_API_DOCUMENTATION.md)** - WebAssembly专用API文档
- **[Jupyter指南](jupyter-guide.md)** - Jupyter Notebook使用说明

## ✨ 特性

- 🔥 **全面张量运算**：数学运算、广播、索引和统计，Phase 8 高级工具
- 🤖 **Transformer架构**：完整的Transformer实现，包含多头注意力机制
- 🧮 **矩阵分解**：SVD、QR、特征值分解，兼容PyTorch
- 🧠 **自动微分**：基于磁带的计算图进行梯度计算
- 🚀 **动态执行引擎**：JIT编译和运行时优化
- 🏗️ **神经网络层**：Linear、Conv1d/2d/3d、ConvTranspose、RNN/LSTM/GRU、BatchNorm、Dropout等
- ⚡ **跨平台优化**：SIMD（AVX2/SSE/NEON）、平台特定和硬件感知优化
- 🎮 **GPU集成**：CUDA/Metal/OpenCL支持，自动设备选择
- 🌐 **WebAssembly支持**：完整的浏览器机器学习，包含神经网络层、计算机视觉和实时推理
- 🎮 **WebGPU集成**：Chrome优化的GPU加速，CPU回退确保跨浏览器兼容性
- 📁 **模型格式支持**：Safetensors、ONNX推理、PyTorch状态字典兼容性
- ✅ **生产就绪**：968个测试通过，统一错误处理系统
- 📐 **增强数学函数**：完整的数学函数集（exp、ln、sin、cos、tan、sqrt、abs、pow）
- 🔧 **高级运算符重载**：张量的完整运算符支持，包含标量运算和就地赋值
- 📈 **高级优化器**：SGD、Adam、AdamW、RMSprop、AdaGrad，配备学习率调度器
- 🔍 **数据验证和质量保证**：统计分析、异常检测、一致性检查、实时监控
- 🐛 **全面调试和日志记录**：结构化日志、性能分析、内存跟踪、自动化警报
- 🎯 **Phase 8 张量工具**: 条件操作 (where, masked_select, masked_fill), 索引操作 (gather, scatter, index_select), 统计操作 (topk, kthvalue), 以及高级工具 (unique, histogram)

## 🚀 快速开始

**📓 完整的Jupyter设置指南，请参见 [README_JUPYTER.md](../../README_JUPYTER.md)**

### Python Jupyter Lab演示

📓 **[完整Jupyter设置指南](../../README_JUPYTER.md)** | **[Jupyter指南](jupyter-guide.md)**

#### 标准CPU演示
一键启动Jupyter Lab中的RusTorch：

```bash
./start_jupyter.sh
```

#### WebGPU加速演示
启动支持WebGPU的RusTorch进行基于浏览器的GPU加速：

```bash
./start_jupyter_webgpu.sh
```

两个脚本都会：
- 📦 自动创建虚拟环境
- 🔧 构建RusTorch Python绑定
- 🚀 启动Jupyter Lab并打开演示笔记本
- 📍 打开准备运行的演示笔记本

**WebGPU特性：**
- 🌐 基于浏览器的GPU加速
- ⚡ 浏览器中的高性能矩阵运算
- 🔄 GPU不可用时自动回退到CPU
- 🎯 Chrome/Edge优化（推荐浏览器）

#### Jupyter的Rust内核
在Jupyter中启动原生Rust内核（evcxr_jupyter）：

```bash
./quick_start_rust_kernel.sh
```

这将：
- 🦀 安装evcxr_jupyter Rust内核
- 📓 创建Rust内核演示笔记本
- 🚀 启动支持原生Rust的Jupyter
- 📍 直接在Rust中进行张量操作

### 安装

将此添加到您的`Cargo.toml`：

```toml
[dependencies]
rustorch = "0.5.10"

# 可选特性
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # 线性代数运算（SVD、QR、特征值）
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # 浏览器机器学习的WebAssembly支持
webgpu = ["rustorch/webgpu"]            # Chrome优化的WebGPU加速

# 禁用linalg特性（避免OpenBLAS/LAPACK依赖）：
rustorch = { version = "0.5.10", default-features = false }
```

### 基本用法

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // 创建张量
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // 使用运算符重载的基本运算
    let c = &a + &b;  // 逐元素加法
    let d = &a - &b;  // 逐元素减法
    let e = &a * &b;  // 逐元素乘法
    let f = &a / &b;  // 逐元素除法
    
    // 标量运算
    let g = &a + 10.0;  // 所有元素加标量
    let h = &a * 2.0;   // 乘以标量
    
    // 数学函数
    let exp_result = a.exp();   // 指数函数
    let ln_result = a.ln();     // 自然对数
    let sin_result = a.sin();   // 正弦函数
    let sqrt_result = a.sqrt(); // 平方根
    
    // 矩阵运算
    let matmul_result = a.matmul(&b);  // 矩阵乘法
    
    // 线性代数运算（需要linalg特性）
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // SVD分解
        let qr_result = a.qr();         // QR分解
        let eig_result = a.eigh();      // 特征值分解
    }
    
    // 高级优化器与学习率调度
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // 5个轮次预热到0.1
    
    println!("形状：{:?}", c.shape());
    println!("结果：{:?}", c.as_slice());
}
```

### WebAssembly用法

对于基于浏览器的机器学习应用：

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // 神经网络层
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // 增强数学函数
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // 统计分布
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // 训练优化器
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // 学习率，动量
    
    // 图像处理
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // 前向传播
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('浏览器机器学习预测：', predictions);
}
```

## 📚 文档

- **[快速入门](../getting-started.md)** - 基本用法和示例
- **[特性](../features.md)** - 完整特性列表和规范
- **[性能](../performance.md)** - 基准测试和优化详情
- **[Jupyter WASM指南](jupyter-guide.md)** - Jupyter Notebook逐步设置

### WebAssembly和浏览器机器学习
- **[WebAssembly指南](../WASM_GUIDE.md)** - 完整的WASM集成和API参考
- **[WebGPU集成](../WEBGPU_INTEGRATION.md)** - Chrome优化的GPU加速

### 生产和运维
- **[GPU加速指南](../GPU_ACCELERATION_GUIDE.md)** - GPU设置和使用
- **[生产指南](../PRODUCTION_GUIDE.md)** - 部署和扩展

## 📊 性能

**最新基准测试结果：**

| 操作 | 性能 | 详情 |
|-----------|-------------|---------|
| **SVD分解** | ~1ms（8x8矩阵） | ✅ 基于LAPACK |
| **QR分解** | ~24μs（8x8矩阵） | ✅ 快速分解 |
| **特征值** | ~165μs（8x8矩阵） | ✅ 对称矩阵 |
| **复数FFT** | 10-312μs（8-64样本） | ✅ Cooley-Tukey优化 |
| **神经网络** | 1-7s训练 | ✅ Boston housing演示 |
| **激活函数** | <1μs | ✅ ReLU、Sigmoid、Tanh |

## 🧪 测试

**968个测试通过** - 具有统一错误处理系统的生产就绪质量保证。

```bash
# 运行所有测试
cargo test --no-default-features

# 运行包含线性代数特性的测试
cargo test --features linalg
```

## 🤝 贡献

我们欢迎贡献！特别需要帮助的领域：

- **🎯 特殊函数精度**：提高数值精度
- **⚡ 性能优化**：SIMD改进、GPU优化
- **🧪 测试**：更全面的测试用例
- **📚 文档**：示例、教程、改进
- **🌐 平台支持**：WebAssembly、移动平台

## 许可证

采用以下任一许可证：

 * Apache许可证2.0版本（[LICENSE-APACHE](../../LICENSE-APACHE)或http://www.apache.org/licenses/LICENSE-2.0）
 * MIT许可证（[LICENSE-MIT](../../LICENSE-MIT)或http://opensource.org/licenses/MIT）

您可以选择其中一个。