# RusTorch WASM Jupyter Notebook 指南

为初学者设计的在Jupyter Notebook中轻松使用RusTorch WASM的逐步指南。

## 📚 目录

1. [要求](#要求)
2. [安装说明](#安装说明)
3. [基本用法](#基本用法)
4. [实际示例](#实际示例)
5. [故障排除](#故障排除)
6. [常见问题](#常见问题)

## 要求

### 最低要求
- **Python 3.8+**
- **Jupyter Notebook** 或 **Jupyter Lab**
- **Node.js 16+**（用于WASM构建）
- **Rust**（最新稳定版本）
- **wasm-pack**（将Rust代码转换为WASM）

### 推荐环境
- 内存：8GB或更多
- 浏览器：Chrome、Firefox、Safari的最新版本
- 操作系统：Windows 10/11、macOS 10.15+、Ubuntu 20.04+

## 安装说明

### 🚀 快速开始（推荐）

**最简单的方法**：一个命令启动Jupyter Lab
```bash
./start_jupyter.sh
```

此脚本会自动：
- 创建并激活虚拟环境
- 安装依赖项（numpy、jupyter、matplotlib）
- 构建RusTorch Python绑定
- 启动Jupyter Lab并打开演示笔记本

### 手动安装

#### 步骤1：安装基础工具

```bash
# 检查Python版本
python --version

# 安装Jupyter Lab
pip install jupyterlab

# 安装Node.js（macOS使用Homebrew）
brew install node

# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 安装wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### 步骤2：构建RusTorch WASM

```bash
# 克隆项目
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# 添加WASM目标
rustup target add wasm32-unknown-unknown

# 使用wasm-pack构建
wasm-pack build --target web --out-dir pkg
```

#### 步骤3：启动Jupyter

```bash
# 启动Jupyter Lab
jupyter lab
```

## 基本用法

### 创建张量

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 一维张量
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('1D张量：', vec.to_array());
    
    // 二维张量（矩阵）
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // 形状：2行3列
    );
    console.log('2D张量形状：', matrix.shape());
});
```

### 基本运算

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // 加法
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // 矩阵乘法
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### 自动微分

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 创建具有梯度跟踪的张量
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // 计算：y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // 反向传播
    y.backward();
    
    // 获取梯度（dy/dx = 2x + 3 = 7，当x=2时）
    console.log('梯度：', x.grad().to_array());
});
```

## 实际示例

### 线性回归

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // 准备数据
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // 初始化参数
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // 训练循环
    for (let epoch = 0; epoch < 100; epoch++) {
        // 预测：y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // 损失：MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // 计算梯度
        loss.backward();
        
        // 更新参数
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // 重置梯度
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`轮次 ${epoch}：损失 = ${loss.item()}`);
        }
    }
    
    console.log(`最终 w：${w.item()}，b：${b.item()}`);
});
```

## 故障排除

### 🚀 加速Rust内核（推荐）
如果初始执行较慢，启用缓存可显著提高性能：

```bash
# 创建缓存目录
mkdir -p ~/.config/evcxr

# 启用500MB缓存
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**效果：**
- 第一次：正常编译时间
- 后续运行：无需重新编译依赖项（快数倍）
- `rustorch`库在首次使用后也会被缓存

### 常见错误

#### "RusTorch is not defined" 错误
**解决方案**：始终等待RusTorchReady
```javascript
window.RusTorchReady.then((rt) => {
    // 在这里使用RusTorch
});
```

#### "Failed to load WASM module" 错误
**解决方案**：
1. 验证`pkg`目录是否正确生成
2. 检查浏览器控制台中的错误消息
3. 确保WASM文件路径正确

#### 内存不足错误
**解决方案**：
```javascript
// 显式释放内存
tensor.free();

// 使用更小的批次大小
const batchSize = 32;  // 使用32而不是1000
```

### 性能提示

1. **使用批处理**：使用批处理而不是循环处理数据
2. **内存管理**：显式释放大型张量
3. **合适的数据类型**：当不需要高精度时使用f32

## 常见问题

### 问：我可以在Google Colab中使用这个吗？
**答**：可以，上传WASM文件并使用自定义JavaScript加载器。

### 问：我可以混合Python和WASM代码吗？
**答**：可以，使用IPython.display.Javascript在Python和JavaScript之间传递数据。

### 问：如何调试？
**答**：使用浏览器开发者工具（F12）并检查Console选项卡中的错误。

### 问：有哪些高级功能可用？
**答**：目前支持基本张量运算、自动微分和简单神经网络。CNN和RNN层正在计划中。

## 后续步骤

1. 📖 [详细的RusTorch WASM API](../wasm.md)
2. 🔬 [高级示例](../examples/)
3. 🚀 [性能优化指南](../wasm-memory-optimization.md)

## 社区和支持

- GitHub：[RusTorch仓库](https://github.com/JunSuzukiJapan/rustorch)
- Issues：在GitHub上报告错误和请求功能

---

使用RusTorch WASM愉快学习！🦀🔥📓